from __future__ import annotations

from dataclasses import dataclass
import math
import os
import numpy as np

import torch
from torch import Tensor

torch.set_num_threads(os.cpu_count() or 1)
torch.set_num_interop_threads(1)


@dataclass(frozen=True, slots=True)
class BuiltClusterLevel:
    child_to_parent: Tensor
    parent_to_child_ids: Tensor
    parent_sizes: Tensor
    exposed_vectors: Tensor


@dataclass(frozen=True, slots=True)
class BuiltHierarchicalFlashHead:
    levels: tuple[BuiltClusterLevel, ...]
    router_matrix: Tensor


def l2_normalize(x: Tensor, dim: int) -> Tensor:
    return x / x.norm(dim=dim, keepdim=True).clamp_min(1e-12)


def build_near_equal_cluster_capacities(
    num_children: int,
    num_parents: int,
    *,
    generator: torch.Generator | None = None,
    device: torch.device | None = None,
) -> Tensor:
    if num_parents < 1:
        raise ValueError("num_parents must be >= 1")
    if num_parents > num_children:
        raise ValueError("num_parents cannot exceed num_children")

    base = num_children // num_parents
    remainder = num_children % num_parents

    capacities = torch.full(
        (num_parents,),
        fill_value=base,
        dtype=torch.long,
        device=device,
    )

    if remainder > 0:
        plus_one_ids = torch.randperm(
            num_parents,
            generator=generator,
            device=device,
        )[:remainder]
        capacities[plus_one_ids] += 1

    return capacities


def build_parent_to_child_ids_padded(
    child_to_parent: Tensor,
    *,
    num_parents: int,
) -> Tensor:
    if child_to_parent.ndim != 1:
        raise ValueError("child_to_parent must have shape [num_children]")

    parent_sizes = torch.bincount(child_to_parent, minlength=num_parents)
    max_children = int(parent_sizes.max().item())

    sorted_child_ids = torch.argsort(child_to_parent, stable=True)
    sorted_parent_ids = child_to_parent[sorted_child_ids]

    parent_to_child_ids = torch.full(
        (num_parents, max_children),
        fill_value=-1,
        dtype=torch.long,
        device=child_to_parent.device,
    )

    offsets = torch.zeros(num_parents + 1, dtype=torch.long, device=child_to_parent.device)
    offsets[1:] = parent_sizes.cumsum(dim=0)

    for parent_id in range(num_parents):
        start = int(offsets[parent_id].item())
        end = int(offsets[parent_id + 1].item())
        size = end - start

        if size == 0:
            continue

        child_ids = sorted_child_ids[start:end]
        if not torch.all(sorted_parent_ids[start:end] == parent_id):
            raise RuntimeError("Grouping mismatch while building parent_to_child_ids")

        parent_to_child_ids[parent_id, :size] = child_ids

    return parent_to_child_ids


@torch.no_grad()
def collect_top_parent_choices(
    vectors: Tensor,
    centroids: Tensor,
    *,
    batch_size: int,
    top_m: int,
) -> tuple[Tensor, Tensor, Tensor]:
    num_children = vectors.shape[0]
    num_parents = centroids.shape[0]
    actual_top_m = min(top_m, num_parents)

    top_parent_ids = torch.empty(
        (num_children, actual_top_m),
        dtype=torch.long,
        device=vectors.device,
    )
    top_parent_scores = torch.empty(
        (num_children, actual_top_m),
        dtype=vectors.dtype,
        device=vectors.device,
    )
    confidence = torch.empty(
        num_children,
        dtype=vectors.dtype,
        device=vectors.device,
    )

    for start in range(0, num_children, batch_size):
        end = min(start + batch_size, num_children)
        batch = vectors[start:end]
        scores = batch @ centroids.transpose(0, 1)

        batch_top_scores, batch_top_ids = torch.topk(scores, k=actual_top_m, dim=-1)
        top_parent_ids[start:end] = batch_top_ids
        top_parent_scores[start:end] = batch_top_scores

        if actual_top_m >= 2:
            confidence[start:end] = batch_top_scores[:, 0] - batch_top_scores[:, 1]
        else:
            confidence[start:end] = 0.0

    return top_parent_ids, top_parent_scores, confidence


@torch.no_grad()
def build_capacity_constrained_assignment(
    vectors: Tensor,
    centroids: Tensor,
    capacities: Tensor,
    *,
    batch_size: int,
    top_m: int = 16,
    verbose: bool = True,
    level_name: str = "level",
) -> tuple[Tensor, int]:
    if vectors.ndim != 2:
        raise ValueError("vectors must have shape [num_children, hidden_size]")
    if centroids.ndim != 2:
        raise ValueError("centroids must have shape [num_parents, hidden_size]")
    if capacities.ndim != 1:
        raise ValueError("capacities must have shape [num_parents]")
    if vectors.shape[1] != centroids.shape[1]:
        raise ValueError("vectors and centroids must have the same hidden size")
    if centroids.shape[0] != capacities.shape[0]:
        raise ValueError("capacities length must match num_parents")
    if int(capacities.sum().item()) != int(vectors.shape[0]):
        raise ValueError("capacities must sum to num_children")

    num_children = vectors.shape[0]

    if verbose:
        print(f"[{level_name}] collecting top-{top_m} parent choices")

    top_parent_ids, _, confidence = collect_top_parent_choices(
        vectors,
        centroids,
        batch_size=batch_size,
        top_m=top_m,
    )

    token_order = confidence.argsort(descending=True)

    top_parent_ids_np = top_parent_ids.cpu().numpy()
    token_order_np = token_order.cpu().numpy()
    remaining_np = capacities.cpu().numpy().copy()
    assignment_np = np.full(num_children, fill_value=-1, dtype=np.int64)

    fallback_count = 0

    if verbose:
        print(f"[{level_name}] assigning {num_children} items with balanced capacities")

    for idx, child_id in enumerate(token_order_np):
        assigned = False

        for parent_id in top_parent_ids_np[child_id]:
            if remaining_np[parent_id] > 0:
                assignment_np[child_id] = parent_id
                remaining_np[parent_id] -= 1
                assigned = True
                break

        if not assigned:
            fallback_count += 1

            available_parent_ids_np = np.flatnonzero(remaining_np > 0)
            available_parent_ids = torch.from_numpy(available_parent_ids_np).to(vectors.device)

            child_vector = vectors[child_id]
            fallback_scores = child_vector @ centroids[available_parent_ids].transpose(0, 1)
            best_local_idx = int(fallback_scores.argmax().item())
            best_parent_id = int(available_parent_ids_np[best_local_idx])

            assignment_np[child_id] = best_parent_id
            remaining_np[best_parent_id] -= 1

        if verbose and (idx + 1) % 20000 == 0:
            print(
                f"[{level_name}] assigned {idx + 1}/{num_children} items "
                f"(fallbacks so far: {fallback_count})"
            )

    if (assignment_np < 0).any():
        raise RuntimeError("Some items were not assigned to a parent cluster")

    if np.any(remaining_np != 0):
        raise RuntimeError("Some parent clusters did not fill to their required capacities")

    child_to_parent = torch.from_numpy(assignment_np).to(vectors.device, dtype=torch.long)
    return child_to_parent, fallback_count


@torch.no_grad()
def cluster_vectors(
    vectors: Tensor,
    *,
    children_per_parent: int = 20,
    num_iters: int = 20,
    normalize_vectors: bool = True,
    seed: int = 0,
    batch_size: int = 4096,
    top_m: int = 64,
    verbose: bool = True,
    level_name: str = "level",
) -> BuiltClusterLevel:
    if vectors.ndim != 2:
        raise ValueError("vectors must have shape [num_children, hidden_size]")
    if children_per_parent < 1:
        raise ValueError("children_per_parent must be >= 1")
    if num_iters < 1:
        raise ValueError("num_iters must be >= 1")

    device = torch.device("cpu")
    vectors = vectors.to(device=device, dtype=torch.float32)

    if normalize_vectors:
        vectors = l2_normalize(vectors, dim=-1)

    num_children, hidden_size = vectors.shape
    num_parents = max(1, math.ceil(num_children / children_per_parent))

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    capacities = build_near_equal_cluster_capacities(
        num_children=num_children,
        num_parents=num_parents,
        generator=generator,
        device=device,
    )

    if verbose:
        print()
        print(
            f"[{level_name}] clustering {num_children} items into {num_parents} parents "
            f"with balanced capacities"
        )
        print(
            f"[{level_name}] target sizes: "
            f"min={int(capacities.min().item())}, "
            f"mean={float(capacities.float().mean().item()):.2f}, "
            f"max={int(capacities.max().item())}"
        )

    init_ids = torch.randperm(num_children, generator=generator, device=device)[:num_parents]
    centroids = vectors[init_ids].clone()

    for iter_idx in range(num_iters):
        child_to_parent, fallback_count = build_capacity_constrained_assignment(
            vectors,
            centroids,
            capacities,
            batch_size=batch_size,
            top_m=top_m,
            verbose=verbose,
            level_name=f"{level_name} iter {iter_idx + 1}/{num_iters}",
        )

        sums = torch.zeros((num_parents, hidden_size), dtype=vectors.dtype, device=device)
        sums.index_add_(0, child_to_parent, vectors)

        counts = torch.bincount(child_to_parent, minlength=num_parents)

        if not torch.equal(counts.cpu(), capacities.cpu()):
            raise RuntimeError("Balanced assignment failed: counts do not match capacities")

        centroids = sums / counts.unsqueeze(-1).to(vectors.dtype)
        if normalize_vectors:
            centroids = l2_normalize(centroids, dim=-1)

        if verbose:
            print(
                f"[{level_name}] finished iter {iter_idx + 1}/{num_iters} "
                f"(fallback assignments: {fallback_count})"
            )

    child_to_parent, fallback_count = build_capacity_constrained_assignment(
        vectors,
        centroids,
        capacities,
        batch_size=batch_size,
        top_m=top_m,
        verbose=verbose,
        level_name=f"{level_name} final",
    )

    sums = torch.zeros((num_parents, hidden_size), dtype=vectors.dtype, device=device)
    sums.index_add_(0, child_to_parent, vectors)

    parent_sizes = torch.bincount(child_to_parent, minlength=num_parents)
    if not torch.equal(parent_sizes.cpu(), capacities.cpu()):
        raise RuntimeError("Final balanced assignment failed: sizes do not match capacities")

    exposed_vectors = sums / parent_sizes.unsqueeze(-1).to(vectors.dtype)
    if normalize_vectors:
        exposed_vectors = l2_normalize(exposed_vectors, dim=-1)

    parent_to_child_ids = build_parent_to_child_ids_padded(
        child_to_parent=child_to_parent,
        num_parents=num_parents,
    )

    if verbose:
        print(
            f"[{level_name}] finished with exact sizes: "
            f"min={int(parent_sizes.min().item())}, "
            f"mean={float(parent_sizes.float().mean().item()):.2f}, "
            f"max={int(parent_sizes.max().item())}"
        )
        print(
            f"[{level_name}] parent_to_child_ids shape={tuple(parent_to_child_ids.shape)} "
            f"(this should now be tight)"
        )

    return BuiltClusterLevel(
        child_to_parent=child_to_parent.cpu(),
        parent_to_child_ids=parent_to_child_ids.cpu(),
        parent_sizes=parent_sizes.cpu(),
        exposed_vectors=exposed_vectors.cpu(),
    )


@torch.no_grad()
def build_hierarchical_flashhead(
    lm_head_vector_table: Tensor,
    *,
    num_cluster_levels: int = 2,
    children_per_parent: int = 20,
    num_iters: int = 20,
    normalize_vectors: bool = True,
    seed: int = 0,
    batch_size: int = 4096,
    top_m: int = 16,
    verbose: bool = True,
) -> BuiltHierarchicalFlashHead:
    if lm_head_vector_table.ndim != 2:
        raise ValueError("lm_head_vector_table must have shape [vocab_size, hidden_size]")
    if num_cluster_levels < 1:
        raise ValueError("num_cluster_levels must be >= 1")

    current_vectors = lm_head_vector_table.detach().float().cpu()
    built_levels: list[BuiltClusterLevel] = []

    if verbose:
        print()
        print(f"building hierarchical flashhead with {num_cluster_levels} cluster level(s)")

    for level_idx in range(num_cluster_levels):
        if current_vectors.shape[0] <= 1:
            break

        level = cluster_vectors(
            current_vectors,
            children_per_parent=children_per_parent,
            num_iters=num_iters,
            normalize_vectors=normalize_vectors,
            seed=seed + level_idx,
            batch_size=batch_size,
            top_m=top_m,
            verbose=verbose,
            level_name=f"level {level_idx}",
        )
        built_levels.append(level)
        current_vectors = level.exposed_vectors

    if not built_levels:
        raise RuntimeError("No hierarchy was built")

    built = BuiltHierarchicalFlashHead(
        levels=tuple(built_levels),
        router_matrix=built_levels[-1].exposed_vectors.clone(),
    )

    if verbose:
        print()
        print("finished hierarchy:")
        for level_idx, level in enumerate(built.levels):
            print(
                f"  level {level_idx}: {level.exposed_vectors.shape[0]} nodes, "
                f"router-to-children shape={tuple(level.parent_to_child_ids.shape)}"
            )
        print(f"  router shape: {tuple(built.router_matrix.shape)}")

    return built

def save_hierarchical_flashhead(path: str, built: BuiltHierarchicalFlashHead) -> None:
    payload = {
        "levels": [
            {
                "child_to_parent": level.child_to_parent,
                "parent_to_child_ids": level.parent_to_child_ids,
                "parent_sizes": level.parent_sizes,
                "exposed_vectors": level.exposed_vectors,
            }
            for level in built.levels
        ],
        "router_matrix": built.router_matrix,
    }
    torch.save(payload, path)


def load_hierarchical_flashhead(path: str) -> BuiltHierarchicalFlashHead:
    payload = torch.load(path, map_location="cpu")

    levels = tuple(
        BuiltClusterLevel(
            child_to_parent=level_payload["child_to_parent"],
            parent_to_child_ids=level_payload["parent_to_child_ids"],
            parent_sizes=level_payload["parent_sizes"],
            exposed_vectors=level_payload["exposed_vectors"],
        )
        for level_payload in payload["levels"]
    )

    return BuiltHierarchicalFlashHead(
        levels=levels,
        router_matrix=payload["router_matrix"],
    )