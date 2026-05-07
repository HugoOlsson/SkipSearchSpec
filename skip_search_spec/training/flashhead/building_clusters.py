from collections.abc import Callable
from dataclasses import dataclass
from torch import Tensor, nn
import torch
import os

torch.set_num_threads(os.cpu_count() or 1)
torch.set_num_interop_threads(1)


@dataclass(frozen=True, slots=True)
class BuiltFlashHeadClusters:
    token_to_cluster_mapping: Tensor  # [vocab_size]
    cluster_to_token_ids: Tensor      # [num_clusters, cluster_size], no padding
    centroids: Tensor                 # [num_clusters, hidden_size]
    cluster_sizes: Tensor             # [num_clusters], all equal


@dataclass(frozen=True, slots=True)
class FlashHeadIndex:
    # [hidden_size, num_clusters], transposed centroids for fast cluster scoring
    centroids_t: Tensor          
    # [num_clusters, cluster_size], token ids contained in each cluster          
    cluster_to_token_ids: Tensor     
    # [num_clusters, cluster_size, hidden_size], LM-head rows grouped by cluster      
    clustered_lm_head: Tensor           
    # [num_clusters, cluster_size] or None, matching LM-head bias grouped by cluster   
    clustered_lm_head_bias: Tensor | None  
    # number of tokens in each cluster
    cluster_size: int          
    # total number of clusters           
    num_clusters: int    
    # total number of tokens in the vocabulary                 
    vocab_size: int                        


@dataclass(frozen=True, slots=True)
class ClusterQualityMetrics:
    mean_assigned_similarity: float
    p05_assigned_similarity: float
    mean_margin_to_best_other: float
    p05_margin_to_best_other: float
    fraction_assigned_to_nearest_centroid: float
    mean_abs_capacity_error: float | None
    max_abs_capacity_error: int | None
    min_cluster_size: int
    max_cluster_size: int
    cluster_size_std: float

    


def l2_normalize(x: Tensor, dim: int) -> Tensor:
    return x / x.norm(dim=dim, keepdim=True).clamp_min(1e-12)


def build_equal_cluster_capacities(
    vocab_size: int,
    num_clusters: int,
    *,
    device: torch.device | None = None,
) -> Tensor:
    if num_clusters < 1:
        raise ValueError("num_clusters must be >= 1")
    if num_clusters > vocab_size:
        raise ValueError("num_clusters cannot exceed vocab_size")
    if vocab_size % num_clusters != 0:
        raise ValueError(
            "Strictly equal-sized clusters require vocab_size % num_clusters == 0, "
            f"but got vocab_size={vocab_size}, num_clusters={num_clusters}, "
            f"remainder={vocab_size % num_clusters}."
        )

    cluster_size = vocab_size // num_clusters

    return torch.full(
        (num_clusters,),
        fill_value=cluster_size,
        dtype=torch.long,
        device=device,
    )


@torch.no_grad()
def assign_to_nearest_centroid(
    vectors: Tensor,
    centroids: Tensor,
) -> Tensor:
    scores = vectors @ centroids.transpose(0, 1)
    return scores.argmax(dim=-1)


@torch.no_grad()
def find_tokens_to_evict_by_regret(
    scores: Tensor,
    token_to_cluster_mapping: Tensor,
    cluster_sizes: Tensor,
    cluster_capacities: Tensor,
) -> Tensor:
    overloaded_cluster_ids = torch.nonzero(
        cluster_sizes > cluster_capacities,
        as_tuple=False,
    ).flatten()

    if overloaded_cluster_ids.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=scores.device)

    if scores.shape[1] < 2:
        return torch.empty(0, dtype=torch.long, device=scores.device)

    top2_scores, _ = scores.topk(k=2, dim=-1)
    regrets = top2_scores[:, 0] - top2_scores[:, 1]

    evicted_token_ids_parts: list[Tensor] = []

    for cluster_id in overloaded_cluster_ids.tolist():
        overflow = int((cluster_sizes[cluster_id] - cluster_capacities[cluster_id]).item())

        member_token_ids = torch.nonzero(
            token_to_cluster_mapping == cluster_id,
            as_tuple=False,
        ).flatten()

        member_regrets = regrets[member_token_ids]

        evicted_member_ids = member_token_ids[
            torch.argsort(member_regrets)[:overflow]
        ]

        evicted_token_ids_parts.append(evicted_member_ids)

    if not evicted_token_ids_parts:
        return torch.empty(0, dtype=torch.long, device=scores.device)

    evicted_token_ids = torch.cat(evicted_token_ids_parts, dim=0)

    # Reassign the easiest-to-move tokens first.
    evicted_token_ids = evicted_token_ids[
        torch.argsort(regrets[evicted_token_ids])
    ]

    return evicted_token_ids


@torch.no_grad()
def assign_with_greedy_capacity_rebalancing(
    vectors: Tensor,
    centroids: Tensor,
    cluster_capacities: Tensor,
) -> tuple[Tensor, Tensor]:
    scores = vectors @ centroids.transpose(0, 1)

    num_clusters = centroids.shape[0]

    if num_clusters == 1:
        token_to_cluster_mapping = torch.zeros(
            vectors.shape[0],
            dtype=torch.long,
            device=vectors.device,
        )
        cluster_sizes = torch.tensor(
            [vectors.shape[0]],
            dtype=torch.long,
            device=vectors.device,
        )
        return token_to_cluster_mapping, cluster_sizes

    token_to_cluster_mapping = scores.argmax(dim=-1)

    cluster_sizes = torch.bincount(
        token_to_cluster_mapping,
        minlength=num_clusters,
    )

    evicted_token_ids = find_tokens_to_evict_by_regret(
        scores=scores,
        token_to_cluster_mapping=token_to_cluster_mapping,
        cluster_sizes=cluster_sizes,
        cluster_capacities=cluster_capacities,
    )

    if evicted_token_ids.numel() == 0:
        if not torch.equal(cluster_sizes, cluster_capacities):
            raise RuntimeError(
                "Expected exact capacities, but assignment was not exact and no evictions occurred"
            )
        return token_to_cluster_mapping, cluster_sizes

    old_cluster_ids = token_to_cluster_mapping[evicted_token_ids]

    token_to_cluster_mapping = token_to_cluster_mapping.clone()
    token_to_cluster_mapping[evicted_token_ids] = -1

    cluster_sizes = cluster_sizes.clone()
    cluster_sizes.scatter_add_(
        0,
        old_cluster_ids,
        -torch.ones_like(old_cluster_ids),
    )

    for token_id in evicted_token_ids.tolist():
        available_cluster_ids = torch.nonzero(
            cluster_sizes < cluster_capacities,
            as_tuple=False,
        ).flatten()

        if available_cluster_ids.numel() == 0:
            raise RuntimeError("No available cluster slots left during rebalancing")

        best_available_pos = scores[token_id, available_cluster_ids].argmax()
        new_cluster_id = int(available_cluster_ids[best_available_pos].item())

        token_to_cluster_mapping[token_id] = new_cluster_id
        cluster_sizes[new_cluster_id] += 1

    if (token_to_cluster_mapping < 0).any():
        raise RuntimeError("Some tokens were left unassigned after rebalancing")

    if not torch.equal(cluster_sizes, cluster_capacities):
        raise RuntimeError(
            "Strict equal-capacity assignment failed: "
            f"min_size={int(cluster_sizes.min().item())}, "
            f"max_size={int(cluster_sizes.max().item())}, "
            f"expected={int(cluster_capacities[0].item())}."
        )

    return token_to_cluster_mapping, cluster_sizes


@torch.no_grad()
def recompute_centroids(
    vectors: Tensor,
    token_to_cluster_mapping: Tensor,
    *,
    num_clusters: int,
    normalize_vectors: bool,
) -> tuple[Tensor, Tensor]:
    if (token_to_cluster_mapping < 0).any():
        raise ValueError("token_to_cluster_mapping contains unassigned tokens")

    hidden_size = vectors.shape[1]

    sums = torch.zeros(
        (num_clusters, hidden_size),
        dtype=vectors.dtype,
        device=vectors.device,
    )

    sums.index_add_(0, token_to_cluster_mapping, vectors)

    cluster_sizes = torch.bincount(
        token_to_cluster_mapping,
        minlength=num_clusters,
    )

    if (cluster_sizes == 0).any():
        empty_cluster_ids = torch.nonzero(cluster_sizes == 0, as_tuple=False).flatten()
        raise RuntimeError(
            "Strict equal clustering produced empty clusters: "
            f"{empty_cluster_ids[:20].tolist()}"
        )

    centroids = sums / cluster_sizes.unsqueeze(-1).to(vectors.dtype)

    if normalize_vectors:
        centroids = l2_normalize(centroids, dim=-1)

    return centroids, cluster_sizes


@torch.no_grad()
def build_clusters(
    lm_head_vector_table: Tensor,
    *,
    num_clusters: int,
    num_iters: int = 100,
    normalize_vectors: bool = True,
    seed: int = 0,
    build_device: torch.device | str | None = None,
    on_iteration_metrics: Callable[[int, dict[str, float]], None] | None = None,
) -> BuiltFlashHeadClusters:
    if lm_head_vector_table.ndim != 2:
        raise ValueError("lm_head_vector_table must have shape [vocab_size, hidden_size]")

    vocab_size, _ = lm_head_vector_table.shape

    if num_clusters < 1:
        raise ValueError("num_clusters must be >= 1")
    if num_clusters > vocab_size:
        raise ValueError("num_clusters cannot exceed vocab_size")
    if vocab_size % num_clusters != 0:
        raise ValueError(
            "Strictly equal-sized clusters require vocab_size % num_clusters == 0, "
            f"but got vocab_size={vocab_size}, num_clusters={num_clusters}, "
            f"remainder={vocab_size % num_clusters}."
        )
    if num_iters < 1:
        raise ValueError("num_iters must be >= 1")

    device = torch.device("cpu" if build_device is None else build_device)
    print("Device to build clusters on:", device)

    vectors = lm_head_vector_table.to(device=device, dtype=torch.float32)

    if normalize_vectors:
        vectors = l2_normalize(vectors, dim=-1)

    generator = torch.Generator(device=vectors.device)
    generator.manual_seed(seed)

    cluster_capacities = build_equal_cluster_capacities(
        vocab_size=vocab_size,
        num_clusters=num_clusters,
        device=vectors.device,
    )

    expected_cluster_size = int(cluster_capacities[0].item())

    init_indices = torch.randperm(
        vocab_size,
        generator=generator,
        device=vectors.device,
    )[:num_clusters]

    centroids = vectors[init_indices].clone()

    if normalize_vectors:
        centroids = l2_normalize(centroids, dim=-1)

    for iter_idx in range(num_iters):
        token_to_cluster_mapping, cluster_sizes = assign_with_greedy_capacity_rebalancing(
            vectors=vectors,
            centroids=centroids,
            cluster_capacities=cluster_capacities,
        )

        centroids, cluster_sizes = recompute_centroids(
            vectors=vectors,
            token_to_cluster_mapping=token_to_cluster_mapping,
            num_clusters=num_clusters,
            normalize_vectors=normalize_vectors,
        )

        metrics_after = compute_clustering_loss(
            vectors=vectors,
            centroids=centroids,
            token_to_cluster_mapping=token_to_cluster_mapping,
        )
        if on_iteration_metrics is not None:
            on_iteration_metrics(iter_idx + 1, metrics_after)

        print(
            f"iter {iter_idx + 1}/{num_iters} "
            f"loss={metrics_after['loss']:.6f} "
            f"mean_sim={metrics_after['mean_assigned_similarity']:.6f} "
            f"p05={metrics_after['p05_assigned_similarity']:.6f} "
            f"min={metrics_after['min_assigned_similarity']:.6f} "
            f"size_min={int(cluster_sizes.min().item())} "
            f"size_max={int(cluster_sizes.max().item())} "
            f"expected_size={expected_cluster_size}"
        )

    token_to_cluster_mapping, cluster_sizes = assign_with_greedy_capacity_rebalancing(
        vectors=vectors,
        centroids=centroids,
        cluster_capacities=cluster_capacities,
    )

    cluster_to_token_ids = build_cluster_to_token_ids_exact(
        token_to_cluster_mapping=token_to_cluster_mapping,
        num_clusters=num_clusters,
    )

    validate_strict_equal_clustering(
        token_to_cluster_mapping=token_to_cluster_mapping,
        cluster_to_token_ids=cluster_to_token_ids,
        cluster_sizes=cluster_sizes,
        vocab_size=vocab_size,
        num_clusters=num_clusters,
    )

    quality = evaluate_clustering_quality(
        vectors=vectors,
        centroids=centroids,
        token_to_cluster_mapping=token_to_cluster_mapping,
        cluster_sizes=cluster_sizes,
        cluster_capacities=cluster_capacities,
        score_batch_size=2048,
    )

    print_clustering_quality(quality)

    return BuiltFlashHeadClusters(
        token_to_cluster_mapping=token_to_cluster_mapping.cpu(),
        cluster_to_token_ids=cluster_to_token_ids.cpu(),
        centroids=centroids.cpu(),
        cluster_sizes=cluster_sizes.cpu(),
    )


def build_cluster_to_token_ids_exact(
    token_to_cluster_mapping: Tensor,
    *,
    num_clusters: int,
) -> Tensor:
    if token_to_cluster_mapping.ndim != 1:
        raise ValueError("token_to_cluster_mapping must have shape [vocab_size]")

    vocab_size = token_to_cluster_mapping.shape[0]

    if vocab_size % num_clusters != 0:
        raise ValueError("vocab_size must be divisible by num_clusters")

    cluster_size = vocab_size // num_clusters

    cluster_sizes = torch.bincount(
        token_to_cluster_mapping,
        minlength=num_clusters,
    )

    if not torch.all(cluster_sizes == cluster_size):
        raise RuntimeError(
            "Clusters are not strictly equal sized: "
            f"min={int(cluster_sizes.min().item())}, "
            f"max={int(cluster_sizes.max().item())}, "
            f"expected={cluster_size}"
        )

    sorted_token_ids = torch.argsort(token_to_cluster_mapping, stable=True)

    return sorted_token_ids.reshape(num_clusters, cluster_size).contiguous()


def validate_strict_equal_clustering(
    *,
    token_to_cluster_mapping: Tensor,
    cluster_to_token_ids: Tensor,
    cluster_sizes: Tensor,
    vocab_size: int,
    num_clusters: int,
) -> None:
    if token_to_cluster_mapping.shape != (vocab_size,):
        raise ValueError("token_to_cluster_mapping has wrong shape")

    if cluster_sizes.shape != (num_clusters,):
        raise ValueError("cluster_sizes has wrong shape")

    if cluster_to_token_ids.ndim != 2:
        raise ValueError("cluster_to_token_ids must have shape [num_clusters, cluster_size]")

    if cluster_to_token_ids.shape[0] != num_clusters:
        raise ValueError("cluster_to_token_ids has wrong num_clusters dimension")

    cluster_size = cluster_to_token_ids.shape[1]

    if vocab_size != num_clusters * cluster_size:
        raise ValueError(
            "Strict equal clusters require vocab_size == num_clusters * cluster_size, "
            f"got vocab_size={vocab_size}, num_clusters={num_clusters}, "
            f"cluster_size={cluster_size}."
        )

    if (cluster_to_token_ids < 0).any():
        raise ValueError("cluster_to_token_ids must not contain padding or negative ids")

    if int(cluster_to_token_ids.max().item()) >= vocab_size:
        raise ValueError("cluster_to_token_ids contains token ids >= vocab_size")

    if not torch.all(cluster_sizes == cluster_size):
        raise ValueError(
            "cluster_sizes are not strictly equal: "
            f"min={int(cluster_sizes.min().item())}, "
            f"max={int(cluster_sizes.max().item())}, "
            f"expected={cluster_size}."
        )

    flat_ids = cluster_to_token_ids.reshape(-1).cpu()
    counts = torch.bincount(flat_ids, minlength=vocab_size)

    if not torch.all(counts == 1):
        duplicate_count = int((counts > 1).sum().item())
        missing_count = int((counts == 0).sum().item())
        raise ValueError(
            "cluster_to_token_ids must contain every token exactly once, "
            f"but found duplicate_token_count={duplicate_count}, "
            f"missing_token_count={missing_count}."
        )


@torch.no_grad()
def build_flashhead_index(
    built: BuiltFlashHeadClusters,
    *,
    lm_head_vector_table: Tensor,
    lm_head_bias: Tensor | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> FlashHeadIndex:
    """
    Builds the fast inference layout.

    Important:
    - lm_head_vector_table should be the original LM-head weight, not L2-normalized.
    - centroids may be normalized because stage 1 is centroid retrieval.
    - clustered_lm_head duplicates/reorders the LM-head vectors by cluster.
      For memory savings, do not also keep the old dense lm_head inside your inference module.
    """
    if lm_head_vector_table.ndim != 2:
        raise ValueError("lm_head_vector_table must have shape [vocab_size, hidden_size]")

    target_device = lm_head_vector_table.device if device is None else torch.device(device)
    target_dtype = lm_head_vector_table.dtype if dtype is None else dtype

    vocab_size, hidden_size = lm_head_vector_table.shape

    if built.centroids.ndim != 2:
        raise ValueError("built.centroids must have shape [num_clusters, hidden_size]")

    if built.cluster_to_token_ids.ndim != 2:
        raise ValueError(
            "built.cluster_to_token_ids must have shape [num_clusters, cluster_size]"
        )

    num_clusters, cluster_size = built.cluster_to_token_ids.shape

    if built.centroids.shape != (num_clusters, hidden_size):
        raise ValueError(
            "centroids shape mismatch: "
            f"expected [{num_clusters}, {hidden_size}], "
            f"got {list(built.centroids.shape)}."
        )

    validate_strict_equal_clustering(
        token_to_cluster_mapping=built.token_to_cluster_mapping,
        cluster_to_token_ids=built.cluster_to_token_ids,
        cluster_sizes=built.cluster_sizes,
        vocab_size=vocab_size,
        num_clusters=num_clusters,
    )

    cluster_to_token_ids = built.cluster_to_token_ids.to(
        device=target_device,
        dtype=torch.long,
    ).contiguous()

    centroids_t = built.centroids.to(
        device=target_device,
        dtype=target_dtype,
    ).transpose(0, 1).contiguous()

    lm_head_vector_table = lm_head_vector_table.to(
        device=target_device,
        dtype=target_dtype,
    )

    flat_token_ids = cluster_to_token_ids.reshape(-1)

    clustered_lm_head = lm_head_vector_table.index_select(
        0,
        flat_token_ids,
    ).reshape(num_clusters, cluster_size, hidden_size).contiguous()

    if lm_head_bias is None:
        clustered_lm_head_bias = None
    else:
        if lm_head_bias.ndim != 1 or lm_head_bias.shape[0] != vocab_size:
            raise ValueError("lm_head_bias must have shape [vocab_size]")

        lm_head_bias = lm_head_bias.to(
            device=target_device,
            dtype=target_dtype,
        )

        clustered_lm_head_bias = lm_head_bias.index_select(
            0,
            flat_token_ids,
        ).reshape(num_clusters, cluster_size).contiguous()

    return FlashHeadIndex(
        centroids_t=centroids_t,
        cluster_to_token_ids=cluster_to_token_ids,
        clustered_lm_head=clustered_lm_head,
        clustered_lm_head_bias=clustered_lm_head_bias,
        cluster_size=int(cluster_size),
        num_clusters=int(num_clusters),
        vocab_size=int(vocab_size),
    )



@torch.no_grad()
def evaluate_clustering_quality(
    vectors: Tensor,
    centroids: Tensor,
    token_to_cluster_mapping: Tensor,
    cluster_sizes: Tensor,
    *,
    cluster_capacities: Tensor | None = None,
    score_batch_size: int = 2048,
) -> ClusterQualityMetrics:
    if vectors.ndim != 2:
        raise ValueError("vectors must have shape [num_vectors, hidden_size]")
    if centroids.ndim != 2:
        raise ValueError("centroids must have shape [num_clusters, hidden_size]")
    if token_to_cluster_mapping.ndim != 1:
        raise ValueError("token_to_cluster_mapping must have shape [num_vectors]")
    if cluster_sizes.ndim != 1:
        raise ValueError("cluster_sizes must have shape [num_clusters]")

    num_vectors = vectors.shape[0]
    device = vectors.device

    assigned_centroids = centroids[token_to_cluster_mapping]
    assigned_scores = (vectors * assigned_centroids).sum(dim=-1).cpu()

    best_other_scores_parts: list[Tensor] = []
    num_assigned_to_nearest = 0

    for start in range(0, num_vectors, score_batch_size):
        end = min(start + score_batch_size, num_vectors)

        batch_vectors = vectors[start:end]
        batch_mapping = token_to_cluster_mapping[start:end]

        scores = batch_vectors @ centroids.transpose(0, 1)

        nearest_cluster_ids = scores.argmax(dim=-1)
        num_assigned_to_nearest += int((nearest_cluster_ids == batch_mapping).sum().item())

        row_ids = torch.arange(end - start, device=device)
        scores[row_ids, batch_mapping] = -torch.inf

        best_other_scores = scores.max(dim=-1).values
        best_other_scores_parts.append(best_other_scores.cpu())

    best_other_scores = torch.cat(best_other_scores_parts, dim=0)
    margins = assigned_scores - best_other_scores

    if cluster_capacities is not None:
        capacity_error = (cluster_sizes.cpu() - cluster_capacities.cpu()).abs()
        mean_abs_capacity_error = float(capacity_error.float().mean().item())
        max_abs_capacity_error = int(capacity_error.max().item())
    else:
        mean_abs_capacity_error = None
        max_abs_capacity_error = None

    return ClusterQualityMetrics(
        mean_assigned_similarity=float(assigned_scores.mean().item()),
        p05_assigned_similarity=float(torch.quantile(assigned_scores, 0.05).item()),
        mean_margin_to_best_other=float(margins.mean().item()),
        p05_margin_to_best_other=float(torch.quantile(margins, 0.05).item()),
        fraction_assigned_to_nearest_centroid=float(num_assigned_to_nearest / num_vectors),
        mean_abs_capacity_error=mean_abs_capacity_error,
        max_abs_capacity_error=max_abs_capacity_error,
        min_cluster_size=int(cluster_sizes.min().item()),
        max_cluster_size=int(cluster_sizes.max().item()),
        cluster_size_std=float(cluster_sizes.float().std(unbiased=False).item()),
    )


def print_clustering_quality(metrics: ClusterQualityMetrics) -> None:
    print()
    print("cluster quality metrics:")
    print(f"  mean_assigned_similarity         = {metrics.mean_assigned_similarity:.6f}")
    print(f"  p05_assigned_similarity          = {metrics.p05_assigned_similarity:.6f}")
    print(f"  mean_margin_to_best_other        = {metrics.mean_margin_to_best_other:.6f}")
    print(f"  p05_margin_to_best_other         = {metrics.p05_margin_to_best_other:.6f}")
    print(f"  fraction_assigned_to_nearest     = {metrics.fraction_assigned_to_nearest_centroid:.6f}")
    print(f"  min_cluster_size                 = {metrics.min_cluster_size}")
    print(f"  max_cluster_size                 = {metrics.max_cluster_size}")
    print(f"  cluster_size_std                 = {metrics.cluster_size_std:.6f}")

    if metrics.mean_abs_capacity_error is not None:
        print(f"  mean_abs_capacity_error          = {metrics.mean_abs_capacity_error:.6f}")
        print(f"  max_abs_capacity_error           = {metrics.max_abs_capacity_error}")


@torch.no_grad()
def compute_clustering_loss(
    vectors: Tensor,
    centroids: Tensor,
    token_to_cluster_mapping: Tensor,
) -> dict[str, float]:
    assigned_centroids = centroids[token_to_cluster_mapping]
    assigned_scores = (vectors * assigned_centroids).sum(dim=-1)

    mean_similarity = float(assigned_scores.mean().item())
    p05_similarity = float(torch.quantile(assigned_scores, 0.05).item())
    min_similarity = float(assigned_scores.min().item())

    return {
        "loss": 1.0 - mean_similarity,
        "mean_assigned_similarity": mean_similarity,
        "p05_assigned_similarity": p05_similarity,
        "min_assigned_similarity": min_similarity,
    }
