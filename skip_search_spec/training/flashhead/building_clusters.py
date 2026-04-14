from dataclasses import dataclass
from torch import Tensor
import torch
import os

torch.set_num_threads(os.cpu_count() or 1)
torch.set_num_interop_threads(1)


@dataclass(frozen=True, slots=True)
class BuiltFlashHeadClusters:
    token_to_cluster_mapping: Tensor      # [vocab_size]
    cluster_to_token_ids: Tensor          # [num_clusters, max_cluster_size], padded with -1
    centroids: Tensor                     # [num_clusters, hidden_size]
    cluster_sizes: Tensor                 # [num_clusters]


def l2_normalize(x: Tensor, dim: int) -> Tensor:
    return x / x.norm(dim=dim, keepdim=True).clamp_min(1e-12)


def build_near_equal_cluster_capacities(
    vocab_size: int,
    num_clusters: int,
    *,
    generator: torch.Generator | None = None,
    device: torch.device | None = None,
) -> Tensor:
    """
    Build a capacity vector whose entries differ by at most 1.

    Example:
      vocab_size=151936, num_clusters=8000
      -> 7936 clusters of size 19 and 64 clusters of size 18
    """
    if num_clusters < 1:
        raise ValueError("num_clusters must be >= 1")
    if num_clusters > vocab_size:
        raise ValueError("num_clusters cannot exceed vocab_size")

    base = vocab_size // num_clusters
    remainder = vocab_size % num_clusters

    capacities = torch.full(
        (num_clusters,),
        fill_value=base,
        dtype=torch.long,
        device=device,
    )

    if remainder > 0:
        # Randomize which clusters get +1 capacity so it is not always the first ones.
        plus_one_cluster_ids = torch.randperm(
            num_clusters,
            generator=generator,
            device=device,
        )[:remainder]
        capacities[plus_one_cluster_ids] += 1

    return capacities


@torch.no_grad()
def build_clusters(
    lm_head_vector_table: Tensor,
    *,
    num_clusters: int,
    num_iters: int = 5,
    normalize_vectors: bool = True,
    seed: int = 0,
    cluster_capacities: Tensor | None = None,
) -> BuiltFlashHeadClusters:
    if lm_head_vector_table.ndim != 2:
        raise ValueError("lm_head_vector_table must have shape [vocab_size, hidden_size]")

    device = torch.device("cpu")
    print("Device to build clusters on:", device)

    vocab_size, hidden_size = lm_head_vector_table.shape

    if num_clusters < 1:
        raise ValueError("num_clusters must be >= 1")
    if num_clusters > vocab_size:
        raise ValueError("num_clusters cannot exceed vocab_size")
    if num_iters < 1:
        raise ValueError("num_iters must be >= 1")

    vectors = lm_head_vector_table.to(device=device, dtype=torch.float32)

    if normalize_vectors:
        vectors = l2_normalize(vectors, dim=-1)

    generator = torch.Generator(device=vectors.device)
    generator.manual_seed(seed)

    if cluster_capacities is None:
        cluster_capacities = build_near_equal_cluster_capacities(
            vocab_size=vocab_size,
            num_clusters=num_clusters,
            generator=generator,
            device=vectors.device,
        )
    else:
        cluster_capacities = cluster_capacities.to(device=vectors.device, dtype=torch.long)

    if cluster_capacities.ndim != 1:
        raise ValueError("cluster_capacities must have shape [num_clusters]")
    if cluster_capacities.shape[0] != num_clusters:
        raise ValueError("cluster_capacities must have shape [num_clusters]")
    if (cluster_capacities < 0).any():
        raise ValueError("cluster_capacities must be non-negative")
    if int(cluster_capacities.sum().item()) != vocab_size:
        raise ValueError(
            f"cluster_capacities must sum to vocab_size={vocab_size}, "
            f"but got sum={int(cluster_capacities.sum().item())}"
        )

    init_indices = torch.randperm(
        vocab_size,
        generator=generator,
        device=vectors.device,
    )[:num_clusters]

    centroids = vectors[init_indices].clone()

    for iter_idx in range(num_iters):
        token_to_cluster_mapping = build_capacity_constrained_token_to_cluster_mapping(
            vectors=vectors,
            centroids=centroids,
            cluster_capacities=cluster_capacities,
        )

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
            raise RuntimeError("Some clusters became empty")

        centroids = sums / cluster_sizes.unsqueeze(-1).to(vectors.dtype)

        if normalize_vectors:
            centroids = l2_normalize(centroids, dim=-1)

        print(f"finished k-means iteration {iter_idx + 1}/{num_iters}")

    token_to_cluster_mapping = build_capacity_constrained_token_to_cluster_mapping(
        vectors=vectors,
        centroids=centroids,
        cluster_capacities=cluster_capacities,
    )

    cluster_sizes = torch.bincount(
        token_to_cluster_mapping,
        minlength=num_clusters,
    )

    cluster_to_token_ids = build_cluster_to_token_ids_padded(
        token_to_cluster_mapping=token_to_cluster_mapping,
        num_clusters=num_clusters,
    )

    return BuiltFlashHeadClusters(
        token_to_cluster_mapping=token_to_cluster_mapping.cpu(),
        cluster_to_token_ids=cluster_to_token_ids.cpu(),
        centroids=centroids.cpu(),
        cluster_sizes=cluster_sizes.cpu(),
    )


@torch.no_grad()
def build_capacity_constrained_token_to_cluster_mapping(
    vectors: Tensor,
    centroids: Tensor,
    cluster_capacities: Tensor,
) -> Tensor:
    if vectors.ndim != 2:
        raise ValueError("vectors must have shape [vocab_size, hidden_size]")

    if centroids.ndim != 2:
        raise ValueError("centroids must have shape [num_clusters, hidden_size]")

    if cluster_capacities.ndim != 1:
        raise ValueError("cluster_capacities must have shape [num_clusters]")

    if vectors.shape[1] != centroids.shape[1]:
        raise ValueError("vectors and centroids must have the same hidden size")

    vocab_size = vectors.shape[0]
    num_clusters = centroids.shape[0]

    if cluster_capacities.shape[0] != num_clusters:
        raise ValueError("cluster_capacities length must equal num_clusters")

    if int(cluster_capacities.sum().item()) != vocab_size:
        raise ValueError(
            f"cluster_capacities must sum to vocab_size={vocab_size}, "
            f"but got sum={int(cluster_capacities.sum().item())}"
        )

    if (cluster_capacities < 0).any():
        raise ValueError("cluster_capacities must be non-negative")

    scores = vectors @ centroids.transpose(0, 1)

    ranked_cluster_ids = scores.argsort(dim=-1, descending=True)

    if num_clusters >= 2:
        top2_scores = scores.topk(k=2, dim=-1).values
        confidence = top2_scores[:, 0] - top2_scores[:, 1]
    else:
        confidence = torch.zeros(vocab_size, dtype=vectors.dtype, device=vectors.device)

    token_order = confidence.argsort(descending=True)

    token_to_cluster_mapping = torch.full(
        (vocab_size,),
        fill_value=-1,
        dtype=torch.long,
        device=vectors.device,
    )

    remaining_capacity = cluster_capacities.clone()

    token_order_np = token_order.cpu().numpy()
    ranked_cluster_ids_np = ranked_cluster_ids.cpu().numpy()

    for token_id in token_order_np:
        preferred_clusters = ranked_cluster_ids_np[token_id]
        for cluster_id in preferred_clusters:
            if remaining_capacity[cluster_id] > 0:
                token_to_cluster_mapping[token_id] = cluster_id
                remaining_capacity[cluster_id] -= 1
                break

    if (token_to_cluster_mapping < 0).any():
        raise RuntimeError("Some tokens were not assigned to a cluster")

    if (remaining_capacity != 0).any():
        raise RuntimeError("Some clusters did not fill to their required capacities")

    return token_to_cluster_mapping


def build_cluster_to_token_ids_padded(
    token_to_cluster_mapping: Tensor,
    *,
    num_clusters: int,
) -> Tensor:
    """
    Returns a padded matrix of shape [num_clusters, max_cluster_size].
    Unused slots are filled with -1.

    Use cluster_sizes to know how many valid token ids each row contains.
    """
    if token_to_cluster_mapping.ndim != 1:
        raise ValueError("token_to_cluster_mapping must have shape [vocab_size]")

    cluster_sizes = torch.bincount(
        token_to_cluster_mapping,
        minlength=num_clusters,
    )

    max_cluster_size = int(cluster_sizes.max().item())

    sorted_token_ids = torch.argsort(token_to_cluster_mapping, stable=True)
    sorted_cluster_ids = token_to_cluster_mapping[sorted_token_ids]

    cluster_to_token_ids = torch.full(
        (num_clusters, max_cluster_size),
        fill_value=-1,
        dtype=torch.long,
        device=token_to_cluster_mapping.device,
    )

    offsets = torch.zeros(
        num_clusters + 1,
        dtype=torch.long,
        device=token_to_cluster_mapping.device,
    )
    offsets[1:] = cluster_sizes.cumsum(dim=0)

    for cluster_id in range(num_clusters):
        start = int(offsets[cluster_id].item())
        end = int(offsets[cluster_id + 1].item())
        size = end - start

        if size > 0:
            token_ids = sorted_token_ids[start:end]
            # Optional safety check
            if not torch.all(sorted_cluster_ids[start:end] == cluster_id):
                raise RuntimeError("Cluster grouping mismatch while building padded token ids")

            cluster_to_token_ids[cluster_id, :size] = token_ids

    return cluster_to_token_ids