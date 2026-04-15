from dataclasses import dataclass
from torch import Tensor
import torch
import os

torch.set_num_threads(os.cpu_count() or 1)
torch.set_num_interop_threads(1)


# The final clutering result

@dataclass(frozen=True, slots=True)
class BuiltFlashHeadClusters:
    token_to_cluster_mapping: Tensor # [vocab_size]
    cluster_to_token_ids: Tensor     # [num_clusters, max_cluster_size] padded with -1
    centroids: Tensor                # [num_clusters, hidden_size]
    cluster_sizes: Tensor            # [num_clusters]

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


# Calculate ideal cluster sizes to keep them as balanced as possible

def build_near_equal_cluster_capacities(
    vocab_size: int,
    num_clusters: int,
    *,
    generator: torch.Generator | None = None,
    device: torch.device | None = None,
) -> Tensor:
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
        plus_one_cluster_ids = torch.randperm(
            num_clusters,
            generator=generator,
            device=device,
        )[:remainder]
        capacities[plus_one_cluster_ids] += 1

    return capacities


@torch.no_grad()
def assign_to_nearest_centroid(
    vectors: Tensor,
    centroids: Tensor,
) -> Tensor:
    scores = vectors @ centroids.transpose(0, 1)
    return scores.argmax(dim=-1)


@torch.no_grad()
def recompute_centroids(
    vectors: Tensor,
    token_to_cluster_mapping: Tensor,
    *,
    num_clusters: int,
    normalize_vectors: bool,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
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

    centroids = torch.zeros_like(sums)

    non_empty = cluster_sizes > 0
    if non_empty.any():
        centroids[non_empty] = (
            sums[non_empty]
            / cluster_sizes[non_empty].unsqueeze(-1).to(vectors.dtype)
        )

    num_empty = int((~non_empty).sum().item())
    if num_empty > 0:
        refill_ids = torch.randperm(
            vectors.shape[0],
            generator=generator,
            device=vectors.device,
        )[:num_empty]
        centroids[~non_empty] = vectors[refill_ids]

    if normalize_vectors:
        centroids = l2_normalize(centroids, dim=-1)

    return centroids, cluster_sizes


@torch.no_grad()
def build_clusters(
    lm_head_vector_table: Tensor,
    *,
    num_clusters: int,
    num_iters: int = 10,
    num_rebalance_rounds: int = 2,
    normalize_vectors: bool = True,
    seed: int = 0,
) -> BuiltFlashHeadClusters:
    
    if lm_head_vector_table.ndim != 2:
        raise ValueError("lm_head_vector_table must have shape [vocab_size, hidden_size]")


    vocab_size, _ = lm_head_vector_table.shape

    if num_clusters < 1:
        raise ValueError("num_clusters must be >= 1")
    if num_clusters > vocab_size:
        raise ValueError("num_clusters cannot exceed vocab_size")
    if num_iters < 1:
        raise ValueError("num_iters must be >= 1")
    

    device = torch.device("cpu")
    print("Device to build clusters on:", device)

    vectors = lm_head_vector_table.to(device=device, dtype=torch.float32)

    if normalize_vectors:
        vectors = l2_normalize(vectors, dim=-1)

    generator = torch.Generator(device=vectors.device)
    generator.manual_seed(seed)

    cluster_capacities = build_near_equal_cluster_capacities(
            vocab_size=vocab_size,
            num_clusters=num_clusters,
            generator=generator,
            device=vectors.device,
        )

    if cluster_capacities.ndim != 1 or cluster_capacities.shape[0] != num_clusters:
        raise ValueError("cluster_capacities must have shape [num_clusters]")
    if int(cluster_capacities.sum().item()) != vocab_size:
        raise ValueError("cluster_capacities must sum to vocab_size")
    if (cluster_capacities < 0).any():
        raise ValueError("cluster_capacities must be non-negative")

    init_indices = torch.randperm(
        vocab_size,
        generator=generator,
        device=vectors.device,
    )[:num_clusters]
    centroids = vectors[init_indices].clone()

    # Stage 1: ordinary spherical k-means
    for iter_idx in range(num_iters):
        token_to_cluster_mapping = assign_to_nearest_centroid(vectors, centroids)

        centroids, cluster_sizes = recompute_centroids(
            vectors=vectors,
            token_to_cluster_mapping=token_to_cluster_mapping,
            num_clusters=num_clusters,
            normalize_vectors=normalize_vectors,
            generator=generator,
        )

        metrics_after = compute_clustering_loss(
            vectors=vectors,
            centroids=centroids,
            token_to_cluster_mapping=token_to_cluster_mapping,
        )

        print(
            f"iter {iter_idx + 1}/{num_iters} "
            f"loss={metrics_after['loss']:.6f} "
            f"mean_sim={metrics_after['mean_assigned_similarity']:.6f} "
            f"p05={metrics_after['p05_assigned_similarity']:.6f} "
            f"min={metrics_after['min_assigned_similarity']:.6f} "
            f"size_min={int(cluster_sizes.min().item())} "
            f"size_max={int(cluster_sizes.max().item())}"
        )

    cluster_to_token_ids = build_cluster_to_token_ids_padded(
        token_to_cluster_mapping=token_to_cluster_mapping,
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


def build_cluster_to_token_ids_padded(
    token_to_cluster_mapping: Tensor,
    *,
    num_clusters: int,
) -> Tensor:
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
            if not torch.all(sorted_cluster_ids[start:end] == cluster_id):
                raise RuntimeError("Cluster grouping mismatch while building padded token ids")

            cluster_to_token_ids[cluster_id, :size] = token_ids

    return cluster_to_token_ids



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