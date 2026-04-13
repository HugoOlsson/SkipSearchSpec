


from dataclasses import dataclass
from torch import Tensor
import torch

from skip_search_spec.helpers.tooling import get_preferred_device

import os
torch.set_num_threads(os.cpu_count() or 1)
torch.set_num_interop_threads(1)


@dataclass(frozen=True, slots=True)
class BuiltFlashHeadClusters:
    token_to_cluster_mapping: Tensor                  # [vocab_size]
    cluster_to_token_ids: Tensor           # [num_clusters, cluster_size]
    centroids: Tensor               # [num_clusters, hidden_size]
    cluster_sizes: Tensor           # [num_clusters]


def l2_normalize(x: Tensor, dim: int) -> Tensor:
    return x / x.norm(dim=dim, keepdim=True).clamp_min(1e-12)


@torch.no_grad()
def build_clusters(
    lm_head_vector_table: Tensor,
    *,
    num_clusters: int,
    num_iters: int = 5,
    normalize_vectors: bool = True,
    seed: int = 0,
    assignment_chunk_size: int = 4096,
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
    if assignment_chunk_size < 1:
        raise ValueError("assignment_chunk_size must be >= 1")
    
    

    # Start from the LM head vector table:
    # shape = [vocab_size, hidden_size]
    # Each row is one token's output vector.
    vectors = lm_head_vector_table.to(device=device, dtype=torch.float32)

    # Optional:
    # normalize each token vector to unit length.
    # Then clustering is based more on "direction" than raw magnitude.
    # This often makes sense because later routing is based on dot products / similarity.
    if normalize_vectors:
        vectors = l2_normalize(vectors, dim=-1)

    # Create a random generator with a fixed seed so clustering is reproducible.
    generator = torch.Generator(device=vectors.device)
    generator.manual_seed(seed)

    # Pick num_clusters random token vectors to serve as the initial cluster representatives.
    # These representatives are the first version of what will become the router matrix rows.
    init_indices = torch.randperm(
        vocab_size,
        generator=generator,
        device=vectors.device,
    )[:num_clusters]

    # centroids shape = [num_clusters, hidden_size]
    # Each row is the current representative vector for one cluster.
    centroids = vectors[init_indices].clone()

    # Repeat the clustering process several times.
    for iter_idx in range(num_iters):

        token_to_cluster_mapping = build_equal_size_token_to_cluster_mapping(
            vectors=vectors,
            centroids=centroids,
        )

        sums = torch.zeros(
            (num_clusters, hidden_size),
            dtype=vectors.dtype,
            device=vectors.device,
        )

        sums.index_add_(0, token_to_cluster_mapping, vectors)

        cluster_size = vocab_size // num_clusters

        centroids = sums / cluster_size

        if normalize_vectors:
            centroids = l2_normalize(centroids, dim=-1)

        print(f"finished k-means iteration {iter_idx + 1}/{num_iters}")

    token_to_cluster_mapping = build_equal_size_token_to_cluster_mapping(
        vectors=vectors,
        centroids=centroids,
    )

    cluster_size = vocab_size // num_clusters
    cluster_sizes = torch.full(
        (num_clusters,),
        fill_value=cluster_size,
        dtype=torch.long,
        device=vectors.device,
    )

    cluster_to_token_ids = build_cluster_to_token_ids(
        token_to_cluster_mapping,
        num_clusters=num_clusters,
    )

    return BuiltFlashHeadClusters(
        token_to_cluster_mapping=token_to_cluster_mapping.cpu(),
        cluster_to_token_ids=cluster_to_token_ids.cpu(),
        centroids=centroids.cpu(),
        cluster_sizes=cluster_sizes.cpu(),
    )


@torch.no_grad()
def build_equal_size_token_to_cluster_mapping(
    vectors: Tensor,
    centroids: Tensor,
) -> Tensor:
    # vectors should be the token-vector table:
    # shape = [vocab_size, hidden_size]
    #
    # Each row is one token's vector.
    if vectors.ndim != 2:
        raise ValueError("vectors must have shape [vocab_size, hidden_size]")

    # centroids should be the current cluster representative vectors:
    # shape = [num_clusters, hidden_size]
    #
    # Each row is one cluster centroid.
    if centroids.ndim != 2:
        raise ValueError("centroids must have shape [num_clusters, hidden_size]")

    # The token vectors and centroid vectors must live in the same hidden space,
    # otherwise dot-product similarity would not make sense.
    if vectors.shape[1] != centroids.shape[1]:
        raise ValueError("vectors and centroids must have the same hidden size")

    vocab_size = vectors.shape[0]
    num_clusters = centroids.shape[0]

    # In this exact equal-size version, every cluster must contain
    # exactly the same number of token vectors.
    #
    # So vocab_size must divide evenly by num_clusters.
    if vocab_size % num_clusters != 0:
        raise ValueError(
            f"Exact equal-size clustering requires vocab_size % num_clusters == 0, "
            f"but got vocab_size={vocab_size}, num_clusters={num_clusters}"
        )

    # This is the exact number of token vectors each cluster is allowed to hold.
    cluster_size = vocab_size // num_clusters

    # Compute similarity between every token vector and every centroid.
    #
    # scores[token_id, cluster_id] tells us how much token vector token_id
    # "likes" centroid cluster_id.
    #
    # shape = [vocab_size, num_clusters]
    scores = vectors @ centroids.transpose(0, 1)

    # For each token, sort all clusters from best match to worst match.
    #
    # ranked_cluster_ids[token_id, 0] is that token's favorite cluster
    # ranked_cluster_ids[token_id, 1] is its second favorite
    # and so on.
    #
    # shape = [vocab_size, num_clusters]
    ranked_cluster_ids = scores.argsort(dim=-1, descending=True)

    # We want to assign the "most certain" tokens first.
    #
    # Intuition:
    # if a token strongly prefers one cluster over all others,
    # we should let it claim that cluster early.
    #
    # We measure this certainty as:
    #   best score - second-best score
    #
    # Large margin = token is very sure where it belongs.
    if num_clusters >= 2:
        top2_scores = scores.topk(k=2, dim=-1).values
        confidence = top2_scores[:, 0] - top2_scores[:, 1]
    else:
        # Degenerate case: only one cluster exists,
        # so confidence does not matter.
        confidence = torch.zeros(vocab_size, dtype=vectors.dtype, device=vectors.device)

    # Process tokens from most certain to least certain.
    #
    # token_order is a permutation of token ids.
    token_order = confidence.argsort(descending=True)

    # This will become the final lookup:
    #
    # token_to_cluster_mapping[token_id] = chosen_cluster_id
    #
    # Start with -1 to mean "not assigned yet".
    token_to_cluster_mapping = torch.full(
        (vocab_size,),
        fill_value=-1,
        dtype=torch.long,
        device=vectors.device,
    )

    # remaining_capacity[cluster_id] says how many more token vectors
    # that cluster is still allowed to take.
    #
    # Every cluster starts with exactly cluster_size available slots.
    remaining_capacity = torch.full(
        (num_clusters,),
        fill_value=cluster_size,
        dtype=torch.long,
        device=vectors.device,
    )

    # Assign each token to the best cluster that still has free capacity.
    #
    # So for each token:
    #   - look through its preferred clusters, best to worst
    #   - choose the first cluster that still has room
    #   - decrease that cluster's remaining capacity by 1
    token_order_np = token_order.numpy()
    ranked_cluster_ids_np = ranked_cluster_ids.numpy()

    for token_id in token_order_np:
        preferred_clusters = ranked_cluster_ids_np[token_id]
        for cluster_id in preferred_clusters:
            if remaining_capacity[cluster_id] > 0:
                token_to_cluster_mapping[token_id] = cluster_id
                remaining_capacity[cluster_id] -= 1
                break

    # Safety check:
    # every token must have been assigned somewhere.
    if (token_to_cluster_mapping < 0).any():
        raise RuntimeError("Some tokens were not assigned to a cluster")

    # Safety check:
    # every cluster should now be exactly full.
    if (remaining_capacity != 0).any():
        raise RuntimeError("Some clusters did not fill to the required size")

    # Return the final token -> cluster mapping.
    return token_to_cluster_mapping



def build_cluster_to_token_ids(
    token_to_cluster_mapping: Tensor,
    *,
    num_clusters: int,
) -> Tensor:
    if token_to_cluster_mapping.ndim != 1:
        raise ValueError("token_to_cluster_mapping must have shape [vocab_size]")

    vocab_size = int(token_to_cluster_mapping.shape[0])
    if vocab_size % num_clusters != 0:
        raise ValueError(
            f"Expected vocab_size to divide num_clusters exactly, "
            f"but got vocab_size={vocab_size}, num_clusters={num_clusters}"
        )

    cluster_size = vocab_size // num_clusters

    sorted_token_ids = torch.argsort(token_to_cluster_mapping, stable=True)
    cluster_to_token_ids = sorted_token_ids.view(num_clusters, cluster_size)

    return cluster_to_token_ids