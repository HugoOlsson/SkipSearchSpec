from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor

from skip_search_spec.training.flashhead.building_clusters import l2_normalize
from skip_search_spec.training.flashhead.storage import load_flashhead


@torch.inference_mode()
def flashhead_next_token(
    hidden_vector: Tensor,
    *,
    centroids: Tensor,
    cluster_to_token_ids: Tensor,
    lm_head_vector_table: Tensor,
    top_k_clusters: int,
    normalize_hidden_for_routing: bool = True,
    lm_head_bias: Tensor | None = None,
) -> Tensor:
    """
    Return the greedy next-token id for one hidden vector using a stored FlashHead.

    The hidden_vector is expected to be the vector that would normally be passed
    to the LM head: shape [hidden_size], already final-normalized if the model
    architecture applies a final norm before lm_head.
    """
    if hidden_vector.ndim != 1:
        raise ValueError("hidden_vector must have shape [hidden_size]")
    if centroids.ndim != 2:
        raise ValueError("centroids must have shape [num_clusters, hidden_size]")
    if cluster_to_token_ids.ndim != 2:
        raise ValueError("cluster_to_token_ids must have shape [num_clusters, max_cluster_size]")
    if lm_head_vector_table.ndim != 2:
        raise ValueError("lm_head_vector_table must have shape [vocab_size, hidden_size]")
    if top_k_clusters < 1:
        raise ValueError("top_k_clusters must be >= 1")

    hidden_size = hidden_vector.shape[0]
    if centroids.shape[1] != hidden_size:
        raise ValueError(
            f"centroids hidden size {centroids.shape[1]} != hidden_vector size {hidden_size}"
        )
    if lm_head_vector_table.shape[1] != hidden_size:
        raise ValueError(
            f"lm_head hidden size {lm_head_vector_table.shape[1]} != hidden_vector size {hidden_size}"
        )
    if cluster_to_token_ids.shape[0] != centroids.shape[0]:
        raise ValueError("cluster_to_token_ids and centroids must agree on num_clusters")

    route_hidden = (
        l2_normalize(hidden_vector, dim=-1)
        if normalize_hidden_for_routing
        else hidden_vector
    )

    actual_top_k = min(top_k_clusters, int(centroids.shape[0]))
    cluster_scores = route_hidden @ centroids.transpose(0, 1)
    top_cluster_ids = torch.topk(cluster_scores, k=actual_top_k).indices

    candidate_token_ids = cluster_to_token_ids[top_cluster_ids].reshape(-1)
    candidate_token_ids = candidate_token_ids[candidate_token_ids >= 0].long()

    if candidate_token_ids.numel() == 0:
        raise RuntimeError("FlashHead routing produced zero valid candidate token ids.")

    candidate_token_vectors = lm_head_vector_table[candidate_token_ids]
    candidate_scores = candidate_token_vectors.float() @ hidden_vector.float()

    if lm_head_bias is not None:
        candidate_scores = candidate_scores + lm_head_bias[candidate_token_ids].float()

    best_candidate_index = candidate_scores.argmax()
    return candidate_token_ids[best_candidate_index]


def build_flashhead_next_token_fn(
    *,
    flashhead_path: str | Path,
    lm_head_vector_table: Tensor,
    top_k_clusters: int,
    normalize_hidden_for_routing: bool = True,
    lm_head_bias: Tensor | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> Callable[[Tensor], Tensor]:
    """
    Load a stored FlashHead and return a function: hidden_vector -> token_id.
    """
    stored = load_flashhead(flashhead_path)

    if device is None:
        device = lm_head_vector_table.device
    if dtype is None:
        dtype = lm_head_vector_table.dtype

    lm_head_vector_table = lm_head_vector_table.to(device=device, dtype=dtype)
    centroids = stored.centroids.to(device=device, dtype=dtype)
    cluster_to_token_ids = stored.cluster_to_token_ids.to(device=device)

    if lm_head_bias is not None:
        lm_head_bias = lm_head_bias.to(device=device, dtype=dtype)

    def next_token(hidden_vector: Tensor) -> Tensor:
        return flashhead_next_token(
            hidden_vector.to(device=device, dtype=dtype),
            centroids=centroids,
            cluster_to_token_ids=cluster_to_token_ids,
            lm_head_vector_table=lm_head_vector_table,
            top_k_clusters=top_k_clusters,
            normalize_hidden_for_routing=normalize_hidden_for_routing,
            lm_head_bias=lm_head_bias,
        )

    return next_token


def build_flashhead_next_token_fn_from_model(
    *,
    model: Any,
    flashhead_path: str | Path,
    top_k_clusters: int,
    normalize_hidden_for_routing: bool = True,
) -> Callable[[Tensor], Tensor]:
    """
    Convenience builder for Hugging Face causal LM models.
    """
    output_embeddings = model.get_output_embeddings()
    if output_embeddings is None:
        raise ValueError("Model does not expose output embeddings via get_output_embeddings().")

    weight = getattr(output_embeddings, "weight", None)
    if not isinstance(weight, Tensor):
        raise ValueError("Output embeddings do not expose a Tensor weight.")

    bias = cast(Tensor | None, getattr(output_embeddings, "bias", None))
    device = weight.device
    dtype = weight.dtype

    return build_flashhead_next_token_fn(
        flashhead_path=flashhead_path,
        lm_head_vector_table=weight,
        lm_head_bias=bias,
        top_k_clusters=top_k_clusters,
        normalize_hidden_for_routing=normalize_hidden_for_routing,
        device=device,
        dtype=dtype,
    )
