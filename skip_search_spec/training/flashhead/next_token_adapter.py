from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from skip_search_spec.training.flashhead.storage import load_flashhead


class FlashHeadModule(nn.Module):
    """
    FlashHead module for greedy next-token lookup.

    Assumes strictly equal-sized clusters:

        cluster_to_token_ids: [num_clusters, cluster_size]

    No padding. No -1 values.
    """

    centroids_t: Tensor
    cluster_to_token_ids: Tensor
    clustered_lm_head: Tensor
    clustered_lm_head_bias: Tensor | None
    top_k_clusters: int
    cluster_size: int

    def __init__(
        self,
        *,
        centroids: Tensor,
        cluster_to_token_ids: Tensor,
        lm_head_vector_table: Tensor,
        top_k_clusters: int,
        lm_head_bias: Tensor | None = None,
    ) -> None:
        super().__init__()

        if top_k_clusters < 1:
            raise ValueError("top_k_clusters must be >= 1")

        if centroids.ndim != 2:
            raise ValueError("centroids must have shape [num_clusters, hidden_size]")

        if cluster_to_token_ids.ndim != 2:
            raise ValueError(
                "cluster_to_token_ids must have shape [num_clusters, cluster_size]"
            )

        if lm_head_vector_table.ndim != 2:
            raise ValueError(
                "lm_head_vector_table must have shape [vocab_size, hidden_size]"
            )

        if (cluster_to_token_ids < 0).any():
            raise ValueError(
                "cluster_to_token_ids must not contain padding for this fast path"
            )

        num_clusters, cluster_size = cluster_to_token_ids.shape
        vocab_size, hidden_size = lm_head_vector_table.shape

        if centroids.shape != (num_clusters, hidden_size):
            raise ValueError("centroids shape does not match cluster/lm_head shape")

        if vocab_size != num_clusters * cluster_size:
            raise ValueError(
                "Strict equal clusters require "
                "vocab_size == num_clusters * cluster_size"
            )

        if lm_head_bias is not None and lm_head_bias.shape != (vocab_size,):
            raise ValueError("lm_head_bias must have shape [vocab_size]")

        cluster_to_token_ids = cluster_to_token_ids.detach().long().contiguous()
        flat_token_ids = cluster_to_token_ids.reshape(-1).to(lm_head_vector_table.device)

        clustered_lm_head = lm_head_vector_table.detach().index_select(
            0,
            flat_token_ids,
        ).reshape(num_clusters, cluster_size, hidden_size).contiguous()

        if lm_head_bias is None:
            clustered_lm_head_bias = None
        else:
            clustered_lm_head_bias = lm_head_bias.detach().index_select(
                0,
                flat_token_ids.to(lm_head_bias.device),
            ).reshape(num_clusters, cluster_size).contiguous()

        self.top_k_clusters = int(top_k_clusters)
        self.cluster_size = int(cluster_size)

        self.register_buffer(
            "centroids_t",
            centroids.detach().transpose(0, 1).contiguous(),
        )
        self.register_buffer(
            "cluster_to_token_ids",
            cluster_to_token_ids,
        )
        self.register_buffer(
            "clustered_lm_head",
            clustered_lm_head,
        )
        self.register_buffer(
            "clustered_lm_head_bias",
            clustered_lm_head_bias,
        )

    @classmethod
    def from_model(
        cls,
        *,
        model: Any,
        flashhead_path: str | Path,
        top_k_clusters: int,
    ) -> FlashHeadModule:
        stored = load_flashhead(flashhead_path)
        lm_head_vector_table, lm_head_bias = extract_lm_head(model)

        return cls(
            centroids=stored.centroids,
            cluster_to_token_ids=stored.cluster_to_token_ids,
            lm_head_vector_table=lm_head_vector_table,
            lm_head_bias=lm_head_bias,
            top_k_clusters=top_k_clusters,
        ).to(device=lm_head_vector_table.device, dtype=lm_head_vector_table.dtype)

    @torch.inference_mode()
    def find_token(self, hidden_vector: Tensor) -> Tensor:
        actual_top_k = min(self.top_k_clusters, self.centroids_t.shape[1])

        cluster_scores = hidden_vector @ self.centroids_t

        top_cluster_ids = torch.topk(
            cluster_scores,
            k=actual_top_k,
            sorted=False,
        ).indices

        candidate_vectors = self.clustered_lm_head[top_cluster_ids]
        candidate_scores = candidate_vectors @ hidden_vector

        if self.clustered_lm_head_bias is not None:
            candidate_scores = (
                candidate_scores
                + self.clustered_lm_head_bias[top_cluster_ids]
            )

        candidate_scores_flat = candidate_scores.reshape(-1)
        candidate_token_ids = self.cluster_to_token_ids[top_cluster_ids].reshape(-1)

        max_score = candidate_scores_flat.max()
        return candidate_token_ids[candidate_scores_flat == max_score].min()


def extract_lm_head(model: Any) -> tuple[Tensor, Tensor | None]:
    output_embeddings = model.get_output_embeddings()

    if output_embeddings is None:
        raise ValueError(
            "Model does not expose output embeddings via get_output_embeddings()"
        )

    weight = getattr(output_embeddings, "weight", None)
    if not isinstance(weight, Tensor):
        raise ValueError("Output embeddings do not expose a Tensor weight")

    bias = getattr(output_embeddings, "bias", None)
    if bias is not None and not isinstance(bias, Tensor):
        raise ValueError("Output embeddings bias is not a Tensor")

    return weight, bias