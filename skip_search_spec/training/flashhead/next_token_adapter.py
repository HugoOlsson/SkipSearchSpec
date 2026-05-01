from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from skip_search_spec.training.flashhead.building_clusters import l2_normalize
from skip_search_spec.training.flashhead.storage import load_flashhead


class FlashHeadModule(nn.Module):
    """
    Minimal FlashHead module for greedy next-token lookup from one hidden vector.

    The hidden vector is expected to be the vector that would normally be passed
    to the LM head: shape [hidden_size], already final-normalized if the model
    architecture applies a final norm before lm_head.
    """

    centroids_t: Tensor
    cluster_to_token_ids: Tensor
    lm_head_vector_table: Tensor
    lm_head_bias: Tensor | None
    top_k_clusters: int
    normalize_hidden_for_routing: bool

    def __init__(
        self,
        *,
        centroids: Tensor,
        cluster_to_token_ids: Tensor,
        lm_head_vector_table: Tensor,
        top_k_clusters: int,
        normalize_hidden_for_routing: bool = True,
        lm_head_bias: Tensor | None = None,
    ) -> None:
        super().__init__()

        if top_k_clusters < 1:
            raise ValueError("top_k_clusters must be >= 1")

        self.top_k_clusters = top_k_clusters
        self.normalize_hidden_for_routing = normalize_hidden_for_routing

        self.register_buffer(
            "centroids_t", centroids.detach().transpose(0, 1).contiguous()
        )
        self.register_buffer(
            "cluster_to_token_ids", cluster_to_token_ids.detach().long()
        )
        self.register_buffer("lm_head_vector_table", lm_head_vector_table.detach())
        self.register_buffer(
            "lm_head_bias", None if lm_head_bias is None else lm_head_bias.detach()
        )

        self._validate_buffers()

    @classmethod
    def from_model(
        cls,
        *,
        model: Any,
        flashhead_path: str | Path,
        top_k_clusters: int,
        normalize_hidden_for_routing: bool = True,
    ) -> FlashHeadModule:
        stored = load_flashhead(flashhead_path)
        lm_head_vector_table, lm_head_bias = extract_lm_head(model)
        return cls(
            centroids=stored.centroids,
            cluster_to_token_ids=stored.cluster_to_token_ids,
            lm_head_vector_table=lm_head_vector_table,
            lm_head_bias=lm_head_bias,
            top_k_clusters=top_k_clusters,
            normalize_hidden_for_routing=normalize_hidden_for_routing,
        ).to(device=lm_head_vector_table.device, dtype=lm_head_vector_table.dtype)

    @torch.inference_mode()
    def find_token(self, hidden_vector: Tensor) -> Tensor:
        route_hidden = (
            l2_normalize(hidden_vector, dim=-1)
            if self.normalize_hidden_for_routing
            else hidden_vector
        )

        actual_top_k = min(self.top_k_clusters, int(self.centroids_t.shape[1]))
        cluster_scores = route_hidden @ self.centroids_t
        top_cluster_ids = torch.topk(cluster_scores, k=actual_top_k).indices

        candidate_token_ids = self.cluster_to_token_ids[top_cluster_ids].reshape(-1)
        candidate_token_ids = candidate_token_ids[candidate_token_ids >= 0].long()

        candidate_token_vectors = self.lm_head_vector_table[candidate_token_ids]
        candidate_scores = candidate_token_vectors @ hidden_vector

        if self.lm_head_bias is not None:
            candidate_scores = candidate_scores + self.lm_head_bias[candidate_token_ids]

        max_score = candidate_scores.max()
        return candidate_token_ids[candidate_scores == max_score].min()

    def _validate_buffers(self) -> None:
        if self.centroids_t.ndim != 2:
            raise ValueError("centroids_t must have shape [hidden_size, num_clusters]")
        if self.cluster_to_token_ids.ndim != 2:
            raise ValueError(
                "cluster_to_token_ids must have shape [num_clusters, max_cluster_size]"
            )
        if self.lm_head_vector_table.ndim != 2:
            raise ValueError(
                "lm_head_vector_table must have shape [vocab_size, hidden_size]"
            )
        if self.cluster_to_token_ids.shape[0] != self.centroids_t.shape[1]:
            raise ValueError(
                "cluster_to_token_ids and centroids must agree on num_clusters"
            )
        if self.lm_head_vector_table.shape[1] != self.centroids_t.shape[0]:
            raise ValueError("lm_head and centroids must agree on hidden_size")
        if self.lm_head_bias is not None and self.lm_head_bias.ndim != 1:
            raise ValueError("lm_head_bias must have shape [vocab_size]")


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
