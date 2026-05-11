from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from skip_search_spec.training.flashhead.storage import load_flashhead


class FlashHeadModule(nn.Module):
    """
    Memory-light FlashHead module for greedy next-token lookup.

    Runtime-owned tensors:
      - centroids_t: [hidden_size, num_clusters]
      - cluster_to_token_ids: [num_clusters, cluster_size]

    Borrowed model tensors:
      - lm_head_vector_table: [vocab_size, hidden_size]
      - lm_head_bias: [vocab_size] or None

    The LM-head weight is not copied or registered as module state.
    """

    centroids_t: Tensor
    cluster_to_token_ids: Tensor
    lm_head_vector_table: Tensor
    lm_head_bias: Tensor | None
    top_k_clusters: int
    cluster_size: int
    num_clusters: int
    vocab_size: int

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
            raise ValueError("cluster_to_token_ids must not contain negative ids")

        num_clusters, cluster_size = cluster_to_token_ids.shape
        vocab_size, hidden_size = lm_head_vector_table.shape

        if centroids.shape != (num_clusters, hidden_size):
            raise ValueError(
                "centroids shape mismatch: "
                f"expected {(num_clusters, hidden_size)}, got {tuple(centroids.shape)}"
            )

        if int(cluster_to_token_ids.max().item()) >= vocab_size:
            raise ValueError("cluster_to_token_ids contains token ids >= vocab_size")

        if vocab_size != num_clusters * cluster_size:
            raise ValueError(
                "Strict equal clusters require "
                "vocab_size == num_clusters * cluster_size"
            )

        if lm_head_bias is not None and lm_head_bias.shape != (vocab_size,):
            raise ValueError("lm_head_bias must have shape [vocab_size]")

        self.top_k_clusters = int(top_k_clusters)
        self.cluster_size = int(cluster_size)
        self.num_clusters = int(num_clusters)
        self.vocab_size = int(vocab_size)

        self.register_buffer(
            "centroids_t",
            centroids.detach().transpose(0, 1).contiguous(),
        )
        self.register_buffer(
            "cluster_to_token_ids",
            cluster_to_token_ids.detach().long().contiguous(),
        )

        # Plain attributes on purpose:
        # these borrow the model's existing LM-head tensors and should not be
        # saved, copied, or moved by FlashHeadModule.to().
        self.lm_head_vector_table = lm_head_vector_table.detach()
        self.lm_head_bias = None if lm_head_bias is None else lm_head_bias.detach()

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

        device = lm_head_vector_table.device
        dtype = lm_head_vector_table.dtype

        return cls(
            centroids=stored.centroids.to(device=device, dtype=dtype),
            cluster_to_token_ids=stored.cluster_to_token_ids.to(device=device),
            lm_head_vector_table=lm_head_vector_table,
            lm_head_bias=lm_head_bias,
            top_k_clusters=top_k_clusters,
        )

    @torch.inference_mode()
    def find_token(self, hidden_vector: Tensor) -> Tensor:
        actual_top_k = min(self.top_k_clusters, self.num_clusters)

        cluster_scores = hidden_vector @ self.centroids_t

        top_cluster_ids = torch.topk(
            cluster_scores,
            k=actual_top_k,
            sorted=False,
        ).indices

        candidate_token_ids = self.cluster_to_token_ids[top_cluster_ids].reshape(-1)

        candidate_vectors = self.lm_head_vector_table.index_select(
            0,
            candidate_token_ids,
        )

        candidate_scores = candidate_vectors @ hidden_vector

        if self.lm_head_bias is not None:
            candidate_scores = candidate_scores + self.lm_head_bias.index_select(
                0,
                candidate_token_ids,
            )

        best_score = candidate_scores.max()
        return candidate_token_ids[candidate_scores == best_score].min()


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