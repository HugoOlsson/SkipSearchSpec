from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class StoredFlashHead:
    cluster_to_token_ids: Tensor  # [num_clusters, cluster_size]
    centroids: Tensor             # [num_clusters, hidden_size]


def _validate_flashhead_tensors(
    *,
    cluster_to_token_ids: Tensor,
    centroids: Tensor,
) -> None:
    if cluster_to_token_ids.ndim != 2:
        raise ValueError(
            "cluster_to_token_ids must have shape [num_clusters, cluster_size]"
        )
    if centroids.ndim != 2:
        raise ValueError("centroids must have shape [num_clusters, hidden_size]")

    num_clusters = int(centroids.shape[0])

    if cluster_to_token_ids.shape[0] != num_clusters:
        raise ValueError(
            "cluster_to_token_ids and centroids must have the same num_clusters"
        )

    if (cluster_to_token_ids < 0).any():
        raise ValueError("cluster_to_token_ids must not contain negative token ids")


def save_flashhead(
    *,
    path: str | Path,
    cluster_to_token_ids: Tensor,
    centroids: Tensor,
) -> None:
    _validate_flashhead_tensors(
        cluster_to_token_ids=cluster_to_token_ids,
        centroids=centroids,
    )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "format": "flashhead_research_v2",
        "cluster_to_token_ids": cluster_to_token_ids.detach().cpu(),
        "centroids": centroids.detach().cpu(),
    }

    torch.save(payload, path)


def load_flashhead(path: str | Path) -> StoredFlashHead:
    path = Path(path)
    payload = cast(dict[str, Any], torch.load(path, map_location="cpu"))

    cluster_to_token_ids = cast(Tensor, payload["cluster_to_token_ids"])
    centroids = cast(Tensor, payload["centroids"])

    _validate_flashhead_tensors(
        cluster_to_token_ids=cluster_to_token_ids,
        centroids=centroids,
    )

    return StoredFlashHead(
        cluster_to_token_ids=cluster_to_token_ids,
        centroids=centroids,
    )