

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, cast

import torch
from torch import Tensor

@dataclass(frozen=True, slots=True)
class StoredFlashHead:
    token_to_cluster_mapping: Tensor          # [vocab_size]
    cluster_to_token_ids: Tensor             # [num_clusters, cluster_size]
    centroids: Tensor                        # [num_clusters, hidden_size]
    cluster_sizes: Tensor                    # [num_clusters]



def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()



def _validate_flashhead_tensors(
    *,
    token_to_cluster_mapping: Tensor,
    cluster_to_token_ids: Tensor,
    centroids: Tensor,
    cluster_sizes: Tensor,
) -> None:
    if token_to_cluster_mapping.ndim != 1:
        raise ValueError("token_to_cluster_mapping must have shape [vocab_size]")
    if cluster_to_token_ids.ndim != 2:
        raise ValueError("cluster_to_token_ids must have shape [num_clusters, cluster_size]")
    if centroids.ndim != 2:
        raise ValueError("centroids must have shape [num_clusters, hidden_size]")
    if cluster_sizes.ndim != 1:
        raise ValueError("cluster_sizes must have shape [num_clusters]")

    num_clusters = int(centroids.shape[0])
    vocab_size = int(token_to_cluster_mapping.shape[0])

    if cluster_to_token_ids.shape[0] != num_clusters:
        raise ValueError("cluster_to_token_ids and centroids must have the same num_clusters")
    if cluster_sizes.shape[0] != num_clusters:
        raise ValueError("cluster_sizes and centroids must have the same num_clusters")
    if int(cluster_sizes.sum().item()) != vocab_size:
        raise ValueError("cluster_sizes must sum to vocab_size")



def save_flashhead(
    *,
    path: str | Path,
    token_to_cluster_mapping: Tensor,
    cluster_to_token_ids: Tensor,
    centroids: Tensor,
    cluster_sizes: Tensor,
) -> None:
    _validate_flashhead_tensors(
        token_to_cluster_mapping=token_to_cluster_mapping,
        cluster_to_token_ids=cluster_to_token_ids,
        centroids=centroids,
        cluster_sizes=cluster_sizes,
    )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "format": "flashhead_research_v1",
        "token_to_cluster_mapping": token_to_cluster_mapping.detach().cpu(),
        "cluster_to_token_ids": cluster_to_token_ids.detach().cpu(),
        "centroids": centroids.detach().cpu(),
        "cluster_sizes": cluster_sizes.detach().cpu(),
    }

    torch.save(payload, path)



def load_flashhead(path: str | Path) -> StoredFlashHead:
    path = Path(path)
    payload = cast(dict[str, Any], torch.load(path, map_location="cpu"))

    token_to_cluster_mapping = cast(Tensor, payload["token_to_cluster_mapping"])
    cluster_to_token_ids = cast(Tensor, payload["cluster_to_token_ids"])
    centroids = cast(Tensor, payload["centroids"])
    cluster_sizes = cast(Tensor, payload["cluster_sizes"])

    _validate_flashhead_tensors(
        token_to_cluster_mapping=token_to_cluster_mapping,
        cluster_to_token_ids=cluster_to_token_ids,
        centroids=centroids,
        cluster_sizes=cluster_sizes,
    )


    return StoredFlashHead(
        token_to_cluster_mapping=token_to_cluster_mapping,
        cluster_to_token_ids=cluster_to_token_ids,
        centroids=centroids,
        cluster_sizes=cluster_sizes,
    )

