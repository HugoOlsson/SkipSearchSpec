from __future__ import annotations

import os
from pathlib import Path
import time
from typing import Any

import torch
from torch import Tensor, nn

from skip_search_spec.training.flashhead.storage import load_flashhead

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


PROFILE_STAGE_NAMES = (
    "cluster_matmul",
    "topk",
    "triton_stage2",
    "triton_final_tie_break",
    "candidate_gather",
    "candidate_matmul",
    "bias_add",
    "flatten_token_ids",
    "tie_break",
)


if triton is not None and tl is not None:

    @triton.jit
    def _triton_stage2_kernel(
        hidden_ptr,
        top_cluster_ids_ptr,
        clustered_lm_head_ptr,
        cluster_to_token_ids_ptr,
        partial_scores_ptr,
        partial_token_ids_ptr,
        num_candidates: tl.constexpr,
        cluster_size: tl.constexpr,
        hidden_size: tl.constexpr,
        BLOCK_CANDIDATES: tl.constexpr,
        BLOCK_HIDDEN: tl.constexpr,
    ):
        program_id = tl.program_id(0)
        candidate_offsets = (
            program_id * BLOCK_CANDIDATES
            + tl.arange(0, BLOCK_CANDIDATES)
        )
        candidate_mask = candidate_offsets < num_candidates
        cluster_slots = candidate_offsets // cluster_size
        token_offsets = candidate_offsets - cluster_slots * cluster_size
        cluster_ids = tl.load(
            top_cluster_ids_ptr + cluster_slots,
            mask=candidate_mask,
            other=0,
        )
        row_offsets = cluster_ids * cluster_size + token_offsets

        scores = tl.zeros((BLOCK_CANDIDATES,), dtype=tl.float32)
        hidden_offsets = tl.arange(0, BLOCK_HIDDEN)
        for hidden_start in range(0, hidden_size, BLOCK_HIDDEN):
            hidden_indices = hidden_start + hidden_offsets
            hidden_mask = hidden_indices < hidden_size
            hidden_values = tl.load(
                hidden_ptr + hidden_indices,
                mask=hidden_mask,
                other=0.0,
            ).to(tl.float32)
            weight_offsets = (
                row_offsets[:, None] * hidden_size
                + hidden_indices[None, :]
            )
            weights = tl.load(
                clustered_lm_head_ptr + weight_offsets,
                mask=candidate_mask[:, None] & hidden_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            scores += tl.sum(weights * hidden_values[None, :], axis=1)

        token_ids = tl.load(
            cluster_to_token_ids_ptr + row_offsets,
            mask=candidate_mask,
            other=2147483647,
        )
        scores = tl.where(candidate_mask, scores, -float("inf"))
        best_score = tl.max(scores, axis=0)
        tie_token_ids = tl.where(scores == best_score, token_ids, 2147483647)
        best_token_id = tl.min(tie_token_ids, axis=0)

        tl.store(partial_scores_ptr + program_id, best_score)
        tl.store(partial_token_ids_ptr + program_id, best_token_id)

else:
    _triton_stage2_kernel = None


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

        self.profile_enabled = _env_flag("SKIP_SEARCH_FLASHHEAD_PROFILE")
        self.profile_sync_device = _env_flag(
            "SKIP_SEARCH_FLASHHEAD_PROFILE_SYNC",
            default=True,
        )
        self.profile_print_every = _env_int(
            "SKIP_SEARCH_FLASHHEAD_PROFILE_PRINT_EVERY",
            default=0,
        )
        self.triton_stage2_enabled = _env_flag("SKIP_SEARCH_FLASHHEAD_TRITON_STAGE2")
        self.triton_block_candidates = _env_int(
            "SKIP_SEARCH_FLASHHEAD_TRITON_BLOCK_CANDIDATES",
            default=128,
        )
        self.triton_block_hidden = _env_int(
            "SKIP_SEARCH_FLASHHEAD_TRITON_BLOCK_HIDDEN",
            default=64,
        )
        self._did_warn_triton_unavailable = False
        self._did_print_triton_enabled = False
        self.reset_profile()

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

        if self.profile_enabled:
            return self._find_token_profiled(
                hidden_vector=hidden_vector,
                actual_top_k=actual_top_k,
            )

        cluster_scores = hidden_vector @ self.centroids_t

        top_cluster_ids = torch.topk(
            cluster_scores,
            k=actual_top_k,
            sorted=False,
        ).indices
        if self._can_use_triton_stage2(hidden_vector):
            return self._find_token_triton_stage2(
                hidden_vector=hidden_vector,
                top_cluster_ids=top_cluster_ids,
            )

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

    def _find_token_profiled(
        self,
        *,
        hidden_vector: Tensor,
        actual_top_k: int,
    ) -> Tensor:
        profile_last_time = self._profile_start(hidden_vector)

        cluster_scores = hidden_vector @ self.centroids_t
        profile_last_time = self._profile_mark(
            "cluster_matmul",
            profile_last_time,
            hidden_vector,
        )

        top_cluster_ids = torch.topk(
            cluster_scores,
            k=actual_top_k,
            sorted=False,
        ).indices
        profile_last_time = self._profile_mark(
            "topk",
            profile_last_time,
            hidden_vector,
        )
        if self._can_use_triton_stage2(hidden_vector):
            partial_scores, partial_token_ids = self._run_triton_stage2_partials(
                hidden_vector=hidden_vector,
                top_cluster_ids=top_cluster_ids,
            )
            profile_last_time = self._profile_mark(
                "triton_stage2",
                profile_last_time,
                hidden_vector,
            )
            max_score = partial_scores.max()
            next_token = partial_token_ids[partial_scores == max_score].min()
            self._profile_finish(
                "triton_final_tie_break",
                profile_last_time,
                hidden_vector,
            )
            return next_token

        candidate_vectors = self.clustered_lm_head[top_cluster_ids]
        profile_last_time = self._profile_mark(
            "candidate_gather",
            profile_last_time,
            hidden_vector,
        )

        candidate_scores = candidate_vectors @ hidden_vector
        profile_last_time = self._profile_mark(
            "candidate_matmul",
            profile_last_time,
            hidden_vector,
        )

        if self.clustered_lm_head_bias is not None:
            candidate_scores = (
                candidate_scores
                + self.clustered_lm_head_bias[top_cluster_ids]
            )
        profile_last_time = self._profile_mark(
            "bias_add",
            profile_last_time,
            hidden_vector,
        )

        candidate_scores_flat = candidate_scores.reshape(-1)
        candidate_token_ids = self.cluster_to_token_ids[top_cluster_ids].reshape(-1)
        profile_last_time = self._profile_mark(
            "flatten_token_ids",
            profile_last_time,
            hidden_vector,
        )

        max_score = candidate_scores_flat.max()
        next_token = candidate_token_ids[candidate_scores_flat == max_score].min()
        self._profile_finish(
            "tie_break",
            profile_last_time,
            hidden_vector,
        )
        return next_token

    def reset_profile(self) -> None:
        self.profile_call_count = 0
        self.profile_seconds = dict.fromkeys(PROFILE_STAGE_NAMES, 0.0)

    def set_profile_enabled(
        self,
        enabled: bool = True,
        *,
        print_every: int | None = None,
        sync_device: bool | None = None,
    ) -> None:
        self.profile_enabled = enabled
        if print_every is not None:
            self.profile_print_every = int(print_every)
        if sync_device is not None:
            self.profile_sync_device = sync_device

    def set_triton_stage2_enabled(self, enabled: bool = True) -> None:
        self.triton_stage2_enabled = enabled

    def _can_use_triton_stage2(self, hidden_vector: Tensor) -> bool:
        if not self.triton_stage2_enabled:
            return False

        reason: str | None = None
        if triton is None or tl is None or _triton_stage2_kernel is None:
            reason = "Triton is not installed"
        elif hidden_vector.device.type != "cuda":
            reason = f"Triton stage-2 requires CUDA, got {hidden_vector.device.type}"
        elif self.clustered_lm_head_bias is not None:
            reason = "Triton stage-2 minimal path does not support LM-head bias"
        elif hidden_vector.ndim != 1:
            reason = "Triton stage-2 expects a 1D hidden vector"
        elif not hidden_vector.is_contiguous():
            reason = "hidden_vector must be contiguous"
        elif not self.clustered_lm_head.is_contiguous():
            reason = "clustered_lm_head must be contiguous"
        elif not self.cluster_to_token_ids.is_contiguous():
            reason = "cluster_to_token_ids must be contiguous"

        if reason is None:
            if not self._did_print_triton_enabled:
                print("FlashHead Triton stage-2 enabled.")
                self._did_print_triton_enabled = True
            return True

        if not self._did_warn_triton_unavailable:
            print(f"FlashHead Triton stage-2 disabled: {reason}.")
            self._did_warn_triton_unavailable = True
        return False

    def _find_token_triton_stage2(
        self,
        *,
        hidden_vector: Tensor,
        top_cluster_ids: Tensor,
    ) -> Tensor:
        if triton is None or _triton_stage2_kernel is None:
            raise RuntimeError("Triton stage-2 requested but Triton is unavailable.")

        partial_scores, partial_token_ids = self._run_triton_stage2_partials(
            hidden_vector=hidden_vector,
            top_cluster_ids=top_cluster_ids,
        )
        max_score = partial_scores.max()
        return partial_token_ids[partial_scores == max_score].min()

    def _run_triton_stage2_partials(
        self,
        *,
        hidden_vector: Tensor,
        top_cluster_ids: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if triton is None or _triton_stage2_kernel is None:
            raise RuntimeError("Triton stage-2 requested but Triton is unavailable.")

        num_candidates = int(top_cluster_ids.numel() * self.cluster_size)
        block_candidates = int(self.triton_block_candidates)
        block_hidden = int(self.triton_block_hidden)
        num_blocks = triton.cdiv(num_candidates, block_candidates)
        partial_scores = torch.empty(
            num_blocks,
            device=hidden_vector.device,
            dtype=torch.float32,
        )
        partial_token_ids = torch.empty(
            num_blocks,
            device=hidden_vector.device,
            dtype=self.cluster_to_token_ids.dtype,
        )

        _triton_stage2_kernel[(num_blocks,)](
            hidden_vector,
            top_cluster_ids,
            self.clustered_lm_head,
            self.cluster_to_token_ids,
            partial_scores,
            partial_token_ids,
            num_candidates,
            self.cluster_size,
            self.clustered_lm_head.shape[2],
            BLOCK_CANDIDATES=block_candidates,
            BLOCK_HIDDEN=block_hidden,
        )
        return partial_scores, partial_token_ids

    def profile_summary(self, *, reset: bool = False) -> dict[str, Any]:
        total_seconds = sum(self.profile_seconds.values())
        calls = self.profile_call_count
        stages: list[dict[str, float | str]] = []

        for name in PROFILE_STAGE_NAMES:
            seconds = self.profile_seconds[name]
            stages.append(
                {
                    "stage": name,
                    "seconds": seconds,
                    "percent": 0.0
                    if total_seconds == 0.0
                    else seconds / total_seconds * 100.0,
                    "milliseconds_per_call": 0.0
                    if calls == 0
                    else seconds / calls * 1000.0,
                }
            )

        summary = {
            "calls": calls,
            "total_seconds": total_seconds,
            "milliseconds_per_call": 0.0
            if calls == 0
            else total_seconds / calls * 1000.0,
            "stages": stages,
        }

        if reset:
            self.reset_profile()

        return summary

    def print_profile_summary(self, *, reset: bool = False) -> None:
        summary = self.profile_summary(reset=reset)
        print("FlashHead find_token profile")
        print(f"  calls: {summary['calls']}")
        print(f"  total_ms: {summary['total_seconds'] * 1000.0:.3f}")
        print(f"  ms/call: {summary['milliseconds_per_call']:.3f}")
        for stage in summary["stages"]:
            print(
                f"  {stage['stage']:>18}: "
                f"{stage['milliseconds_per_call']:.3f} ms/call "
                f"({stage['percent']:.1f}%)"
            )

    def _profile_start(self, tensor: Tensor) -> float | None:
        if not self.profile_enabled:
            return None

        self._profile_sync(tensor)
        return time.perf_counter()

    def _profile_mark(
        self,
        name: str,
        previous_time: float | None,
        tensor: Tensor,
    ) -> float | None:
        if previous_time is None:
            return None

        self._profile_sync(tensor)
        now = time.perf_counter()
        self.profile_seconds[name] += now - previous_time
        return now

    def _profile_finish(
        self,
        name: str,
        previous_time: float | None,
        tensor: Tensor,
    ) -> None:
        if previous_time is None:
            return

        self._profile_mark(name, previous_time, tensor)
        self.profile_call_count += 1

        if (
            self.profile_print_every > 0
            and self.profile_call_count % self.profile_print_every == 0
        ):
            self.print_profile_summary(reset=False)

    def _profile_sync(self, tensor: Tensor) -> None:
        if not self.profile_sync_device:
            return

        device = tensor.device
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elif device.type == "mps" and hasattr(torch, "mps"):
            torch.mps.synchronize()


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


def _env_flag(name: str, *, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default

    return value.lower().strip() not in {"", "0", "false", "no", "off"}


def _env_int(name: str, *, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default

    return int(value)
