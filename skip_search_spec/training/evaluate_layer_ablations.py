from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, cast
import json
import math
import time
from pathlib import Path
import re

import torch

from skip_search_spec.helpers.tooling import (
    distribution_similarity_metrics,
    get_preferred_device,
    get_preferred_float_dtype,
    load_model_and_tokenizer,
)

from skip_search_spec.protocols.windows import (
    DatasetSpec,
    ModelAndTokenizer,
)

# Rename this import path to wherever you placed the shared helper file.
from skip_search_spec.helpers.shared_decoding_tools import (
    build_fixed_window_dataloader,
    cross_entropy_next_token,
    forward_model_logits,
    forward_with_layer_mask,
    get_decoder_layers,
    kl_teacher_to_student_next_token,
    make_layer_pattern,
    shift_next_token_logits,
    stage as shared_stage,
)


@dataclass(frozen=True, slots=True)
class AblationMaskSpec:
    name: str
    keep_layer_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class AblationResult:
    mask_name: str
    kept_layers: tuple[int, ...]
    binary_mask: tuple[int, ...]
    visual_mask: str
    num_kept_layers: int
    num_removed_layers: int
    num_total_layers: int
    keep_fraction: float
    remove_fraction: float
    mean_ce_masked: float
    mean_ce_full: float
    mean_ce_gap: float
    mean_kl_full_to_masked: float
    kl_per_removed_layer: float | None
    mean_js: float
    mean_top1_agreement: float
    mean_overlap: float
    mean_p_masked_on_full_argmax: float


def _stage(
    message: str,
    *,
    model_name: str | None = None,
    ablation_idx: int | None = None,
    num_ablations: int | None = None,
) -> None:
    """
    Small local wrapper around the shared timestamp logger.

    The shared helper owns timestamp formatting.
    This wrapper just adds ablation/model context for this file.
    """
    prefix_parts: list[str] = []

    if ablation_idx is not None and num_ablations is not None:
        prefix_parts.append(f"ablation={ablation_idx}/{num_ablations}")

    if model_name is not None:
        prefix_parts.append(f"[model={model_name}]")

    prefix = ""
    if prefix_parts:
        prefix = " ".join(prefix_parts) + " "

    shared_stage(f"{prefix}{message}")


def _safe_div(numerator: float, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return numerator / float(denominator)


def _format_optional_float(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "NA"
    return f"{value:.6f}"


def _format_binary_mask(binary_mask: tuple[int, ...]) -> str:
    return "[" + ",".join(str(x) for x in binary_mask) + "]"


def _sanitize_for_filename(text: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    sanitized = sanitized.strip("._")
    return sanitized or "model"


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]

    return str(value)


def _result_to_json_dict(
    result: AblationResult,
    *,
    ablation_index: int,
) -> dict[str, Any]:
    return {
        "ablation_index": ablation_index,
        "mask_name": result.mask_name,
        "visual_mask": result.visual_mask,
        "binary_mask": list(result.binary_mask),
        "binary_mask_string": _format_binary_mask(result.binary_mask),
        "kept_layers": list(result.kept_layers),
        "num_kept_layers": result.num_kept_layers,
        "num_removed_layers": result.num_removed_layers,
        "num_total_layers": result.num_total_layers,
        "keep_fraction": result.keep_fraction,
        "remove_fraction": result.remove_fraction,
        "mean_ce_masked": result.mean_ce_masked,
        "mean_ce_full": result.mean_ce_full,
        "mean_ce_gap": result.mean_ce_gap,
        "mean_kl_full_to_masked": result.mean_kl_full_to_masked,
        "kl_per_removed_layer": result.kl_per_removed_layer,
        "mean_js": result.mean_js,
        "mean_top1_agreement": result.mean_top1_agreement,
        "mean_overlap": result.mean_overlap,
        "mean_p_masked_on_full_argmax": result.mean_p_masked_on_full_argmax,
    }


def _save_ablation_results_json(
    *,
    model_name: str,
    dataset_spec: DatasetSpec,
    context_len: int,
    max_examples: int,
    num_windows_to_use: int,
    batch_size: int,
    num_layers: int,
    num_ablations: int,
    num_batches: int,
    device: torch.device,
    compute_dtype: torch.dtype,
    model_kwargs: dict[str, Any] | None,
    results: list[AblationResult],
    output_dir: str | Path = "ablation_results",
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp_for_filename = time.strftime("%Y%m%d_%H%M%S")
    timestamp_readable = time.strftime("%Y-%m-%d %H:%M:%S")

    output_path = output_dir / (
        f"layer_ablations_{_sanitize_for_filename(model_name)}_{timestamp_for_filename}.json"
    )

    payload = {
        "schema_version": 2,
        "created_at": timestamp_readable,
        "model_name": model_name,
        "num_layers": num_layers,
        "evaluation_config": {
            "context_len": context_len,
            "max_examples": max_examples,
            "num_windows_to_use": num_windows_to_use,
            "batch_size": batch_size,
            "num_batches": num_batches,
            "num_ablations": num_ablations,
            "num_results": len(results),
            "device": str(device),
            "compute_dtype": str(compute_dtype),
            "dataset_spec_repr": repr(dataset_spec),
            "model_kwargs": _json_safe(model_kwargs or {}),
        },
        "results": [
            _result_to_json_dict(result, ablation_index=i)
            for i, result in enumerate(results, start=1)
        ],
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print()
    print(f"Saved ablation results JSON to: {output_path}")
    return output_path


@dataclass(frozen=True, slots=True)
class AblationGenerationConfig:
    """
    Centralized ablation-budget settings.

    Default approximate count:
      keep_all:       1
      early_exit:    24
      late_begin:     8
      internal_gaps:  8 * 5 = 40
      periodic:      18
      ------------------
      total:         ~91 before deduplication
    """

    include_keep_all: bool = True

    # 1. Early exits: keep prefix [0, k)
    num_early_exit_masks: int = 24
    early_exit_min_keep: int = 1
    early_exit_max_keep: int | None = None

    # 2. Late-begin: keep suffix [num_layers - k, num_layers)
    num_late_begin_masks: int = 8
    late_begin_min_keep: int = 1
    late_begin_max_keep: int | None = None

    # 3. Internal gaps: remove one contiguous internal block
    num_internal_gap_lengths: int = 8
    internal_gap_positions_per_length: int = 5
    internal_gap_min_length: int = 2
    internal_gap_max_remove_fraction: float = 0.85

    # For each gap length, decide how much of the non-skipped budget goes before the gap.
    # Low value  => gap is early, more layers remain at the end.
    # High value => gap is late, more layers remain at the start.
    internal_gap_left_keep_share_min: float = 0.15
    internal_gap_left_keep_share_max: float = 0.85

    # 4. Periodic masks
    periodic_steps: tuple[int, ...] = (2, 3, 4)

    # drop_every_n = skip one layer every n layers.
    # Example: drop_every_3 keeps roughly 2/3 of layers.
    include_periodic_drop_every_n: bool = True

    # keep_every_n = keep one layer every n layers.
    # Example: keep_every_3 keeps roughly 1/3 of layers.
    include_periodic_keep_every_n: bool = True

    # None means include all phases.
    # For steps (2, 3, 4), all phases gives 2 + 3 + 4 = 9 masks per periodic family.
    periodic_max_phases_per_step: int | None = None

    # If True, always force layer 0 and layer num_layers - 1 to be kept for periodic masks.
    periodic_anchor_edges: bool = False


def _downsample_sorted_values(
    values: list[int],
    count: int,
) -> tuple[int, ...]:
    if count <= 0 or len(values) == 0:
        return ()

    values = sorted(set(values))

    if len(values) <= count:
        return tuple(values)

    if count == 1:
        return (values[len(values) // 2],)

    selected_indices = [
        round(i * (len(values) - 1) / (count - 1))
        for i in range(count)
    ]

    return tuple(values[i] for i in sorted(set(selected_indices)))


def _sample_int_range(
    *,
    low: int,
    high: int,
    count: int,
    anchors: Iterable[int] = (),
) -> tuple[int, ...]:
    """
    Sample at most `count` integers from [low, high].

    Uses:
      - explicit anchors first
      - evenly spaced values to fill the rest

    This keeps the number of masks bounded and stable across model sizes.
    """
    if count <= 0 or high < low:
        return ()

    full_range_size = high - low + 1
    if full_range_size <= count:
        return tuple(range(low, high + 1))

    anchor_values = sorted({
        int(x)
        for x in anchors
        if low <= int(x) <= high
    })

    # Oversample evenly spaced candidates, then downsample.
    candidate_values: set[int] = set(anchor_values)

    oversample_count = max(count * 4, count)
    if oversample_count == 1:
        candidate_values.add((low + high) // 2)
    else:
        for i in range(oversample_count):
            t = i / float(oversample_count - 1)
            candidate_values.add(round(low + t * (high - low)))

    candidates = sorted(candidate_values)

    if len(candidates) <= count:
        return tuple(candidates)

    if len(anchor_values) >= count:
        return _downsample_sorted_values(anchor_values, count)

    remaining_count = count - len(anchor_values)
    non_anchor_values = [x for x in candidates if x not in set(anchor_values)]
    filler_values = _downsample_sorted_values(non_anchor_values, remaining_count)

    return tuple(sorted(set(anchor_values).union(filler_values)))


def _sample_float_range(
    *,
    low: float,
    high: float,
    count: int,
) -> tuple[float, ...]:
    if count <= 0:
        return ()

    if count == 1:
        return ((low + high) / 2.0,)

    return tuple(
        low + (i / float(count - 1)) * (high - low)
        for i in range(count)
    )


def _add_unique_mask(
    *,
    dedup: dict[tuple[int, ...], AblationMaskSpec],
    num_layers: int,
    name: str,
    indices: Iterable[int],
) -> None:
    kept = tuple(sorted({
        int(i)
        for i in indices
        if 0 <= int(i) < num_layers
    }))

    if len(kept) == 0:
        return

    if kept not in dedup:
        dedup[kept] = AblationMaskSpec(
            name=name,
            keep_layer_indices=kept,
        )


def _make_early_exit_masks(
    *,
    num_layers: int,
    config: AblationGenerationConfig,
    dedup: dict[tuple[int, ...], AblationMaskSpec],
) -> None:
    max_keep = config.early_exit_max_keep
    if max_keep is None:
        max_keep = num_layers - 1

    low = max(1, config.early_exit_min_keep)
    high = min(num_layers - 1, max_keep)

    anchors = [
        1,
        2,
        3,
        4,
        6,
        8,
        num_layers // 8,
        num_layers // 6,
        num_layers // 4,
        num_layers // 3,
        num_layers // 2,
        (2 * num_layers) // 3,
        (3 * num_layers) // 4,
        num_layers - 2,
        num_layers - 1,
    ]

    keep_counts = _sample_int_range(
        low=low,
        high=high,
        count=config.num_early_exit_masks,
        anchors=anchors,
    )

    for k in keep_counts:
        _add_unique_mask(
            dedup=dedup,
            num_layers=num_layers,
            name=f"keep_prefix__k_{k}",
            indices=range(0, k),
        )


def _make_late_begin_masks(
    *,
    num_layers: int,
    config: AblationGenerationConfig,
    dedup: dict[tuple[int, ...], AblationMaskSpec],
) -> None:
    max_keep = config.late_begin_max_keep
    if max_keep is None:
        max_keep = num_layers - 1

    low = max(1, config.late_begin_min_keep)
    high = min(num_layers - 1, max_keep)

    anchors = [
        1,
        2,
        4,
        num_layers // 8,
        num_layers // 4,
        num_layers // 3,
        num_layers // 2,
        (3 * num_layers) // 4,
        num_layers - 1,
    ]

    keep_counts = _sample_int_range(
        low=low,
        high=high,
        count=config.num_late_begin_masks,
        anchors=anchors,
    )

    for k in keep_counts:
        _add_unique_mask(
            dedup=dedup,
            num_layers=num_layers,
            name=f"keep_suffix__k_{k}",
            indices=range(num_layers - k, num_layers),
        )


def _make_internal_gap_masks(
    *,
    num_layers: int,
    config: AblationGenerationConfig,
    dedup: dict[tuple[int, ...], AblationMaskSpec],
) -> None:
    """
    Build internal contiguous gaps.

    A gap is represented as:

        kept prefix | skipped gap | kept suffix

    The gap never touches the first or last layer.
    """
    if num_layers < 4:
        return

    min_gap_len = max(1, config.internal_gap_min_length)
    max_gap_len = min(
        num_layers - 2,
        math.floor(config.internal_gap_max_remove_fraction * num_layers),
    )

    if min_gap_len > max_gap_len:
        return

    length_anchors = [
        2,
        3,
        4,
        num_layers // 8,
        num_layers // 6,
        num_layers // 4,
        num_layers // 3,
        num_layers // 2,
        (2 * num_layers) // 3,
        math.floor(0.80 * num_layers),
    ]

    gap_lengths = _sample_int_range(
        low=min_gap_len,
        high=max_gap_len,
        count=config.num_internal_gap_lengths,
        anchors=length_anchors,
    )

    left_keep_shares = _sample_float_range(
        low=config.internal_gap_left_keep_share_min,
        high=config.internal_gap_left_keep_share_max,
        count=config.internal_gap_positions_per_length,
    )

    for gap_len in gap_lengths:
        remaining_kept = num_layers - gap_len

        # Need at least one kept layer before and after the gap.
        if remaining_kept < 2:
            continue

        left_keep_values: set[int] = set()

        for share in left_keep_shares:
            left_keep = round(share * remaining_kept)
            left_keep = max(1, min(remaining_kept - 1, left_keep))
            left_keep_values.add(left_keep)

        for left_keep in sorted(left_keep_values):
            right_keep = remaining_kept - left_keep
            gap_start = left_keep
            gap_end = gap_start + gap_len

            if gap_start <= 0:
                continue

            if gap_end >= num_layers:
                continue

            kept_indices = list(range(0, gap_start)) + list(range(gap_end, num_layers))

            _add_unique_mask(
                dedup=dedup,
                num_layers=num_layers,
                name=(
                    f"drop_internal_gap__start_{gap_start}"
                    f"__len_{gap_len}"
                    f"__left_{left_keep}"
                    f"__right_{right_keep}"
                ),
                indices=kept_indices,
            )


def _sample_periodic_phases(
    *,
    step: int,
    max_phases_per_step: int | None,
) -> tuple[int, ...]:
    phases = list(range(step))

    if max_phases_per_step is None:
        return tuple(phases)

    return _downsample_sorted_values(
        phases,
        count=max_phases_per_step,
    )


def _maybe_anchor_edges(
    *,
    kept: set[int],
    num_layers: int,
    anchor_edges: bool,
) -> set[int]:
    if anchor_edges:
        kept.add(0)
        kept.add(num_layers - 1)

    return kept


def _make_periodic_masks(
    *,
    num_layers: int,
    config: AblationGenerationConfig,
    dedup: dict[tuple[int, ...], AblationMaskSpec],
) -> None:
    for step in config.periodic_steps:
        if step <= 1 or step > num_layers:
            continue

        phases = _sample_periodic_phases(
            step=step,
            max_phases_per_step=config.periodic_max_phases_per_step,
        )

        for phase in phases:
            if config.include_periodic_drop_every_n:
                kept = {
                    i
                    for i in range(num_layers)
                    if (i % step) != phase
                }

                kept = _maybe_anchor_edges(
                    kept=kept,
                    num_layers=num_layers,
                    anchor_edges=config.periodic_anchor_edges,
                )

                _add_unique_mask(
                    dedup=dedup,
                    num_layers=num_layers,
                    name=f"drop_every_{step}__phase_{phase}",
                    indices=kept,
                )

            if config.include_periodic_keep_every_n:
                kept = {
                    i
                    for i in range(num_layers)
                    if (i % step) == phase
                }

                kept = _maybe_anchor_edges(
                    kept=kept,
                    num_layers=num_layers,
                    anchor_edges=config.periodic_anchor_edges,
                )

                _add_unique_mask(
                    dedup=dedup,
                    num_layers=num_layers,
                    name=f"keep_every_{step}__phase_{phase}",
                    indices=kept,
                )


def _make_basic_ablation_masks(
    num_layers: int,
    config: AblationGenerationConfig | None = None,
) -> list[AblationMaskSpec]:
    """
    Strategic bounded ablation set.

    Main families:
      1. early exits / prefix keeps
      2. internal contiguous gaps with different lengths and placements
      3. late-begin / suffix keeps
      4. periodic skip/keep patterns

    The number of masks is controlled by AblationGenerationConfig rather than
    scaling explosively with num_layers.
    """
    if num_layers <= 0:
        return []

    if config is None:
        config = AblationGenerationConfig()

    dedup: dict[tuple[int, ...], AblationMaskSpec] = {}

    if config.include_keep_all:
        _add_unique_mask(
            dedup=dedup,
            num_layers=num_layers,
            name="keep_all",
            indices=range(num_layers),
        )

    _make_early_exit_masks(
        num_layers=num_layers,
        config=config,
        dedup=dedup,
    )

    _make_internal_gap_masks(
        num_layers=num_layers,
        config=config,
        dedup=dedup,
    )

    _make_late_begin_masks(
        num_layers=num_layers,
        config=config,
        dedup=dedup,
    )

    _make_periodic_masks(
        num_layers=num_layers,
        config=config,
        dedup=dedup,
    )

    return list(dedup.values())


def _print_ranking(
    *,
    title: str,
    ranked_results: list[AblationResult],
    score_getter: Any,
    top_k: int = 15,
) -> None:
    print()
    print("=" * 120)
    print(title)
    print("=" * 120)

    for rank, result in enumerate(ranked_results[:top_k], start=1):
        score = score_getter(result)
        score_str = (
            _format_optional_float(score)
            if isinstance(score, (float, type(None)))
            else str(score)
        )

        print(
            f"{rank:>2}. "
            f"{result.mask_name:<40} "
            f"score={score_str:<12} "
            f"kept={result.num_kept_layers:>2}/{result.num_total_layers:<2} "
            f"removed={result.num_removed_layers:>2} "
            f"ce_gap={result.mean_ce_gap:.6f} "
            f"top1={result.mean_top1_agreement:.6f}"
        )
        print(f"    visual={result.visual_mask}")
        print(f"    binary={_format_binary_mask(result.binary_mask)}")


def evaluate_layer_ablations(
    *,
    model_name: str,
    dataset_spec: DatasetSpec,
    context_len: int,
    max_examples: int,
    num_windows_to_use: int = 10,
    batch_size: int = 2,
    model_kwargs: dict[str, Any] | None = None,
    json_output_dir: str | Path = "ablation_results",
) -> list[AblationResult]:
    _stage("evaluate_layer_ablations: start", model_name=model_name)

    device = get_preferred_device()
    compute_dtype = get_preferred_float_dtype(device)

    print("DEVICE:", device)
    print("COMPUTE_DTYPE:", compute_dtype)

    model_and_tokenizer: ModelAndTokenizer = load_model_and_tokenizer(
        model_name,
        model_kwargs={
            # HF expects torch_dtype, not dtype.
            "torch_dtype": compute_dtype,
            **(model_kwargs or {}),
        },
    )

    model = cast(Any, model_and_tokenizer.model)
    model.eval()
    model.to(device=device)

    num_layers = len(get_decoder_layers(model))
    _stage(f"model has {num_layers} decoder layers", model_name=model_name)

    _stage("building dataloader", model_name=model_name)
    dataloader = build_fixed_window_dataloader(
        dataset_spec=dataset_spec,
        model_and_tokenizer=model_and_tokenizer,
        context_len=context_len,
        max_examples=max_examples,
        num_windows_to_use=num_windows_to_use,
        batch_size=batch_size,
        device=device,
        shuffle=False,
    )

    masks = _make_basic_ablation_masks(
        num_layers,
        config=AblationGenerationConfig(
            num_early_exit_masks=20,
            num_late_begin_masks=6,
            num_internal_gap_lengths=8,
            internal_gap_positions_per_length=5,
            periodic_steps=(2, 3, 4),
            include_periodic_drop_every_n=True,
            include_periodic_keep_every_n=True,
        ),
    )
    num_ablations = len(masks)

    print(f"Generated {num_ablations} unique ablation masks")

    print()
    print("=" * 120)
    print(f"Evaluating {num_ablations} ablations for model: {model_name}")
    print("=" * 120)

    for idx, mask_spec in enumerate(masks, start=1):
        pattern = make_layer_pattern(
            num_layers=num_layers,
            active_layer_indices=mask_spec.keep_layer_indices,
        )

        print(
            f"{idx:>2}. {mask_spec.name:<40} "
            f"kept={len(pattern.active_layer_indices):>2}/{num_layers}"
        )
        print(f"    visual={pattern.visual_mask}")
        print(f"    binary={pattern.binary_string}")

    results: list[AblationResult] = []

    for ablation_idx, mask_spec in enumerate(masks, start=1):
        pattern = make_layer_pattern(
            num_layers=num_layers,
            active_layer_indices=mask_spec.keep_layer_indices,
        )

        if len(pattern.active_layer_indices) == 0:
            print(f"Skipping mask {mask_spec.name} because it keeps zero layers.")
            continue

        keep_set = set(pattern.active_layer_indices)

        ce_masked_values: list[float] = []
        ce_full_values: list[float] = []
        kl_values: list[float] = []
        js_values: list[float] = []
        top1_values: list[float] = []
        overlap_values: list[float] = []
        p_masked_on_full_argmax_values: list[float] = []

        _stage(
            f"evaluating mask={mask_spec.name} kept={len(pattern.active_layer_indices)}/{num_layers}",
            model_name=model_name,
            ablation_idx=ablation_idx,
            num_ablations=num_ablations,
        )
        print(f"  visual={pattern.visual_mask}")
        print(f"  binary={pattern.binary_string}")
        print(f"  kept_layers={pattern.active_layer_indices}")
        print(f"  skipped_layers={pattern.skipped_layer_indices}")

        for batch_idx, (input_ids, attention_mask) in enumerate(dataloader):
            input_ids = input_ids.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)
            labels = input_ids

            with torch.no_grad():
                logits_full = forward_model_logits(
                    model=model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

                logits_masked = forward_with_layer_mask(
                    model=model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    keep_layer_indices=keep_set,
                )

                ce_full = cross_entropy_next_token(
                    logits=logits_full,
                    labels=labels,
                )

                ce_masked = cross_entropy_next_token(
                    logits=logits_masked,
                    labels=labels,
                )

                kl_full_to_masked = kl_teacher_to_student_next_token(
                    logits_teacher=logits_full,
                    logits_student=logits_masked,
                    attention_mask=attention_mask,
                )

                sim_metrics = distribution_similarity_metrics(
                    shift_logits_mid=shift_next_token_logits(logits_masked),
                    shift_logits_full=shift_next_token_logits(logits_full),
                )

            ce_full_values.append(ce_full.item())
            ce_masked_values.append(ce_masked.item())
            kl_values.append(kl_full_to_masked.item())
            js_values.append(sim_metrics["js"].item())
            top1_values.append(sim_metrics["top1_agreement"].item())
            overlap_values.append(sim_metrics["overlap"].item())
            p_masked_on_full_argmax_values.append(
                sim_metrics["p_mid_on_full_argmax"].item()
            )

            print(
                f"  ablation={ablation_idx}/{num_ablations} "
                f"[model={model_name}] "
                f"batch={batch_idx + 1}/{len(dataloader)} "
                f"ce_masked={ce_masked.item():.4f} "
                f"ce_full={ce_full.item():.4f} "
                f"ce_gap={(ce_masked.item() - ce_full.item()):.4f} "
                f"kl={kl_full_to_masked.item():.4f}"
            )

        mean_ce_masked = sum(ce_masked_values) / len(ce_masked_values)
        mean_ce_full = sum(ce_full_values) / len(ce_full_values)
        mean_kl = sum(kl_values) / len(kl_values)
        mean_js = sum(js_values) / len(js_values)
        mean_top1 = sum(top1_values) / len(top1_values)
        mean_overlap = sum(overlap_values) / len(overlap_values)
        mean_p_masked = (
            sum(p_masked_on_full_argmax_values) / len(p_masked_on_full_argmax_values)
        )

        num_kept_layers = len(pattern.active_layer_indices)
        num_removed_layers = num_layers - num_kept_layers

        result = AblationResult(
            mask_name=mask_spec.name,
            kept_layers=pattern.active_layer_indices,
            binary_mask=pattern.binary_mask,
            visual_mask=pattern.visual_mask,
            num_kept_layers=num_kept_layers,
            num_removed_layers=num_removed_layers,
            num_total_layers=num_layers,
            keep_fraction=num_kept_layers / num_layers,
            remove_fraction=num_removed_layers / num_layers,
            mean_ce_masked=mean_ce_masked,
            mean_ce_full=mean_ce_full,
            mean_ce_gap=(mean_ce_masked - mean_ce_full),
            mean_kl_full_to_masked=mean_kl,
            kl_per_removed_layer=_safe_div(mean_kl, num_removed_layers),
            mean_js=mean_js,
            mean_top1_agreement=mean_top1,
            mean_overlap=mean_overlap,
            mean_p_masked_on_full_argmax=mean_p_masked,
        )
        results.append(result)

        print(
            f"[done] ablation={ablation_idx}/{num_ablations} [model={model_name}] "
            f"{result.mask_name} | "
            f"kept={result.num_kept_layers}/{result.num_total_layers} | "
            f"removed={result.num_removed_layers} | "
            f"ce_masked={result.mean_ce_masked:.4f} | "
            f"ce_full={result.mean_ce_full:.4f} | "
            f"ce_gap={result.mean_ce_gap:.4f} | "
            f"kl={result.mean_kl_full_to_masked:.4f} | "
            f"kl_per_removed={_format_optional_float(result.kl_per_removed_layer)} | "
            f"top1={result.mean_top1_agreement:.4f}"
        )

    results_by_ce = sorted(
        results,
        key=lambda x: (
            x.mean_ce_gap,
            x.mean_kl_full_to_masked,
            -x.num_removed_layers,
        ),
    )

    results_by_kl = sorted(
        results,
        key=lambda x: (
            x.mean_kl_full_to_masked,
            x.mean_ce_gap,
            -x.num_removed_layers,
        ),
    )

    results_by_kl_per_removed = sorted(
        [x for x in results if x.kl_per_removed_layer is not None],
        key=lambda x: (
            cast(float, x.kl_per_removed_layer),
            x.mean_kl_full_to_masked,
            x.mean_ce_gap,
        ),
    )

    _print_ranking(
        title=f"Top ablations by lowest KL | model={model_name}",
        ranked_results=results_by_kl,
        score_getter=lambda r: r.mean_kl_full_to_masked,
        top_k=min(15, len(results_by_kl)),
    )

    _print_ranking(
        title=f"Top ablations by lowest KL per removed layer | model={model_name}",
        ranked_results=results_by_kl_per_removed,
        score_getter=lambda r: r.kl_per_removed_layer,
        top_k=min(15, len(results_by_kl_per_removed)),
    )

    _print_ranking(
        title=f"Top ablations by lowest CE gap | model={model_name}",
        ranked_results=results_by_ce,
        score_getter=lambda r: r.mean_ce_gap,
        top_k=min(15, len(results_by_ce)),
    )

    json_path = _save_ablation_results_json(
        model_name=model_name,
        dataset_spec=dataset_spec,
        context_len=context_len,
        max_examples=max_examples,
        num_windows_to_use=num_windows_to_use,
        batch_size=batch_size,
        num_layers=num_layers,
        num_ablations=num_ablations,
        num_batches=len(dataloader),
        device=device,
        compute_dtype=compute_dtype,
        model_kwargs=model_kwargs,
        results=results,
        output_dir=json_output_dir,
    )

    _stage(f"saved results json to {json_path}", model_name=model_name)
    _stage("finished evaluation", model_name=model_name)

    return results_by_ce