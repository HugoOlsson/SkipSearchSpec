from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast, Iterable
import json
import math
import time
from contextlib import ExitStack
from pathlib import Path
import re

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets.arrow_dataset import Dataset

from skip_search_spec.helpers.tooling import (
    distribution_similarity_metrics,
    get_preferred_device,
    get_preferred_float_dtype,
    load_dataset,
    load_model_and_tokenizer,
)
from skip_search_spec.helpers.window_building import (
    WindowDataset,
    build_all_training_windows,
    collate_windows,
    tokenize_dataset_to_examples,
)
from skip_search_spec.protocols.windows import DatasetSpec, ModelAndTokenizer, WindowSettings


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
    prefix_parts: list[str] = []

    if ablation_idx is not None and num_ablations is not None:
        prefix_parts.append(f"ablation={ablation_idx}/{num_ablations}")

    if model_name is not None:
        prefix_parts.append(f"[model={model_name}]")

    prefix = ""
    if prefix_parts:
        prefix = " ".join(prefix_parts) + " "

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {prefix}{message}", flush=True)


def _get_backbone(model: Any) -> Any:
    if hasattr(model, "model") and hasattr(model.model, "layers") and hasattr(model.model, "norm"):
        return model.model

    raise TypeError(
        "Unsupported model structure. Expected a decoder-only HF model with "
        "`model.layers` and `model.norm`."
    )


def _get_decoder_layers(model: Any) -> Any:
    backbone = _get_backbone(model)
    return backbone.layers


def _mask_to_binary_list(
    *,
    num_layers: int,
    keep_layer_indices: tuple[int, ...],
) -> list[int]:
    keep_set = set(keep_layer_indices)
    return [1 if i in keep_set else 0 for i in range(num_layers)]


def _mask_to_binary_tuple(
    *,
    num_layers: int,
    keep_layer_indices: tuple[int, ...],
) -> tuple[int, ...]:
    return tuple(
        _mask_to_binary_list(
            num_layers=num_layers,
            keep_layer_indices=keep_layer_indices,
        )
    )


def _mask_to_binary_string(
    *,
    num_layers: int,
    keep_layer_indices: tuple[int, ...],
) -> str:
    binary_list = _mask_to_binary_list(
        num_layers=num_layers,
        keep_layer_indices=keep_layer_indices,
    )
    return "[" + ",".join(str(x) for x in binary_list) + "]"


def _mask_to_visual_string(
    *,
    num_layers: int,
    keep_layer_indices: tuple[int, ...],
) -> str:
    binary_list = _mask_to_binary_list(
        num_layers=num_layers,
        keep_layer_indices=keep_layer_indices,
    )
    return "".join("█" if x == 1 else "·" for x in binary_list)


def _safe_div(numerator: float, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return numerator / float(denominator)


def _format_optional_float(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "NA"
    return f"{value:.6f}"


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
        "schema_version": 1,
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


@torch.no_grad()
def _forward_with_layer_mask(
    *,
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    keep_layer_indices: set[int],
) -> torch.Tensor:
    """
    Runs the model normally, but replaces skipped decoder layers with identity.

    This is much more robust than manually replaying the model internals because
    the original HF forward still handles:
    - RoPE / position embeddings
    - causal masks
    - sliding-window masks
    - any model-family-specific kwargs
    """
    backbone = _get_backbone(model)
    layers = backbone.layers

    def make_skip_hook() -> Any:
        def skip_hook(module: Any, inputs: tuple[Any, ...], output: Any) -> Any:
            if len(inputs) == 0 or not isinstance(inputs[0], torch.Tensor):
                raise TypeError(
                    f"Expected first layer input to be hidden_states tensor, got "
                    f"{type(inputs[0]) if len(inputs) > 0 else 'empty inputs'}"
                )

            hidden_in = inputs[0]

            if isinstance(output, torch.Tensor):
                return hidden_in

            if isinstance(output, tuple) and len(output) > 0:
                return type(output)((hidden_in, *output[1:]))

            raise TypeError(
                f"Unexpected decoder layer output type when skipping layer: {type(output)}"
            )

        return skip_hook

    with ExitStack() as stack:
        for layer_idx, layer in enumerate(layers):
            if layer_idx not in keep_layer_indices:
                handle = layer.register_forward_hook(make_skip_hook())
                stack.callback(handle.remove)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            use_cache=False,
            return_dict=True,
        )

    return cast(torch.Tensor, outputs.logits)


def _cross_entropy_next_token(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )


def _kl_full_to_masked_next_token(
    *,
    logits_full: torch.Tensor,
    logits_masked: torch.Tensor,
) -> torch.Tensor:
    shift_logits_full = logits_full[:, :-1, :].contiguous()
    shift_logits_masked = logits_masked[:, :-1, :].contiguous()

    log_p_masked = F.log_softmax(shift_logits_masked.float(), dim=-1)
    log_p_full = F.log_softmax(shift_logits_full.float(), dim=-1)

    kl_per_token = F.kl_div(
        log_p_masked,
        log_p_full,
        reduction="none",
        log_target=True,
    ).sum(dim=-1)

    return kl_per_token.mean()

def _make_dense_keep_edges_masks(
    *,
    num_layers: int,
    min_remove_fraction: float = 0.60,
    max_remove_fraction: float = 0.90,
    min_edge_keep: int = 1,
    include_all_splits: bool = True,
    max_splits_per_total: int | None = None,
) -> list[AblationMaskSpec]:
    if num_layers <= 0:
        return []

    dedup: dict[tuple[int, ...], AblationMaskSpec] = {}

    def add_mask(name: str, indices: Iterable[int]) -> None:
        kept = tuple(sorted({i for i in indices if 0 <= i < num_layers}))
        if len(kept) == 0:
            return
        if kept not in dedup:
            dedup[kept] = AblationMaskSpec(
                name=name,
                keep_layer_indices=kept,
            )

    min_keep_total = max(
        2 * min_edge_keep,
        math.ceil((1.0 - max_remove_fraction) * num_layers),
    )
    max_keep_total = min(
        num_layers - 1,
        math.floor((1.0 - min_remove_fraction) * num_layers),
    )

    if min_keep_total > max_keep_total:
        return []

    for total_kept in range(min_keep_total, max_keep_total + 1):
        candidate_pairs: list[tuple[int, int]] = []

        for left in range(min_edge_keep, total_kept - min_edge_keep + 1):
            right = total_kept - left
            if right < min_edge_keep:
                continue
            if left + right >= num_layers:
                continue
            candidate_pairs.append((left, right))

        if not candidate_pairs:
            continue

        if include_all_splits:
            selected_pairs = candidate_pairs
        else:
            # keep a smaller but still informative subset:
            # left-heavy, balanced, right-heavy
            mid = len(candidate_pairs) // 2
            selected_pairs = [candidate_pairs[0], candidate_pairs[mid], candidate_pairs[-1]]

            # dedup while preserving order
            seen: set[tuple[int, int]] = set()
            selected_pairs = [
                pair for pair in selected_pairs
                if not (pair in seen or seen.add(pair))
            ]

        if max_splits_per_total is not None and len(selected_pairs) > max_splits_per_total:
            if max_splits_per_total <= 1:
                selected_pairs = [selected_pairs[len(selected_pairs) // 2]]
            else:
                # evenly subsample across asymmetry range
                indices = [
                    round(i * (len(selected_pairs) - 1) / (max_splits_per_total - 1))
                    for i in range(max_splits_per_total)
                ]
                selected_pairs = [selected_pairs[i] for i in sorted(set(indices))]

        for left, right in selected_pairs:
            kept = list(range(0, left)) + list(range(num_layers - right, num_layers))
            add_mask(
                f"keep_edges__left_{left}__right_{right}__total_{total_kept}__remove_{num_layers - total_kept}",
                kept,
            )

    return list(dedup.values())


def _make_basic_ablation_masks(num_layers: int) -> list[AblationMaskSpec]:
    if num_layers <= 0:
        return []

    dedup: dict[tuple[int, ...], AblationMaskSpec] = {}

    def normalize(indices: Iterable[int]) -> tuple[int, ...]:
        return tuple(sorted({i for i in indices if 0 <= i < num_layers}))

    def add_mask(name: str, indices: Iterable[int]) -> None:
        kept = normalize(indices)
        if len(kept) == 0:
            return
        # Keep the FIRST canonical name for a given kept-layer pattern.
        if kept not in dedup:
            dedup[kept] = AblationMaskSpec(
                name=name,
                keep_layer_indices=kept,
            )

    def unique_counts(*values: int, upper: int | None = None, lower: int = 1) -> list[int]:
        max_allowed = num_layers if upper is None else upper
        return sorted({
            int(v)
            for v in values
            if lower <= int(v) <= max_allowed
        })

    def centered_block(size: int) -> tuple[int, ...]:
        size = max(1, min(size, num_layers))
        start = max(0, (num_layers - size) // 2)
        end = min(num_layers, start + size)
        return tuple(range(start, end))

    def remove_block(start: int, length: int) -> tuple[int, ...]:
        start = max(0, min(start, num_layers))
        end = max(start, min(num_layers, start + length))
        removed = set(range(start, end))
        return tuple(i for i in range(num_layers) if i not in removed)

    def periodic_indices(step: int, keep_count: int, phase: int) -> tuple[int, ...]:
        return tuple(
            i for i in range(num_layers)
            if ((i + phase) % step) < keep_count
        )

    def periodic_indices_anchored(step: int, keep_count: int, phase: int) -> tuple[int, ...]:
        kept = {
            i for i in range(num_layers)
            if ((i + phase) % step) < keep_count
        }
        kept.add(0)
        kept.add(num_layers - 1)
        return tuple(sorted(kept))

    # ------------------------------------------------------------------
    # 1) Baseline
    # ------------------------------------------------------------------
    add_mask("keep_all", range(num_layers))

    # ------------------------------------------------------------------
    # 2) Prefix-only (canonical early-exit family)
    # ------------------------------------------------------------------
    prefix_sizes = unique_counts(
        1,
        2,
        3,
        4,
        5,
        6,
        num_layers // 4 - 1,
        num_layers // 4,
        num_layers // 4 + 1,
        num_layers // 3 - 1,
        num_layers // 3,
        num_layers // 3 + 1,
        num_layers // 2 - 2,
        num_layers // 2 - 1,
        num_layers // 2,
        num_layers // 2 + 1,
        num_layers // 2 + 2,
        (2 * num_layers) // 3 - 1,
        (2 * num_layers) // 3,
        (2 * num_layers) // 3 + 1,
        (3 * num_layers) // 4 - 1,
        (3 * num_layers) // 4,
        (3 * num_layers) // 4 + 1,
        num_layers - 2,
        num_layers - 1,
        upper=max(1, num_layers - 1),
    )
    for k in prefix_sizes:
        add_mask(f"keep_prefix__k_{k}", range(0, k))

    # ------------------------------------------------------------------
    # 3) Suffix-only (canonical late-begin family)
    # ------------------------------------------------------------------
    suffix_sizes = prefix_sizes
    for k in suffix_sizes:
        add_mask(f"keep_suffix__k_{k}", range(num_layers - k, num_layers))

    # ------------------------------------------------------------------
    # 4) Internal single-layer drop
    #    Internal only: idx in [1, num_layers-2]
    # ------------------------------------------------------------------
    # if num_layers >= 3:
    #     for idx in range(1, num_layers - 1):
    #         add_mask(
    #             f"drop_internal_single__idx_{idx}",
    #             remove_block(start=idx, length=1),
    #         )

    # ------------------------------------------------------------------
    # 5) Internal contiguous block drop
    #    Internal only: block must not touch first/last layer
    # ------------------------------------------------------------------
    if num_layers >= 5:
        block_lengths = unique_counts(
            2,
            3,
            4,
            num_layers // 8,
            num_layers // 6,
            num_layers // 4,
            upper=max(2, num_layers - 2),
            lower=2,
        )

        for length in block_lengths:
            if length >= num_layers - 1:
                continue

            candidate_starts = sorted({
                1,
                max(1, (num_layers // 4) - (length // 2)),
                max(1, (num_layers // 2) - (length // 2)),
                max(1, ((3 * num_layers) // 4) - (length // 2)),
                max(1, num_layers - length - 1),
            })

            for start in candidate_starts:
                if start <= 0:
                    continue
                if start + length >= num_layers:
                    continue
                add_mask(
                    f"drop_internal_block__start_{start}__len_{length}",
                    remove_block(start=start, length=length),
                )

    # ------------------------------------------------------------------
    # 6) Center contiguous keep-block
    # ------------------------------------------------------------------
    center_lengths = unique_counts(
        1,
        2,
        4,
        num_layers // 8,
        num_layers // 6,
        num_layers // 4,
        num_layers // 3,
        num_layers // 2,
        upper=max(1, num_layers - 1),
    )
    for length in center_lengths:
        add_mask(
            f"keep_center_block__len_{length}",
            centered_block(length),
        )

   # ------------------------------------------------------------------
    # 7) Edge-only family
    #    Dense middle-gap sweep:
    #    keep left prefix + right suffix, remove contiguous middle block,
    #    focused on removing 60-90% of layers.
    # ------------------------------------------------------------------
    dense_keep_edges = _make_dense_keep_edges_masks(
        num_layers=num_layers,
        min_remove_fraction=0.60,
        max_remove_fraction=0.90,
        min_edge_keep=1,
        include_all_splits=True,      # set False if too many
        max_splits_per_total=None,    # e.g. 7 if you want to cap it
    )

    for spec in dense_keep_edges:
        add_mask(spec.name, spec.keep_layer_indices)

    # ------------------------------------------------------------------
    # 8) Periodic / strided family
    #    Canonical replacement for every_nth, group_k_of_n, two_on_two_off, etc.
    # ------------------------------------------------------------------
    periodic_specs = [
        (2, 1),
        (3, 1),
        (3, 2),
        (4, 1),
        (4, 2),
        (4, 3),
    ]

    for step, keep_count in periodic_specs:
        if step > num_layers:
            continue
        if keep_count <= 0 or keep_count >= step:
            continue

        for phase in range(step):
            add_mask(
                f"keep_periodic__step_{step}__keep_{keep_count}__phase_{phase}",
                periodic_indices(step=step, keep_count=keep_count, phase=phase),
            )

    # ------------------------------------------------------------------
    # 9) Anchored periodic family
    #    Same as periodic, but always keeps first and last layer
    # ------------------------------------------------------------------
    for step, keep_count in periodic_specs:
        if step > num_layers:
            continue
        if keep_count <= 0 or keep_count >= step:
            continue

        for phase in range(step):
            add_mask(
                f"keep_periodic_anchored__step_{step}__keep_{keep_count}__phase_{phase}",
                periodic_indices_anchored(step=step, keep_count=keep_count, phase=phase),
            )

    # ------------------------------------------------------------------
    # 10) Landmark family
    # ------------------------------------------------------------------
    quarter_idx = num_layers // 4
    mid_idx = num_layers // 2
    third_1 = num_layers // 3
    third_2 = (2 * num_layers) // 3
    three_quarter_idx = (3 * num_layers) // 4
    last_idx = num_layers - 1

    landmark_schemes: list[tuple[str, list[int]]] = [
        ("first_mid_last", [0, mid_idx, last_idx]),
        ("first_quarter_mid_threequarter_last", [0, quarter_idx, mid_idx, three_quarter_idx, last_idx]),
        ("first_thirds_last", [0, third_1, third_2, last_idx]),
        ("quarter_mid_threequarter", [quarter_idx, mid_idx, three_quarter_idx]),
    ]

    for scheme_name, indices in landmark_schemes:
        add_mask(
            f"keep_landmarks__scheme_{scheme_name}",
            indices,
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
        score_str = _format_optional_float(score) if isinstance(score, (float, type(None))) else str(score)

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
        print(f"    binary={list(result.binary_mask)}")


def evaluate_layer_ablations(
    *,
    model_name: str,
    dataset_spec: DatasetSpec,
    context_len: int,
    max_examples: int,
    num_windows_to_use: int = 10,
    batch_size: int = 2,
    model_kwargs: dict[str, Any] | None = None,
    custom_masks: list[AblationMaskSpec] | None = None,
    json_output_dir: str | Path = "ablation_results",
) -> list[AblationResult]:
    _stage("evaluate_layer_ablations: start", model_name=model_name)

    device = get_preferred_device()
    compute_dtype = get_preferred_float_dtype(device)

    model_and_tokenizer: ModelAndTokenizer = load_model_and_tokenizer(
        model_name,
        model_kwargs={
            "dtype": compute_dtype,
            **(model_kwargs or {}),
        },
    )

    model = cast(Any, model_and_tokenizer.model)
    model.eval()
    model.to(device=device, dtype=compute_dtype)

    dataset: Dataset = load_dataset(dataset_spec)
    window_settings = WindowSettings(C1=context_len)

    _stage("tokenizing dataset", model_name=model_name)
    tokenized_examples = tokenize_dataset_to_examples(
        dataset,
        model_and_tokenizer.tokenizer,
        dataset_spec,
        max_examples=max_examples,
    )

    _stage("building windows", model_name=model_name)
    all_windows = build_all_training_windows(
        tokenized_examples,
        window_settings,
        dataset_spec,
    )

    if len(all_windows) < num_windows_to_use:
        raise ValueError(
            f"Requested {num_windows_to_use} windows, but only built {len(all_windows)} windows."
        )

    selected_windows = all_windows[:num_windows_to_use]
    window_dataset = WindowDataset(selected_windows)

    dataloader = DataLoader(
        window_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_windows,
        pin_memory=(device.type == "cuda"),
    )

    num_layers = len(_get_decoder_layers(model))
    _stage(f"model has {num_layers} decoder layers", model_name=model_name)

    masks = custom_masks if custom_masks is not None else _make_basic_ablation_masks(num_layers)
    num_ablations = len(masks)

    print(f"Generated {num_ablations} unique ablation masks")

    print()
    print("=" * 120)
    print(f"Evaluating {num_ablations} ablations for model: {model_name}")
    print("=" * 120)
    for idx, mask_spec in enumerate(masks, start=1):
        binary_mask = _mask_to_binary_tuple(
            num_layers=num_layers,
            keep_layer_indices=mask_spec.keep_layer_indices,
        )
        visual_mask = _mask_to_visual_string(
            num_layers=num_layers,
            keep_layer_indices=mask_spec.keep_layer_indices,
        )
        print(
            f"{idx:>2}. {mask_spec.name:<40} "
            f"kept={len(mask_spec.keep_layer_indices):>2}/{num_layers}"
        )
        print(f"    visual={visual_mask}")
        print(f"    binary={list(binary_mask)}")

    results: list[AblationResult] = []

    for ablation_idx, mask_spec in enumerate(masks, start=1):
        if len(mask_spec.keep_layer_indices) == 0:
            print(f"Skipping mask {mask_spec.name} because it keeps zero layers.")
            continue

        invalid_indices = [i for i in mask_spec.keep_layer_indices if not (0 <= i < num_layers)]
        if invalid_indices:
            raise ValueError(
                f"Mask {mask_spec.name} contains invalid layer indices: {invalid_indices}"
            )

        keep_set = set(mask_spec.keep_layer_indices)
        binary_mask = _mask_to_binary_tuple(
            num_layers=num_layers,
            keep_layer_indices=mask_spec.keep_layer_indices,
        )
        visual_mask = _mask_to_visual_string(
            num_layers=num_layers,
            keep_layer_indices=mask_spec.keep_layer_indices,
        )

        ce_masked_values: list[float] = []
        ce_full_values: list[float] = []
        kl_values: list[float] = []
        js_values: list[float] = []
        top1_values: list[float] = []
        overlap_values: list[float] = []
        p_masked_on_full_argmax_values: list[float] = []

        _stage(
            f"evaluating mask={mask_spec.name} kept={len(mask_spec.keep_layer_indices)}/{num_layers}",
            model_name=model_name,
            ablation_idx=ablation_idx,
            num_ablations=num_ablations,
        )
        print(f"  visual={visual_mask}")
        print(f"  binary={list(binary_mask)}")
        print(f"  kept_layers={mask_spec.keep_layer_indices}")

        for batch_idx, (input_ids, attention_mask) in enumerate(dataloader):
            input_ids = input_ids.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)
            labels = input_ids

            with torch.no_grad():
                full_outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=False,
                    use_cache=False,
                    return_dict=True,
                )
                logits_full = cast(torch.Tensor, full_outputs.logits)

                logits_masked = _forward_with_layer_mask(
                    model=model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    keep_layer_indices=keep_set,
                )

                ce_full = _cross_entropy_next_token(logits_full, labels)
                ce_masked = _cross_entropy_next_token(logits_masked, labels)
                kl_full_to_masked = _kl_full_to_masked_next_token(
                    logits_full=logits_full,
                    logits_masked=logits_masked,
                )

                shift_logits_masked = logits_masked[:, :-1, :].contiguous()
                shift_logits_full = logits_full[:, :-1, :].contiguous()

                sim_metrics = distribution_similarity_metrics(
                    shift_logits_mid=shift_logits_masked,
                    shift_logits_full=shift_logits_full,
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

        num_kept_layers = len(mask_spec.keep_layer_indices)
        num_removed_layers = num_layers - num_kept_layers
        kl_per_removed_layer = _safe_div(mean_kl, num_removed_layers)

        result = AblationResult(
            mask_name=mask_spec.name,
            kept_layers=mask_spec.keep_layer_indices,
            binary_mask=binary_mask,
            visual_mask=visual_mask,
            num_kept_layers=num_kept_layers,
            num_removed_layers=num_removed_layers,
            num_total_layers=num_layers,
            keep_fraction=num_kept_layers / num_layers,
            remove_fraction=num_removed_layers / num_layers,
            mean_ce_masked=mean_ce_masked,
            mean_ce_full=mean_ce_full,
            mean_ce_gap=(mean_ce_masked - mean_ce_full),
            mean_kl_full_to_masked=mean_kl,
            kl_per_removed_layer=kl_per_removed_layer,
            mean_js=mean_js,
            mean_top1_agreement=mean_top1,
            mean_overlap=mean_overlap,
            mean_p_masked_on_full_argmax=mean_p_masked,
        )
        results.append(result)

        print(
            f"[done] ablation={ablation_idx}/{num_ablations} [model={model_name}] {result.mask_name} | "
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
        key=lambda x: (x.mean_ce_gap, x.mean_kl_full_to_masked, -x.num_removed_layers),
    )
    results_by_kl = sorted(
        results,
        key=lambda x: (x.mean_kl_full_to_masked, x.mean_ce_gap, -x.num_removed_layers),
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