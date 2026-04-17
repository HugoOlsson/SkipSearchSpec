from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast, Iterable
import math
import time
from contextlib import ExitStack

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


def _stage(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


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


def _make_basic_ablation_masks(num_layers: int) -> list[AblationMaskSpec]:
    all_layers = tuple(range(num_layers))

    def keep(indices: Iterable[int]) -> tuple[int, ...]:
        return tuple(sorted(i for i in set(indices) if 0 <= i < num_layers))

    def add_mask(
        masks: list[AblationMaskSpec],
        name: str,
        indices: Iterable[int],
    ) -> None:
        kept = keep(indices)
        if len(kept) == 0:
            return
        masks.append(
            AblationMaskSpec(
                name=name,
                keep_layer_indices=kept,
            )
        )

    masks: list[AblationMaskSpec] = []

    # ------------------------------------------------------------------
    # Baselines
    # ------------------------------------------------------------------
    add_mask(masks, "full_model", all_layers)

    for frac_num, frac_den in [(1, 4), (1, 3), (1, 2), (2, 3), (3, 4)]:
        k = max(1, (num_layers * frac_num) // frac_den)
        add_mask(masks, f"early_exit_{frac_num}_{frac_den}", range(0, k))
        add_mask(masks, f"late_begin_{frac_num}_{frac_den}", range(num_layers - k, num_layers))

    # ------------------------------------------------------------------
    # Periodic masks
    # ------------------------------------------------------------------
    for step in [2, 3, 4, 5]:
        for offset in range(step):
            add_mask(
                masks,
                f"every_{step}th_from_{offset}",
                (i for i in range(num_layers) if i % step == offset),
            )

    # keep K out of every M consecutive layers
    for group_size, keep_count in [
        (2, 1),
        (3, 2),
        (4, 2),
        (4, 3),
        (5, 2),
        (5, 3),
    ]:
        for offset in range(group_size):
            add_mask(
                masks,
                f"group_{keep_count}_of_{group_size}_offset_{offset}",
                (
                    i for i in range(num_layers)
                    if ((i + offset) % group_size) < keep_count
                ),
            )

    # ------------------------------------------------------------------
    # Middle-only and edge-only
    # ------------------------------------------------------------------
    quarter = max(1, num_layers // 4)
    third = max(1, num_layers // 3)
    half = max(1, num_layers // 2)

    add_mask(masks, "middle_half", range(num_layers // 4, num_layers - num_layers // 4))
    add_mask(masks, "middle_third", range(num_layers // 3, num_layers - num_layers // 3))

    add_mask(
        masks,
        "outer_halves_only",
        list(range(0, quarter)) + list(range(num_layers - quarter, num_layers)),
    )

    add_mask(
        masks,
        "outer_thirds_only",
        list(range(0, third)) + list(range(num_layers - third, num_layers)),
    )

    for size_name, size in [
        ("center_2", 2),
        ("center_4", 4),
        ("center_quarter", quarter),
        ("center_half", half),
    ]:
        start = max(0, (num_layers - size) // 2)
        end = min(num_layers, start + size)
        add_mask(masks, size_name, range(start, end))

    # ------------------------------------------------------------------
    # Hybrid: keep one region dense, another sparse
    # ------------------------------------------------------------------
    add_mask(
        masks,
        "first_half_dense_second_half_even",
        list(range(0, num_layers // 2))
        + [i for i in range(num_layers // 2, num_layers) if i % 2 == 0],
    )
    add_mask(
        masks,
        "first_half_dense_second_half_odd",
        list(range(0, num_layers // 2))
        + [i for i in range(num_layers // 2, num_layers) if i % 2 == 1],
    )

    add_mask(
        masks,
        "first_half_even_second_half_dense",
        [i for i in range(0, num_layers // 2) if i % 2 == 0]
        + list(range(num_layers // 2, num_layers)),
    )
    add_mask(
        masks,
        "first_half_odd_second_half_dense",
        [i for i in range(0, num_layers // 2) if i % 2 == 1]
        + list(range(num_layers // 2, num_layers)),
    )

    t1 = num_layers // 3
    t2 = (2 * num_layers) // 3

    add_mask(
        masks,
        "first_third_dense_rest_even",
        list(range(0, t1)) + [i for i in range(t1, num_layers) if i % 2 == 0],
    )
    add_mask(
        masks,
        "last_third_dense_rest_even",
        [i for i in range(0, t2) if i % 2 == 0] + list(range(t2, num_layers)),
    )
    add_mask(
        masks,
        "middle_third_dense_rest_even",
        [i for i in range(0, t1) if i % 2 == 0]
        + list(range(t1, t2))
        + [i for i in range(t2, num_layers) if i % 2 == 0],
    )

    # ------------------------------------------------------------------
    # Two-on / two-off and related motifs
    # ------------------------------------------------------------------
    for offset in range(4):
        add_mask(
            masks,
            f"two_on_two_off_offset_{offset}",
            (
                i for i in range(num_layers)
                if ((i + offset) % 4) in (0, 1)
            ),
        )

    for offset in range(6):
        add_mask(
            masks,
            f"three_on_three_off_offset_{offset}",
            (
                i for i in range(num_layers)
                if ((i + offset) % 6) in (0, 1, 2)
            ),
        )

    # ------------------------------------------------------------------
    # Local holes in otherwise dense model
    # ------------------------------------------------------------------
    hole_sizes = [1, 2, 4]
    anchor_points = [
        0,
        num_layers // 4,
        num_layers // 2,
        (3 * num_layers) // 4,
        max(0, num_layers - 1),
    ]

    for hole_size in hole_sizes:
        for anchor in anchor_points:
            start = min(max(0, anchor), max(0, num_layers - hole_size))
            removed = set(range(start, start + hole_size))
            add_mask(
                masks,
                f"full_minus_hole_size_{hole_size}_start_{start}",
                (i for i in range(num_layers) if i not in removed),
            )

    # ------------------------------------------------------------------
    # Sparse masks with anchors
    # ------------------------------------------------------------------
    for step in [2, 3, 4]:
        for offset in range(step):
            interior = [
                i for i in range(1, num_layers - 1)
                if i % step == offset
            ]
            add_mask(
                masks,
                f"anchored_every_{step}th_from_{offset}",
                [0] + interior + [num_layers - 1],
            )

    edge_width = max(1, num_layers // 6)
    add_mask(
        masks,
        "keep_edges_sparse_middle_even",
        list(range(0, edge_width))
        + [i for i in range(edge_width, num_layers - edge_width) if i % 2 == 0]
        + list(range(num_layers - edge_width, num_layers)),
    )
    add_mask(
        masks,
        "keep_edges_sparse_middle_odd",
        list(range(0, edge_width))
        + [i for i in range(edge_width, num_layers - edge_width) if i % 2 == 1]
        + list(range(num_layers - edge_width, num_layers)),
    )

    # ------------------------------------------------------------------
    # Stair-step masks
    # ------------------------------------------------------------------
    add_mask(
        masks,
        "dense_early_medium_middle_sparse_late",
        list(range(0, t1))
        + [i for i in range(t1, t2) if i % 2 == 0]
        + [i for i in range(t2, num_layers) if i % 3 == 0],
    )
    add_mask(
        masks,
        "sparse_early_medium_middle_dense_late",
        [i for i in range(0, t1) if i % 3 == 0]
        + [i for i in range(t1, t2) if i % 2 == 0]
        + list(range(t2, num_layers)),
    )

    dedup: dict[tuple[int, ...], AblationMaskSpec] = {}
    for mask in masks:
        dedup[mask.keep_layer_indices] = mask

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
) -> list[AblationResult]:
    _stage("evaluate_layer_ablations: start")

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

    _stage("tokenizing dataset")
    tokenized_examples = tokenize_dataset_to_examples(
        dataset,
        model_and_tokenizer.tokenizer,
        dataset_spec,
        max_examples=max_examples,
    )

    _stage("building windows")
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
    _stage(f"model has {num_layers} decoder layers")

    masks = custom_masks if custom_masks is not None else _make_basic_ablation_masks(num_layers)

    print()
    print("=" * 120)
    print(f"Evaluating {len(masks)} ablations for {model_name}")
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

    for mask_spec in masks:
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
            f"evaluating mask={mask_spec.name} "
            f"kept={len(mask_spec.keep_layer_indices)}/{num_layers}"
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
                f"  batch={batch_idx + 1}/{len(dataloader)} "
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
            f"[done] {result.mask_name} | "
            f"kept={result.num_kept_layers}/{result.num_total_layers} | "
            f"removed={result.num_removed_layers} | "
            f"ce_masked={result.mean_ce_masked:.4f} | "
            f"ce_full={result.mean_ce_full:.4f} | "
            f"ce_gap={result.mean_ce_gap:.4f} | "
            f"kl={result.mean_kl_full_to_masked:.4f} | "
            f"kl_per_removed={_format_optional_float(result.kl_per_removed_layer)} | "
            f"top1={result.mean_top1_agreement:.4f}"
        )

    results_by_ce = sorted(results, key=lambda x: (x.mean_ce_gap, x.mean_kl_full_to_masked, -x.num_removed_layers))
    results_by_kl = sorted(results, key=lambda x: (x.mean_kl_full_to_masked, x.mean_ce_gap, -x.num_removed_layers))
    results_by_kl_per_removed = sorted(
        [x for x in results if x.kl_per_removed_layer is not None],
        key=lambda x: (cast(float, x.kl_per_removed_layer), x.mean_kl_full_to_masked, x.mean_ce_gap),
    )

    _print_ranking(
        title="Top ablations by lowest KL",
        ranked_results=results_by_kl,
        score_getter=lambda r: r.mean_kl_full_to_masked,
        top_k=min(15, len(results_by_kl)),
    )

    _print_ranking(
        title="Top ablations by lowest KL per removed layer",
        ranked_results=results_by_kl_per_removed,
        score_getter=lambda r: r.kl_per_removed_layer,
        top_k=min(15, len(results_by_kl_per_removed)),
    )

    _print_ranking(
        title="Top ablations by lowest CE gap",
        ranked_results=results_by_ce,
        score_getter=lambda r: r.mean_ce_gap,
        top_k=min(15, len(results_by_ce)),
    )

    _stage("finished evaluation")
    return results_by_ce