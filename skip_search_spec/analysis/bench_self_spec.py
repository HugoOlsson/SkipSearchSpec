from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import math
import os
from pathlib import Path
import textwrap
from typing import Any, Literal

import torch

from skip_search_spec.experiments.inference_prompts import (
    CHAT_TEST_PROMPTS,
    INFERENCE_TEST_PROMPTS_CONCRETE,
    INFERENCE_TEST_PROMPTS_CONCRETE_SWEDISH,
    INFERENCE_TEST_PROMPTS_EASY,
    INFERENCE_TEST_PROMPTS_HARD,
)
from skip_search_spec.inference.normal_inference import generate_normal
from skip_search_spec.inference.self_spec_inference import BridgeSelfSpeculator
from skip_search_spec.training.bridged_gap_model import BridgedGapModel


PromptSetName = Literal[
    "completion-style",
    "chat-style",
    "hard-completion-style",
    "concrete-completion-style",
    "swedish-concrete-completion-style",
]
VariantMode = Literal["auto", "flashhead", "no-flashhead", "both"]
VariantOrder = Literal["no-flashhead-first", "flashhead-first"]


PROMPT_SETS: dict[PromptSetName, tuple[list[tuple[str, str]], bool]] = {
    "chat-style": (CHAT_TEST_PROMPTS, True),
    "completion-style": (INFERENCE_TEST_PROMPTS_EASY, False),
    "hard-completion-style": (INFERENCE_TEST_PROMPTS_HARD, False),
    "concrete-completion-style": (INFERENCE_TEST_PROMPTS_CONCRETE, False),
    "swedish-concrete-completion-style": (
        INFERENCE_TEST_PROMPTS_CONCRETE_SWEDISH,
        False,
    ),
}


@dataclass(frozen=True, slots=True)
class PromptBenchResult:
    index: int
    name: str
    included_in_metrics: bool
    self_spec_seconds: float
    self_spec_generated_tokens: int
    verifier_calls: int
    drafted_tokens: int
    accepted_draft_tokens: int
    acceptance_rate: float
    dense_head_seconds: float
    flashhead_seconds: float
    drafter_registration_seconds: float
    drafter_teardown_seconds: float
    normal_seconds: float | None
    normal_generated_tokens: int | None
    exact_match: bool | None
    speedup_seconds: float | None
    speedup_per_generated_token: float | None
    first_mismatch_index: int | None


@dataclass(frozen=True, slots=True)
class BenchSummary:
    prompt_count_total: int
    prompt_count_included: int
    warmup_prompts: int
    total_self_spec_seconds: float
    total_normal_seconds: float | None
    total_self_spec_generated_tokens: int
    total_normal_generated_tokens: int | None
    total_drafted_tokens: int
    total_accepted_draft_tokens: int
    total_acceptance_rate: float
    mean_prompt_acceptance_rate: float
    total_speedup_seconds: float | None
    total_speedup_per_generated_token: float | None
    mean_prompt_speedup_per_generated_token: float | None
    std_prompt_speedup_per_generated_token: float | None
    exact_match_count: int | None
    exact_match_rate: float | None
    internal_dense_head_seconds: float | None
    internal_flashhead_seconds: float | None
    internal_head_seconds: float | None
    internal_body_seconds_estimate: float | None
    internal_drafter_registration_seconds: float | None
    internal_drafter_teardown_seconds: float | None


def run_cli(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        prog="skip_search_spec bench_self_spec",
        description="Benchmark self-speculation and save results as JSON.",
    )
    parser.add_argument("draft_block_size", type=int)
    parser.add_argument("bridge_checkpoint_path")
    parser.add_argument("flashhead_path", nargs="?")
    parser.add_argument(
        "--prompt-set",
        choices=tuple(PROMPT_SETS),
        default="completion-style",
    )
    parser.add_argument(
        "--warmup-prompts",
        type=int,
        default=0,
        help="Run but exclude the first N prompts from all aggregate metrics.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument(
        "--compare-to-normal",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run normal greedy generation for exact-match and speedup metrics.",
    )
    parser.add_argument(
        "--measure-internal-timings",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--flashhead-top-k-clusters",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--variants",
        choices=("auto", "flashhead", "no-flashhead", "both"),
        default="auto",
        help=(
            "Which self-spec variants to run. auto runs both when a flashhead "
            "path is supplied, otherwise no-flashhead only."
        ),
    )
    parser.add_argument(
        "--variant-order",
        choices=("no-flashhead-first", "flashhead-first"),
        default="no-flashhead-first",
        help="Order used when --variants=both or auto resolves to both.",
    )
    parser.add_argument(
        "--bridge-dtype",
        choices=("float32", "float16", "bfloat16", "model"),
        default="float32",
    )
    parser.add_argument(
        "--output-dir",
        default="benchmarks/self_spec",
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Optional basename prefix for the saved JSON file.",
    )

    args = parser.parse_args(argv)
    bench_self_spec(
        draft_block_size=args.draft_block_size,
        bridge_checkpoint_path=args.bridge_checkpoint_path,
        flashhead_path=args.flashhead_path,
        prompt_set=args.prompt_set,
        warmup_prompts=args.warmup_prompts,
        max_new_tokens=args.max_new_tokens,
        compare_to_normal=args.compare_to_normal,
        measure_internal_timings=args.measure_internal_timings,
        flashhead_top_k_clusters=args.flashhead_top_k_clusters,
        variants=args.variants,
        variant_order=args.variant_order,
        bridge_dtype=args.bridge_dtype,
        output_dir=args.output_dir,
        output_prefix=args.output_prefix,
    )


def plot_cli(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        prog="skip_search_spec plot_self_spec_bench",
        description="Plot a self-spec benchmark distribution from a saved JSON file.",
    )
    parser.add_argument("json_path")
    parser.add_argument(
        "--output-path",
        default=None,
        help="PNG path. Defaults to the JSON path with a .png suffix.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional figure title. Defaults to a title derived from the JSON metadata.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=None,
        help="Optional fixed number of histogram bins.",
    )
    args = parser.parse_args(argv)

    plot_self_spec_bench_json(
        json_path=args.json_path,
        output_path=args.output_path,
        title=args.title,
        bins=args.bins,
    )


def bench_self_spec(
    *,
    draft_block_size: int,
    bridge_checkpoint_path: str | Path,
    flashhead_path: str | Path | None = None,
    prompt_set: PromptSetName = "completion-style",
    warmup_prompts: int = 0,
    max_new_tokens: int = 200,
    compare_to_normal: bool = True,
    measure_internal_timings: bool = True,
    flashhead_top_k_clusters: int = 100,
    variants: VariantMode = "auto",
    variant_order: VariantOrder = "no-flashhead-first",
    bridge_dtype: str = "float32",
    output_dir: str | Path = "benchmarks/self_spec",
    output_prefix: str | None = None,
) -> Path:
    if draft_block_size < 1:
        raise ValueError("draft_block_size must be >= 1.")
    if warmup_prompts < 0:
        raise ValueError("warmup_prompts must be >= 0.")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be >= 1.")

    prompts, use_chat_template = PROMPT_SETS[prompt_set]
    if warmup_prompts >= len(prompts):
        raise ValueError(
            "warmup_prompts must be smaller than the number of prompts "
            f"({len(prompts)})."
        )

    bridge_path = Path(bridge_checkpoint_path).expanduser()
    flash_path = Path(flashhead_path).expanduser() if flashhead_path else None

    print("Loading bridge checkpoint:", bridge_path)
    bridged = BridgedGapModel.load_from_checkpoint(
        bridge_checkpoint_path=bridge_path,
        bridge_dtype=_parse_bridge_dtype(bridge_dtype),
    )

    variant_specs = _resolve_variant_specs(
        flash_path=flash_path,
        variants=variants,
        variant_order=variant_order,
    )
    normal_cache: dict[int, Any] = {}
    variant_results = []

    for variant in variant_specs:
        print()
        print(f"Running variant: {variant['label']}")
        variant_results.append(
            _run_bench_variant(
                bridged=bridged,
                variant=variant,
                prompts=prompts,
                normal_cache=normal_cache,
                use_chat_template=use_chat_template,
                warmup_prompts=warmup_prompts,
                max_new_tokens=max_new_tokens,
                draft_block_size=draft_block_size,
                compare_to_normal=compare_to_normal,
                measure_internal_timings=measure_internal_timings,
                flashhead_top_k_clusters=flashhead_top_k_clusters,
                bridge_path=bridge_path,
                prompt_set=prompt_set,
                bridge_dtype=bridge_dtype,
            )
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_stem = output_prefix or _default_run_stem(
        bridge_path=bridge_path,
        prompt_set=prompt_set,
    )
    json_path = output_dir / f"{run_stem}.json"

    payload = {
        "schema_version": 2,
        "common_metadata": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "bridge_checkpoint_path": str(bridge_path.resolve(strict=False)),
            "prompt_set": prompt_set,
            "variant_order": [variant["key"] for variant in variant_specs],
        },
        "variant_results": variant_results,
    }
    if len(variant_results) == 1:
        payload["metadata"] = variant_results[0]["metadata"]
        payload["summary"] = variant_results[0]["summary"]
        payload["prompt_results"] = variant_results[0]["prompt_results"]

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print()
    print("Saved benchmark JSON:", json_path)
    print(
        "Plot with:",
        f"poetry run skip_search_spec plot_self_spec_bench {json_path}",
    )
    return json_path


def _resolve_variant_specs(
    *,
    flash_path: Path | None,
    variants: VariantMode,
    variant_order: VariantOrder,
) -> list[dict[str, Any]]:
    if variants == "auto":
        variants = "both" if flash_path is not None else "no-flashhead"

    if variants in {"flashhead", "both"} and flash_path is None:
        raise ValueError("A flashhead_path is required for FlashHead benchmarking.")

    no_flashhead = {
        "key": "no_flashhead",
        "label": "Without FH",
        "flash_path": None,
    }
    with_flashhead = {
        "key": "flashhead",
        "label": "With FH",
        "flash_path": flash_path,
    }

    if variants == "no-flashhead":
        return [no_flashhead]
    if variants == "flashhead":
        return [with_flashhead]
    if variant_order == "flashhead-first":
        return [with_flashhead, no_flashhead]
    return [no_flashhead, with_flashhead]


def _run_bench_variant(
    *,
    bridged: BridgedGapModel,
    variant: dict[str, Any],
    prompts: list[tuple[str, str]],
    normal_cache: dict[int, Any],
    use_chat_template: bool,
    warmup_prompts: int,
    max_new_tokens: int,
    draft_block_size: int,
    compare_to_normal: bool,
    measure_internal_timings: bool,
    flashhead_top_k_clusters: int,
    bridge_path: Path,
    prompt_set: str,
    bridge_dtype: str,
) -> dict[str, Any]:
    flash_path = variant["flash_path"]
    speculator = BridgeSelfSpeculator(
        bridged_model=bridged,
        flashhead_path=flash_path,
        flashhead_top_k_clusters=flashhead_top_k_clusters,
    )
    prompt_results: list[PromptBenchResult] = []

    for prompt_index, (prompt_name, prompt) in enumerate(prompts, start=1):
        included = prompt_index > warmup_prompts
        phase = "measure" if included else "warmup"
        print()
        print(
            f"[{variant['label']} / {phase}] "
            f"Prompt {prompt_index}/{len(prompts)}: {prompt_name}"
        )

        self_spec_result = speculator.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            draft_block_size=draft_block_size,
            use_chat_template=use_chat_template,
            build_token_trace=False,
            measure_internal_timings=measure_internal_timings,
        )
        timings = self_spec_result.timings
        acceptance_rate = self_spec_result.accepted_draft_tokens / max(
            self_spec_result.drafted_tokens, 1
        )

        normal_seconds: float | None = None
        normal_generated_tokens: int | None = None
        exact_match: bool | None = None
        speedup_seconds: float | None = None
        speedup_per_generated_token: float | None = None
        first_mismatch_index: int | None = None

        if compare_to_normal:
            normal_result = normal_cache.get(prompt_index)
            if normal_result is None:
                normal_result = generate_normal(
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    use_chat_template=use_chat_template,
                    use_cache=True,
                    model=bridged.model,
                    tokenizer=bridged.tokenizer,
                    device=bridged.device,
                )
                normal_cache[prompt_index] = normal_result

            normal_seconds = normal_result.inference_seconds
            normal_generated_tokens = normal_result.num_generated_tokens
            exact_match = self_spec_result.text == normal_result.text
            speedup_seconds = _safe_div(normal_seconds, timings.total_seconds)
            speedup_per_generated_token = _speedup_per_generated_token(
                self_spec_tokens=self_spec_result.num_generated_tokens,
                self_spec_seconds=timings.total_seconds,
                normal_tokens=normal_result.num_generated_tokens,
                normal_seconds=normal_seconds,
            )
            if not exact_match:
                first_mismatch_index = _first_mismatch_index(
                    self_spec_result.text,
                    normal_result.text,
                )

        result = PromptBenchResult(
            index=prompt_index,
            name=prompt_name,
            included_in_metrics=included,
            self_spec_seconds=timings.total_seconds,
            self_spec_generated_tokens=self_spec_result.num_generated_tokens,
            verifier_calls=self_spec_result.verifier_calls,
            drafted_tokens=self_spec_result.drafted_tokens,
            accepted_draft_tokens=self_spec_result.accepted_draft_tokens,
            acceptance_rate=acceptance_rate,
            dense_head_seconds=timings.dense_head_seconds,
            flashhead_seconds=timings.flashhead_seconds,
            drafter_registration_seconds=timings.drafter_registration_seconds,
            drafter_teardown_seconds=timings.drafter_teardown_seconds,
            normal_seconds=normal_seconds,
            normal_generated_tokens=normal_generated_tokens,
            exact_match=exact_match,
            speedup_seconds=speedup_seconds,
            speedup_per_generated_token=speedup_per_generated_token,
            first_mismatch_index=first_mismatch_index,
        )
        prompt_results.append(result)

        print(
            {
                "acceptance_rate": acceptance_rate,
                "self_spec_seconds": timings.total_seconds,
                "speedup_per_generated_token": speedup_per_generated_token,
                "exact_match": exact_match,
            }
        )

    metadata = _build_metadata(
        bridged=bridged,
        speculator=speculator,
        bridge_path=bridge_path,
        flash_path=flash_path,
        prompt_set=prompt_set,
        use_chat_template=use_chat_template,
        draft_block_size=draft_block_size,
        max_new_tokens=max_new_tokens,
        compare_to_normal=compare_to_normal,
        measure_internal_timings=measure_internal_timings,
        flashhead_top_k_clusters=flashhead_top_k_clusters,
        bridge_dtype=bridge_dtype,
    )
    summary = _summarize(
        prompt_results=prompt_results,
        measure_internal_timings=measure_internal_timings,
    )

    return {
        "variant_key": variant["key"],
        "variant_label": variant["label"],
        "metadata": metadata,
        "summary": asdict(summary),
        "prompt_results": [asdict(result) for result in prompt_results],
    }


def _parse_bridge_dtype(value: str) -> torch.dtype | Literal["model"]:
    if value == "model":
        return "model"
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return dtype_map[value]


def _summarize(
    *,
    prompt_results: list[PromptBenchResult],
    measure_internal_timings: bool,
) -> BenchSummary:
    included = [result for result in prompt_results if result.included_in_metrics]
    speedups = [
        result.speedup_per_generated_token
        for result in included
        if result.speedup_per_generated_token is not None
    ]

    total_self_spec_seconds = sum(result.self_spec_seconds for result in included)
    total_self_spec_tokens = sum(
        result.self_spec_generated_tokens for result in included
    )
    total_drafted = sum(result.drafted_tokens for result in included)
    total_accepted = sum(result.accepted_draft_tokens for result in included)

    total_normal_seconds: float | None = None
    total_normal_tokens: int | None = None
    total_speedup_seconds: float | None = None
    total_speedup_per_token: float | None = None
    exact_match_count: int | None = None
    exact_match_rate: float | None = None

    compared = [result for result in included if result.normal_seconds is not None]
    if compared:
        total_normal_seconds = sum(result.normal_seconds or 0.0 for result in compared)
        total_normal_tokens = sum(
            result.normal_generated_tokens or 0 for result in compared
        )
        total_speedup_seconds = _safe_div(
            total_normal_seconds,
            total_self_spec_seconds,
        )
        total_speedup_per_token = _speedup_per_generated_token(
            self_spec_tokens=total_self_spec_tokens,
            self_spec_seconds=total_self_spec_seconds,
            normal_tokens=total_normal_tokens,
            normal_seconds=total_normal_seconds,
        )
        exact_match_count = sum(1 for result in compared if result.exact_match)
        exact_match_rate = _safe_div(exact_match_count, len(compared))

    dense_head_seconds = sum(result.dense_head_seconds for result in included)
    flashhead_seconds = sum(result.flashhead_seconds for result in included)
    registration_seconds = sum(
        result.drafter_registration_seconds for result in included
    )
    teardown_seconds = sum(result.drafter_teardown_seconds for result in included)
    head_seconds = dense_head_seconds + flashhead_seconds
    body_seconds = max(
        0.0,
        total_self_spec_seconds
        - head_seconds
        - registration_seconds
        - teardown_seconds,
    )

    return BenchSummary(
        prompt_count_total=len(prompt_results),
        prompt_count_included=len(included),
        warmup_prompts=len(prompt_results) - len(included),
        total_self_spec_seconds=total_self_spec_seconds,
        total_normal_seconds=total_normal_seconds,
        total_self_spec_generated_tokens=total_self_spec_tokens,
        total_normal_generated_tokens=total_normal_tokens,
        total_drafted_tokens=total_drafted,
        total_accepted_draft_tokens=total_accepted,
        total_acceptance_rate=_safe_div(total_accepted, total_drafted) or 0.0,
        mean_prompt_acceptance_rate=_mean(
            [result.acceptance_rate for result in included]
        ),
        total_speedup_seconds=total_speedup_seconds,
        total_speedup_per_generated_token=total_speedup_per_token,
        mean_prompt_speedup_per_generated_token=_mean_or_none(speedups),
        std_prompt_speedup_per_generated_token=_sample_std_or_none(speedups),
        exact_match_count=exact_match_count,
        exact_match_rate=exact_match_rate,
        internal_dense_head_seconds=dense_head_seconds
        if measure_internal_timings
        else None,
        internal_flashhead_seconds=flashhead_seconds
        if measure_internal_timings
        else None,
        internal_head_seconds=head_seconds if measure_internal_timings else None,
        internal_body_seconds_estimate=body_seconds
        if measure_internal_timings
        else None,
        internal_drafter_registration_seconds=registration_seconds
        if measure_internal_timings
        else None,
        internal_drafter_teardown_seconds=teardown_seconds
        if measure_internal_timings
        else None,
    )


def _build_metadata(
    *,
    bridged: BridgedGapModel,
    speculator: BridgeSelfSpeculator,
    bridge_path: Path,
    flash_path: Path | None,
    prompt_set: str,
    use_chat_template: bool,
    draft_block_size: int,
    max_new_tokens: int,
    compare_to_normal: bool,
    measure_internal_timings: bool,
    flashhead_top_k_clusters: int,
    bridge_dtype: str,
) -> dict[str, Any]:
    model_config = getattr(bridged.model, "config", None)
    backend = (
        os.environ.get("SKIP_SEARCH_ATTN_IMPLEMENTATION")
        or getattr(model_config, "_attn_implementation", None)
        or getattr(model_config, "attn_implementation", None)
        or "unknown"
    )
    model_dtype = str(next(bridged.model.parameters()).dtype)
    loaded_bridge_dtype = str(next(bridged.bridge.parameters()).dtype)
    flashhead = speculator.flashhead

    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "bridge_checkpoint_path": str(bridge_path.resolve(strict=False)),
        "flashhead_path": str(flash_path.resolve(strict=False)) if flash_path else None,
        "prompt_set": prompt_set,
        "use_chat_template": use_chat_template,
        "draft_block_size": draft_block_size,
        "max_new_tokens": max_new_tokens,
        "compare_to_normal": compare_to_normal,
        "measure_internal_timings": measure_internal_timings,
        "flashhead_enabled": flashhead is not None,
        "flashhead_top_k_clusters": flashhead_top_k_clusters
        if flashhead is not None
        else None,
        "flashhead_loaded_num_clusters": int(flashhead.cluster_to_token_ids.shape[0])
        if flashhead is not None
        else None,
        "flashhead_loaded_cluster_size": int(flashhead.cluster_size)
        if flashhead is not None
        else None,
        "model_name": bridged.config.model_name,
        "gap_start": bridged.gap.start,
        "gap_length": bridged.gap.length,
        "gap_end": bridged.gap.end,
        "reference_hidden_source": bridged.config.reference_hidden_source,
        "device": str(bridged.device),
        "requested_bridge_dtype": bridge_dtype,
        "loaded_bridge_dtype": loaded_bridge_dtype,
        "model_dtype": model_dtype,
        "torch_dtype_env": os.environ.get("SKIP_SEARCH_TORCH_DTYPE"),
        "attention_backend": str(backend),
        "attention_backend_env": os.environ.get("SKIP_SEARCH_ATTN_IMPLEMENTATION"),
        "torch_version": torch.__version__,
    }


def plot_self_spec_bench_json(
    *,
    json_path: str | Path,
    output_path: str | Path | None = None,
    title: str | None = None,
    bins: int | None = None,
) -> Path:
    json_path = Path(json_path)
    if output_path is None:
        output_path = json_path.with_suffix(".png")
    plot_path = Path(output_path)

    with json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    variants = _variant_payloads(payload)
    _plot_variant_distribution(
        variants=variants,
        output_path=plot_path,
        title=title,
        bins=bins,
    )
    print("Saved benchmark plot:", plot_path)
    return plot_path


def _variant_payloads(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if "variant_results" in payload:
        return list(payload["variant_results"])
    return [
        {
            "variant_key": "flashhead"
            if payload.get("metadata", {}).get("flashhead_enabled")
            else "no_flashhead",
            "variant_label": "With FH"
            if payload.get("metadata", {}).get("flashhead_enabled")
            else "Without FH",
            "metadata": payload["metadata"],
            "summary": payload["summary"],
            "prompt_results": payload["prompt_results"],
        }
    ]


def _plot_variant_distribution(
    *,
    variants: list[dict[str, Any]],
    output_path: Path,
    title: str | None,
    bins: int | None,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(output_path.parent / ".matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(output_path.parent / ".cache"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    from matplotlib.ticker import FuncFormatter, MultipleLocator

    styled_variants = [
        _styled_variant_payload(variant) for variant in _ordered_plot_variants(variants)
    ]
    all_speedups = [
        speedup for variant in styled_variants for speedup in variant["speedups"]
    ]
    if not all_speedups:
        raise ValueError(
            "No speedup values found. Run the benchmark with normal comparison."
        )

    common_metadata = styled_variants[0]["metadata"]
    model_name = str(common_metadata.get("model_name", ""))
    fig = plt.figure(figsize=(11.25, 6.25))
    fig.patch.set_facecolor("white")

    fig.text(
        0.5,
        0.955,
        title or "Self-speculation inference speedup",
        ha="center",
        va="top",
        fontsize=20,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.905,
        _model_display_name(model_name),
        ha="center",
        va="top",
        fontsize=12.5,
        fontweight="bold",
        color="#4A4A4A",
    )

    ax = fig.add_axes([0.055, 0.385, 0.89, 0.44])
    rounded_panel = FancyBboxPatch(
        (0, 0),
        1,
        1,
        boxstyle="round,pad=0.012",
        transform=ax.transAxes,
        facecolor="white",
        edgecolor="#DCDCDC",
        linewidth=0.9,
        zorder=-10,
        clip_on=False,
    )
    ax.add_patch(rounded_panel)

    bin_edges = _histogram_bin_edges(all_speedups, bins)
    max_count = 0
    for variant in styled_variants:
        counts, _, _ = ax.hist(
            variant["speedups"],
            bins=bin_edges,
            histtype="bar",
            color=variant["fill"],
            edgecolor=variant["edge"],
            linewidth=2.0,
            alpha=0.34,
            label=variant["label"],
        )
        max_count = max(max_count, int(max(counts)) if len(counts) else 0)
    _plot_normal_fit_curves(ax, styled_variants, bin_edges, max_count)

    ax.set_axisbelow(True)
    ax.xaxis.set_major_locator(MultipleLocator(0.1))
    ax.xaxis.set_minor_locator(MultipleLocator(0.02))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}x"))
    ax.grid(axis="x", which="major", color="#E7E7E7", linewidth=1.0)
    ax.grid(axis="x", which="minor", color="#EFEFEF", linewidth=0.65)
    ax.grid(axis="y", color="#ECECEC", linewidth=0.8)
    ax.tick_params(axis="y", length=0, labelsize=12)
    ax.tick_params(axis="x", length=0, labelsize=12, pad=8)
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    x_min = min(bin_edges)
    x_max = max(bin_edges)
    ax.set_xlim(x_min - 0.03, x_max + 0.03)
    ax.set_ylim(0, max(max_count * 1.42, 1.0))

    _annotate_variant_means(ax, styled_variants, max_count)

    info_ax = fig.add_axes([0.055, 0.105, 0.89, 0.205])
    _draw_info_strip(info_ax, _info_sections(styled_variants))
    file_ax = fig.add_axes([0.055, 0.025, 0.89, 0.055])
    _draw_file_strip(file_ax, _file_rows(styled_variants))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _ordered_plot_variants(variants: list[dict[str, Any]]) -> list[dict[str, Any]]:
    order = {"no_flashhead": 0, "flashhead": 1}
    return sorted(
        variants,
        key=lambda variant: order.get(str(variant.get("variant_key")), 10),
    )


def _styled_variant_payload(variant: dict[str, Any]) -> dict[str, Any]:
    key = str(variant.get("variant_key", ""))
    metadata = variant["metadata"]
    summary = variant["summary"]
    prompt_results = variant["prompt_results"]
    if key == "flashhead" or metadata.get("flashhead_enabled"):
        colors = {
            "edge": "#9B0079",
            "fill": "#D790C6",
            "text": "#9B0079",
        }
        label = "With FH"
    else:
        colors = {
            "edge": "#007399",
            "fill": "#7FB6C8",
            "text": "#174C56",
        }
        label = "Without FH"

    speedups = [
        float(result["speedup_per_generated_token"])
        for result in prompt_results
        if result.get("included_in_metrics")
        and result.get("speedup_per_generated_token") is not None
    ]
    return {
        "key": key,
        "label": label,
        "metadata": metadata,
        "summary": summary,
        "speedups": speedups,
        **colors,
    }


def _histogram_bin_edges(values: list[float], bins: int | None) -> list[float]:
    value_min = min(values)
    value_max = max(values)
    if value_min == value_max:
        value_min -= 0.05
        value_max += 0.05
    padding = max((value_max - value_min) * 0.18, 0.04)
    lower = math.floor((value_min - padding) / 0.02) * 0.02
    upper = math.ceil((value_max + padding) / 0.02) * 0.02
    bin_count = bins or min(14, max(6, int(math.sqrt(len(values))) + 4))
    width = (upper - lower) / bin_count
    return [lower + width * i for i in range(bin_count + 1)]


def _annotate_variant_means(
    ax: Any,
    variants: list[dict[str, Any]],
    max_count: int,
) -> None:
    y_top = max(max_count * 1.22, 1.0)
    for index, variant in enumerate(variants):
        summary = variant["summary"]
        mean = summary.get("mean_prompt_speedup_per_generated_token")
        if mean is None:
            continue
        y = y_top - index * max(max_count * 0.16, 0.2)
        ax.text(
            float(mean),
            y,
            f"{variant['label']} avg speedup = {_fmt_x(float(mean))}",
            ha="center",
            va="bottom",
            fontsize=10.8,
            fontweight="semibold",
            color=variant["text"],
        )


def _plot_normal_fit_curves(
    ax: Any,
    variants: list[dict[str, Any]],
    bin_edges: list[float],
    max_count: int,
) -> None:
    if len(bin_edges) < 2:
        return
    bin_width = bin_edges[1] - bin_edges[0]
    x_min = bin_edges[0]
    x_max = bin_edges[-1]
    xs = [x_min + (x_max - x_min) * i / 399 for i in range(400)]
    for variant in variants:
        speedups = variant["speedups"]
        if len(speedups) < 2:
            continue
        mean = sum(speedups) / len(speedups)
        std = _sample_std_or_none(speedups)
        if std is None or std <= 0:
            continue
        ys = [
            len(speedups)
            * bin_width
            * (1.0 / (std * math.sqrt(2.0 * math.pi)))
            * math.exp(-0.5 * ((x - mean) / std) ** 2)
            for x in xs
        ]
        if max_count > 0 and max(ys) > max_count * 1.08:
            scale = (max_count * 1.08) / max(ys)
            ys = [y * scale for y in ys]
        ax.plot(
            xs,
            ys,
            color=variant["edge"],
            linewidth=2.0,
            alpha=0.9,
        )


def _info_sections(
    variants: list[dict[str, Any]],
) -> list[tuple[str, list[tuple[str, str]]]]:
    first = variants[0]
    metadata = first["metadata"]
    setup = [
        ("prompt set", _value(metadata.get("prompt_set"))),
        ("block size", _value(metadata.get("draft_block_size"))),
        ("warmup prompts", _value(first["summary"].get("warmup_prompts"))),
        ("measured prompts", _value(first["summary"].get("prompt_count_included"))),
    ]
    runtime = [
        ("backend", _value(metadata.get("attention_backend"))),
        ("model dtype", _dtype_label(metadata.get("model_dtype"))),
        ("bridge dtype", _dtype_label(metadata.get("loaded_bridge_dtype"))),
        (
            "internal timing",
            _yes_no(bool(metadata.get("measure_internal_timings"))),
        ),
    ]

    results: list[tuple[str, str]] = []
    timings: list[tuple[str, str]] = []
    flashhead_meta = next(
        (variant["metadata"] for variant in variants if variant["metadata"].get("flashhead_enabled")),
        None,
    )
    for variant in variants:
        summary = variant["summary"]
        prefix = "FH" if variant["metadata"].get("flashhead_enabled") else "No FH"
        results.extend(
            [
                (
                    f"{prefix} speedup",
                    _fmt_x(summary.get("mean_prompt_speedup_per_generated_token")),
                ),
                (f"{prefix} acceptance", _fmt_pct(summary.get("total_acceptance_rate"))),
                (f"{prefix} exact match", _fmt_pct(summary.get("exact_match_rate"))),
            ]
        )
        timings.append(
            (
                f"{prefix} head/body",
                (
                    f"{_fmt_seconds(summary.get('internal_head_seconds'))} / "
                    f"{_fmt_seconds(summary.get('internal_body_seconds_estimate'))}"
                ),
            )
        )

    if flashhead_meta:
        timings.extend(
            [
                (
                    "FH index",
                    (
                        f"{_value(flashhead_meta.get('flashhead_loaded_num_clusters'))} "
                        f"clusters; top-k "
                        f"{_value(flashhead_meta.get('flashhead_top_k_clusters'))}"
                    ),
                ),
            ]
        )

    return [
        ("Setup", setup),
        ("Runtime", runtime),
        ("Results", results),
        ("Head", timings),
    ]


def _file_rows(variants: list[dict[str, Any]]) -> list[tuple[str, str]]:
    metadata = variants[0]["metadata"]
    rows = [("bridge file", _value(metadata.get("bridge_checkpoint_path")))]
    flashhead_path = next(
        (
            variant["metadata"].get("flashhead_path")
            for variant in variants
            if variant["metadata"].get("flashhead_path")
        ),
        None,
    )
    if flashhead_path:
        rows.append(("flashhead file", _value(flashhead_path)))
    return rows


def _draw_info_strip(
    ax: Any,
    sections: list[tuple[str, list[tuple[str, str]]]],
) -> None:
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    title_color = "#173F46"
    label_color = "#587078"
    value_color = "#243E45"
    column_width = 0.238
    column_gap = 0.016
    y_top = 0.95
    body_top = 0.73

    for column, (title, rows) in enumerate(sections):
        x = column * (column_width + column_gap)
        ax.text(
            x,
            y_top,
            title,
            ha="left",
            va="top",
            fontsize=9.4,
            fontweight="bold",
            color=title_color,
            clip_on=True,
        )
        ax.plot(
            [x, x + column_width],
            [0.79, 0.79],
            color="#D9E1E3",
            linewidth=0.8,
            solid_capstyle="round",
            clip_on=True,
        )

        y = body_top
        crowded = len(rows) > 4
        row_step = 0.15 if not crowded else 0.102
        font_size = 7.95 if not crowded else 7.45
        wrap_width = 48
        for label, value in rows:
            line = _metadata_line(label, value, width=wrap_width)
            ax.text(
                x,
                y,
                line,
                ha="left",
                va="top",
                fontsize=font_size,
                color=value_color,
                linespacing=1.18,
                clip_on=True,
            )
            label_text = line.split("=", 1)[0]
            ax.text(
                x,
                y,
                label_text,
                ha="left",
                va="top",
                fontsize=font_size,
                color=label_color,
                linespacing=1.18,
                clip_on=True,
            )
            y -= row_step * max(1, line.count("\n") + 1)


def _draw_file_strip(ax: Any, rows: list[tuple[str, str]]) -> None:
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    label_color = "#587078"
    value_color = "#243E45"
    max_line_length = max((len(label) + len(value) + 3 for label, value in rows), default=0)
    font_size = 5.8
    if max_line_length > 190:
        font_size = 4.8
    if max_line_length > 230:
        font_size = 4.3
    y = 0.82
    for label, value in rows:
        prefix = f"{label} = "
        ax.text(
            0,
            y,
            prefix,
            ha="left",
            va="top",
            fontsize=font_size,
            color=label_color,
            clip_on=True,
        )
        ax.text(
            0.071,
            y,
            value,
            ha="left",
            va="top",
            fontsize=font_size,
            color=value_color,
            clip_on=True,
        )
        y -= 0.42


def _model_display_name(model_name: str) -> str:
    display = model_name.split("/")[-1].replace("_", ".")
    return display.replace("-", " ")


def _metadata_line(label: str, value: str, *, width: int) -> str:
    prefix = f"{label} = "
    wrapped = textwrap.fill(
        prefix + value,
        width=width,
        subsequent_indent=" " * len(prefix),
        break_long_words=False,
        break_on_hyphens=False,
    )
    return wrapped


def _dtype_label(value: Any) -> str:
    text = _value(value)
    return text.replace("torch.", "")


def _value(value: Any) -> str:
    if value is None:
        return "n/a"
    return str(value)


def _shorten_path(value: Any, *, max_chars: int = 24) -> str:
    if value is None:
        return "n/a"
    path = Path(str(value))
    filename = path.name or str(value)
    if len(filename) <= max_chars:
        return filename
    keep_left = max_chars // 2 - 1
    keep_right = max_chars - keep_left - 3
    return f"{filename[:keep_left]}...{filename[-keep_right:]}"


def _first_mismatch_index(left: str, right: str) -> int:
    for idx, (left_char, right_char) in enumerate(zip(left, right)):
        if left_char != right_char:
            return idx
    return min(len(left), len(right))


def _speedup_per_generated_token(
    *,
    self_spec_tokens: int,
    self_spec_seconds: float,
    normal_tokens: int,
    normal_seconds: float,
) -> float | None:
    self_tps = _safe_div(self_spec_tokens, self_spec_seconds)
    normal_tps = _safe_div(normal_tokens, normal_seconds)
    if self_tps is None or normal_tps is None:
        return None
    return _safe_div(self_tps, normal_tps)


def _safe_div(numerator: float | int, denominator: float | int) -> float | None:
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return _mean(values)


def _sample_std_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    mean = _mean(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def _default_run_stem(*, bridge_path: Path, prompt_set: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"bench_self_spec__{bridge_path.stem}__{prompt_set}__{timestamp}"


def _default_plot_title(metadata: dict[str, Any]) -> str:
    prompt_set = str(metadata.get("prompt_set", "unknown prompt set"))
    return f"Self-spec speedup distribution ({prompt_set})"


def _metric_value(value: Any) -> str:
    if value is None:
        return "n/a"
    return str(value)


def _fmt_float(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def _fmt_seconds(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}s"


def _fmt_x(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}x"


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * value:.2f}%"


def _fmt_count(value: int | None, total: int) -> str:
    if value is None:
        return "n/a"
    return f"{value}/{total}"


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"
