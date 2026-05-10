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
        description="Benchmark self-speculation and save a rigorous report plot.",
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
        help="Optional basename prefix for the saved PNG and JSON files.",
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
        bridge_dtype=args.bridge_dtype,
        output_dir=args.output_dir,
        output_prefix=args.output_prefix,
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
    bridge_dtype: str = "float32",
    output_dir: str | Path = "benchmarks/self_spec",
    output_prefix: str | None = None,
) -> tuple[Path, Path]:
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
        print(f"[{phase}] Prompt {prompt_index}/{len(prompts)}: {prompt_name}")

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
            normal_result = generate_normal(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                use_chat_template=use_chat_template,
                use_cache=True,
                model=bridged.model,
                tokenizer=bridged.tokenizer,
                device=bridged.device,
            )
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

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_stem = output_prefix or _default_run_stem(
        bridge_path=bridge_path,
        prompt_set=prompt_set,
    )
    json_path = output_dir / f"{run_stem}.json"
    plot_path = output_dir / f"{run_stem}.png"

    payload = {
        "schema_version": 1,
        "metadata": metadata,
        "summary": asdict(summary),
        "prompt_results": [asdict(result) for result in prompt_results],
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    _plot_report(
        plot_path=plot_path,
        metadata=metadata,
        summary=summary,
        prompt_results=prompt_results,
    )

    print()
    print("Saved benchmark JSON:", json_path)
    print("Saved benchmark plot:", plot_path)
    return json_path, plot_path


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


def _plot_report(
    *,
    plot_path: Path,
    metadata: dict[str, Any],
    summary: BenchSummary,
    prompt_results: list[PromptBenchResult],
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(plot_path.parent / ".matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(plot_path.parent / ".cache"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt

    included = [result for result in prompt_results if result.included_in_metrics]
    speedup_results = [
        result for result in included if result.speedup_per_generated_token is not None
    ]
    speedups = [
        float(result.speedup_per_generated_token)
        for result in speedup_results
        if result.speedup_per_generated_token is not None
    ]

    fig = plt.figure(figsize=(18, 11), constrained_layout=True)
    grid = fig.add_gridspec(
        nrows=2,
        ncols=3,
        width_ratios=[1.4, 1.4, 1.1],
        height_ratios=[1.0, 1.15],
    )
    ax_hist = fig.add_subplot(grid[0, 0:2])
    ax_bars = fig.add_subplot(grid[1, 0:2])
    ax_metrics = fig.add_subplot(grid[:, 2])

    fig.suptitle(
        f"Self-Spec Benchmark: {metadata['model_name']}",
        fontsize=17,
        fontweight="bold",
    )

    _plot_speedup_histogram(ax_hist, speedups, summary)
    _plot_prompt_speedups(ax_bars, speedup_results)
    _plot_metric_panel(ax_metrics, metadata, summary)

    fig.savefig(plot_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_speedup_histogram(
    ax: Any,
    speedups: list[float],
    summary: BenchSummary,
) -> None:
    ax.set_title("Distribution of Speedup Per Generated Token")
    ax.set_xlabel("Self-spec tokens/sec divided by normal tokens/sec")
    ax.set_ylabel("Prompt count")
    ax.grid(axis="y", alpha=0.25)

    if not speedups:
        ax.text(
            0.5,
            0.5,
            "No speedup data available.\nRun with --compare-to-normal.",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return

    bin_count = min(20, max(3, int(math.sqrt(len(speedups))) + 1))
    counts, bins, _ = ax.hist(
        speedups,
        bins=bin_count,
        color="#2f6f73",
        edgecolor="white",
        alpha=0.86,
    )

    mean = summary.mean_prompt_speedup_per_generated_token
    std = summary.std_prompt_speedup_per_generated_token
    if mean is not None:
        ax.axvline(mean, color="#101820", linewidth=2, label=f"mean={mean:.3f}x")

    if mean is not None and std is not None and std > 0:
        x_min = min(bins[0], mean - 4 * std)
        x_max = max(bins[-1], mean + 4 * std)
        xs = [x_min + (x_max - x_min) * i / 399 for i in range(400)]
        bin_width = float(bins[1] - bins[0]) if len(bins) > 1 else 1.0
        ys = [
            len(speedups)
            * bin_width
            * (1.0 / (std * math.sqrt(2.0 * math.pi)))
            * math.exp(-0.5 * ((x - mean) / std) ** 2)
            for x in xs
        ]
        ax.plot(
            xs,
            ys,
            color="#c8553d",
            linewidth=2.2,
            label=f"normal fit std={std:.3f}",
        )
    elif mean is not None:
        ax.text(
            0.98,
            0.92,
            "std=0.000",
            ha="right",
            va="top",
            transform=ax.transAxes,
        )

    max_count = max(counts) if len(counts) else 1
    if summary.total_speedup_per_generated_token is not None:
        ax.axvline(
            summary.total_speedup_per_generated_token,
            color="#8c1c13",
            linestyle="--",
            linewidth=1.8,
            label=f"total={summary.total_speedup_per_generated_token:.3f}x",
        )
    ax.set_ylim(0, max(max_count * 1.25, 1.0))
    ax.legend(loc="best", frameon=False)


def _plot_prompt_speedups(
    ax: Any,
    speedup_results: list[PromptBenchResult],
) -> None:
    ax.set_title("Per-Prompt Speedup")
    ax.set_xlabel("Speedup per generated token")
    ax.grid(axis="x", alpha=0.25)

    if not speedup_results:
        ax.text(
            0.5,
            0.5,
            "No per-prompt speedups to plot.",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_yticks([])
        return

    ordered = sorted(
        speedup_results,
        key=lambda result: result.speedup_per_generated_token or 0.0,
    )
    labels = [_shorten_label(result.name) for result in ordered]
    values = [float(result.speedup_per_generated_token or 0.0) for result in ordered]
    colors = ["#2f6f73" if result.exact_match else "#c8553d" for result in ordered]

    y_positions = list(range(len(ordered)))
    ax.barh(y_positions, values, color=colors, alpha=0.88)
    ax.axvline(1.0, color="#101820", linestyle=":", linewidth=1.6)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=max(6, min(9, 220 // max(len(labels), 1))))

    x_max = max(values) if values else 1.0
    ax.set_xlim(0, max(1.15, x_max * 1.12))
    for y_pos, value in zip(y_positions, values):
        ax.text(
            value + max(0.01, x_max * 0.012),
            y_pos,
            f"{value:.2f}x",
            va="center",
            fontsize=7,
        )


def _plot_metric_panel(
    ax: Any,
    metadata: dict[str, Any],
    summary: BenchSummary,
) -> None:
    ax.axis("off")
    sections = [
        (
            "Run configuration",
            [
                ("Prompt set", metadata["prompt_set"]),
                ("Warmup skipped", summary.warmup_prompts),
                ("Prompts measured", summary.prompt_count_included),
                ("Block size", metadata["draft_block_size"]),
                ("Max new tokens", metadata["max_new_tokens"]),
                ("Internal timings", _yes_no(metadata["measure_internal_timings"])),
                ("FlashHead", _yes_no(metadata["flashhead_enabled"])),
                ("FlashHead clusters", metadata["flashhead_loaded_num_clusters"]),
                ("FlashHead top-k", metadata["flashhead_top_k_clusters"]),
            ],
        ),
        (
            "Aggregate metrics",
            [
                ("Total acceptance", _fmt_pct(summary.total_acceptance_rate)),
                ("Mean acceptance", _fmt_pct(summary.mean_prompt_acceptance_rate)),
                (
                    "Total speedup/token",
                    _fmt_x(summary.total_speedup_per_generated_token),
                ),
                (
                    "Mean prompt speedup",
                    _fmt_x(summary.mean_prompt_speedup_per_generated_token),
                ),
                (
                    "Prompt speedup std",
                    _fmt_float(summary.std_prompt_speedup_per_generated_token),
                ),
                ("Exact match rate", _fmt_pct(summary.exact_match_rate)),
                (
                    "Exact matches",
                    _fmt_count(
                        summary.exact_match_count, summary.prompt_count_included
                    ),
                ),
                ("Self-spec tokens", summary.total_self_spec_generated_tokens),
                ("Normal tokens", summary.total_normal_generated_tokens),
            ],
        ),
        (
            "Timing totals",
            [
                ("Self-spec seconds", _fmt_seconds(summary.total_self_spec_seconds)),
                ("Normal seconds", _fmt_seconds(summary.total_normal_seconds)),
                ("Head seconds", _fmt_seconds(summary.internal_head_seconds)),
                (
                    "Dense head seconds",
                    _fmt_seconds(summary.internal_dense_head_seconds),
                ),
                ("FlashHead seconds", _fmt_seconds(summary.internal_flashhead_seconds)),
                (
                    "Body seconds est.",
                    _fmt_seconds(summary.internal_body_seconds_estimate),
                ),
                (
                    "Hook registration",
                    _fmt_seconds(summary.internal_drafter_registration_seconds),
                ),
                (
                    "Hook teardown",
                    _fmt_seconds(summary.internal_drafter_teardown_seconds),
                ),
            ],
        ),
        (
            "Runtime and files",
            [
                ("Model dtype", metadata["model_dtype"]),
                ("Bridge dtype", metadata["loaded_bridge_dtype"]),
                ("Backend", metadata["attention_backend"]),
                ("Device", metadata["device"]),
                ("Bridge file", metadata["bridge_checkpoint_path"]),
                ("FlashHead file", metadata["flashhead_path"]),
            ],
        ),
    ]

    wrapped_sections: list[tuple[str, list[tuple[str, str]]]] = []
    total_lines = 0
    for title, rows in sections:
        wrapped_rows: list[tuple[str, str]] = []
        total_lines += 2
        for key, value in rows:
            wrapped = _wrap_value(value)
            wrapped_rows.append((key, wrapped))
            total_lines += max(1, wrapped.count("\n") + 1)
        total_lines += 1
        wrapped_sections.append((title, wrapped_rows))

    line_step = min(0.025, 0.95 / max(total_lines, 1))
    scale = line_step / 0.025
    title_font_size = max(7.8, min(11.0, 11.0 * scale))
    row_font_size = max(5.6, min(8.2, 8.2 * scale))

    y = 0.99
    for title, rows in wrapped_sections:
        ax.text(
            0.0,
            y,
            title,
            fontsize=title_font_size,
            fontweight="bold",
            va="top",
            transform=ax.transAxes,
        )
        y -= line_step * 1.35
        for key, wrapped in rows:
            text = f"{key}: {wrapped}"
            ax.text(
                0.0,
                y,
                text,
                fontsize=row_font_size,
                family="monospace",
                va="top",
                transform=ax.transAxes,
            )
            line_count = max(1, text.count("\n") + 1)
            y -= line_step * line_count
        y -= line_step * 0.85


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


def _shorten_label(label: str, max_len: int = 42) -> str:
    if len(label) <= max_len:
        return label
    return label[: max_len - 3] + "..."


def _wrap_value(value: Any, width: int = 43) -> str:
    if value is None:
        return "n/a"
    text = str(value)
    return textwrap.fill(
        text,
        width=width,
        subsequent_indent="  ",
        break_long_words=True,
        break_on_hyphens=False,
    )


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
