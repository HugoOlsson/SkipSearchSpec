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
from skip_search_spec.helpers.versioning import get_git_revision
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
PhaseName = Literal["warmup", "profile", "speed"]


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
    """One real speed-benchmark prompt.

    These rows are generated with measure_internal_timings=False. Profile timings
    are stored separately in ProfilePromptResult / ProfileSummary.
    """

    index: int
    name: str
    included_in_metrics: bool
    self_spec_seconds: float
    self_spec_generated_tokens: int
    verifier_calls: int
    drafted_tokens: int
    accepted_draft_tokens: int
    acceptance_rate: float
    drafter_total_seconds: float
    drafter_body_seconds: float
    dense_head_seconds: float
    flashhead_seconds: float
    drafter_overhead_seconds: float
    drafter_registration_seconds: float
    drafter_teardown_seconds: float
    self_spec_peak_allocated_bytes: int | None
    normal_peak_allocated_bytes: int | None
    normal_seconds: float | None
    normal_generated_tokens: int | None
    exact_match: bool | None
    speedup_seconds: float | None
    speedup_per_generated_token: float | None
    first_mismatch_index: int | None


@dataclass(frozen=True, slots=True)
class ProfilePromptResult:
    """One internally timed profiling prompt.

    These rows are generated with measure_internal_timings=True and are not used
    for speedup numbers.
    """

    index: int
    name: str
    profile_seconds: float
    generated_tokens: int

    verifier_seconds: float
    verifier_calls: int

    drafter_total_seconds: float
    drafter_body_seconds: float
    drafter_dense_head_seconds: float
    drafter_flashhead_seconds: float
    drafter_head_seconds: float
    drafter_overhead_seconds: float
    drafter_calls: int

    drafter_registration_seconds: float
    drafter_teardown_seconds: float


@dataclass(frozen=True, slots=True)
class PhaseRunResult:
    prompt_results: list[PromptBenchResult]
    profile_results: list[ProfilePromptResult]


@dataclass(frozen=True, slots=True)
class ProfileSummary:
    profile_prompt_count: int
    profile_generated_tokens: int

    profile_total_seconds: float
    verifier_seconds: float

    drafter_total_seconds: float
    drafter_body_seconds: float
    drafter_dense_head_seconds: float
    drafter_flashhead_seconds: float
    drafter_head_seconds: float
    drafter_overhead_seconds: float

    drafter_body_fraction: float | None
    drafter_head_fraction: float | None
    drafter_overhead_fraction: float | None
    drafter_to_verifier_fraction: float | None

    drafter_registration_seconds: float
    drafter_teardown_seconds: float


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
    mean_self_spec_generated_tokens_per_prompt: float
    mean_normal_generated_tokens_per_prompt: float | None
    normal_tokens_per_second: float | None
    self_spec_tokens_per_second: float | None


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
        default="concrete-completion-style",
    )
    parser.add_argument(
        "--warmup-prompts",
        type=int,
        default=5,
        help=(
            "Run N warmup prompts before profiling and benchmarking. Warmup "
            "results are discarded."
        ),
    )
    parser.add_argument(
        "--profile-prompts",
        type=int,
        default=15,
        help=(
            "Run M prompts with internal timings after warmup. These profile "
            "runs are saved separately and are not used for speedup metrics."
        ),
    )
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=None,
        help="Optional maximum number of prompts to run from the selected prompt set.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument(
        "--compare-to-normal",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run normal greedy generation for exact-match and speedup metrics.",
    )
    parser.add_argument(
        "--flashhead-top-k-clusters",
        type=int,
        default=50,
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
        "--debug-argmax-ties",
        action="store_true",
        help="Print the first exact argmax tie per generation for self-spec and normal inference.",
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
        profile_prompts=args.profile_prompts,
        max_prompts=args.max_prompts,
        max_new_tokens=args.max_new_tokens,
        compare_to_normal=args.compare_to_normal,
        flashhead_top_k_clusters=args.flashhead_top_k_clusters,
        variants=args.variants,
        variant_order=args.variant_order,
        bridge_dtype=args.bridge_dtype,
        output_dir=args.output_dir,
        output_prefix=args.output_prefix,
        debug_argmax_ties=args.debug_argmax_ties,
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
    profile_prompts: int = 5,
    max_prompts: int | None = None,
    max_new_tokens: int = 200,
    compare_to_normal: bool = True,
    flashhead_top_k_clusters: int = 100,
    variants: VariantMode = "auto",
    variant_order: VariantOrder = "no-flashhead-first",
    bridge_dtype: str = "float32",
    output_dir: str | Path = "benchmarks/self_spec",
    output_prefix: str | None = None,
    debug_argmax_ties: bool = False,
) -> Path:
    if draft_block_size < 1:
        raise ValueError("draft_block_size must be >= 1.")
    if warmup_prompts < 0:
        raise ValueError("warmup_prompts must be >= 0.")
    if profile_prompts < 0:
        raise ValueError("profile_prompts must be >= 0.")
    if max_prompts is not None and max_prompts < 1:
        raise ValueError("max_prompts must be >= 1.")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be >= 1.")

    prompts, use_chat_template = PROMPT_SETS[prompt_set]
    original_prompt_count = len(prompts)
    if max_prompts is not None:
        prompts = prompts[:max_prompts]
    if warmup_prompts + profile_prompts > len(prompts):
        raise ValueError(
            "warmup_prompts + profile_prompts must be <= the number of selected prompts "
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
                profile_prompts=profile_prompts,
                max_new_tokens=max_new_tokens,
                draft_block_size=draft_block_size,
                compare_to_normal=compare_to_normal,
                flashhead_top_k_clusters=flashhead_top_k_clusters,
                bridge_path=bridge_path,
                prompt_set=prompt_set,
                bridge_dtype=bridge_dtype,
                requested_max_prompts=max_prompts,
                available_prompt_count=original_prompt_count,
                debug_argmax_ties=debug_argmax_ties,
            )
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_stem = output_prefix or _default_run_stem(
        bridge_path=bridge_path,
        prompt_set=prompt_set,
        draft_block_size=draft_block_size,
        max_new_tokens=max_new_tokens,
        warmup_prompts=warmup_prompts,
        profile_prompts=profile_prompts,
        variants=variants,
        flash_path=flash_path,
        bridged=bridged,
    )
    json_path = output_dir / f"{run_stem}.json"

    payload = {
        "schema_version": 3,
        "common_metadata": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "bridge_checkpoint_path": str(bridge_path.resolve(strict=False)),
            "prompt_set": prompt_set,
            "max_prompts": max_prompts,
            "available_prompt_count": original_prompt_count,
            "run_prompt_count": len(prompts),
            "warmup_prompts": warmup_prompts,
            "profile_prompts": profile_prompts,
            "variant_order": [variant["key"] for variant in variant_specs],
            "speed_measure_internal_timings": False,
            "profile_measure_internal_timings": profile_prompts > 0,
        },
        "variant_results": variant_results,
    }
    if len(variant_results) == 1:
        payload["metadata"] = variant_results[0]["metadata"]
        payload["summary"] = variant_results[0]["summary"]
        payload["profile_summary"] = variant_results[0]["profile_summary"]
        payload["profile_results"] = variant_results[0]["profile_results"]
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
        "label": "Without ANNH",
        "flash_path": None,
    }
    with_flashhead = {
        "key": "flashhead",
        "label": "With ANNH",
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
    profile_prompts: int,
    max_new_tokens: int,
    draft_block_size: int,
    compare_to_normal: bool,
    flashhead_top_k_clusters: int,
    bridge_path: Path,
    prompt_set: str,
    bridge_dtype: str,
    requested_max_prompts: int | None,
    available_prompt_count: int,
    debug_argmax_ties: bool,
) -> dict[str, Any]:
    flash_path = variant["flash_path"]
    speculator = BridgeSelfSpeculator(
        bridged_model=bridged,
        flashhead_path=flash_path,
        flashhead_top_k_clusters=flashhead_top_k_clusters,
    )

    warmup_slice = prompts[:warmup_prompts]
    profile_slice = prompts[warmup_prompts : warmup_prompts + profile_prompts]
    speed_slice = prompts

    _ = _run_prompt_phase(
        phase="warmup",
        speculator=speculator,
        bridged=bridged,
        prompts=warmup_slice,
        start_index=1,
        normal_cache=normal_cache,
        use_chat_template=use_chat_template,
        max_new_tokens=max_new_tokens,
        draft_block_size=draft_block_size,
        compare_to_normal=compare_to_normal,
        variant_label=variant["label"],
        debug_argmax_ties=debug_argmax_ties,
    )

    profile_phase = _run_prompt_phase(
        phase="profile",
        speculator=speculator,
        bridged=bridged,
        prompts=profile_slice,
        start_index=warmup_prompts + 1,
        normal_cache=normal_cache,
        use_chat_template=use_chat_template,
        max_new_tokens=max_new_tokens,
        draft_block_size=draft_block_size,
        compare_to_normal=False,
        variant_label=variant["label"],
        debug_argmax_ties=debug_argmax_ties,
    )
    profile_results = profile_phase.profile_results
    profile_summary = _summarize_profile(
        profile_results,
        draft_block_size=draft_block_size,
    )

    speed_phase = _run_prompt_phase(
        phase="speed",
        speculator=speculator,
        bridged=bridged,
        prompts=speed_slice,
        start_index=1,
        normal_cache=normal_cache,
        use_chat_template=use_chat_template,
        max_new_tokens=max_new_tokens,
        draft_block_size=draft_block_size,
        compare_to_normal=compare_to_normal,
        variant_label=variant["label"],
        debug_argmax_ties=debug_argmax_ties,
    )
    prompt_results = speed_phase.prompt_results

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
        flashhead_top_k_clusters=flashhead_top_k_clusters,
        bridge_dtype=bridge_dtype,
        requested_max_prompts=requested_max_prompts,
        available_prompt_count=available_prompt_count,
        run_prompt_count=len(prompts),
        warmup_prompts=warmup_prompts,
        profile_prompts=profile_prompts,
        profile_prompt_count=len(profile_results),
        speed_prompt_count=len(prompt_results),
    )
    summary = _summarize(
        prompt_results=prompt_results,
        warmup_prompts=warmup_prompts,
    )

    metadata["mean_self_spec_peak_allocated_bytes"] = _mean_int_or_none(
        [
            result.self_spec_peak_allocated_bytes
            for result in prompt_results
            if result.self_spec_peak_allocated_bytes is not None
        ]
    )

    metadata["mean_normal_peak_allocated_bytes"] = _mean_int_or_none(
        [
            result.normal_peak_allocated_bytes
            for result in prompt_results
            if result.normal_peak_allocated_bytes is not None
        ]
    )

    return {
        "variant_key": variant["key"],
        "variant_label": variant["label"],
        "metadata": metadata,
        "summary": asdict(summary),
        "profile_summary": asdict(profile_summary) if profile_summary else None,
        "profile_results": [asdict(result) for result in profile_results],
        "prompt_results": [asdict(result) for result in prompt_results],
    }


def _run_prompt_phase(
    *,
    phase: PhaseName,
    speculator: BridgeSelfSpeculator,
    bridged: BridgedGapModel,
    prompts: list[tuple[str, str]],
    start_index: int,
    normal_cache: dict[int, Any],
    use_chat_template: bool,
    max_new_tokens: int,
    draft_block_size: int,
    compare_to_normal: bool,
    variant_label: str,
    debug_argmax_ties: bool
) -> PhaseRunResult:
    if not prompts:
        return PhaseRunResult(prompt_results=[], profile_results=[])

    measure_internal_timings = phase == "profile"
    should_record_profile = phase == "profile"
    should_record_speed = phase == "speed"
    should_run_normal = compare_to_normal and phase in {"warmup", "speed"}

    print()
    print(
        f"{phase.capitalize()} phase for {variant_label}: "
        f"{len(prompts)} prompt(s); "
        f"internal timings {'enabled' if measure_internal_timings else 'disabled'}"
    )

    prompt_results: list[PromptBenchResult] = []
    profile_results: list[ProfilePromptResult] = []

    for offset, (prompt_name, prompt) in enumerate(prompts):
        prompt_index = start_index + offset

        print(
            f"[{variant_label} / {phase}] Prompt {offset + 1}/{len(prompts)} "
            f"(original index {prompt_index}): {prompt_name}"
        )

        _reset_cuda_peak_memory_stats()
        self_spec_result = speculator.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            draft_block_size=draft_block_size,
            use_chat_template=use_chat_template,
            build_token_trace=False,
            measure_internal_timings=measure_internal_timings,
            debug_argmax_ties=debug_argmax_ties
        )
        self_spec_peak_allocated_bytes = _cuda_peak_allocated_bytes()
        timings = self_spec_result.timings
        acceptance_rate = self_spec_result.accepted_draft_tokens / max(
            self_spec_result.drafted_tokens,
            1,
        )

        if should_record_profile:
            profile_result = ProfilePromptResult(
                index=prompt_index,
                name=prompt_name,
                profile_seconds=timings.total_seconds,
                generated_tokens=self_spec_result.num_generated_tokens,
                verifier_seconds=timings.verifier_seconds,
                drafter_total_seconds=timings.drafter_total_seconds,
                drafter_body_seconds=timings.drafter_body_seconds,
                drafter_dense_head_seconds=timings.dense_head_seconds,
                drafter_flashhead_seconds=timings.flashhead_seconds,
                drafter_head_seconds=timings.head_seconds,
                drafter_overhead_seconds=timings.drafter_overhead_seconds,
                drafter_registration_seconds=timings.drafter_registration_seconds,
                drafter_teardown_seconds=timings.drafter_teardown_seconds,
                drafter_calls=self_spec_result.drafted_tokens,
                verifier_calls=self_spec_result.verifier_calls
            )
            profile_results.append(profile_result)

            print(
                {
                    "profile_seconds": profile_result.profile_seconds,
                    "drafter_body_fraction": _safe_div(
                        profile_result.drafter_body_seconds,
                        profile_result.drafter_total_seconds,
                    ),
                    "drafter_head_fraction": _safe_div(
                        profile_result.drafter_head_seconds,
                        profile_result.drafter_total_seconds,
                    ),
                    "drafter_overhead_fraction": _safe_div(
                        profile_result.drafter_overhead_seconds,
                        profile_result.drafter_total_seconds,
                    ),
                   "drafter_to_verifier_fraction": _safe_div(
                        _safe_div(profile_result.drafter_total_seconds, draft_block_size),
                        profile_result.verifier_seconds,
                    ),
                }
            )

        normal_seconds: float | None = None
        normal_generated_tokens: int | None = None
        exact_match: bool | None = None
        speedup_seconds: float | None = None
        speedup_per_generated_token: float | None = None
        first_mismatch_index: int | None = None
        normal_peak_allocated_bytes: int | None = None

        if should_run_normal:
            normal_cache_entry = normal_cache.get(prompt_index)

            if normal_cache_entry is None:
                _reset_cuda_peak_memory_stats()
                normal_result = generate_normal(
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    use_chat_template=use_chat_template,
                    use_cache=True,
                    model=bridged.model,
                    tokenizer=bridged.tokenizer,
                    device=bridged.device,
                    #measure_internal_timings=measure_internal_timings,
                    debug_argmax_ties=debug_argmax_ties
                )
                normal_peak_allocated_bytes = _cuda_peak_allocated_bytes()
                normal_cache_entry = {
                    "result": normal_result,
                    "peak_allocated_bytes": normal_peak_allocated_bytes,
                }

                if phase == "speed":
                    normal_cache[prompt_index] = normal_cache_entry
            else:
                normal_result = normal_cache_entry["result"]
                normal_peak_allocated_bytes = normal_cache_entry[
                    "peak_allocated_bytes"
                ]

            if should_record_speed:
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

        if should_record_speed:
            prompt_result = PromptBenchResult(
                index=prompt_index,
                name=prompt_name,
                included_in_metrics=True,
                self_spec_seconds=timings.total_seconds,
                self_spec_generated_tokens=self_spec_result.num_generated_tokens,
                verifier_calls=self_spec_result.verifier_calls,
                drafted_tokens=self_spec_result.drafted_tokens,
                accepted_draft_tokens=self_spec_result.accepted_draft_tokens,
                acceptance_rate=acceptance_rate,
                drafter_total_seconds=timings.drafter_total_seconds,
                drafter_body_seconds=timings.drafter_body_seconds,
                dense_head_seconds=timings.dense_head_seconds,
                flashhead_seconds=timings.flashhead_seconds,
                drafter_overhead_seconds=timings.drafter_overhead_seconds,
                drafter_registration_seconds=timings.drafter_registration_seconds,
                drafter_teardown_seconds=timings.drafter_teardown_seconds,
                normal_seconds=normal_seconds,
                normal_generated_tokens=normal_generated_tokens,
                exact_match=exact_match,
                speedup_seconds=speedup_seconds,
                speedup_per_generated_token=speedup_per_generated_token,
                first_mismatch_index=first_mismatch_index,
                self_spec_peak_allocated_bytes=self_spec_peak_allocated_bytes,
                normal_peak_allocated_bytes=normal_peak_allocated_bytes,
            )
            prompt_results.append(prompt_result)

            print(
                {
                    "acceptance_rate": acceptance_rate,
                    "self_spec_seconds": timings.total_seconds,
                    "speedup_per_generated_token": speedup_per_generated_token,
                    "exact_match": exact_match,
                }
            )

    return PhaseRunResult(
        prompt_results=prompt_results,
        profile_results=profile_results,
    )


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
    warmup_prompts: int,
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
    self_spec_tokens_per_second = _safe_div(
        total_self_spec_tokens,
        total_self_spec_seconds,
    )

    normal_tokens_per_second: float | None = None

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

        normal_tokens_per_second = _safe_div(
            total_normal_tokens,
            total_normal_seconds,
        )

    return BenchSummary(
        prompt_count_total=len(prompt_results),
        prompt_count_included=len(included),
        warmup_prompts=warmup_prompts,
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
        mean_self_spec_generated_tokens_per_prompt=_safe_div(
            total_self_spec_tokens,
            len(included),
        )
        or 0.0,
        mean_normal_generated_tokens_per_prompt=(
            _safe_div(total_normal_tokens, len(compared)) if compared else None
        ),
        normal_tokens_per_second=normal_tokens_per_second,
        self_spec_tokens_per_second=self_spec_tokens_per_second,
    )


def _summarize_profile(
    profile_results: list[ProfilePromptResult],
    *,
    draft_block_size: int,
) -> ProfileSummary | None:
    if not profile_results:
        return None

    profile_total = sum(result.profile_seconds for result in profile_results)
    generated_tokens = sum(result.generated_tokens for result in profile_results)

    verifier_calls_minus_prefill = sum(result.verifier_calls - 1 for result in profile_results) #Removing one because we are not timing the prefill verifier call

    verifier_seconds = sum(result.verifier_seconds for result in profile_results)

    time_per_verifier_call = verifier_seconds/verifier_calls_minus_prefill

    drafter_total = sum(result.drafter_total_seconds for result in profile_results)

    drafter_calls = sum(result.drafter_calls for result in profile_results)

    time_per_drafter_call = drafter_total/drafter_calls


    drafter_body = sum(result.drafter_body_seconds for result in profile_results)
    drafter_dense_head = sum(
        result.drafter_dense_head_seconds for result in profile_results
    )
    drafter_flashhead = sum(
        result.drafter_flashhead_seconds for result in profile_results
    )
    drafter_head = drafter_dense_head + drafter_flashhead
    drafter_overhead = max(0.0, drafter_total - drafter_body - drafter_head)

    registration = sum(
        result.drafter_registration_seconds for result in profile_results
    )
    teardown = sum(result.drafter_teardown_seconds for result in profile_results)

    return ProfileSummary(
        profile_prompt_count=len(profile_results),
        profile_generated_tokens=generated_tokens,
        profile_total_seconds=profile_total,
        verifier_seconds=verifier_seconds,
        drafter_total_seconds=drafter_total,
        drafter_body_seconds=drafter_body,
        drafter_dense_head_seconds=drafter_dense_head,
        drafter_flashhead_seconds=drafter_flashhead,
        drafter_head_seconds=drafter_head,
        drafter_overhead_seconds=drafter_overhead,
        drafter_body_fraction=_safe_div(drafter_body, drafter_total),
        drafter_head_fraction=_safe_div(drafter_head, drafter_total),
        drafter_overhead_fraction=_safe_div(drafter_overhead, drafter_total),
        drafter_to_verifier_fraction=_safe_div(
            time_per_drafter_call,
            time_per_verifier_call,
        ),
        drafter_registration_seconds=registration,
        drafter_teardown_seconds=teardown,
    )


def _mean_int_or_none(values: list[int]) -> float | None:
    if not values:
        return None
    return float(sum(values)) / float(len(values))


def _gpu_display_name() -> str:
    if not torch.cuda.is_available():
        return "CPU / CUDA unavailable"

    device_index = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device_index)
    return props.name


def _num_model_layers(model: Any) -> int | None:
    config = getattr(model, "config", None)
    if config is None:
        return None

    for attr in (
        "num_hidden_layers",
        "n_layer",
        "num_layers",
        "n_layers",
    ):
        value = getattr(config, attr, None)
        if value is not None:
            return int(value)

    return None


def _vocab_size(model: Any, tokenizer: Any) -> int | None:
    config = getattr(model, "config", None)

    value = getattr(config, "vocab_size", None) if config is not None else None
    if value is not None:
        return int(value)

    try:
        return int(len(tokenizer))
    except TypeError:
        return None


def _parameter_count(module: Any) -> int:
    return sum(param.numel() for param in module.parameters())


def _lm_head_parameter_count(model: Any) -> int | None:
    lm_head = getattr(model, "lm_head", None)
    if lm_head is not None:
        return _parameter_count(lm_head)

    output_embeddings = None
    if hasattr(model, "get_output_embeddings"):
        output_embeddings = model.get_output_embeddings()

    if output_embeddings is not None:
        return _parameter_count(output_embeddings)

    return None


def _lm_head_parameter_fraction(model: Any) -> float | None:
    total_params = _parameter_count(model)
    head_params = _lm_head_parameter_count(model)

    if head_params is None or total_params == 0:
        return None

    return head_params / total_params


def _flashhead_acceptance_ratio(
    variants: list[dict[str, Any]],
) -> float | None:
    without_fh: float | None = None
    with_fh: float | None = None

    for variant in variants:
        summary = variant["summary"]
        rate = summary.get("total_acceptance_rate")
        if rate is None:
            continue

        if variant["metadata"].get("flashhead_enabled"):
            with_fh = float(rate)
        else:
            without_fh = float(rate)

    if with_fh is None or without_fh is None or without_fh == 0.0:
        return None

    return with_fh / without_fh


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
    flashhead_top_k_clusters: int,
    bridge_dtype: str,
    requested_max_prompts: int | None,
    available_prompt_count: int,
    run_prompt_count: int,
    warmup_prompts: int,
    profile_prompts: int,
    profile_prompt_count: int,
    speed_prompt_count: int,
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

    lm_head_params = _lm_head_parameter_count(bridged.model)
    lm_total_params = _parameter_count(bridged.model)
    lm_head_fraction = _lm_head_parameter_fraction(bridged.model)

    repo_state = get_git_revision()

    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "git_commit": repo_state.commit if repo_state else None,
        "bridge_checkpoint_path": str(bridge_path.resolve(strict=False)),
        "flashhead_path": str(flash_path.resolve(strict=False)) if flash_path else None,
        "prompt_set": prompt_set,
        "max_prompts": requested_max_prompts,
        "available_prompt_count": available_prompt_count,
        "run_prompt_count": run_prompt_count,
        "warmup_prompts": warmup_prompts,
        "requested_profile_prompts": profile_prompts,
        "profile_prompt_count": profile_prompt_count,
        "speed_prompt_count": speed_prompt_count,
        "use_chat_template": use_chat_template,
        "draft_block_size": draft_block_size,
        "max_new_tokens": max_new_tokens,
        "compare_to_normal": compare_to_normal,
        "speed_measure_internal_timings": False,
        "profile_measure_internal_timings": profile_prompt_count > 0,
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
        "gpu": _gpu_display_name(),
        "gap_tuple": [bridged.gap.start, bridged.gap.length],
        "num_model_layers": _num_model_layers(bridged.model),
        "lm_vocab_size": _vocab_size(bridged.model, bridged.tokenizer),
        "lm_total_parameters": lm_total_params,
        "lm_head_parameters": lm_head_params,
        "lm_head_parameter_fraction": lm_head_fraction,
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
            "variant_label": "With ANNH"
            if payload.get("metadata", {}).get("flashhead_enabled")
            else "Without ANNH",
            "metadata": payload["metadata"],
            "summary": payload["summary"],
            "profile_summary": payload.get("profile_summary"),
            "profile_results": payload.get("profile_results", []),
            "prompt_results": payload["prompt_results"],
        }
    ]


def _draw_git_revision(fig: Any, metadata: dict[str, Any]) -> None:
    commit = metadata.get("git_commit_short") or metadata.get("git_commit")
    tag = metadata.get("git_tag")

    if not commit:
        return

    commit = str(commit)[:8]
    text = f"git commit {commit}"
    if tag:
        text += f" ({tag})"

    fig.text(
        0.985,
        0.018,
        text,
        ha="right",
        va="bottom",
        fontsize=11,
        color="#292929",
    )


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
    from matplotlib.ticker import FixedLocator, FuncFormatter, MultipleLocator

    plot_variants = [
        _styled_variant_payload(variant) for variant in _ordered_plot_variants(variants)
    ]
    all_speedups = [value for variant in plot_variants for value in variant["speedups"]]
    if not all_speedups:
        raise ValueError(
            "No speedup values found. Run the benchmark with normal comparison."
        )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.unicode_minus": False,
        }
    )

    metadata = plot_variants[0]["metadata"]
    fig = plt.figure(figsize=(14.6, 9.0))
    fig.patch.set_facecolor("white")

    fig.text(
        0.5,
        0.945,
        title or "Self-speculation per-token speedup",
        ha="center",
        va="top",
        fontsize=22,
        fontweight="bold",
        color="#050505",
    )
    fig.text(
        0.5,
        0.883,
        _model_display_name(str(metadata.get("model_name", ""))),
        ha="center",
        va="top",
        fontsize=17,
        fontweight="bold",
        color="#4A4A4A",
    )

    ax = fig.add_axes([0.075, 0.39, 0.85, 0.445])
    ax.set_facecolor("none")
    panel = FancyBboxPatch(
        (0.0, 0.0),
        1.0,
        1.0,
        boxstyle="round,pad=0.0,rounding_size=0.009",
        transform=ax.transAxes,
        facecolor="white",
        edgecolor="#DDDDDD",
        linewidth=1.0,
        zorder=-20,
        clip_on=False,
    )
    ax.add_patch(panel)

    x_min, x_max = _plot_x_limits(all_speedups)
    bin_edges = _histogram_bin_edges(all_speedups, bins, lower=x_min, upper=x_max)
    max_count = 0
    for variant in plot_variants:
        counts, _, _ = ax.hist(
            variant["speedups"],
            bins=bin_edges,
            color=variant["fill"],
            edgecolor=variant["edge"],
            linewidth=2.2,
            alpha=0.56,
        )
        max_count = max(max_count, int(max(counts)) if len(counts) else 0)

    normal_curves = _normal_fit_curves(
        variants=plot_variants,
        bin_edges=bin_edges,
        x_min=x_min,
        x_max=x_max,
    )
    max_curve_y = max(
        (max(curve["ys"]) for curve in normal_curves if curve["ys"]),
        default=0.0,
    )
    y_max = max(max(max_count, max_curve_y) * 1.65, 1.0)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0, y_max)
    for curve in normal_curves:
        ax.plot(
            curve["xs"],
            curve["ys"],
            color=curve["color"],
            linewidth=1.8,
            alpha=0.9,
            solid_capstyle="round",
        )

    for variant in plot_variants:
        speedup = variant["aggregate_speedup"]
        if speedup is None:
            continue
        ax.vlines(
            speedup,
            0,
            y_max * 0.94,
            color=variant["edge"],
            linewidth=1.15,
            linestyles=(0, (4, 5)),
            alpha=0.72,
        )

    ax.set_axisbelow(True)
    ax.xaxis.set_major_locator(FixedLocator(_major_ticks(x_min, x_max)))
    ax.xaxis.set_minor_locator(MultipleLocator(0.05))
    ax.xaxis.set_major_formatter(FuncFormatter(_speedup_tick_label))
    ax.grid(axis="x", which="major", color="#E8E8E8", linewidth=1.05)
    ax.grid(axis="x", which="minor", color="#F0F0F0", linewidth=0.75)
    ax.tick_params(axis="x", length=0, labelsize=15, pad=12, colors="#050505")
    ax.tick_params(axis="y", length=0)
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    _draw_inline_legend(ax, plot_variants)
    footer_ax = fig.add_axes([0.085, 0.095, 0.83, 0.225])
    _draw_report_footer(footer_ax, _info_sections(plot_variants))
    _draw_git_revision(fig, metadata)

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
    profile_summary = variant.get("profile_summary")
    prompt_results = variant["prompt_results"]
    profile_results = variant.get("profile_results", [])
    if key == "flashhead" or metadata.get("flashhead_enabled"):
        colors = {
            "edge": "#9B007F",
            "fill": "#DCA6D2",
        }
        label = "With skipped layers + ANN head:"
        short_label = "w ANNH"
    else:
        colors = {
            "edge": "#006E90",
            "fill": "#86BDCB",
        }
        label = "With skipped layers:"
        short_label = "w/o ANNH"

    speedups = [
        float(result["speedup_per_generated_token"])
        for result in prompt_results
        if result.get("included_in_metrics")
        and result.get("speedup_per_generated_token") is not None
    ]
    return {
        "key": key,
        "label": label,
        "short_label": short_label,
        "metadata": metadata,
        "summary": summary,
        "profile_summary": profile_summary,
         "profile_results": profile_results,
        "speedups": speedups,
        "aggregate_speedup": _summary_speedup(summary),
        **colors,
    }


def _summary_speedup(summary: dict[str, Any]) -> float | None:
    for key in (
        #"mean_prompt_speedup_per_generated_token",
        "total_speedup_per_generated_token",
        #"total_speedup_seconds",
    ):
        value = summary.get(key)
        if value is not None:
            return float(value)
    return None


def _plot_x_limits(values: list[float]) -> tuple[float, float]:
    lower = math.floor((min(values) - 0.16) / 0.1) * 0.1
    upper = math.ceil((max(values) + 0.12) / 0.1) * 0.1
    lower = min(lower, 0.5)
    upper = max(upper, 1.4)
    if upper - lower < 0.7:
        midpoint = (upper + lower) / 2.0
        lower = math.floor((midpoint - 0.35) / 0.1) * 0.1
        upper = math.ceil((midpoint + 0.35) / 0.1) * 0.1
    return lower, upper


def _histogram_bin_edges(
    _values: list[float],
    bins: int | None,
    *,
    lower: float,
    upper: float,
) -> list[float]:
    if bins is not None:
        bin_count = max(1, bins)
    else:
        span = upper - lower
        bin_count = max(8, min(16, round(span / 0.06)))
    width = (upper - lower) / bin_count
    return [lower + width * index for index in range(bin_count + 1)]


def _major_ticks(lower: float, upper: float) -> list[float]:
    start = math.ceil((lower + 0.001) * 10.0) / 10.0
    stop = math.floor((upper - 0.001) * 10.0) / 10.0
    ticks: list[float] = []
    current = start
    while current <= stop + 0.0001:
        ticks.append(round(current, 1))
        current += 0.1
    return ticks


def _speedup_tick_label(value: float, _: Any) -> str:
    if abs(value - round(value)) < 0.001:
        return f"{int(round(value))}x"
    return f"{value:.1f}x"


def _normal_fit_curves(
    *,
    variants: list[dict[str, Any]],
    bin_edges: list[float],
    x_min: float,
    x_max: float,
) -> list[dict[str, Any]]:
    if len(bin_edges) < 2:
        return []

    bin_width = bin_edges[1] - bin_edges[0]
    xs = [x_min + (x_max - x_min) * index / 399 for index in range(400)]
    curves: list[dict[str, Any]] = []
    for variant in variants:
        speedups = variant["speedups"]
        std = _sample_std_or_none(speedups)
        if std is None or std <= 0.0:
            continue

        mean = _mean(speedups)
        normalizer = 1.0 / (std * math.sqrt(2.0 * math.pi))
        ys = [
            len(speedups)
            * bin_width
            * normalizer
            * math.exp(-0.5 * ((x - mean) / std) ** 2)
            for x in xs
        ]
        curves.append(
            {
                "xs": xs,
                "ys": ys,
                "color": variant["edge"],
            }
        )
    return curves


def _draw_inline_legend(ax: Any, variants: list[dict[str, Any]]) -> None:
    for index, variant in enumerate(variants):
        speedup = variant["aggregate_speedup"]
        suffix = "" if speedup is None else f" {_fmt_x(speedup)}"
        ax.text(
            0.016,
            0.955 - index * 0.085,
            f"{variant['label']}{suffix}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=15.5,
            fontweight="semibold",
            color=variant["edge"],
        )


def _fmt_int(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{int(value):,}"


def _fmt_pct_2(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * value:.2f}%"


def _fmt_bytes(value: Any) -> str:
    if value is None:
        return "n/a"

    value = float(value)
    gib = value / (1024**3)
    if gib >= 1.0:
        return f"{gib:.2f} GiB"

    mib = value / (1024**2)
    if mib >= 1.0:
        return f"{mib:.1f} MiB"

    kib = value / 1024
    if kib >= 1.0:
        return f"{kib:.1f} KiB"

    return f"{int(value)} B"


def _gap_label(metadata: dict[str, Any]) -> str:
    gap_start = metadata.get("gap_start")
    gap_end = metadata.get("gap_end")
    num_layers = metadata.get("num_model_layers")

    if gap_start is None or gap_end is None or num_layers is None:
        return "n/a"

    kept_start = int(gap_start)
    kept_end = int(num_layers) - int(gap_end)

    return f"({kept_start}, {kept_end})"


def _fmt_compact_number(value: Any) -> str:
    if value is None:
        return "n/a"

    value = float(value)
    abs_value = abs(value)

    if abs_value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if abs_value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if abs_value >= 1_000:
        return f"{value / 1_000:.1f}K"

    return str(int(value))

def _flashhead_head_speedup(
    variants: list[dict[str, Any]],
) -> float | None:
    dense_head_seconds: float | None = None
    flashhead_seconds: float | None = None

    for variant in variants:
        profile_summary = variant.get("profile_summary") or {}
        if not profile_summary:
            continue

        if variant["metadata"].get("flashhead_enabled"):
            flashhead_seconds = profile_summary.get("drafter_flashhead_seconds")
        else:
            dense_head_seconds = profile_summary.get("drafter_dense_head_seconds")

    return _safe_div(dense_head_seconds, flashhead_seconds)

def _info_sections(
    variants: list[dict[str, Any]],
) -> list[tuple[str, list[tuple[str, str]]]]:
    first = variants[0]
    metadata = first["metadata"]
    first_profile = first.get("profile_summary") or {}
    setup = [
        ("Prompt set", _prompt_set_display_name(metadata.get("prompt_set"))),
        ("Gap", _gap_label(metadata)),
        ("Block size", _value(metadata.get("draft_block_size"))),
        ("Layers", _value(metadata.get("num_model_layers"))),
        ("LM vocab", _fmt_int(metadata.get("lm_vocab_size"))),
        ("LM params", _fmt_compact_number(metadata.get("lm_total_parameters"))),
        ("Head params", _fmt_compact_number(metadata.get("lm_head_parameters"))),
        ("Head portion", _fmt_pct_2(metadata.get("lm_head_parameter_fraction"))),
    ]
    runtime = [
        ("GPU", _value(metadata.get("gpu"))),
        ("Backend", _value(metadata.get("attention_backend"))),
        ("Model dtype", _dtype_label(metadata.get("model_dtype"))),
        ("Bridge dtype", _dtype_label(metadata.get("loaded_bridge_dtype"))),
        ("Speed internals", "no"),
        (
            "Profile prompts",
            _value(
                first_profile.get("profile_prompt_count")
                if first_profile
                else metadata.get("profile_prompt_count")
            ),
        ),
        ("Warmup prompts", _value(first["summary"].get("warmup_prompts"))),
        ("Measured prompts", _value(first["summary"].get("prompt_count_included"))),
    ]

    profile_rows: list[tuple[str, str]] = []
    flashhead_meta = next(
        (
            variant["metadata"]
            for variant in variants
            if variant["metadata"].get("flashhead_enabled")
        ),
        None,
    )
    results: list[tuple[str, str]] = []
    normal_peak_memory = first["metadata"].get("mean_normal_peak_allocated_bytes")

    if normal_peak_memory is not None:
        results.append(
            (
                "Peak mem normal",
                _fmt_bytes(normal_peak_memory),
            )
        )

    for variant in variants:
        summary = variant["summary"]
        variant_metadata = variant["metadata"]
        prefix = variant["short_label"]

        results.extend(
            [
                (
                    f"Mean speedup ({prefix})",
                    _fmt_x(_summary_speedup(summary)),
                ),
                (
                    f"Acceptance rate ({prefix})",
                    _fmt_pct(summary.get("total_acceptance_rate")),
                ),
                (
                    f"Exact match ({prefix})",
                    _fmt_pct(summary.get("exact_match_rate")),
                ),
                (
                    f"Peak mem self ({prefix})",
                    _fmt_bytes(
                        variant_metadata.get("mean_self_spec_peak_allocated_bytes")
                    ),
                ),
            ]
        )


        profile_summary = variant.get("profile_summary") or {}
        if profile_summary:
            profile_rows.append(
                (
                    f"Drafter split ({prefix})",
                    _fmt_drafter_split(
                        profile_summary.get("drafter_body_fraction"),
                        profile_summary.get("drafter_head_fraction"),
                        profile_summary.get("drafter_overhead_fraction"),
                    ),
                )
            )
            normal_time_per_token = _safe_div(
                summary.get("total_normal_seconds"),
                summary.get("total_normal_generated_tokens"),
            )

            verifier_calls_minus_prefill = sum(
                result["verifier_calls"] - 1
                for result in variant.get("profile_results", [])
            )

            verifier_time_per_call = _safe_div(
                profile_summary.get("verifier_seconds"),
                verifier_calls_minus_prefill,
            )

            drafter_calls = sum(
                result["drafter_calls"]
                for result in variant.get("profile_results", [])
            )

            drafter_time_per_call = _safe_div(
                profile_summary.get("drafter_total_seconds"),
                drafter_calls,
            )

            verifier_to_normal_ratio = _safe_div(
                verifier_time_per_call,
                normal_time_per_token,
            )

            drafter_to_normal_ratio = _safe_div(
                drafter_time_per_call,
                normal_time_per_token,
            )

            profile_rows.append(
                (
                    f"Verifier/normal ({prefix})",
                    _fmt_x(verifier_to_normal_ratio),
                )
            )

            profile_rows.append(
                (
                    f"Drafter/normal ({prefix})",
                    _fmt_pct(drafter_to_normal_ratio),
                )
            )
            # profile_rows.append(
            #     (
            #         f"Drafter sec ({prefix})",
            #         _fmt_head_body_overhead(
            #             profile_summary.get("drafter_body_seconds"),
            #             profile_summary.get("drafter_head_seconds"),
            #             profile_summary.get("drafter_overhead_seconds"),
            #         ),
            #     )
            # )

    results.extend(
            [
                (
                    f"Tokens gen",
                    (
                        f"s {_fmt_int(summary.get('total_self_spec_generated_tokens'))} / "
                        f"n {_fmt_int(summary.get('total_normal_generated_tokens'))}"
                    ),
                ),
            ]
        )
    
    
    
    fh_accuracy = _flashhead_acceptance_ratio(variants)
    if fh_accuracy is not None:
        profile_rows.append(("ANNH acceptance ratio", _fmt_pct(fh_accuracy)))

    fh_head_speedup = _flashhead_head_speedup(variants)
    if fh_head_speedup is not None:
        profile_rows.append(("ANNH head speedup", _fmt_x(fh_head_speedup)))

    if flashhead_meta:
        profile_rows.extend(
            [
                (
                    "ANNH index",
                    (
                        f"{_value(flashhead_meta.get('flashhead_loaded_num_clusters'))} "
                        f"clusters; top-k "
                        f"{_value(flashhead_meta.get('flashhead_top_k_clusters'))}"
                    ),
                ),
            ]
        )

    sections = [
        ("Setup", setup),
        ("Runtime", runtime),
        ("Results", results),
    ]

    if profile_rows:
        sections.append(("Profile", profile_rows))

    return sections


def _draw_report_footer(
    ax: Any,
    sections: list[tuple[str, list[tuple[str, str]]]],
) -> None:
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    column_xs = [0.0, 0.265, 0.495, 0.755]
    column_rights = [0.23, 0.455, 0.745, 1.1]
    title_color = "#050505"
    label_color = "#526D73"
    value_color = "#243E45"
    for section_index, (heading, rows) in enumerate(sections):
        x = column_xs[min(section_index, len(column_xs) - 1)]
        right = column_rights[min(section_index, len(column_rights) - 1)]
        ax.text(
            x,
            0.98,
            heading,
            ha="left",
            va="top",
            fontsize=17.5,
            fontweight="bold",
            color=title_color,
        )
        y = 0.76
        row_step = 0.115
        for label, value in rows:
            _draw_labeled_value(
                ax,
                x,
                y,
                label,
                value,
                label_color=label_color,
                value_color=value_color,
                right=right,
            )
            y -= row_step


def _draw_labeled_value(
    ax: Any,
    x: float,
    y: float,
    label: str,
    value: str,
    *,
    label_color: str,
    value_color: str,
    right: float,
) -> None:
    prefix = f"{label} = "
    fontsize = _footer_font_size(ax, x, right, prefix + value)
    prefix_artist = ax.text(
        x,
        y,
        prefix,
        ha="left",
        va="top",
        fontsize=fontsize,
        color=label_color,
    )
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    prefix_box = prefix_artist.get_window_extent(renderer=renderer)
    axes_box = ax.get_window_extent(renderer=renderer)
    value_x = x + prefix_box.width / axes_box.width
    ax.text(
        value_x,
        y,
        value,
        ha="left",
        va="top",
        fontsize=fontsize,
        fontweight="bold",
        color=value_color,
    )


def _footer_font_size(ax: Any, x: float, right: float, text: str) -> float:
    font_size = 10.7
    minimum = 7.8
    while font_size > minimum:
        probe = ax.text(
            x,
            0.5,
            text,
            fontsize=font_size,
            fontweight="bold",
            alpha=0.0,
        )
        ax.figure.canvas.draw()
        renderer = ax.figure.canvas.get_renderer()
        text_width = probe.get_window_extent(renderer=renderer).width
        axes_width = ax.get_window_extent(renderer=renderer).width
        probe.remove()
        if x + text_width / axes_width <= right:
            return font_size
        font_size -= 0.4
    return minimum


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


def _prompt_set_display_name(value: Any) -> str:
    labels = {
        "chat-style": "chat",
        "completion-style": "completion",
        "hard-completion-style": "hard completion",
        "concrete-completion-style": "concrete",
        "swedish-concrete-completion-style": "swedish concrete",
    }
    text = _value(value)
    return labels.get(text, text[:18] + "..." if len(text) > 21 else text)


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


def _safe_div(numerator: float | int | None, denominator: float | int | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0:
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


def _cuda_peak_allocated_bytes() -> int | None:
    if not torch.cuda.is_available():
        return None
    return int(torch.cuda.max_memory_allocated())


def _reset_cuda_peak_memory_stats() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _default_run_stem(
    *,
    bridge_path: Path,
    prompt_set: str,
    draft_block_size: int,
    max_new_tokens: int,
    warmup_prompts: int,
    profile_prompts: int,
    variants: VariantMode,
    flash_path: Path | None,
    bridged: BridgedGapModel,
) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    model_name = _slugify_filename_part(
        _model_display_name(str(bridged.config.model_name))
    )

    prompt_name = _slugify_filename_part(prompt_set)

    num_layers = _num_model_layers(bridged.model)
    gap_label = _gap_file_label(
        gap_start=bridged.gap.start,
        gap_end=bridged.gap.end,
        num_layers=num_layers,
    )

    if variants == "auto":
        variant_label = "both" if flash_path is not None else "no-fh"
    elif variants == "flashhead":
        variant_label = "fh"
    elif variants == "no-flashhead":
        variant_label = "no-fh"
    else:
        variant_label = "both"

    return (
        f"bench_self_spec"
        f"__{model_name}"
        f"__{prompt_name}"
        f"__keep-{gap_label}"
        f"__block-{draft_block_size}"
        f"__max-{max_new_tokens}"
        f"__warmup-{warmup_prompts}"
        f"__profile-{profile_prompts}"
        f"__{variant_label}"
        f"__{timestamp}"
    )


def _gap_file_label(
    *,
    gap_start: int,
    gap_end: int,
    num_layers: int | None,
) -> str:
    if num_layers is None:
        return f"gap-{gap_start}-{gap_end}"

    kept_start = max(0, int(gap_start))
    kept_end = max(0, int(num_layers) - int(gap_end))
    return f"{kept_start}-{kept_end}"


def _slugify_filename_part(value: str) -> str:
    cleaned = value.strip().lower()
    chars: list[str] = []

    previous_was_sep = False
    for char in cleaned:
        if char.isalnum():
            chars.append(char)
            previous_was_sep = False
        elif not previous_was_sep:
            chars.append("-")
            previous_was_sep = True

    slug = "".join(chars).strip("-")
    return slug or "unknown"


def _fmt_seconds_compact(value: float | None) -> str:
    if value is None:
        return "n/a"
    if abs(value) >= 1.0:
        return f"{value:.1f}s"
    return f"{value:.3f}s"


def _fmt_head_body(head_seconds: Any, body_seconds: Any) -> str:
    head = float(head_seconds) if head_seconds is not None else None
    body = float(body_seconds) if body_seconds is not None else None
    return f"{_fmt_seconds_compact(head)}/{_fmt_seconds_compact(body)}"


def _fmt_head_body_overhead(
    body_seconds: Any,
    head_seconds: Any,
    overhead_seconds: Any,
) -> str:
    body = float(body_seconds) if body_seconds is not None else None
    head = float(head_seconds) if head_seconds is not None else None
    overhead = float(overhead_seconds) if overhead_seconds is not None else None
    return (
        f"B {_fmt_seconds_compact(body)} / "
        f"H {_fmt_seconds_compact(head)} / "
        f"O {_fmt_seconds_compact(overhead)}"
    )


def _fmt_drafter_split(
    body_fraction: Any,
    head_fraction: Any,
    overhead_fraction: Any,
) -> str:
    body = float(body_fraction) if body_fraction is not None else None
    head = float(head_fraction) if head_fraction is not None else None
    overhead = float(overhead_fraction) if overhead_fraction is not None else None
    return f"B {_fmt_pct(body)} / H {_fmt_pct(head)} / O {_fmt_pct(overhead)}"


def _fmt_x(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}x"


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * value:.1f}%"


def _fmt_count(value: int | None, total: int) -> str:
    if value is None:
        return "n/a"
    return f"{value}/{total}"


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"
