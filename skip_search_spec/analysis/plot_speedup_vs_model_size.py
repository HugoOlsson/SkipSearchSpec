from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any


DEFAULT_BENCHMARK_JSON_PATHS = [
    "benchmarks/self_spec/L4_V2/bench_self_spec__llama-3-2-1b-instruct__concrete-completion-style__keep-1-1__block-2__max-200__warmup-5__profile-15__both__20260517_120248.json",
    "benchmarks/self_spec/L4_V2/bench_self_spec__llama-3-2-3b-instruct__concrete-completion-style__keep-1-1__block-2__max-200__warmup-5__profile-15__both__20260517_123233.json",
    "benchmarks/self_spec/L4_V2/bench_self_spec__qwen3-4b-instruct-2507__concrete-completion-style__keep-1-1__block-2__max-200__warmup-5__profile-15__both__20260517_131002.json",
    "benchmarks/self_spec/L4_V2/bench_self_spec__mistral-7b-instruct-v0-3__concrete-completion-style__keep-1-1__block-2__max-200__warmup-5__profile-15__both__20260517_131146.json",
    "benchmarks/self_spec/L4_V2/bench_self_spec__llama-3-1-8b-instruct__concrete-completion-style__keep-1-1__block-2__max-200__warmup-5__profile-15__both__20260517_133323.json",
]

DEFAULT_OUTPUT_PATH = (
    "thesis_report/my-figures/plots/benches/speedup_vs_model_size_annh.png"
)

MODEL_LABELS = {
    "meta-llama/Llama-3.1-8B-Instruct": "Llama 3.1 8B",
    "meta-llama/Llama-3.2-1B-Instruct": "Llama 3.2 1B",
    "meta-llama/Llama-3.2-3B-Instruct": "Llama 3.2 3B",
    "mistralai/Mistral-7B-Instruct-v0.3": "Mistral 7B",
    "Qwen/Qwen3-4B-Instruct-2507": "Qwen3 4B",
}

LABEL_OFFSETS = {
    "Llama 3.2 1B": (8, -2),
    "Llama 3.2 3B": (8, 4),
    "Qwen3 4B": (8, -12),
    "Mistral 7B": (-8, 13),
    "Llama 3.1 8B": (8, -14),
}

# Matches the self-spec benchmark result plot theme in bench_self_spec.py.
ANNH_EDGE = "#9B007F"
ANNH_FILL = "#DCA6D2"
TEXT_DARK = "#243E45"
TEXT_MUTED = "#526D73"
AXIS_DARK = "#050505"
GRID_MAJOR = "#E8E8E8"
GRID_MINOR = "#F0F0F0"


@dataclass(frozen=True)
class SpeedupPoint:
    label: str
    model_name: str
    parameters_b: float
    speedup: float
    source_path: Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot ANNH self-speculation speedup against model size from benchmark "
            "JSON files."
        )
    )
    parser.add_argument(
        "benchmark_jsons",
        nargs="*",
        help=(
            "Benchmark JSON files to plot. If omitted, the thesis L4 benchmark "
            "JSONs are used."
        ),
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output image path. Default: {DEFAULT_OUTPUT_PATH}",
    )
    parser.add_argument(
        "--no-fit",
        action="store_true",
        help="Do not draw the least-squares fit line.",
    )
    args = parser.parse_args()

    repo_root = Path.cwd()
    input_paths = args.benchmark_jsons or DEFAULT_BENCHMARK_JSON_PATHS
    points = [_load_point(_resolve_path(path, repo_root)) for path in input_paths]
    output_path = _resolve_path(args.output, repo_root)

    _plot(points, output_path=output_path, draw_fit=not args.no_fit)

    print(f"Wrote {output_path}")
    for point in sorted(points, key=lambda item: item.parameters_b):
        print(
            f"{point.label}: {point.parameters_b:.3f}B params, "
            f"{point.speedup:.3f}x speedup"
        )


def _resolve_path(path: str | Path, repo_root: Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return repo_root / path


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return payload


def _load_point(path: Path) -> SpeedupPoint:
    payload = _load_json(path)
    variant = _flashhead_variant(payload, path)
    metadata = variant["metadata"]
    summary = variant["summary"]

    model_name = str(metadata["model_name"])
    parameters_b = float(metadata["lm_total_parameters"]) / 1e9
    speedup = float(summary["total_speedup_per_generated_token"])
    label = MODEL_LABELS.get(model_name, model_name.split("/")[-1].replace("-Instruct", ""))

    return SpeedupPoint(
        label=label,
        model_name=model_name,
        parameters_b=parameters_b,
        speedup=speedup,
        source_path=path,
    )


def _flashhead_variant(payload: dict[str, Any], path: Path) -> dict[str, Any]:
    for variant in payload.get("variant_results", []):
        metadata = variant.get("metadata", {})
        if (
            variant.get("variant_key") == "flashhead"
            or metadata.get("flashhead_enabled")
        ):
            return variant
    raise ValueError(f"No flashhead/ANNH variant found in {path}")


def _pearson_r(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x == 0.0 or var_y == 0.0:
        return None
    return cov / math.sqrt(var_x * var_y)


def _linear_fit(xs: list[float], ys: list[float]) -> tuple[float, float] | None:
    if len(xs) < 2:
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    var_x = sum((x - mean_x) ** 2 for x in xs)
    if var_x == 0.0:
        return None
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / var_x
    intercept = mean_y - slope * mean_x
    return intercept, slope


def _plot(
    points: list[SpeedupPoint],
    *,
    output_path: Path,
    draw_fit: bool,
) -> None:
    if not points:
        raise ValueError("No points to plot.")

    cache_root = Path(tempfile.gettempdir()) / "skip_search_spec_matplotlib"
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "config"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "cache"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator

    sorted_points = sorted(points, key=lambda point: point.parameters_b)
    xs = [point.parameters_b for point in sorted_points]
    ys = [point.speedup for point in sorted_points]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.unicode_minus": False,
        }
    )

    fig, ax = plt.subplots(figsize=(8.6, 5.0), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.scatter(
        xs,
        ys,
        s=78,
        color=ANNH_FILL,
        edgecolors=ANNH_EDGE,
        linewidths=1.7,
        zorder=4,
    )

    r = _pearson_r(xs, ys)
    fit = _linear_fit(xs, ys)
    if draw_fit and fit is not None:
        intercept, slope = fit
        x_fit = [0.0, max(xs) + 0.7]
        y_fit = [intercept + slope * x for x in x_fit]
        ax.plot(
            x_fit,
            y_fit,
            color=ANNH_EDGE,
            linewidth=2.0,
            linestyle=(0, (5, 4)),
            alpha=0.92,
            zorder=2,
        )
        if r is not None:
            ax.text(
                0.98,
                0.96,
                f"linear fit, r = {r:.2f}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=10,
                color=TEXT_MUTED,
            )

    for point in sorted_points:
        dx, dy = LABEL_OFFSETS.get(point.label, (7, 7))
        ax.annotate(
            point.label,
            xy=(point.parameters_b, point.speedup),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=9.5,
            color=TEXT_DARK,
        )

    x_max = max(xs) + 0.9
    y_min = max(1.15, min(ys) - 0.08)
    y_max = max(ys) + 0.08
    ax.set_xlim(0.0, x_max)
    ax.set_ylim(y_min, y_max)

    ax.xaxis.set_major_locator(MultipleLocator(2.0))
    ax.xaxis.set_minor_locator(MultipleLocator(1.0))
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))

    ax.grid(axis="both", which="major", color=GRID_MAJOR, linewidth=1.0)
    ax.grid(axis="both", which="minor", color=GRID_MINOR, linewidth=0.7)
    ax.set_axisbelow(True)

    ax.set_xlabel("Model size (billion parameters)", fontsize=11, color=TEXT_DARK)
    ax.set_ylabel("Speedup with ANNH", fontsize=11, color=TEXT_DARK)
    ax.tick_params(axis="both", colors=AXIS_DARK, labelsize=9.5, length=0)

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS_DARK)
        ax.spines[side].set_linewidth(1.1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=1.1)
    fig.savefig(output_path)
    plt.close(fig)


if __name__ == "__main__":
    main()
