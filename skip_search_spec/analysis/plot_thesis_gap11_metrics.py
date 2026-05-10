from __future__ import annotations

from argparse import ArgumentParser
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Literal

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


MetricAlias = Literal[
    "top1", "kl", "top1_drafter_matches_verifier", "kl_verifier_to_drafter"
]


THESIS_GAP_1_1_JSON_PATHS = [
    "measurements/2026-05-09-50b567/middle_gap_skip/for_thesis_17580058_MY09__meta-llama_Llama-3_1-8B-Instruct_1_30_1/run.json",
    "measurements/2026-05-09-9551a4/middle_gap_skip/for_thesis_18451838_MY09__mistralai_Mistral-7B-Instruct-v0_3_1_30_1/run.json",
    "measurements/2026-05-09-96fe48/middle_gap_skip/for_thesis_13582146_MY09__meta-llama_Llama-3_2-1B-Instruct_1_14_1/run.json",
    "measurements/2026-05-09-99c392/middle_gap_skip/for_thesis_15465604_MY09__Qwen_Qwen3-1_7B_1_26_1/run.json",
    "measurements/2026-05-09-9af545/middle_gap_skip/for_thesis_17360671_MY09__Qwen_Qwen2_5-0_5B-Instruct_1_22_1/run.json",
    "measurements/2026-05-09-87166f/middle_gap_skip/for_thesis_16125454_MY09__Qwen_Qwen2_5-14B-Instruct_1_46_1/run.json",
    "measurements/2026-05-09-f9a0d3/middle_gap_skip/for_thesis_15154000_MY09__meta-llama_Llama-3_2-3B-Instruct_1_26_1/run.json",
]

THESIS_GAP_2_2_JSON_PATHS = [
    "measurements/2026-05-10-f7dea8/middle_gap_skip/for_thesis_day2_10500516_MY10__meta-llama_Llama-3_2-1B-Instruct_2_12_2/run.json",
    "measurements/2026-05-10-076f2b/middle_gap_skip/for_thesis_day2_11122915_MY10__meta-llama_Llama-3_2-3B-Instruct_2_24_2/run.json",
    "measurements/2026-05-10-215d9b/middle_gap_skip/for_thesis_day2_11401601_MY10__meta-llama_Llama-3_1-8B-Instruct_2_28_2/run.json",
    "measurements/2026-05-10-381530/middle_gap_skip/for_thesis_day2_12422559_MY10__Qwen_Qwen2_5-0_5B-Instruct_2_20_2/run.json",
    "measurements/2026-05-10-381530/middle_gap_skip/for_thesis_day2_12214343_MY10__Qwen_Qwen3-1_7B_2_24_2/run.json",
    "measurements/2026-05-10-7ad273/middle_gap_skip/for_thesis_day2_10081712_MY10__mistralai_Mistral-7B-Instruct-v0_3_2_28_2/run.json",
]

THESIS_GAP_2_0_JSON_PATHS = [
    "measurements/2026-05-10-123fc3/middle_gap_skip/for_thesis_day2_14180470_MY10__meta-llama_Llama-3_2-1B-Instruct_2_14_0/run.json",
    "measurements/2026-05-10-123fc3/middle_gap_skip/for_thesis_day2_13554184_MY10__meta-llama_Llama-3_2-3B-Instruct_2_26_0/run.json",
    "measurements/2026-05-10-123fc3/middle_gap_skip/for_thesis_day2_15181164_MY10__Qwen_Qwen2_5-0_5B-Instruct_2_22_0/run.json",
    "measurements/2026-05-10-123fc3/middle_gap_skip/for_thesis_day2_14591395_MY10__Qwen_Qwen3-1_7B_2_26_0/run.json",
    "measurements/2026-05-10-123fc3/middle_gap_skip/for_thesis_day2_14312370_MY10__mistralai_Mistral-7B-Instruct-v0_3_2_30_0/run.json",
]

THESIS_GAP_JSON_PATHS: dict[tuple[int, int], list[str]] = {
    (1, 1): THESIS_GAP_1_1_JSON_PATHS,
    (2, 2): THESIS_GAP_2_2_JSON_PATHS,
    (2, 0): THESIS_GAP_2_0_JSON_PATHS,
}

METRIC_ALIASES: dict[str, str] = {
    "top1": "top1_drafter_matches_verifier",
    "kl": "kl_verifier_to_drafter",
    "top1_drafter_matches_verifier": "top1_drafter_matches_verifier",
    "kl_verifier_to_drafter": "kl_verifier_to_drafter",
}

METRIC_Y_LABELS = {
    "top1_drafter_matches_verifier": "Top-1 agreement",
    "kl_verifier_to_drafter": "KL(verifier || drafter)",
}

METRIC_TITLE_PREFIXES = {
    "top1_drafter_matches_verifier": "Top-1 Agreement",
    "kl_verifier_to_drafter": "Verifier-to-Drafter KL",
}

MODEL_LABELS = {
    "meta-llama/Llama-3.1-8B-Instruct": "Llama 3.1 8B Instruct",
    "meta-llama/Llama-3.2-1B-Instruct": "Llama 3.2 1B Instruct",
    "meta-llama/Llama-3.2-3B-Instruct": "Llama 3.2 3B Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3": "Mistral 7B v0.3 Instruct",
    "Qwen/Qwen2.5-0.5B-Instruct": "Qwen2.5 0.5B Instruct",
    "Qwen/Qwen2.5-14B-Instruct": "Qwen2.5 14B Instruct",
    "Qwen/Qwen3-1.7B": "Qwen3 1.7B Instruct",
}

THESIS_COLORS = [
    "#2F6FB0",
    "#E17C05",
    "#1B9E77",
    "#C04E8B",
    "#4E9A9A",
    "#C9A227",
    "#6A51A3",
]

MODEL_COLORS = {
    "meta-llama/Llama-3.1-8B-Instruct": "#2F6FB0",
    "mistralai/Mistral-7B-Instruct-v0.3": "#E17C05",
    "meta-llama/Llama-3.2-1B-Instruct": "#1B9E77",
    "Qwen/Qwen3-1.7B": "#C04E8B",
    "Qwen/Qwen2.5-0.5B-Instruct": "#4E9A9A",
    "Qwen/Qwen2.5-14B-Instruct": "#C9A227",
    "meta-llama/Llama-3.2-3B-Instruct": "#6A51A3",
}

MODEL_ORDER = list(MODEL_COLORS)

DEFAULT_SMOOTH_WINDOW = 5
RAW_PREFIX_POINTS = 10
FINAL_AVG_POINTS = 20
LINE_WIDTH = 1.35


class MetricSeries:
    def __init__(
        self,
        label: str,
        steps: list[int],
        values: list[float],
        git_commit: str,
        color: str,
        model_name: str,
    ) -> None:
        self.label = label
        self.steps = steps
        self.values = values
        self.git_commit = git_commit
        self.color = color
        self.model_name = model_name


def _resolve_metric(metric: str) -> str:
    try:
        return METRIC_ALIASES[metric]
    except KeyError as exc:
        valid = ", ".join(sorted(METRIC_ALIASES))
        raise ValueError(f"Unknown metric {metric!r}. Choose one of: {valid}") from exc


def _safe_path_part(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.=\-+]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, dict):
        raise TypeError(f"Expected top-level JSON object in {path}")

    return payload


def _model_name(payload: dict[str, Any], path: Path) -> str:
    context = payload.get("context", {})
    model_names = context.get("model_names") or []
    run_config = context.get("run_config", {})
    model_name = (
        model_names[0]
        if model_names
        else run_config.get("model_name", path.parent.name)
    )

    if not isinstance(model_name, str):
        model_name = str(model_name)

    return model_name


def _model_label(payload: dict[str, Any], path: Path) -> str:
    model_name = _model_name(payload, path)
    return MODEL_LABELS.get(model_name, model_name.split("/")[-1].replace("_", "."))


def _model_color(payload: dict[str, Any], path: Path) -> str:
    model_name = _model_name(payload, path)
    if model_name in MODEL_COLORS:
        return MODEL_COLORS[model_name]

    digest = hashlib.sha1(model_name.encode("utf-8")).hexdigest()
    color_index = int(digest[:8], 16) % len(THESIS_COLORS)
    return THESIS_COLORS[color_index]


def _model_sort_key(series: MetricSeries) -> tuple[int, str]:
    try:
        return MODEL_ORDER.index(series.model_name), series.model_name
    except ValueError:
        return len(MODEL_ORDER), series.model_name


def _git_commit_label(payload: dict[str, Any]) -> str:
    git_commit = payload.get("context", {}).get("git_commit")
    if not git_commit:
        return "no_git"
    return str(git_commit)[:8]


def _gap_metadata(payload: dict[str, Any], path: Path) -> tuple[int, int, int]:
    context = payload.get("context", {})
    run_config = context.get("run_config", {})
    try:
        gap_start = int(run_config["gap_start"])
        gap_end = int(run_config["gap_end"])
        num_layers = int(run_config["num_layers"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Could not verify gap metadata for {path}") from exc

    return gap_start, gap_end, num_layers


def _gap_shape_text(gap_shape: tuple[int, int]) -> str:
    left_kept, right_kept = gap_shape
    return f"({left_kept}, {right_kept})"


def _gap_shape_slug(gap_shape: tuple[int, int]) -> str:
    left_kept, right_kept = gap_shape
    return f"gap{left_kept}{right_kept}"


def _verify_gap_shape(
    payload: dict[str, Any], path: Path, gap_shape: tuple[int, int]
) -> None:
    gap_start, gap_end, num_layers = _gap_metadata(payload, path)
    actual_gap_shape = (gap_start, num_layers - gap_end)
    if actual_gap_shape != gap_shape:
        raise ValueError(
            f"{path} is not a gap {_gap_shape_text(gap_shape)} run: "
            f"left_kept={actual_gap_shape[0]}, right_kept={actual_gap_shape[1]}"
        )


def _skipped_layers_label(payload: dict[str, Any], path: Path) -> str:
    _, _, num_layers = _gap_metadata(payload, path)
    run_config = payload.get("context", {}).get("run_config", {})
    skipped_layers = run_config.get("skipped_layers")

    if isinstance(skipped_layers, list):
        num_skipped = len(skipped_layers)
    else:
        gap_start, gap_end, _ = _gap_metadata(payload, path)
        num_skipped = gap_end - gap_start

    skipped_percent = 100.0 * num_skipped / num_layers
    return f"{num_skipped}/{num_layers} skipped, {skipped_percent:.0f}%"


def _load_metric_series(
    path: Path,
    *,
    metric_name: str,
    phase: str,
    gap_shape: tuple[int, int],
) -> MetricSeries:
    payload = _load_json(path)
    _verify_gap_shape(payload, path, gap_shape)

    points_by_step: dict[int, float] = {}
    for event in payload.get("metric_events", []):
        if not isinstance(event, dict):
            continue
        if event.get("name") != metric_name or event.get("phase") != phase:
            continue
        step = event.get("step")
        value = event.get("value")
        if step is None or value is None:
            continue
        points_by_step[int(step)] = float(value)

    if not points_by_step:
        raise ValueError(
            f"No {phase!r} metric events named {metric_name!r} found in {path}"
        )

    steps = sorted(points_by_step)
    values = [points_by_step[step] for step in steps]
    model_name = _model_name(payload, path)
    label = f"{_model_label(payload, path)} ({_skipped_layers_label(payload, path)})"
    return MetricSeries(
        label=label,
        steps=steps,
        values=values,
        git_commit=_git_commit_label(payload),
        color=_model_color(payload, path),
        model_name=model_name,
    )


def _moving_average(values: list[float], window: int) -> list[float]:
    if window <= 1 or len(values) <= RAW_PREFIX_POINTS:
        return values

    raw_prefix = values[:RAW_PREFIX_POINTS]
    tail = values[RAW_PREFIX_POINTS:]
    smoothed: list[float] = []
    left_width = (window - 1) // 2
    right_width = window // 2

    for i in range(len(tail)):
        start = max(0, i - left_width)
        end = min(len(tail), i + right_width + 1)
        smoothed.append(sum(tail[start:end]) / (end - start))

    return [*raw_prefix, *smoothed]


def _mean_last_n(values: list[float], n: int) -> float:
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    tail = values[-n:]
    return sum(tail) / len(tail)


def _format_axes(ax: plt.Axes, *, metric_name: str, log_y: bool) -> None:
    ax.set_xlabel("Training step", labelpad=7)
    ax.set_ylabel(METRIC_Y_LABELS[metric_name], labelpad=7)
    ax.grid(True, axis="y", color="#D7D7D7", linewidth=0.7)
    ax.grid(True, axis="x", color="#ECECEC", linewidth=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.margins(x=0.01)

    if metric_name == "top1_drafter_matches_verifier":
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        ax.set_ylim(bottom=0.0)

    if log_y:
        ax.set_yscale("log")


def _format_final_value(value: float, *, metric_name: str) -> str:
    if metric_name == "top1_drafter_matches_verifier":
        return f"{100.0 * value:.1f}%"
    return f"{value:.2f}"


def _format_final_panel(
    ax: plt.Axes,
    *,
    series_list: list[MetricSeries],
    metric_name: str,
) -> None:
    final_values = [
        _mean_last_n(series.values, FINAL_AVG_POINTS) for series in series_list
    ]
    y_positions = list(range(len(series_list)))
    colors = [series.color for series in series_list]

    x_min = min(final_values)
    x_max = max(final_values)
    x_span = max(x_max - x_min, 1e-12)
    x_lower = max(0.0, x_min - 0.12 * x_span)
    x_upper = x_max + 0.24 * x_span

    bars = ax.barh(
        y_positions,
        [value - x_lower for value in final_values],
        left=x_lower,
        height=0.48,
        color=colors,
        alpha=0.9,
        zorder=3,
    )

    for bar, value in zip(bars, final_values):
        ax.annotate(
            _format_final_value(value, metric_name=metric_name),
            xy=(value, bar.get_y() + bar.get_height() / 2),
            xytext=(5, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=8.0,
            color="#222222",
        )

    ax.set_title(
        f"Final avg.\nlast {FINAL_AVG_POINTS} datapoints",
        fontsize=9.4,
        pad=8,
    )
    ax.set_yticks(y_positions)
    ax.set_yticklabels([])
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=8, pad=2)
    ax.grid(True, axis="x", color="#E1E1E1", linewidth=0.6)
    ax.grid(False, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.invert_yaxis()

    ax.set_xlim(x_lower, x_upper)

    if metric_name == "top1_drafter_matches_verifier":
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))


def _git_commits_text(series_list: list[MetricSeries]) -> str:
    unique_commits: list[str] = []
    for series in series_list:
        if series.git_commit not in unique_commits:
            unique_commits.append(series.git_commit)
    return "Training git commits: " + ", ".join(unique_commits)


def plot_thesis_gap11_metric(
    json_paths: Iterable[str | Path] | None = None,
    *,
    metric: MetricAlias = "top1",
    phase: str = "train",
    gap_shape: tuple[int, int] = (1, 1),
    output_dir: str | Path | None = None,
    smooth_window: int = DEFAULT_SMOOTH_WINDOW,
    log_y: bool = False,
) -> list[Path]:
    metric_name = _resolve_metric(metric)
    if json_paths is None:
        try:
            json_paths = THESIS_GAP_JSON_PATHS[gap_shape]
        except KeyError as exc:
            raise ValueError(f"No default run.json paths for gap {gap_shape}") from exc

    paths = [Path(path) for path in json_paths]
    if not paths:
        raise ValueError("Provide at least one run.json path.")

    series_list = [
        _load_metric_series(
            path,
            metric_name=metric_name,
            phase=phase,
            gap_shape=gap_shape,
        )
        for path in paths
    ]
    series_list.sort(key=_model_sort_key)

    if smooth_window <= 0:
        raise ValueError(f"smooth_window must be positive, got {smooth_window}")

    plt.rcParams.update(
        {
            "axes.labelsize": 10.5,
            "axes.titlesize": 14,
            "font.size": 10,
            "legend.fontsize": 8.2,
            "savefig.bbox": "tight",
        }
    )

    fig = plt.figure(figsize=(9.2, 5.75))
    gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[4.6, 1.15], wspace=0.1)
    ax = fig.add_subplot(gs[0, 0])
    final_ax = fig.add_subplot(gs[0, 1])
    fig.subplots_adjust(top=0.83, bottom=0.34, left=0.095, right=0.975)

    subtitle = ""
    if smooth_window > 1:
        subtitle += (
            f"First {RAW_PREFIX_POINTS} points raw, "
            f"then {smooth_window}-point centered moving average"
        )

    fig.suptitle(
        (
            f"{METRIC_TITLE_PREFIXES[metric_name]} "
            f"for Training {_gap_shape_text(gap_shape)} Gap"
        ),
        y=0.965,
        fontsize=15,
        fontweight="semibold",
    )
    fig.text(
        0.5,
        0.91,
        subtitle,
        ha="center",
        va="center",
        fontsize=9.6,
        color="#555555",
    )

    for series in series_list:
        values = _moving_average(series.values, smooth_window)
        ax.plot(
            series.steps,
            values,
            color=series.color,
            linewidth=LINE_WIDTH,
            label=series.label,
        )

    _format_axes(ax, metric_name=metric_name, log_y=log_y)
    _format_final_panel(final_ax, series_list=series_list, metric_name=metric_name)

    legend = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
        frameon=False,
        handlelength=2.4,
        columnspacing=1.6,
        labelspacing=0.55,
        borderaxespad=0.8,
    )
    fig.canvas.draw()
    legend_bbox = legend.get_window_extent(fig.canvas.get_renderer()).transformed(
        fig.transFigure.inverted()
    )
    fig.text(
        0.5,
        max(0.01, legend_bbox.y0 - 0.025),
        _git_commits_text(series_list),
        ha="center",
        va="top",
        fontsize=8.2,
        color="#666666",
    )

    output_dir = (
        Path(output_dir)
        if output_dir is not None
        else Path("plots") / f"thesis_{_gap_shape_slug(gap_shape)}_metrics"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = (
        f"__smooth_{smooth_window}" if smooth_window != DEFAULT_SMOOTH_WINDOW else ""
    )
    gap_part = _gap_shape_slug(gap_shape)
    output_stem = f"thesis_{gap_part}__{_safe_path_part(metric_name)}__{phase}{suffix}"

    output_path = output_dir / f"{output_stem}.png"
    fig.savefig(output_path, dpi=300)

    plt.close(fig)
    return [output_path]


def main() -> None:
    parser = ArgumentParser(
        description="Plot thesis-ready gap training metrics from run.json files."
    )
    parser.add_argument(
        "--metric",
        choices=sorted(METRIC_ALIASES),
        default="top1",
        help="Metric to plot. Friendly aliases: top1, kl.",
    )
    parser.add_argument("--phase", default="train")
    parser.add_argument(
        "--gap-shape",
        nargs=2,
        type=int,
        metavar=("LEFT_KEPT", "RIGHT_KEPT"),
        default=(1, 1),
        help="Gap shape to plot and verify. Defaults to 1 1. Use 2 2 for for_thesis_day2 runs.",
    )
    parser.add_argument(
        "--json-path",
        dest="json_paths",
        action="append",
        default=None,
        help=(
            "Explicit run.json path. Can be passed multiple times. "
            "Defaults to the configured paths for --gap-shape."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=DEFAULT_SMOOTH_WINDOW,
        help="Centered moving-average window over logged metric points. Use 1 for raw traces.",
    )
    parser.add_argument(
        "--log-y", action="store_true", help="Use a logarithmic y-axis."
    )
    args = parser.parse_args()

    gap_shape = tuple(args.gap_shape)

    output_paths = plot_thesis_gap11_metric(
        json_paths=args.json_paths,
        metric=args.metric,
        phase=args.phase,
        gap_shape=gap_shape,
        output_dir=args.output_dir,
        smooth_window=args.smooth_window,
        log_y=args.log_y,
    )

    for output_path in output_paths:
        print(f"Saved plot to: {output_path}")


if __name__ == "__main__":
    main()
