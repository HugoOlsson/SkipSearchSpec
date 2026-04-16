from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from skip_search_spec.protocols.measurements import MeasurementRun, MetricEvent


def _events_to_series(
    events: Iterable[MetricEvent],
) -> dict[str, tuple[list[float], list[float]]]:
    by_name: dict[str, list[tuple[float, float]]] = {}
    by_name_seen: dict[str, int] = {}

    for event in events:
        if event.phase != "train":
            continue

        seen = by_name_seen.get(event.name, 0)
        x = float(event.step) if event.step is not None else float(seen)
        by_name_seen[event.name] = seen + 1

        if event.name not in by_name:
            by_name[event.name] = []
        by_name[event.name].append((x, float(event.value)))

    series: dict[str, tuple[list[float], list[float]]] = {}
    for name, points in by_name.items():
        points.sort(key=lambda p: p[0])
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        series[name] = (xs, ys)
    return series


def _plot_panel(
    ax: Axes,
    series: dict[str, tuple[list[float], list[float]]],
    metric_names: list[str],
) -> None:
    plotted = 0
    for name in metric_names:
        if name not in series:
            continue
        xs, ys = series[name]
        if not xs:
            continue

        ax.plot(xs, ys, label=name, linewidth=1.6)
        plotted += 1

    if plotted == 0:
        ax.set_visible(False)
        return

    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    ax.set_xlabel("step")


def _resolve_early_exit_layer(run: MeasurementRun) -> str:
    config = run.context.run_config
    candidates = (
        "early_exit_layer",
        "inner_exit_layer_index",
        "exit_layer",
    )
    for key in candidates:
        if key in config:
            return str(config[key])
    return "n/a"


def _resolve_total_layers(run: MeasurementRun) -> str:
    config = run.context.run_config
    candidates = (
        "total_layers",
        "num_layers",
        "model_num_layers",
    )
    for key in candidates:
        if key in config:
            return str(config[key])
    return "n/a"


def plot_training_run(
    *,
    run: MeasurementRun,
    output_path: Path,
) -> Path:
    series = _events_to_series(run.metric_events)
    if not series:
        raise ValueError("No metric events found for phase='train'.")

    known_panels: list[tuple[str, list[str]]] = [
        ("Losses", ["loss", "ce_mid", "ce_full"]),
        ("Divergence", ["ce_gap", "kl_full_to_mid", "js"]),
        ("Agreement", ["top1_agreement", "overlap", "p_mid_on_full_argmax"]),
    ]
    used_metrics: set[str] = set()
    active_panels: list[tuple[str, list[str]]] = []

    for title, names in known_panels:
        available = [name for name in names if name in series]
        if available:
            active_panels.append((title, available))
            used_metrics.update(available)

    other_metrics = sorted(name for name in series.keys() if name not in used_metrics)
    if other_metrics:
        active_panels.append(("Other Metrics", other_metrics))

    if not active_panels:
        active_panels.append(("Metrics", sorted(series.keys())))

    nrows = len(active_panels)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=1,
        figsize=(12, max(3.2 * nrows, 4.5)),
        squeeze=False,
    )
    flat_axes = [axes[i][0] for i in range(nrows)]

    for ax, (panel_title, names) in zip(flat_axes, active_panels):
        _plot_panel(ax, series, names)
        if ax.get_visible():
            ax.set_title(panel_title)

    model_label = ", ".join(run.context.model_names) if run.context.model_names else "n/a"
    dataset_label = run.context.dataset_name or "n/a"
    git_tag = run.context.git_tag or "n/a"
    git_commit_short = run.context.git_commit[:12]
    early_exit_layer = _resolve_early_exit_layer(run)
    total_layers = _resolve_total_layers(run)
    if early_exit_layer != "n/a" and total_layers != "n/a":
        early_exit_layer_label = f"{early_exit_layer}/{total_layers}"
    else:
        early_exit_layer_label = early_exit_layer
    fig.suptitle(
        (
            f"run_id={run.context.run_id} | phase=train | model={model_label}\n"
            f"dataset={dataset_label} | early_exit_layer={early_exit_layer_label}\n"
            f"git_tag={git_tag} | git_commit={git_commit_short}"
        ),
        fontsize=11,
        linespacing=1.4,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _default_output_path(run_json_path: Path) -> Path:
    return run_json_path.with_name(f"{run_json_path.stem}_plots.png")


def main() -> None:
    parser = ArgumentParser(description="Plot useful training metrics from a MeasurementRun JSON.")
    parser.add_argument("run_json", type=Path, help="Path to run.json")
    args = parser.parse_args()

    run_json_path = args.run_json
    with run_json_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    run = MeasurementRun.from_dict(raw)

    output_path = _default_output_path(run_json_path)
    plot_training_run(run=run, output_path=output_path)

    print(f"Saved plot to: {output_path}")


if __name__ == "__main__":
    main()
