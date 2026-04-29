# skip_search_spec/analysis/plot_training_metric_average_bars.py

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt

from skip_search_spec.analysis.plot_training_metrics import (
    MetricSeries,
    _load_metric_series,
    _safe_path_part,
)


def _mean_last_n(values: list[float], n: int) -> tuple[float, int]:
    if n <= 0:
        raise ValueError(f"last_n_points must be positive, got {n}")

    used_values = values[-n:]
    return sum(used_values) / len(used_values), len(used_values)


def plot_training_metric_average_bars_jsons(
    json_paths: Iterable[str | Path],
    *,
    metric_name: str = "kl_full_to_mid",
    phase: str = "train",
    last_n_points: int = 20,
    output_dir: str | Path = "measurements/training_metric_bar_plots",
    y_log_scale: bool = False,
    sort_by_value: bool = True,
    lower_is_better: bool = True,
) -> Path:
    paths = [Path(p) for p in json_paths]

    if not paths:
        raise ValueError("Must provide at least one JSON path.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    series_list: list[MetricSeries] = [
        _load_metric_series(path, metric_name=metric_name, phase=phase)
        for path in paths
    ]

    rows: list[tuple[str, float, int, int]] = []

    for series in series_list:
        avg_value, used_points = _mean_last_n(series.values, last_n_points)
        total_points = len(series.values)

        rows.append(
            (
                series.label,
                avg_value,
                used_points,
                total_points,
            )
        )

    if sort_by_value:
        rows.sort(key=lambda row: row[1], reverse=not lower_is_better)

    labels = [
        f"{label}\npoints={total_points}, avg_last={used_points}"
        for label, _, used_points, total_points in rows
    ]
    values = [avg_value for _, avg_value, _, _ in rows]

    fig_height = max(5.0, 0.6 * len(rows) + 1.5)
    fig, ax = plt.subplots(figsize=(13, fig_height))

    y_positions = list(range(len(rows)))

    ax.barh(y_positions, values, height=0.7)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()

    ax.set_title(f"{metric_name}: average of last {last_n_points} {phase} points")
    ax.set_xlabel(f"average {metric_name}")
    ax.set_ylabel("run")

    if y_log_scale:
        ax.set_xscale("log")

    ax.grid(True, axis="x", alpha=0.3)

    x_max = max(values)
    x_min = min(values)
    x_padding = max((x_max - x_min) * 0.01, abs(x_max) * 0.01, 1e-12)

    for y, (_, avg_value, used_points, total_points) in zip(y_positions, rows):
        ax.text(
            avg_value + x_padding,
            y,
            f"{avg_value:.5g}  ({used_points}/{total_points})",
            va="center",
            fontsize=8,
        )

    fig.tight_layout()

    output_name = (
        f"{_safe_path_part(metric_name)}"
        f"__{phase}"
        f"__last_{last_n_points}_avg"
        f"__{len(paths)}_runs.png"
    )
    output_path = output_dir / output_name

    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    print(f"[plot_training_metric_average_bars_jsons] wrote {output_path}", flush=True)

    return output_path