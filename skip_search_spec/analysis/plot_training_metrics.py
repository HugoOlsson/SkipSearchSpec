# skip_search_spec/analysis/plot_training_metrics.py

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt


@dataclass(frozen=True, slots=True)
class MetricSeries:
    label: str
    steps: list[int]
    values: list[float]


def _safe_path_part(text: str) -> str:
    text = text.replace("/", "_")
    text = re.sub(r"[^A-Za-z0-9_.=\-+]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")[:180]


def _short_model_name(model_name: str) -> str:
    return model_name.split("/")[-1].replace(".", "_")

def _make_gap_text(run_config: dict[str, Any]) -> str:
    gap_start = int(run_config["gap_start"])
    gap_end = int(run_config["gap_end"])
    gap_length = int(run_config["gap_length"])
    num_layers = int(run_config["num_layers"])

    start = gap_start
    gap = gap_length
    end = num_layers - gap_end

    return f"{start}:{gap}:{end}"


def _make_label(data: dict[str, Any], path: Path) -> str:
    context = data.get("context", {})
    run_config = context.get("run_config", {})

    run_id = str(context.get("run_id", path.stem))

    git_commit = context.get("git_commit")
    git_short = str(git_commit)[:8] if git_commit else "no_git"

    model_name = str(
        run_config.get("model_name")
        or (context.get("model_names") or ["unknown_model"])[0]
    )

    model_short = _short_model_name(model_name)
    gap_text = _make_gap_text(run_config)

    return f"{model_short} {run_id} git={git_short} gap={gap_text}"


def _load_metric_series(
    json_path: str | Path,
    *,
    metric_name: str,
    phase: str,
) -> MetricSeries:
    path = Path(json_path)

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    by_step: dict[int, float] = {}

    for event in data.get("metric_events", []):
        if event.get("name") != metric_name:
            continue
        if event.get("phase") != phase:
            continue

        step = event.get("step")
        value = event.get("value")

        if step is None or value is None:
            continue

        by_step[int(step)] = float(value)

    if not by_step:
        raise ValueError(
            f"No metric events found for name={metric_name!r}, phase={phase!r} in {path}"
        )

    steps = sorted(by_step)
    values = [by_step[step] for step in steps]

    return MetricSeries(
        label=_make_label(data, path),
        steps=steps,
        values=values,
    )


def plot_training_metric_jsons(
    json_paths: Iterable[str | Path],
    *,
    metric_name: str = "kl_full_to_mid",
    phase: str = "train",
    output_dir: str | Path = "measurements/training_metric_plots",
    y_log_scale: bool = False,
) -> Path:
    paths = [Path(p) for p in json_paths]

    if not paths:
        raise ValueError("Must provide at least one JSON path.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    series_list = [
        _load_metric_series(path, metric_name=metric_name, phase=phase)
        for path in paths
    ]

    fig, ax = plt.subplots(figsize=(13, 7))

    for series in series_list:
        ax.plot(
            series.steps,
            series.values,
            linewidth=1.8,
            label=series.label,
        )

    ax.set_title(f"{metric_name} over training")
    ax.set_xlabel("step")
    ax.set_ylabel(metric_name)

    if y_log_scale:
        ax.set_yscale("log")

    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()

    output_name = f"{_safe_path_part(metric_name)}__{phase}__{len(paths)}_runs.png"
    output_path = output_dir / output_name

    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    print(f"[plot_training_metric_jsons] wrote {output_path}", flush=True)

    return output_path