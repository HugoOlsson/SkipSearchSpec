from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from matplotlib.patches import Patch
import matplotlib.pyplot as plt


DEFAULT_METRIC = "kl_per_removed_layer"
DEFAULT_NUM_COLUMNS = 3

CATEGORY_COLORS = {
    "early_exit": "#0BB2B2",    
    "late_start": "#710088", 
    "single_left_out": "#5F8CFF",
    "internal_gap": "#007F1E", 
    "other": "#005278",
}

def classify_ablation(mask_name: str) -> str:
    family = mask_name.partition("__")[0]

    if family == "drop_internal_single":
        return "single_left_out"
    
    if family == "keep_suffix":
            return "late_start"

    if family == "drop_internal_block" or family == "keep_edges":
        return "internal_gap"

    if family == "keep_prefix":
        return "early_exit"

    return "other"

def _load_results_from_json(json_path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = Path(json_path)
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    metadata: dict[str, Any] = {}

    if isinstance(payload, list):
        results = payload
    elif isinstance(payload, dict):
        metadata = {k: v for k, v in payload.items() if k not in {"results", "ablation_results"}}
        if "results" in payload:
            results = payload["results"]
        elif "ablation_results" in payload:
            results = payload["ablation_results"]
        else:
            raise ValueError(
                "JSON file must contain either a top-level list or a dict with a 'results' key."
            )
    else:
        raise TypeError(f"Unsupported JSON root type: {type(payload)}")

    if not isinstance(results, list):
        raise TypeError(f"Expected results to be a list, got {type(results)}")

    return metadata, results


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _sanitize_filename(text: str) -> str:
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")
    sanitized = "".join(c if c in allowed else "_" for c in text)
    sanitized = sanitized.strip("._")
    return sanitized or "plot"


def _pick_ablation_index(result: dict[str, Any], fallback_index_1_based: int) -> int:
    for key in (
        "ablation_index",
        "source_ablation_index",
        "result_index",
        "index",
        "source_index",
    ):
        value = _coerce_int(result.get(key))
        if value is not None:
            return value
    return fallback_index_1_based


def _prepare_rows(
    *,
    json_path: str | Path,
    metric: str,
    top_k: int | None,
    ascending: bool,
    include_full_model: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metadata, results = _load_results_from_json(json_path)

    rows: list[dict[str, Any]] = []
    for source_pos, result in enumerate(results, start=1):
        if not isinstance(result, dict):
            continue

        visual = result.get("visual_mask", result.get("visual"))
        if not isinstance(visual, str) or len(visual) == 0:
            continue

        metric_value = _coerce_float(result.get(metric))
        if metric_value is None:
            continue

        removed = result.get("num_removed_layers")
        removed_int = int(removed) if removed is not None else None

        if not include_full_model and removed_int == 0:
            continue

        row = dict(result)
        row["ablation_index"] = _pick_ablation_index(result, source_pos)
        row["visual"] = visual
        row["metric"] = metric_value
        row["removed"] = removed_int

        rows.append(row)

    if not rows:
        raise ValueError(
            f"No plottable rows found for metric '{metric}' in {json_path}."
        )

    rows.sort(key=lambda row: row["metric"], reverse=not ascending)
    if top_k is not None:
        rows = rows[:top_k]

    return metadata, rows


def _chunk_rows_left_to_right(rows: list[dict[str, Any]], num_columns: int) -> list[list[dict[str, Any]]]:
    if len(rows) == 0:
        return []

    num_columns = max(1, min(num_columns, len(rows)))
    rows_per_column = math.ceil(len(rows) / num_columns)

    chunks: list[list[dict[str, Any]]] = []
    for start in range(0, len(rows), rows_per_column):
        chunks.append(rows[start:start + rows_per_column])

    return chunks


def plot_ablation_json(
    json_path: str | Path,
    *,
    metric: str = DEFAULT_METRIC,
    output_path: str | Path | None = None,
    title: str | None = None,
    top_k: int | None = None,
    ascending: bool = True,
    include_full_model: bool = False,
    dpi: int = 220,
    num_columns: int = DEFAULT_NUM_COLUMNS,
    bar_height: float = 0.7,
) -> Path:
    """
    Read ablation-results JSON and save a multi-column horizontal bar chart.

    Each vertical stack is split into three aligned regions:
    [ablation index] [visual mask] [bar + raw value]

    All bar regions use the same global x-limits, so the normalization is shared
    across the full figure.

    Set ascending=False to sort from biggest to smallest.
    """
    json_path = Path(json_path)
    metadata, rows = _prepare_rows(
        json_path=json_path,
        metric=metric,
        top_k=top_k,
        ascending=ascending,
        include_full_model=include_full_model,
    )

    chunks = _chunk_rows_left_to_right(rows, num_columns)
    actual_num_columns = len(chunks)
    max_rows_in_any_column = max(len(chunk) for chunk in chunks)

    values = [row["metric"] for row in rows]
    x_min_value = min(values)
    x_max_value = max(values)

    x_lower = min(0.0, x_min_value)
    x_upper = x_max_value
    x_span = max(x_upper - x_lower, 1e-12)
    left_pad = 0.04 * x_span
    right_pad = 0.3 * x_span
    global_xlim = (x_lower - left_pad, x_upper + right_pad)

    label_fontsize = min(8.0, max(3.0, 240.0 / max(max_rows_in_any_column, 1)))
    value_fontsize = max(6, label_fontsize - 0.2)
    max_visual_chars = max(len(row["visual"]) for row in rows)
    layer_count = max_visual_chars
    t = (layer_count - 20) / (48 - 20)
    t = max(0.0, min(1.0, t))
    visual_divisor = 1.0 + t * (1.4 - 1.0)
    visual_fontsize = label_fontsize / visual_divisor
    title_fontsize = 13.0

    # Inches needed for the visual column, given monospace at visual_fontsize.
    # ~0.6 em per char; em ≈ fontsize in points; 72 pt/inch.
    char_width_in = 0.6 * visual_fontsize / 72
    visual_width_in = max_visual_chars * char_width_in + 0.15  # small right pad

    # Treat width_ratios as inches so the visual_ax gets exactly visual_width_in.
    idx_width_in = 0.45
    bar_width_in = 2.5
    column_width_in = idx_width_in + visual_width_in + bar_width_in

    fig_width = max(12.0, column_width_in * actual_num_columns)
    fig_height = max(4.6, 0.25 * max_rows_in_any_column + 1.55)

    fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
    outer = fig.add_gridspec(1, actual_num_columns, wspace=0)

    if title is None:
        model_name = metadata.get("model_name")
        if isinstance(model_name, str) and len(model_name) > 0:
            title = f"{metric} | {model_name}"
        else:
            title = metric

    code_version = metadata.get("code_version") or {}
    commit = code_version.get("commit")
    short_commit = commit[:8] if isinstance(commit, str) else None

    running_rank = 1
    bar_axes: list[Any] = []

    for outer_idx, chunk in enumerate(chunks):
        inner = outer[0, outer_idx].subgridspec(
            1,
            3,
            width_ratios=[idx_width_in, visual_width_in, bar_width_in],
            wspace=0.03,
        )
        idx_ax = fig.add_subplot(inner[0, 0])
        visual_ax = fig.add_subplot(inner[0, 1])
        bar_ax = fig.add_subplot(inner[0, 2], sharex=bar_axes[0] if bar_axes else None)
        bar_axes.append(bar_ax)

        y_positions = list(range(len(chunk)))
        chunk_values = [row["metric"] for row in chunk]
        chunk_colors = [
            CATEGORY_COLORS[classify_ablation(row["mask_name"])]
            for row in chunk
        ]

        for text_ax in (idx_ax, visual_ax):
            text_ax.set_xlim(0.0, 1.0)
            text_ax.set_ylim(-0.5, len(chunk) - 0.5)
            text_ax.invert_yaxis()
            text_ax.axis("off")

        bar_ax.barh(
            y_positions,
            chunk_values,
            height=bar_height,
            color=chunk_colors,
        )
        bar_ax.set_yticks([])
        bar_ax.set_ylim(-0.5, len(chunk) - 0.5)
        bar_ax.invert_yaxis()
        bar_ax.set_xlim(*global_xlim)
        bar_ax.margins(y=0.01)
        bar_ax.grid(axis="x", alpha=0.25)
        bar_ax.set_axisbelow(True)
        bar_ax.axvline(0.0, linewidth=0.8, alpha=0.45)

        value_offset = 0.05 * x_span
        for y, row in enumerate(chunk):
            row_color = CATEGORY_COLORS[classify_ablation(row["mask_name"])]

            metric_value = row["metric"]
            full_kl_value = row["mean_kl_full_to_masked"]

            metric_str = f"{metric_value:.1f}" if abs(metric_value) > 5 else f"{metric_value:.3f}"
            full_kl_str = f"{full_kl_value:.1f}" if abs(full_kl_value) > 5 else f"{full_kl_value:.2f}"

            label = f"{metric_str} ({full_kl_str})"

            idx_ax.text(
                0.98,
                y,
                str(row["ablation_index"]),
                ha="right",
                va="center",
                fontsize=label_fontsize,
                fontfamily="monospace",
                clip_on=True,
            )
            visual_ax.text(
                0.01,
                y,
                row["visual"],
                ha="left",
                va="center",
                fontsize=visual_fontsize,
                fontfamily="monospace",
                clip_on=True,
                color=row_color,
            )
            bar_ax.text(
                row["metric"] + value_offset,
                y,
                label,
                ha="left",
                va="center",
                fontsize=value_fontsize,
                fontfamily="monospace",
                clip_on=True,
            )

        start_rank = running_rank
        end_rank = running_rank + len(chunk) - 1
        visual_ax.set_title(f"#{start_rank}–#{end_rank}", fontsize=9.0)
        running_rank = end_rank + 1

    for bar_ax in bar_axes[1:]:
        bar_ax.tick_params(axis="x", labelbottom=True)

    fig.suptitle(metric, fontsize=title_fontsize, y=0.98)
    fig.text(
        0.5,
        0.955,
        f"Model: {model_name}",
        ha="center",
        va="top",
        fontsize=title_fontsize/1.3,
    )

    if short_commit:
        fig.text(
            0.995,
            0.005,
            f"commit {short_commit}",
            ha="right",
            va="bottom",
            fontsize=9,
            color="#666",
            family="monospace",
        )

    used_categories = {
        classify_ablation(row["mask_name"])
        for row in rows
    }

    legend_handles = [
        Patch(facecolor=color, label=category)
        for category, color in CATEGORY_COLORS.items()
        if category in used_categories
    ]

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 0.94),
    )

    fig.subplots_adjust(
        left=0.02,
        right=0.995,
        top=0.86,
        bottom=0.08,
    )

    if output_path is None:
        stem = _sanitize_filename(json_path.stem)
        output_path = json_path.parent / "plots" / f"{stem}_{_sanitize_filename(metric)}_multicolumn.png"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot ablation results from a JSON file as a multi-column horizontal bar chart."
    )
    parser.add_argument("json_path", help="Path to the ablation-results JSON file")
    parser.add_argument(
        "--metric",
        default=DEFAULT_METRIC,
        help="Metric key to plot, for example: kl_per_removed_layer, mean_kl_full_to_masked, mean_ce_gap",
    )
    parser.add_argument("--output-path", default=None, help="Where to save the PNG")
    parser.add_argument("--title", default=None, help="Optional custom title")
    parser.add_argument("--top-k", type=int, default=None, help="Plot only the top K rows after sorting")
    parser.add_argument(
        "--descending",
        action="store_true",
        help="Sort from highest to lowest instead of lowest to highest",
    )
    parser.add_argument(
        "--include-full-model",
        action="store_true",
        help="Include entries with zero removed layers",
    )
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument(
        "--num-columns",
        type=int,
        default=DEFAULT_NUM_COLUMNS,
        help="How many side-by-side vertical stacks to split the bars into",
    )
    parser.add_argument(
        "--bar-height",
        type=float,
        default=0.86,
        help="Height of each horizontal bar inside its stack",
    )
    args = parser.parse_args()

    output_path = plot_ablation_json(
        args.json_path,
        metric=args.metric,
        output_path=args.output_path,
        title=args.title,
        top_k=args.top_k,
        ascending=not args.descending,
        include_full_model=args.include_full_model,
        dpi=args.dpi,
        num_columns=args.num_columns,
        bar_height=args.bar_height,
    )
    print(output_path)


if __name__ == "__main__":
    main()
