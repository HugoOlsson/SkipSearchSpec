from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_SORT_METRIC = "kl_per_removed_layer"


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
                "JSON file must contain either a top-level list or a dict with a 'results' or 'ablation_results' key."
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


def _sanitize_filename(text: str) -> str:
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")
    sanitized = "".join(c if c in allowed else "_" for c in text)
    sanitized = sanitized.strip("._")
    return sanitized or "plot"


def _extract_binary_mask(result: dict[str, Any]) -> list[int] | None:
    raw = result.get("binary_mask")
    if not isinstance(raw, list) and not isinstance(raw, tuple):
        return None

    mask: list[int] = []
    for item in raw:
        if isinstance(item, bool):
            mask.append(1 if item else 0)
        elif isinstance(item, int):
            mask.append(1 if item != 0 else 0)
        elif isinstance(item, float):
            mask.append(1 if item != 0.0 else 0)
        else:
            return None
    return mask


def _prepare_rows(
    results: list[dict[str, Any]],
    *,
    sort_metric: str,
    ascending: bool,
    include_full_model: bool,
    top_k: int | None,
) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    expected_num_layers: int | None = None

    for result in results:
        if not isinstance(result, dict):
            continue

        visual = result.get("visual_mask", result.get("visual"))
        if not isinstance(visual, str) or len(visual) == 0:
            continue

        binary_mask = _extract_binary_mask(result)
        if binary_mask is None or len(binary_mask) == 0:
            continue

        if expected_num_layers is None:
            expected_num_layers = len(binary_mask)
        elif len(binary_mask) != expected_num_layers:
            raise ValueError(
                "All results must have the same binary_mask length. "
                f"Saw both {expected_num_layers} and {len(binary_mask)}."
            )

        num_removed = result.get("num_removed_layers")
        if not include_full_model and num_removed == 0:
            continue

        sort_value = _coerce_float(result.get(sort_metric))
        if sort_value is None:
            continue

        rows.append(
            {
                "visual": visual,
                "binary_mask": binary_mask,
                "sort_value": sort_value,
                "mask_name": result.get("mask_name"),
                "num_removed_layers": num_removed,
            }
        )

    if not rows:
        raise ValueError(
            f"No plottable rows found for sort metric '{sort_metric}'."
        )

    rows.sort(key=lambda row: row["sort_value"], reverse=not ascending)
    if top_k is not None:
        rows = rows[:top_k]

    assert expected_num_layers is not None
    return rows, expected_num_layers


def plot_ablation_heatmap_from_json(
    json_path: str | Path,
    *,
    output_path: str | Path | None = None,
    sort_metric: str = DEFAULT_SORT_METRIC,
    ascending: bool = True,
    include_full_model: bool = False,
    top_k: int | None = None,
    show_metric_strip: bool = True,
    title: str | None = None,
    dpi: int = 220,
) -> Path:
    """
    Render a heatmap over layers from an ablation-results JSON file.

    Rows are ablation variants.
    Columns are layer indices.
    Cell value 1 means the layer is kept, 0 means the layer is removed.

    The y-axis uses only the visual mask string as the identifier.

    Expected JSON structure:
      - either a top-level list of result dicts
      - or a dict with a top-level 'results' or 'ablation_results' key

    Required per result:
      - 'binary_mask'
      - 'visual_mask' (or 'visual')
      - the chosen sort metric, by default 'kl_per_removed_layer'
    """
    json_path = Path(json_path)
    metadata, results = _load_results_from_json(json_path)
    rows, num_layers = _prepare_rows(
        results,
        sort_metric=sort_metric,
        ascending=ascending,
        include_full_model=include_full_model,
        top_k=top_k,
    )

    heatmap = np.asarray([row["binary_mask"] for row in rows], dtype=float)
    labels = [row["visual"] for row in rows]
    metric_values = np.asarray([row["sort_value"] for row in rows], dtype=float)

    n_rows = heatmap.shape[0]

    if show_metric_strip:
        fig, (ax_heatmap, ax_metric) = plt.subplots(
            1,
            2,
            figsize=(max(10.0, 0.22 * num_layers + 3.0), max(4.0, 0.14 * n_rows + 1.2)),
            dpi=dpi,
            gridspec_kw={"width_ratios": [max(6, num_layers), 1.5], "wspace": 0.04},
        )
    else:
        fig, ax_heatmap = plt.subplots(
            1,
            1,
            figsize=(max(10.0, 0.22 * num_layers + 3.0), max(4.0, 0.14 * n_rows + 1.2)),
            dpi=dpi,
        )
        ax_metric = None

    # Binary mask heatmap: 1=kept, 0=removed.
    im = ax_heatmap.imshow(
        heatmap,
        aspect="auto",
        interpolation="nearest",
        origin="upper",
        cmap="gray_r",
        vmin=0.0,
        vmax=1.0,
    )
    _ = im

    ax_heatmap.set_xlabel("Layer index")
    ax_heatmap.set_ylabel("Ablation visual")

    if title is None:
        model_name = metadata.get("model_name")
        if isinstance(model_name, str) and len(model_name) > 0:
            title = f"Layer-ablation heatmap | {model_name} | sorted by {sort_metric}"
        else:
            title = f"Layer-ablation heatmap | sorted by {sort_metric}"
    ax_heatmap.set_title(title)

    # Use only the visual pattern on the y-axis, as requested.
    label_fontsize = min(7.5, max(2.8, 220.0 / max(n_rows, 1)))
    ax_heatmap.set_yticks(np.arange(n_rows))
    ax_heatmap.set_yticklabels(labels, fontsize=label_fontsize, fontfamily="monospace")
    ax_heatmap.tick_params(axis="y", pad=1)

    # Show only some x ticks if there are many layers.
    if num_layers <= 16:
        x_tick_step = 1
    elif num_layers <= 32:
        x_tick_step = 2
    elif num_layers <= 64:
        x_tick_step = 4
    else:
        x_tick_step = max(1, num_layers // 16)

    x_ticks = np.arange(0, num_layers, x_tick_step)
    ax_heatmap.set_xticks(x_ticks)
    ax_heatmap.set_xticklabels([str(x) for x in x_ticks], fontsize=8)

    # Light grid so layer boundaries remain readable even with many rows.
    ax_heatmap.set_xticks(np.arange(-0.5, num_layers, 1), minor=True)
    ax_heatmap.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax_heatmap.grid(which="minor", linestyle="-", linewidth=0.15, alpha=0.35)
    ax_heatmap.tick_params(which="minor", bottom=False, left=False)

    if ax_metric is not None:
        metric_column = metric_values.reshape(-1, 1)
        ax_metric.imshow(
            metric_column,
            aspect="auto",
            interpolation="nearest",
            origin="upper",
            cmap="viridis",
        )
        ax_metric.set_title(sort_metric, fontsize=9)
        ax_metric.set_xticks([])
        ax_metric.set_yticks([])

    fig.tight_layout(pad=0.45)

    if output_path is None:
        stem = _sanitize_filename(json_path.stem)
        output_path = json_path.with_name(
            f"{stem}_heatmap_{_sanitize_filename(sort_metric)}.png"
        )
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Create a layer-ablation heatmap from an ablation-results JSON file.")
    parser.add_argument("json_path", help="Path to the ablation-results JSON file")
    parser.add_argument("--output-path", default=None, help="Where to save the PNG")
    parser.add_argument(
        "--sort-metric",
        default=DEFAULT_SORT_METRIC,
        help="Metric to sort rows by, for example: kl_per_removed_layer, mean_kl_full_to_masked, mean_ce_gap",
    )
    parser.add_argument(
        "--descending",
        action="store_true",
        help="Sort from highest to lowest instead of lowest to highest",
    )
    parser.add_argument(
        "--include-full-model",
        action="store_true",
        help="Include rows with zero removed layers",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Only include the top K rows after sorting",
    )
    parser.add_argument(
        "--hide-metric-strip",
        action="store_true",
        help="Hide the small metric strip on the right",
    )
    parser.add_argument("--title", default=None, help="Optional custom title")
    parser.add_argument("--dpi", type=int, default=220)
    args = parser.parse_args()

    output_path = plot_ablation_heatmap_from_json(
        args.json_path,
        output_path=args.output_path,
        sort_metric=args.sort_metric,
        ascending=not args.descending,
        include_full_model=args.include_full_model,
        top_k=args.top_k,
        show_metric_strip=not args.hide_metric_strip,
        title=args.title,
        dpi=args.dpi,
    )
    print(output_path)


if __name__ == "__main__":
    main()
