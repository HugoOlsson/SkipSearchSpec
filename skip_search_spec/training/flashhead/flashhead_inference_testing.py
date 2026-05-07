from dataclasses import dataclass
import os
from pathlib import Path
import tempfile
from typing import Iterable

import torch
from torch import Tensor
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from skip_search_spec.helpers.versioning import get_git_revision
from skip_search_spec.training.flashhead.next_token_adapter import FlashHeadModule


@dataclass(frozen=True, slots=True)
class TopKContainmentMetrics:
    top_k_clusters: int
    num_positions: int
    top1_match_rate: float
    top3_containment: float


DEFAULT_TOP_K_CLUSTER_SWEEP = (1, 10, 20, 50, 100, 300, 500, 1000)


def save_top1_match_rate_table_image(
    *,
    metrics_by_top_k: Iterable[TopKContainmentMetrics],
    path: str | Path,
    git_commit: str,
    num_windows: int,
    window_length: int,
    num_clusters: int,
) -> Path:
    matplotlib_cache_root = Path(tempfile.gettempdir()) / "skip_search_spec_matplotlib"
    mpl_config_dir = matplotlib_cache_root / "config"
    xdg_cache_dir = matplotlib_cache_root / "xdg"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    xdg_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache_dir))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics_by_top_k = tuple(metrics_by_top_k)
    if not metrics_by_top_k:
        raise ValueError("metrics_by_top_k must not be empty")

    table_path = Path(path)
    table_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [
        (
            str(metrics.top_k_clusters),
            f"{100 * metrics.top_k_clusters / num_clusters:.2f}%",
            f"{metrics.top1_match_rate:.6f}",
        )
        for metrics in metrics_by_top_k
    ]

    fig_height = 0.55 + 0.24 * (len(rows) + 1)
    fig, ax = plt.subplots(figsize=(6.8, fig_height))
    ax.axis("off")
    ax.set_title("Top-1 match rate by probing top_k_clusters", pad=3)

    table = ax.table(
        cellText=rows,
        colLabels=(
            "top_k_clusters",
            "% of all clusters probed",
            "top1_match_rate",
        ),
        cellLoc="left",
        colLoc="left",
        bbox=(0.03, 0.10, 0.94, 0.80),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#d0d7de")
        cell.set_text_props(ha="left")
        if row == 0:
            cell.set_facecolor("#f6f8fa")
            cell.set_text_props(weight="bold")

    ax.text(
        0.5,
        0.02,
        (
            f"commit={git_commit[:7]}  "
            f"windows={num_windows}  "
            f"window_length={window_length}  "
            f"total_clusters={num_clusters}"
        ),
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=8,
        color="#57606a",
    )

    fig.tight_layout()
    fig.savefig(table_path, dpi=180)
    plt.close(fig)

    return table_path


@torch.inference_mode()
def evaluate_topk_containment_on_token_windows(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    flashhead: FlashHeadModule,
    token_windows: Iterable[Tensor],
    *,
    max_windows: int | None = None,
    max_positions_per_window: int | None = None,
    print_every_windows: int | None = 10,
    print_first_n_mismatches: int = 5,
    print_git_commit: bool = True,
) -> TopKContainmentMetrics:
    model_device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype

    flashhead = flashhead.to(device=model_device, dtype=model_dtype)

    num_windows_used = 0
    num_positions = 0
    top1_hits = 0
    top3_hits = 0
    num_printed_mismatches = 0

    if print_git_commit:
        git_commit = get_git_revision().commit
        print(f"git_commit={git_commit}")
    print("Starting top-k containment evaluation...")
    print(f"  top_k_clusters={flashhead.top_k_clusters}")
    print(f"  max_windows={max_windows}")
    print(f"  max_positions_per_window={max_positions_per_window}")
    print()

    for window_input_ids in token_windows:
        if max_windows is not None and num_windows_used >= max_windows:
            break

        if window_input_ids.ndim != 1:
            raise ValueError("Each token window must have shape [seq_len]")

        if window_input_ids.numel() < 2:
            continue

        if (
            print_every_windows is not None
            and num_windows_used > 0
            and num_windows_used % print_every_windows == 0
        ):
            print(
                f"Processed {num_windows_used} windows, "
                f"{num_positions} positions, "
                f"top1={top1_hits / num_positions:.6f}, "
                f"top3={top3_hits / num_positions:.6f}"
            )

        input_ids = window_input_ids.unsqueeze(0).to(model_device)
        attention_mask = torch.ones_like(input_ids, device=model_device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )

        # Position t predicts token t+1
        dense_logits_all = outputs.logits[0, :-1, :]  # [seq_len - 1, vocab]
        hidden_states_all = outputs.hidden_states[-1][
            0, :-1, :
        ]  # [seq_len - 1, hidden]

        num_eval_positions = dense_logits_all.shape[0]
        if max_positions_per_window is not None:
            num_eval_positions = min(num_eval_positions, max_positions_per_window)

        for pos in range(num_eval_positions):
            dense_logits = dense_logits_all[pos]
            hidden_vector = hidden_states_all[pos]

            routed_top1_id = int(flashhead.find_token(hidden_vector).item())

            dense_top1_id = int(dense_logits.argmax().item())
            dense_top3_ids = torch.topk(dense_logits, k=3).indices

            top1_match = routed_top1_id == dense_top1_id
            top3_match = bool((dense_top3_ids == routed_top1_id).any().item())

            if top1_match:
                top1_hits += 1

            if top3_match:
                top3_hits += 1

            if (not top1_match) and num_printed_mismatches < print_first_n_mismatches:
                dense_top1_text = tokenizer.decode([dense_top1_id])
                routed_top1_text = tokenizer.decode([routed_top1_id])

                print(
                    f"  mismatch #{num_printed_mismatches + 1}: "
                    f"window={num_windows_used}, pos={pos}, "
                    f"dense_top1_id={dense_top1_id}, dense_top1_text={dense_top1_text!r}, "
                    f"routed_top1_id={routed_top1_id}, routed_top1_text={routed_top1_text!r}"
                )
                num_printed_mismatches += 1

            num_positions += 1

        num_windows_used += 1

    if num_positions == 0:
        raise RuntimeError("No evaluation positions were collected.")

    metrics = TopKContainmentMetrics(
        top_k_clusters=flashhead.top_k_clusters,
        num_positions=num_positions,
        top1_match_rate=top1_hits / num_positions,
        top3_containment=top3_hits / num_positions,
    )

    print()
    print("Finished top-k containment evaluation.")
    print(f"  num_windows_used={num_windows_used}")
    print(f"  top_k_clusters={metrics.top_k_clusters}")
    print(f"  num_positions={metrics.num_positions}")
    print(f"  top1_match_rate={metrics.top1_match_rate:.6f}")
    print(f"  top3_containment={metrics.top3_containment:.6f}")

    return metrics


@torch.inference_mode()
def evaluate_topk_cluster_sweep_on_token_windows(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    flashhead: FlashHeadModule,
    token_windows: Iterable[Tensor],
    *,
    top_k_clusters_values: Iterable[int] = DEFAULT_TOP_K_CLUSTER_SWEEP,
    max_windows: int = 30,
    max_positions_per_window: int | None = None,
    print_every_windows: int | None = 10,
    print_first_n_mismatches: int = 0,
    top1_match_rate_table_image_path: str | Path = (
        "topk_cluster_sweep_top1_match_rate.png"
    ),
    window_length: int | None = None,
) -> tuple[TopKContainmentMetrics, ...]:
    if max_windows < 1:
        raise ValueError("max_windows must be >= 1")

    top_k_clusters_values = tuple(top_k_clusters_values)
    if not top_k_clusters_values:
        raise ValueError("top_k_clusters_values must not be empty")

    invalid_top_k_values = [value for value in top_k_clusters_values if value < 1]
    if invalid_top_k_values:
        raise ValueError(
            f"top_k_clusters_values must all be >= 1: {invalid_top_k_values}"
        )

    fixed_token_windows: list[Tensor] = []
    for window_input_ids in token_windows:
        if len(fixed_token_windows) >= max_windows:
            break
        fixed_token_windows.append(window_input_ids.detach().cpu())

    if not fixed_token_windows:
        raise RuntimeError("No token windows were collected for the sweep.")

    plotted_window_length = (
        int(fixed_token_windows[0].numel())
        if window_length is None
        else int(window_length)
    )
    num_clusters = int(flashhead.centroids_t.shape[1])

    git_commit = get_git_revision().commit
    print(f"git_commit={git_commit}")
    print("Starting top-k cluster sweep...")
    print(f"  top_k_clusters_values={top_k_clusters_values}")
    print(f"  num_windows={len(fixed_token_windows)}")
    print(f"  window_length={plotted_window_length}")
    print(f"  num_clusters={num_clusters}")
    print(f"  max_positions_per_window={max_positions_per_window}")
    print()

    original_top_k_clusters = flashhead.top_k_clusters
    metrics_by_top_k: list[TopKContainmentMetrics] = []

    try:
        for top_k_clusters in top_k_clusters_values:
            print(f"Evaluating top_k_clusters={top_k_clusters}")
            flashhead.top_k_clusters = top_k_clusters
            metrics = evaluate_topk_containment_on_token_windows(
                model=model,
                tokenizer=tokenizer,
                flashhead=flashhead,
                token_windows=fixed_token_windows,
                max_windows=len(fixed_token_windows),
                max_positions_per_window=max_positions_per_window,
                print_every_windows=print_every_windows,
                print_first_n_mismatches=print_first_n_mismatches,
                print_git_commit=False,
            )
            metrics_by_top_k.append(metrics)
            print()
    finally:
        flashhead.top_k_clusters = original_top_k_clusters

    table_path = save_top1_match_rate_table_image(
        metrics_by_top_k=metrics_by_top_k,
        path=top1_match_rate_table_image_path,
        git_commit=git_commit,
        num_windows=len(fixed_token_windows),
        window_length=plotted_window_length,
        num_clusters=num_clusters,
    )
    print(f"top1_match_rate_table_image_path={table_path}")

    return tuple(metrics_by_top_k)
