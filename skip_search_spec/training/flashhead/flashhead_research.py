from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, cast

from torch import Tensor
from transformers import PreTrainedModel, PreTrainedTokenizerBase
import matplotlib.pyplot as plt
from skip_search_spec.helpers.tooling import (
    get_preferred_device,
    get_preferred_float_dtype,
    load_model_and_tokenizer,
)
from skip_search_spec.helpers.versioning import get_git_revision

from skip_search_spec.helpers.shared_decoding_tools import build_fixed_window_dataloader
from skip_search_spec.protocols.windows import DatasetSpec, ModelAndTokenizer
from skip_search_spec.training.flashhead.building_clusters import build_clusters
from skip_search_spec.training.flashhead.flashhead_inference_testing import (
    evaluate_topk_cluster_sweep_on_token_windows,
)
from skip_search_spec.training.flashhead.next_token_adapter import ANNHModule
from skip_search_spec.training.flashhead.storage import save_flashhead


@dataclass(frozen=True, slots=True)
class LoadedFlashHeadBase:
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    lm_head_vector_table: Tensor


def extract_lm_head_vector_table(model: PreTrainedModel) -> Tensor:
    output_embeddings = model.get_output_embeddings()
    if output_embeddings is None:
        raise ValueError(
            "Model does not expose output embeddings via get_output_embeddings()"
        )

    weight = getattr(output_embeddings, "weight", None)
    if not isinstance(weight, Tensor):
        raise ValueError("Output embeddings do not expose a Tensor weight")

    return weight


def load_flashhead_base(model_name: str) -> LoadedFlashHeadBase:
    model_and_tokenizer = load_model_and_tokenizer(model_name)
    model = model_and_tokenizer.model
    tokenizer = model_and_tokenizer.tokenizer

    lm_head_vector_table = extract_lm_head_vector_table(model).detach().float().cpu()

    return LoadedFlashHeadBase(
        model=model,
        tokenizer=tokenizer,
        lm_head_vector_table=lm_head_vector_table,
    )


def get_computer_label() -> str:
    if sys.platform == "darwin":
        try:
            chip = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                text=True,
            ).strip()
            memory_bytes = int(
                subprocess.check_output(
                    ["sysctl", "-n", "hw.memsize"],
                    text=True,
                ).strip()
            )
            memory_gb = round(memory_bytes / 1024**3)
            return f"{chip}, {memory_gb}GB"
        except (OSError, subprocess.CalledProcessError, ValueError):
            return "Apple M5, 24GB"

    if sys.platform.startswith("linux"):
        try:
            gpu_line = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            ).splitlines()[0]
            gpu_name, memory_mb = [part.strip() for part in gpu_line.split(",", 1)]
            memory_gb = round(int(memory_mb) / 1024)
            return f"{gpu_name}, {memory_gb}GB"
        except (IndexError, OSError, subprocess.CalledProcessError, ValueError):
            return "linux"

    return sys.platform


def save_mean_similarity_plot(
    *,
    mean_similarity_by_iteration: tuple[float, ...],
    store_path: str,
    model_name: str,
    git_commit: str,
    cluster_build_seconds: float,
    num_clusters: int,
    computer_label: str,
    dtype: str,
) -> Path:
    plot_path = Path(store_path).with_name(
        f"{Path(store_path).stem}_mean_similarity.png"
    )
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    iterations = range(1, len(mean_similarity_by_iteration) + 1)
    num_iterations = len(mean_similarity_by_iteration)
    tick_step = max(1, num_iterations // 8)
    x_ticks = sorted(
        {
            1,
            num_iterations,
            *range(tick_step, num_iterations + 1, tick_step),
        }
    )

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(
        iterations,
        mean_similarity_by_iteration,
        color="#005278",
        marker="o",
        linewidth=1.5,
    )
    ax.set_title("Flashhead clustering", pad=22)
    ax.text(
        0.5,
        1.035,
        model_name,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=9,
    )
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Mean assigned similarity")
    ax.set_xticks(x_ticks)
    ax.grid(True, alpha=0.3)
    ax.text(
        0.99,
        0.01,
        (
            f"commit={git_commit[:7]}\n"
            f"clustering_time={cluster_build_seconds:.3f}s\n"
            f"clusters={num_clusters}\n"
            f"computer={computer_label}\n"
            f"dtype={dtype}"
        ),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        multialignment="left",
        linespacing=1.25,
        fontsize=8,
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "alpha": 0.85,
            "edgecolor": "none",
        },
    )
    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)

    return plot_path


def build_flashhead_head(store_path: str, model_name: str) -> None:
    git_commit = get_git_revision().commit
    num_clusters = 9496
    print(f"git_commit={git_commit}")

    loaded = load_flashhead_base(model_name)

    vocab_size, hidden_size = loaded.lm_head_vector_table.shape

    print(f"model_name={model_name}")
    print(f"vocab_size={vocab_size}")
    print(f"hidden_size={hidden_size}")
    print(f"dtype={loaded.lm_head_vector_table.dtype}")
    print(f"device={loaded.lm_head_vector_table.device}")

    print()
    print("first 3 token vectors:")
    print(loaded.lm_head_vector_table[:3, :8])

    print()
    print("building clusters...")

    mean_similarity_by_iteration: list[float] = []
    cluster_build_start_seconds = time.perf_counter()
    built_clusters = build_clusters(
        lm_head_vector_table=loaded.lm_head_vector_table,
        num_clusters=num_clusters,
        num_iters=40,
        normalize_vectors=True,
        seed=0,
        on_iteration_metrics=lambda _iteration, metrics: mean_similarity_by_iteration.append(
            metrics["mean_assigned_similarity"]
        ),
    )
    cluster_build_seconds = time.perf_counter() - cluster_build_start_seconds
    print(f"cluster_build_seconds={cluster_build_seconds:.3f}")

    plot_path = save_mean_similarity_plot(
        mean_similarity_by_iteration=tuple(mean_similarity_by_iteration),
        store_path=store_path,
        model_name=model_name,
        git_commit=git_commit,
        cluster_build_seconds=cluster_build_seconds,
        num_clusters=num_clusters,
        computer_label=get_computer_label(),
        dtype=str(loaded.lm_head_vector_table.dtype).removeprefix("torch."),
    )
    print(f"mean_similarity_plot_path={plot_path}")


    save_flashhead(
        path=store_path,
        cluster_to_token_ids=built_clusters.cluster_to_token_ids,
        centroids=built_clusters.centroids,
    )


def evaluate_flashhead(stored_path: str, model_name: str) -> None:
    loaded = load_flashhead_base(model_name)
    device = get_preferred_device()
    compute_dtype = get_preferred_float_dtype(device)
    model = cast(Any, loaded.model)

    model.to(device=device, dtype=compute_dtype)
    model.eval()

    flashhead = ANNHModule.from_model(
        model=model,
        flashhead_path=stored_path,
        top_k_clusters=500,
    )

    dataset_spec = DatasetSpec(
        name="Cosmopedia-100k",
        huggingface_path="HuggingFaceTB/cosmopedia-100k",
        config_name=None,  # or "default" if your loader requires a string
        split="train",
        text_field="text",
    )

    max_examples = 100
    context_len = 200
    num_windows_to_use = 30
    batch_size = 8

    dataloader = build_fixed_window_dataloader(
        dataset_spec=dataset_spec,
        model_and_tokenizer=ModelAndTokenizer(
            model=loaded.model,
            tokenizer=loaded.tokenizer,
        ),
        context_len=context_len,
        max_examples=max_examples,
        num_windows_to_use=num_windows_to_use,
        batch_size=batch_size,
        device=device,
        shuffle=False,
    )

    def iter_token_windows():
        for input_ids, _attention_mask in dataloader:
            for window_input_ids in input_ids:
                yield window_input_ids

    metrics = evaluate_topk_cluster_sweep_on_token_windows(
        model=loaded.model,
        flashhead=flashhead,
        token_windows=iter_token_windows(),
        max_windows=num_windows_to_use,
        max_positions_per_window=256,
        tokenizer=loaded.tokenizer,
        top1_match_rate_table_image_path=Path(stored_path).with_name(
            f"{Path(stored_path).stem}_topk_sweep_top1_match_rate.png"
        ),
        window_length=context_len,
    )

    print(metrics)
    plt.show()
