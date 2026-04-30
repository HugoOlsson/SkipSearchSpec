from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor
from transformers import PreTrainedModel, PreTrainedTokenizerBase
import matplotlib.pyplot as plt
from skip_search_spec.helpers.tooling import load_model_and_tokenizer
from sklearn.decomposition import PCA

from skip_search_spec.helpers.shared_decoding_tools import build_fixed_window_dataloader
from skip_search_spec.protocols.windows import DatasetSpec, ModelAndTokenizer
from skip_search_spec.training.flashhead.building_clusters import build_clusters
from skip_search_spec.training.flashhead.old.inference_testing import compare_dense_vs_routed_until_mismatch
from skip_search_spec.training.flashhead.flashhead_inference_testing import evaluate_topk_containment_on_token_windows
from skip_search_spec.training.flashhead.old.inspection import inspect_cluster_tokens, visualize_centroids_2d, visualize_cluster_sizes, visualize_sampled_token_vectors_2d
from skip_search_spec.training.flashhead.storage import load_flashhead, save_flashhead


@dataclass(frozen=True, slots=True)
class LoadedFlashHeadBase:
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    lm_head_vector_table: Tensor


def extract_lm_head_vector_table(model: PreTrainedModel) -> Tensor:
    output_embeddings = model.get_output_embeddings()
    if output_embeddings is None:
        raise ValueError("Model does not expose output embeddings via get_output_embeddings()")

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

def build_flashhead_head(store_path: str, model_name: str) -> None:
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

    built_clusters = build_clusters(
        lm_head_vector_table=loaded.lm_head_vector_table,
        num_clusters=8000,
        num_iters=30,
        normalize_vectors=True,
        seed=0,
    )

    print()
    print("largest cluster size:")
    print(int(built_clusters.cluster_sizes.max().item()))

    print()
    print("smallest cluster size:")
    print(int(built_clusters.cluster_sizes.min().item()))

    save_flashhead(
        path=store_path,
        token_to_cluster_mapping=built_clusters.token_to_cluster_mapping,
        cluster_to_token_ids=built_clusters.cluster_to_token_ids,
        centroids=built_clusters.centroids,
        cluster_sizes=built_clusters.cluster_sizes,
    )

   

def evaluate_flashhead(stored_path: str, model_name: str) -> None:
    loaded = load_flashhead_base(model_name)
    stored = load_flashhead(stored_path)

    dataset_spec = DatasetSpec(
        name="FineWeb-Edu-1B",
        huggingface_path="codelion/fineweb-edu-1B",
        config_name="default",
        split="train",
        text_field="text",
    )

    max_examples = 100
    context_len = 200
    num_windows_to_use = 20
    batch_size = 8
    model_device = next(loaded.model.parameters()).device

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
        device=model_device,
        shuffle=False,
    )

    def iter_token_windows():
        for input_ids, _attention_mask in dataloader:
            for window_input_ids in input_ids:
                yield window_input_ids

    metrics = evaluate_topk_containment_on_token_windows(
        model=loaded.model,
        centroids=stored.centroids,
        cluster_to_token_ids=stored.cluster_to_token_ids,
        token_windows=iter_token_windows(),
        top_k_clusters=500,
        max_windows=num_windows_to_use,
        max_positions_per_window=256,
        normalize_hidden_for_routing=True,
        tokenizer=loaded.tokenizer,
    )

    print(metrics)
    plt.show()
