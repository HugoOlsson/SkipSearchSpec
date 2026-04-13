from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from transformers import PreTrainedModel, PreTrainedTokenizerBase
import matplotlib.pyplot as plt
from skip_search_spec.helpers.tooling import load_model_and_tokenizer
from sklearn.decomposition import PCA

from skip_search_spec.training.flashhead.building_clusters import build_clusters, build_equal_size_token_to_cluster_mapping
from skip_search_spec.training.flashhead.inference_testing import compare_dense_vs_routed_until_mismatch
from skip_search_spec.training.flashhead.inspection import inspect_cluster_tokens, visualize_centroids_2d, visualize_cluster_sizes, visualize_sampled_token_vectors_2d
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
        loaded.lm_head_vector_table,
        num_clusters=2374,
        num_iters=50,
        normalize_vectors=True,
        seed=0,
        assignment_chunk_size=4096,
    )

    print()
    print("finished building clusters")

    print()
    print(f"token_to_cluster_mapping shape: {built_clusters.token_to_cluster_mapping.shape}")
    print(f"centroids shape: {built_clusters.centroids.shape}")
    print(f"cluster_sizes shape: {built_clusters.cluster_sizes.shape}")

    print()
    print("first 20 token -> cluster assignments:")
    print(built_clusters.token_to_cluster_mapping[:20])

    print()
    print("first 10 cluster sizes:")
    print(built_clusters.cluster_sizes[:10])

    print()
    print("largest cluster size:")
    print(int(built_clusters.cluster_sizes.max().item()))

    print()
    print("smallest cluster size:")
    print(int(built_clusters.cluster_sizes.min().item()))

    print()
    print("first 3 centroid vectors:")
    print(built_clusters.centroids[:3, :8])

    print()
    print(f"cluster_to_token_ids shape: {built_clusters.cluster_to_token_ids.shape}")

    print()
    print("first cluster, first 20 token ids:")
    print(built_clusters.cluster_to_token_ids[0, :20])

    save_flashhead(
        path=store_path,
        token_to_cluster_mapping=built_clusters.token_to_cluster_mapping,
        cluster_to_token_ids=built_clusters.cluster_to_token_ids,
        centroids=built_clusters.centroids,
        cluster_sizes=built_clusters.cluster_sizes,
    )

   

def evaluate_flashhead(stored_path:str, model_name: str):
    loaded = load_flashhead_base(model_name)
    stored = load_flashhead(stored_path)

    inspect_cluster_tokens(
        loaded.tokenizer,
        stored.cluster_to_token_ids,
        cluster_id=0,
        max_tokens=30,
    )

    compare_dense_vs_routed_until_mismatch(
        model=loaded.model,
        tokenizer=loaded.tokenizer,
        lm_head_vector_table=loaded.lm_head_vector_table,
        centroids=stored.centroids,
        cluster_to_token_ids=stored.cluster_to_token_ids,
        prompt="Who were the last 100 presedents of the United States?",
        top_k_clusters=300,
        max_new_tokens=5000,
        normalize_hidden_for_routing=True,
    )

    visualize_cluster_sizes(stored.cluster_sizes)
    visualize_centroids_2d(stored.centroids)
    visualize_sampled_token_vectors_2d(
        loaded.lm_head_vector_table,
        stored.token_to_cluster_mapping,
    )
    plt.show()
