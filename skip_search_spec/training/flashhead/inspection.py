


from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from torch import Tensor
import torch
from transformers import PreTrainedTokenizerBase


def inspect_cluster_tokens(
    tokenizer: PreTrainedTokenizerBase,
    cluster_to_token_ids: Tensor,
    cluster_id: int,
    *,
    max_tokens: int = 30,
) -> None:
    token_ids = cluster_to_token_ids[cluster_id, :max_tokens].tolist()

    print(f"cluster_id={cluster_id}")
    for token_id in token_ids:
        text = tokenizer.decode([token_id])
        print(token_id, repr(text))



def visualize_cluster_sizes(cluster_sizes: Tensor) -> None:
    sizes = cluster_sizes.cpu().numpy()

    plt.figure(figsize=(8, 4))
    plt.hist(sizes, bins=100)
    plt.xlabel("Cluster size")
    plt.ylabel("Number of clusters")
    plt.title("Distribution of cluster sizes")
    plt.tight_layout()


def visualize_centroids_2d(centroids: Tensor) -> None:
    x = centroids.cpu().numpy()
    pca = PCA(n_components=2)
    xy = pca.fit_transform(x)

    plt.figure(figsize=(6, 6))
    plt.scatter(xy[:, 0], xy[:, 1], s=20)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Cluster centroids projected to 2D")
    plt.tight_layout()


def visualize_sampled_token_vectors_2d(
    lm_head_vector_table: Tensor,
    token_to_cluster_mapping: Tensor,
    *,
    num_points: int = 5000,
    seed: int = 0,
) -> None:
    vocab_size = lm_head_vector_table.shape[0]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    sample_ids = torch.randperm(vocab_size, generator=generator)[:num_points]
    sampled_vectors = lm_head_vector_table[sample_ids].cpu().numpy()
    sampled_clusters = token_to_cluster_mapping[sample_ids].cpu().numpy()

    pca = PCA(n_components=2)
    xy = pca.fit_transform(sampled_vectors)

    plt.figure(figsize=(7, 7))
    plt.scatter(xy[:, 0], xy[:, 1], c=sampled_clusters, s=4)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Sampled token vectors colored by cluster")
    plt.tight_layout()