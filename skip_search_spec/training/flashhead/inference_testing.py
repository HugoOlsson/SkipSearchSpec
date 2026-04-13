


from dataclasses import dataclass
from typing import cast

from torch import Tensor
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from skip_search_spec.training.flashhead.building_clusters import l2_normalize


@dataclass(frozen=True, slots=True)
class RouteAndRescoreResult:
    top_cluster_ids: Tensor
    top_cluster_scores: Tensor
    candidate_token_ids: Tensor
    candidate_scores: Tensor
    best_token_id: Tensor  # scalar tensor on device


def route_and_rescore_one_hidden_vector(
    hidden_vector: Tensor,
    centroids: Tensor,
    cluster_to_token_ids: Tensor,
    lm_head_vector_table: Tensor,
    *,
    top_k_clusters: int,
    normalize_hidden_for_routing: bool = True,
) -> RouteAndRescoreResult:
    if hidden_vector.ndim != 1:
        raise ValueError("hidden_vector must have shape [hidden_size]")

    if normalize_hidden_for_routing:
        route_hidden = l2_normalize(hidden_vector, dim=-1)
    else:
        route_hidden = hidden_vector

    cluster_scores = route_hidden @ centroids.transpose(0, 1)
    actual_top_k = min(top_k_clusters, int(centroids.shape[0]))

    top_cluster_scores, top_cluster_ids = torch.topk(cluster_scores, k=actual_top_k)

    selected_cluster_token_ids = cluster_to_token_ids[top_cluster_ids]
    candidate_token_ids = selected_cluster_token_ids.reshape(-1)

    candidate_token_vectors = lm_head_vector_table[candidate_token_ids]
    candidate_scores = candidate_token_vectors @ hidden_vector

    best_candidate_index = candidate_scores.argmax()
    best_token_id = candidate_token_ids[best_candidate_index]

    return RouteAndRescoreResult(
        top_cluster_ids=top_cluster_ids,
        top_cluster_scores=top_cluster_scores,
        candidate_token_ids=candidate_token_ids,
        candidate_scores=candidate_scores,
        best_token_id=best_token_id,
    )

@torch.inference_mode()
def get_last_hidden_vector_for_prompt(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
) -> Tensor:
    model_device = next(model.parameters()).device

    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = cast(Tensor, encoded["input_ids"]).to(model_device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = cast(Tensor, attention_mask).to(model_device)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )

    return outputs.hidden_states[-1][0, -1, :]

@torch.inference_mode()
def compare_dense_vs_routed_until_mismatch(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    lm_head_vector_table: Tensor,
    centroids: Tensor,
    cluster_to_token_ids: Tensor,
    *,
    prompt: str,
    top_k_clusters: int,
    max_new_tokens: int,
    normalize_hidden_for_routing: bool = True,
) -> None:
    model_device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype

    centroids = centroids.to(device=model_device, dtype=model_dtype)
    cluster_to_token_ids = cluster_to_token_ids.to(device=model_device)
    lm_head_vector_table = lm_head_vector_table.to(device=model_device, dtype=model_dtype)

    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = cast(Tensor, encoded["input_ids"]).to(model_device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = cast(Tensor, attention_mask).to(model_device)

    print()
    print(f"starting prompt={prompt!r}")
    print(f"top_k_clusters={top_k_clusters}")
    print(f"max_new_tokens={max_new_tokens}")

    past_key_values = None
    current_input_ids = input_ids
    current_attention_mask = attention_mask

    for step_idx in range(max_new_tokens):
        outputs = model(
            input_ids=current_input_ids,
            attention_mask=current_attention_mask,
            output_hidden_states=True,
            use_cache=True,
            past_key_values=past_key_values,
            return_dict=True,
        )

        past_key_values = outputs.past_key_values

        # Dense result comes directly from model logits.
        dense_logits = outputs.logits[0, -1, :]
        dense_best_token_id = int(dense_logits.argmax().item())

        # Routed result uses final hidden state of last token.
        hidden_vector = outputs.hidden_states[-1][0, -1, :]

        routed = route_and_rescore_one_hidden_vector(
            hidden_vector=hidden_vector,
            centroids=centroids,
            cluster_to_token_ids=cluster_to_token_ids,
            lm_head_vector_table=lm_head_vector_table,
            top_k_clusters=top_k_clusters,
            normalize_hidden_for_routing=normalize_hidden_for_routing,
        )
        routed_best_token_id = int(routed.best_token_id.item())

        dense_best_text = tokenizer.decode([dense_best_token_id])
        routed_best_text = tokenizer.decode([routed_best_token_id])

        if dense_best_token_id != routed_best_token_id:
            print()
            print(f"mismatch at generated step {step_idx + 1}")
            print(f"dense token_id={dense_best_token_id}, text={dense_best_text!r}")
            print(f"routed token_id={routed_best_token_id}, text={routed_best_text!r}")
            print(
                "dense_best_in_candidates="
                f"{bool((routed.candidate_token_ids == dense_best_token_id).any().item())}"
            )
            return

        print(f"step {step_idx + 1}: matched on {dense_best_text!r}")

        next_token = torch.tensor([[dense_best_token_id]], dtype=torch.long, device=model_device)

        # After the first step, only feed the new token.
        current_input_ids = next_token

        if current_attention_mask is not None:
            next_attention = torch.ones((1, 1), dtype=current_attention_mask.dtype, device=model_device)
            current_attention_mask = torch.cat([current_attention_mask, next_attention], dim=1)