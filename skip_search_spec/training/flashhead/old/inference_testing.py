from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from skip_search_spec.training.flashhead.building_clusters import l2_normalize


@dataclass(frozen=True, slots=True)
class RouteCandidatesResult:
    top_cluster_ids: Tensor
    top_cluster_scores: Tensor
    candidate_token_ids: Tensor


@dataclass(frozen=True, slots=True)
class RescoreResult:
    candidate_token_ids: Tensor
    candidate_scores: Tensor
    best_token_id: Tensor  # scalar tensor on device


def route_one_hidden_vector(
    hidden_vector: Tensor,
    centroids: Tensor,
    cluster_to_token_ids: Tensor,
    *,
    top_k_clusters: int,
    normalize_hidden_for_routing: bool = True,
) -> RouteCandidatesResult:
    if hidden_vector.ndim != 1:
        raise ValueError("hidden_vector must have shape [hidden_size]")

    if centroids.ndim != 2:
        raise ValueError("centroids must have shape [num_clusters, hidden_size]")

    if cluster_to_token_ids.ndim != 2:
        raise ValueError("cluster_to_token_ids must have shape [num_clusters, max_cluster_size]")

    if centroids.shape[0] != cluster_to_token_ids.shape[0]:
        raise ValueError("centroids and cluster_to_token_ids must agree on num_clusters")

    if normalize_hidden_for_routing:
        route_hidden = l2_normalize(hidden_vector, dim=-1)
    else:
        route_hidden = hidden_vector

    cluster_scores = route_hidden @ centroids.transpose(0, 1)
    actual_top_k = min(top_k_clusters, int(centroids.shape[0]))
    top_cluster_scores, top_cluster_ids = torch.topk(cluster_scores, k=actual_top_k)

    selected_cluster_token_ids = cluster_to_token_ids[top_cluster_ids].reshape(-1)

    # IMPORTANT:
    # cluster_to_token_ids is padded with -1, and -1 is valid indexing in PyTorch,
    # so we must remove padded entries before using these as token ids.
    candidate_token_ids = selected_cluster_token_ids[selected_cluster_token_ids >= 0]

    if candidate_token_ids.numel() == 0:
        raise RuntimeError("Routing produced zero valid candidate token ids")

    return RouteCandidatesResult(
        top_cluster_ids=top_cluster_ids,
        top_cluster_scores=top_cluster_scores,
        candidate_token_ids=candidate_token_ids,
    )


def rescore_candidate_token_ids_with_dense_logits(
    candidate_token_ids: Tensor,
    dense_logits: Tensor,
) -> RescoreResult:
    if candidate_token_ids.ndim != 1:
        raise ValueError("candidate_token_ids must have shape [num_candidates]")

    if dense_logits.ndim != 1:
        raise ValueError("dense_logits must have shape [vocab_size]")

    candidate_scores = dense_logits[candidate_token_ids]
    max_score = candidate_scores.max()
    tied_mask = candidate_scores == max_score
    best_token_id = candidate_token_ids[tied_mask].min()

    return RescoreResult(
        candidate_token_ids=candidate_token_ids,
        candidate_scores=candidate_scores,
        best_token_id=best_token_id,
    )


def rescore_candidate_token_ids_with_manual_lm_head(
    candidate_token_ids: Tensor,
    hidden_vector: Tensor,
    lm_head_vector_table: Tensor,
    *,
    lm_head_bias: Tensor | None = None,
) -> RescoreResult:
    """
    This is the fast path you would use for real approximate inference,
    assuming lm_head_vector_table is the exact raw lm-head weight matrix.

    For correctness testing against the dense model, prefer
    rescore_candidate_token_ids_with_dense_logits(...).
    """
    if candidate_token_ids.ndim != 1:
        raise ValueError("candidate_token_ids must have shape [num_candidates]")

    if hidden_vector.ndim != 1:
        raise ValueError("hidden_vector must have shape [hidden_size]")

    if lm_head_vector_table.ndim != 2:
        raise ValueError("lm_head_vector_table must have shape [vocab_size, hidden_size]")

    candidate_token_vectors = lm_head_vector_table[candidate_token_ids].float()
    candidate_scores = candidate_token_vectors @ hidden_vector.float()

    if lm_head_bias is not None:
        candidate_scores = candidate_scores + lm_head_bias[candidate_token_ids].float()

    best_candidate_index = candidate_scores.argmax()
    best_token_id = candidate_token_ids[best_candidate_index]

    return RescoreResult(
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
    """
    This comparison now does exactly what you described conceptually:

    1. Route hidden_vector into the routing matrix / centroids
    2. Select top-k clusters
    3. Gather all token ids from those clusters into one big candidate set
    4. Score only those candidates with the *true dense logits*

    Because step 4 uses dense_logits[candidate_token_ids], this has the property:
    if the dense-best token is in candidate_token_ids, routed must match dense.
    """
    model_device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype

    centroids = centroids.to(device=model_device, dtype=model_dtype)
    cluster_to_token_ids = cluster_to_token_ids.to(device=model_device)
    lm_head_vector_table = lm_head_vector_table.to(device=model_device, dtype=model_dtype)

    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = cast(Tensor, encoded["input_ids"]).to(model_device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = cast(Tensor, encoded["attention_mask"]).to(model_device)

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

        dense_logits = outputs.logits[0, -1, :]
        dense_best_token_id = int(dense_logits.argmax().item())

        hidden_vector = outputs.hidden_states[-1][0, -1, :]

        routed_candidates = route_one_hidden_vector(
            hidden_vector=hidden_vector,
            centroids=centroids,
            cluster_to_token_ids=cluster_to_token_ids,
            top_k_clusters=top_k_clusters,
            normalize_hidden_for_routing=normalize_hidden_for_routing,
        )

        rescored = rescore_candidate_token_ids_with_dense_logits(
            candidate_token_ids=routed_candidates.candidate_token_ids,
            dense_logits=dense_logits,
        )
        routed_best_token_id = int(rescored.best_token_id.item())

        dense_best_text = tokenizer.decode([dense_best_token_id])
        routed_best_text = tokenizer.decode([routed_best_token_id])

        if dense_best_token_id != routed_best_token_id:
            dense_best_in_candidates = bool(
                (rescored.candidate_token_ids == dense_best_token_id).any().item()
            )

            print()
            print(f"mismatch at generated step {step_idx + 1}")
            print(f"dense token_id={dense_best_token_id}, text={dense_best_text!r}")
            print(f"routed token_id={routed_best_token_id}, text={routed_best_text!r}")
            print(f"dense_best_in_candidates={dense_best_in_candidates}")

            # Extra debugging info:
            if dense_best_in_candidates:
                dense_pos = (rescored.candidate_token_ids == dense_best_token_id).nonzero(as_tuple=True)[0][0]
                routed_pos = (rescored.candidate_token_ids == routed_best_token_id).nonzero(as_tuple=True)[0][0]

                print(f"dense score for dense token = {float(rescored.candidate_scores[dense_pos].item())}")
                print(f"dense score for routed token = {float(rescored.candidate_scores[routed_pos].item())}")
                print("This should not happen with exact dense rescoring.")
            else:
                print("The dense winner was not present in the routed candidate set.")

            return

        print(f"step {step_idx + 1}: matched on {dense_best_text!r}")

        next_token = torch.tensor([[dense_best_token_id]], dtype=torch.long, device=model_device)
        current_input_ids = next_token

        if current_attention_mask is not None:
            next_attention = torch.ones((1, 1), dtype=current_attention_mask.dtype, device=model_device)
            current_attention_mask = torch.cat([current_attention_mask, next_attention], dim=1)