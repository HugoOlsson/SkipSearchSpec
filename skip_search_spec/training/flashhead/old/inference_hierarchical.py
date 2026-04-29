from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, cast

import torch
from torch import Tensor
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from skip_search_spec.training.flashhead.old.building_hierarchical_clusters import (
    BuiltHierarchicalFlashHead,
    l2_normalize,
)


@dataclass(frozen=True, slots=True)
class HierarchicalRouteResult:
    candidate_token_ids: Tensor
    candidate_scores: Tensor
    best_token_id: Tensor


def normalize_paths_per_level(
    paths_per_level: int | Sequence[int],
    *,
    num_levels: int,
) -> tuple[int, ...]:
    if isinstance(paths_per_level, int):
        if paths_per_level < 1:
            raise ValueError("paths_per_level must be >= 1")
        return (paths_per_level,) * num_levels

    normalized = tuple(int(x) for x in paths_per_level)

    if len(normalized) != num_levels:
        raise ValueError(
            f"paths_per_level must have length {num_levels} for this hierarchy, "
            f"but got length {len(normalized)}"
        )

    if any(x < 1 for x in normalized):
        raise ValueError("all entries in paths_per_level must be >= 1")

    return normalized


def choose_best_token_id_with_dense_tie_break(
    candidate_token_ids: Tensor,
    candidate_scores: Tensor,
) -> Tensor:
    """
    Match dense argmax tie-breaking behavior:
    among all candidate tokens with the maximum score,
    pick the smallest token id.
    """
    if candidate_token_ids.ndim != 1:
        raise ValueError("candidate_token_ids must have shape [num_candidates]")

    if candidate_scores.ndim != 1:
        raise ValueError("candidate_scores must have shape [num_candidates]")

    if candidate_token_ids.shape[0] != candidate_scores.shape[0]:
        raise ValueError("candidate_token_ids and candidate_scores must have the same length")

    max_score = candidate_scores.max()
    tied_mask = candidate_scores == max_score
    best_token_id = candidate_token_ids[tied_mask].min()
    return best_token_id


@torch.no_grad()
def route_and_rescore_one_hidden_vector(
    hidden_vector: Tensor,
    hierarchical_head: BuiltHierarchicalFlashHead,
    lm_head_vector_table: Tensor,
    *,
    paths_per_level: int | Sequence[int],
    normalize_hidden_for_routing: bool = True,
    dense_logits: Tensor | None = None,
) -> HierarchicalRouteResult:
    if hidden_vector.ndim != 1:
        raise ValueError("hidden_vector must have shape [hidden_size]")

    num_levels = len(hierarchical_head.levels)
    if num_levels == 0:
        raise ValueError("hierarchical_head.levels must not be empty")

    path_budget = normalize_paths_per_level(
        paths_per_level,
        num_levels=num_levels,
    )

    if normalize_hidden_for_routing:
        route_hidden = l2_normalize(hidden_vector, dim=-1)
    else:
        route_hidden = hidden_vector

    # Step 1: top router matrix -> highest level node ids.
    router_scores = route_hidden @ hierarchical_head.router_matrix.transpose(0, 1)
    current_ids = torch.topk(
        router_scores,
        k=min(path_budget[0], int(hierarchical_head.router_matrix.shape[0])),
    ).indices

    # Step 2: descend the hierarchy.
    for descent_idx, level_idx in enumerate(range(num_levels - 1, -1, -1)):
        level = hierarchical_head.levels[level_idx]

        child_ids = level.parent_to_child_ids[current_ids].reshape(-1)
        child_ids = child_ids[child_ids >= 0]

        if child_ids.numel() == 0:
            raise RuntimeError("Hierarchy descent produced zero valid child ids")

        if level_idx == 0:
            candidate_token_ids = child_ids

            if dense_logits is not None:
                candidate_scores = dense_logits[candidate_token_ids]
            else:
                candidate_token_vectors = lm_head_vector_table[candidate_token_ids]
                candidate_scores = candidate_token_vectors @ hidden_vector

            best_token_id = choose_best_token_id_with_dense_tie_break(
                candidate_token_ids=candidate_token_ids,
                candidate_scores=candidate_scores,
            )

            return HierarchicalRouteResult(
                candidate_token_ids=candidate_token_ids,
                candidate_scores=candidate_scores,
                best_token_id=best_token_id,
            )

        lower_level = hierarchical_head.levels[level_idx - 1]
        lower_vectors = lower_level.exposed_vectors[child_ids]
        lower_scores = route_hidden @ lower_vectors.transpose(0, 1)

        next_budget = path_budget[descent_idx + 1]
        keep = min(next_budget, int(child_ids.numel()))
        current_ids = child_ids[torch.topk(lower_scores, k=keep).indices]

    raise RuntimeError("Hierarchy descent failed")


@torch.inference_mode()
def compare_dense_vs_hierarchical_until_mismatch(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    lm_head_vector_table: Tensor,
    hierarchical_head: BuiltHierarchicalFlashHead,
    *,
    prompt: str,
    paths_per_level: int | Sequence[int],
    max_new_tokens: int,
    normalize_hidden_for_routing: bool = True,
) -> None:
    model_device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype

    lm_head_vector_table = lm_head_vector_table.to(device=model_device, dtype=model_dtype)

    hierarchical_head = BuiltHierarchicalFlashHead(
        levels=tuple(
            type(level)(
                child_to_parent=level.child_to_parent.to(model_device),
                parent_to_child_ids=level.parent_to_child_ids.to(model_device),
                parent_sizes=level.parent_sizes.to(model_device),
                exposed_vectors=level.exposed_vectors.to(device=model_device, dtype=model_dtype),
            )
            for level in hierarchical_head.levels
        ),
        router_matrix=hierarchical_head.router_matrix.to(device=model_device, dtype=model_dtype),
    )

    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = cast(Tensor, encoded["input_ids"]).to(model_device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = cast(Tensor, attention_mask).to(model_device)

    print()
    print(f"prompt={prompt!r}")
    print(f"paths_per_level={paths_per_level}")
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

        routed = route_and_rescore_one_hidden_vector(
            hidden_vector=hidden_vector,
            hierarchical_head=hierarchical_head,
            lm_head_vector_table=lm_head_vector_table,
            paths_per_level=paths_per_level,
            normalize_hidden_for_routing=normalize_hidden_for_routing,
            dense_logits=dense_logits,
        )
        routed_best_token_id = int(routed.best_token_id.item())

        dense_text = tokenizer.decode([dense_best_token_id])
        routed_text = tokenizer.decode([routed_best_token_id])

        if dense_best_token_id != routed_best_token_id:
            print()
            print(f"mismatch at generated step {step_idx + 1}")
            print(f"dense : token_id={dense_best_token_id}, text={dense_text!r}")
            print(f"routed: token_id={routed_best_token_id}, text={routed_text!r}")
            print(
                "dense_best_in_candidates="
                f"{bool((routed.candidate_token_ids == dense_best_token_id).any().item())}"
            )
            print(f"num_candidate_tokens={int(routed.candidate_token_ids.numel())}")

            if bool((routed.candidate_token_ids == dense_best_token_id).any().item()):
                dense_pos = (routed.candidate_token_ids == dense_best_token_id).nonzero(as_tuple=True)[0][0]
                routed_pos = (routed.candidate_token_ids == routed_best_token_id).nonzero(as_tuple=True)[0][0]

                print(f"dense score for dense token = {float(routed.candidate_scores[dense_pos].item())}")
                print(f"dense score for routed token = {float(routed.candidate_scores[routed_pos].item())}")

            return

        print(f"step {step_idx + 1}: matched on {dense_text!r}")

        next_token = torch.tensor([[dense_best_token_id]], dtype=torch.long, device=model_device)
        current_input_ids = next_token

        if current_attention_mask is not None:
            next_attention = torch.ones((1, 1), dtype=current_attention_mask.dtype, device=model_device)
            current_attention_mask = torch.cat([current_attention_mask, next_attention], dim=1)