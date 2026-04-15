from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from skip_search_spec.training.flashhead.inference_testing import (
    rescore_candidate_token_ids_with_dense_logits,
    route_one_hidden_vector,
)


@dataclass(frozen=True, slots=True)
class TopKContainmentMetrics:
    num_positions: int
    top1_containment: float
    top3_containment: float
    dense_winner_in_candidate_set_rate: float
    mean_candidate_count: float


@torch.inference_mode()
def evaluate_topk_containment_on_token_windows(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    centroids: Tensor,
    cluster_to_token_ids: Tensor,
    token_windows: Iterable[Tensor],
    *,
    top_k_clusters: int,
    max_windows: int | None = None,
    max_positions_per_window: int | None = None,
    normalize_hidden_for_routing: bool = True,
    print_every_windows: int | None = 10,
    print_first_n_mismatches: int = 5,
) -> TopKContainmentMetrics:
    model_device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype

    centroids = centroids.to(device=model_device, dtype=model_dtype)
    cluster_to_token_ids = cluster_to_token_ids.to(device=model_device)

    num_windows_used = 0
    num_positions = 0
    top1_hits = 0
    top3_hits = 0
    dense_winner_in_candidates = 0
    total_candidate_count = 0
    num_printed_mismatches = 0

    print("Starting top-k containment evaluation...")
    print(f"  top_k_clusters={top_k_clusters}")
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
                f"top3={top3_hits / num_positions:.6f}, "
                f"dense_in_candidates={dense_winner_in_candidates / num_positions:.6f}"
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
        dense_logits_all = outputs.logits[0, :-1, :]              # [seq_len - 1, vocab]
        hidden_states_all = outputs.hidden_states[-1][0, :-1, :]  # [seq_len - 1, hidden]

        num_eval_positions = dense_logits_all.shape[0]
        if max_positions_per_window is not None:
            num_eval_positions = min(num_eval_positions, max_positions_per_window)

        for pos in range(num_eval_positions):
            dense_logits = dense_logits_all[pos]
            hidden_vector = hidden_states_all[pos]

            routed_candidates = route_one_hidden_vector(
                hidden_vector=hidden_vector,
                centroids=centroids,
                cluster_to_token_ids=cluster_to_token_ids,
                top_k_clusters=top_k_clusters,
                normalize_hidden_for_routing=normalize_hidden_for_routing,
            )

            candidate_token_ids = routed_candidates.candidate_token_ids
            total_candidate_count += int(candidate_token_ids.numel())

            dense_top1_id = int(dense_logits.argmax().item())
            dense_top3_ids = torch.topk(dense_logits, k=3).indices

            rescored = rescore_candidate_token_ids_with_dense_logits(
                candidate_token_ids=candidate_token_ids,
                dense_logits=dense_logits,
            )
            routed_top1_id = int(rescored.best_token_id.item())

            top1_match = routed_top1_id == dense_top1_id
            top3_match = bool((dense_top3_ids == routed_top1_id).any().item())
            dense_in_candidates = bool((candidate_token_ids == dense_top1_id).any().item())

            if top1_match:
                top1_hits += 1

            if top3_match:
                top3_hits += 1

            if dense_in_candidates:
                dense_winner_in_candidates += 1

            if (not top1_match) and num_printed_mismatches < print_first_n_mismatches:
                dense_top1_text = tokenizer.decode([dense_top1_id])
                routed_top1_text = tokenizer.decode([routed_top1_id])

                print(
                    f"  mismatch #{num_printed_mismatches + 1}: "
                    f"window={num_windows_used}, pos={pos}, "
                    f"dense_top1_id={dense_top1_id}, dense_top1_text={dense_top1_text!r}, "
                    f"routed_top1_id={routed_top1_id}, routed_top1_text={routed_top1_text!r}, "
                    f"dense_in_candidates={dense_in_candidates}, "
                    f"candidate_count={int(candidate_token_ids.numel())}"
                )
                num_printed_mismatches += 1

            num_positions += 1

        num_windows_used += 1

    if num_positions == 0:
        raise RuntimeError("No evaluation positions were collected.")

    metrics = TopKContainmentMetrics(
        num_positions=num_positions,
        top1_containment=top1_hits / num_positions,
        top3_containment=top3_hits / num_positions,
        dense_winner_in_candidate_set_rate=dense_winner_in_candidates / num_positions,
        mean_candidate_count=total_candidate_count / num_positions,
    )

    print()
    print("Finished top-k containment evaluation.")
    print(f"  num_windows_used={num_windows_used}")
    print(f"  num_positions={metrics.num_positions}")
    print(f"  top1_containment={metrics.top1_containment:.6f}")
    print(f"  top3_containment={metrics.top3_containment:.6f}")
    print(f"  dense_winner_in_candidate_set_rate={metrics.dense_winner_in_candidate_set_rate:.6f}")
    print(f"  mean_candidate_count={metrics.mean_candidate_count:.2f}")

    return metrics