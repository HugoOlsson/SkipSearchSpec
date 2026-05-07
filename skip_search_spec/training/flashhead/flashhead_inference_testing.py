from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from skip_search_spec.training.flashhead.next_token_adapter import FlashHeadModule


@dataclass(frozen=True, slots=True)
class TopKContainmentMetrics:
    top_k_clusters: int
    num_positions: int
    top1_match_rate: float
    top3_containment: float


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
) -> TopKContainmentMetrics:
    model_device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype

    flashhead = flashhead.to(device=model_device, dtype=model_dtype)

    num_windows_used = 0
    num_positions = 0
    top1_hits = 0
    top3_hits = 0
    num_printed_mismatches = 0

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
