from __future__ import annotations

from typing import Any

import torch

from skip_search_spec.helpers.storage import load_early_exit_checkpoint
from skip_search_spec.helpers.tooling import get_preferred_device, get_preferred_float_dtype
from skip_search_spec.training.old.train_early_exit import EarlyExitModel


@torch.no_grad()
def draft_tokens(
    early_exit_model: EarlyExitModel,
    input_ids: torch.Tensor,
    draft_steps: int,
    eos_token_id: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Draft up to draft_steps tokens with the true early-exit path.

    Returns:
        drafted_ids: original sequence plus drafted suffix
        drafted_tokens: only the drafted suffix, shape [1, K]
    """
    drafted_ids = input_ids
    drafted_tokens_list: list[torch.Tensor] = []

    for _ in range(draft_steps):
        attention_mask = torch.ones_like(drafted_ids)
        logits_mid = early_exit_model.draft_logits(
            input_ids=drafted_ids,
            attention_mask=attention_mask,
        )

        next_token = logits_mid[:, -1, :].argmax(dim=-1, keepdim=True)
        drafted_ids = torch.cat([drafted_ids, next_token], dim=1)
        drafted_tokens_list.append(next_token)

        if eos_token_id is not None and int(next_token.item()) == eos_token_id:
            break

    if len(drafted_tokens_list) == 0:
        empty = torch.empty(
            (input_ids.shape[0], 0),
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        return drafted_ids, empty

    drafted_tokens = torch.cat(drafted_tokens_list, dim=1)
    return drafted_ids, drafted_tokens


@torch.no_grad()
def verify_draft(
    early_exit_model: EarlyExitModel,
    input_ids: torch.Tensor,
    drafted_ids: torch.Tensor,
    drafted_tokens: torch.Tensor,
    round_start_len: int,
) -> tuple[torch.Tensor, int]:
    """
    Verify drafted tokens with the full model.

    Returns:
        new_input_ids
        accepted_count
    """
    drafted_count = drafted_tokens.shape[1]

    attention_mask = torch.ones_like(drafted_ids)
    logits_full = early_exit_model.full_logits(
        input_ids=drafted_ids,
        attention_mask=attention_mask,
    )

    verify_logits = logits_full[
        :,
        round_start_len - 1 : round_start_len - 1 + drafted_count,
        :,
    ]
    full_pred_tokens = verify_logits.argmax(dim=-1)

    mismatch_index: int | None = None
    for j in range(drafted_count):
        if full_pred_tokens[0, j].item() != drafted_tokens[0, j].item():
            mismatch_index = j
            break

    if mismatch_index is None:
        return drafted_ids, drafted_count

    accepted_prefix = drafted_tokens[:, :mismatch_index]
    correction = full_pred_tokens[:, mismatch_index : mismatch_index + 1]

    parts = [input_ids]
    if accepted_prefix.shape[1] > 0:
        parts.append(accepted_prefix)
    parts.append(correction)

    new_input_ids = torch.cat(parts, dim=1)
    return new_input_ids, mismatch_index


@torch.no_grad()
def greedy_self_speculate(
    early_exit_model: EarlyExitModel,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = 64,
    draft_steps: int = 4,
) -> tuple[str, int, int]:
    """
    Minimal greedy self-speculative decoding.

    Notes:
    - batch size 1
    - exact token-match verification
    - no KV cache
    - no bonus token
    """
    device = next(early_exit_model.parameters()).device

    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    eos_token_id = tokenizer.eos_token_id

    prompt_len = input_ids.shape[1]
    accepted_tokens = 0
    proposed_tokens = 0

    early_exit_model.eval()

    while (input_ids.shape[1] - prompt_len) < max_new_tokens:
        round_start_len = input_ids.shape[1]
        remaining = max_new_tokens - (round_start_len - prompt_len)
        steps_this_round = min(draft_steps, remaining)

        drafted_ids, drafted_tokens = draft_tokens(
            early_exit_model=early_exit_model,
            input_ids=input_ids,
            draft_steps=steps_this_round,
            eos_token_id=eos_token_id,
        )

        drafted_count = drafted_tokens.shape[1]
        proposed_tokens += drafted_count

        if drafted_count == 0:
            break

        input_ids, accepted_this_round = verify_draft(
            early_exit_model=early_exit_model,
            input_ids=input_ids,
            drafted_ids=drafted_ids,
            drafted_tokens=drafted_tokens,
            round_start_len=round_start_len,
        )
        accepted_tokens += accepted_this_round

        last_token = int(input_ids[0, -1].item())
        if eos_token_id is not None and last_token == eos_token_id:
            break

    text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return text, accepted_tokens, proposed_tokens


@torch.no_grad()
def run_self_speculation(
    prompt: str = "Once upon a time there were",
    max_new_tokens: int = 200,
    draft_steps: int = 2,
    checkpoint_path: str = "checkpoints/early_exit_model.pt",
    print_output: bool = True,
    early_exit_model: EarlyExitModel | None = None,
    tokenizer: Any | None = None,
) -> tuple[str, int, int, float]:
    if max_new_tokens <= 0:
        raise ValueError(f"max_new_tokens must be > 0, got {max_new_tokens}")
    if draft_steps <= 0:
        raise ValueError(f"draft_steps must be > 0, got {draft_steps}")

    provided_model = early_exit_model is not None
    provided_tokenizer = tokenizer is not None
    if provided_model != provided_tokenizer:
        raise ValueError(
            "early_exit_model and tokenizer must either both be provided or both be omitted."
        )

    if early_exit_model is None or tokenizer is None:
        device = get_preferred_device()
        compute_dtype = get_preferred_float_dtype(device)

        early_exit_model, tokenizer = load_early_exit_checkpoint(
            checkpoint_path=checkpoint_path,
            device=device,
            compute_dtype=compute_dtype,
        )

    early_exit_model.eval()

    text, accepted_tokens, proposed_tokens = greedy_self_speculate(
        early_exit_model=early_exit_model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        draft_steps=draft_steps,
    )

    acceptance_rate = accepted_tokens / proposed_tokens if proposed_tokens > 0 else 0.0

    if print_output:
        print("PROMPT:")
        print(prompt)
        print()
        print("OUTPUT:")
        print(text)
        print()
        print(f"max_new_tokens={max_new_tokens}")
        print(f"draft_steps={draft_steps}")
        print(f"accepted_tokens={accepted_tokens}")
        print(f"proposed_tokens={proposed_tokens}")
        print(f"acceptance_rate={acceptance_rate:.4f}")

    return text, accepted_tokens, proposed_tokens, acceptance_rate


def benchmark_self_speculation(
    checkpoint_path: str = "checkpoints/early_exit_model15B.pt",
) -> None:
    device = get_preferred_device()
    compute_dtype = get_preferred_float_dtype(device)

    early_exit_model, tokenizer = load_early_exit_checkpoint(
        checkpoint_path=checkpoint_path,
        device=device,
        compute_dtype=compute_dtype,
    )
    early_exit_model.eval()

    cases: list[tuple[str, int, int]] = [
        ("Once upon a time there were", 120, 1),
        ("Once upon a time there were", 120, 2),
        ("Once upon a time there were", 120, 4),
        ("Hello, what is happening?", 80, 1),
        ("Hello, what is happening?", 80, 2),
        ("Hello, what is happening?", 80, 4),
        ("The little girl looked out the window and saw", 100, 2),
        ("Tom and Anna went to the park because", 100, 2),
    ]

    results: list[tuple[str, int, int, int, int, float]] = []

    print("STARTING SELF-SPEC BENCHMARK")
    print(f"num_cases={len(cases)}")
    print()

    for i, (prompt, max_new_tokens, draft_steps) in enumerate(cases, start=1):
        text, accepted_tokens, proposed_tokens, acceptance_rate = run_self_speculation(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            draft_steps=draft_steps,
            print_output=False,
            early_exit_model=early_exit_model,
            tokenizer=tokenizer,
        )

        results.append((
            prompt,
            max_new_tokens,
            draft_steps,
            accepted_tokens,
            proposed_tokens,
            acceptance_rate,
        ))

        print(f"[{i}/{len(cases)}]")
        print(f"prompt={prompt!r}")
        print(f"max_new_tokens={max_new_tokens}")
        print(f"draft_steps={draft_steps}")
        print(f"accepted_tokens={accepted_tokens}")
        print(f"proposed_tokens={proposed_tokens}")
        print(f"acceptance_rate={acceptance_rate:.4f}")
        print(f"output_preview={text[:160]!r}")
        print()

    total_accepted = sum(r[3] for r in results)
    total_proposed = sum(r[4] for r in results)

    macro_acceptance = sum(r[5] for r in results) / len(results) if results else 0.0
    micro_acceptance = total_accepted / total_proposed if total_proposed > 0 else 0.0

    print("SUMMARY")
    print(f"num_cases={len(results)}")
    print(f"macro_acceptance_rate={macro_acceptance:.4f}")
    print(f"micro_acceptance_rate={micro_acceptance:.4f}")
    print()

    for draft_steps in sorted({r[2] for r in results}):
        group = [r for r in results if r[2] == draft_steps]
        group_total_accepted = sum(r[3] for r in group)
        group_total_proposed = sum(r[4] for r in group)
        group_acceptance = (
            group_total_accepted / group_total_proposed
            if group_total_proposed > 0 else 0.0
        )

        print(
            f"draft_steps={draft_steps}  "
            f"cases={len(group)}  "
            f"acceptance_rate={group_acceptance:.4f}"
        )