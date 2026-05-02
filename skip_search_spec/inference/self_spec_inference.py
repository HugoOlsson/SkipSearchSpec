from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Any, cast

import torch

from skip_search_spec.training.bridged_gap_model import BridgedGapModel


@dataclass(slots=True)
class SelfSpecResult:
    text: str
    output_ids: torch.Tensor
    verifier_calls: int
    drafted_tokens: int
    accepted_draft_tokens: int
    trace_json_path: Path | None = None


@dataclass(frozen=True, slots=True)
class TokenData:
    token_id: int
    type: str  # "prompt", "draft", "bonus"
    status: str | None = None  # "accepted" / "rejected" only for draft tokens
    draft_block_index: int | None = None


class BridgeSelfSpeculator:
    """
    Minimal greedy self-speculation using BridgedGapModel.

    This class only owns the speculative decoding loop.

    BridgedGapModel owns:
      - model loading
      - bridge loading
      - skipped-layer hooks
      - verifier hidden capture
      - drafter bridge injection
      - reference hidden source, for example "reentry" vs "final"
    """

    def __init__(
        self,
        *,
        bridged_model: BridgedGapModel,
    ) -> None:
        self.bridged = bridged_model
        self.model = bridged_model.model
        self.tokenizer = bridged_model.tokenizer
        self.device = bridged_model.device

        self.bridged.eval_all()

    @torch.inference_mode()
    def generate(
        self,
        *,
        prompt: str,
        max_new_tokens: int,
        draft_block_size: int = 4,
        stop_on_eos: bool = True,
        use_chat_template: bool = True,
        enable_thinking: bool = False,
        trace_json_path: str | Path | None = None,
    ) -> SelfSpecResult:
        model_prompt = prompt
        if use_chat_template:
            model_prompt = format_user_chat_prompt(
                tokenizer=self.tokenizer,
                prompt=prompt,
                enable_thinking=enable_thinking,
            )

        encoded_prompt = self.tokenizer(
            model_prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )

        input_ids = cast(torch.Tensor, encoded_prompt["input_ids"]).to(self.device)
        prompt_len = input_ids.size(1)

        # START: FOR VISUALIZATION
        token_trace: list[TokenData] = [
            TokenData(
                token_id=token_id,
                type="prompt",
                status=None,
                draft_block_index=None,
            )
            for token_id in _token_ids_1d(input_ids)
        ]

        draft_block_index = 0
        # END: FOR VISUALIZATION

        if input_ids.size(0) != 1:
            raise ValueError("This minimal self-spec test only supports batch size 1.")


        verifier_calls = 0
        drafted_tokens = 0
        accepted_draft_tokens = 0
        generated_tokens = 0

        # 1. Verify the prompt once to create the base.
        verifier = self.bridged.run_verifier(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
        )
        verifier_calls += 1

        # KV-CACHE HANDLING START
        verifier_past_key_values: Any | None = verifier.past_key_values
        if verifier_past_key_values is None:
            raise RuntimeError("Verifier did not return past_key_values.")

        verifier_cache_len = input_ids.size(1)
        # KV-CACHE HANDLING END

        bonus_token = verifier.logits[:, -1, :].argmax(
            dim=-1,
            keepdim=True,
        )

        accepted_ids = torch.cat([input_ids, bonus_token], dim=1)
        verifier_reference_hidden = verifier.reference_hidden
        generated_tokens += 1

        token_trace.append(
            TokenData(
                token_id=_one_token_id(bonus_token),
                type="bonus",
                status=None,
                draft_block_index=None,
            )
        )

        while generated_tokens < max_new_tokens:

            # 2. Draft from the current accepted prefix.
            remaining_tokens = max_new_tokens - generated_tokens
            block_size = min(draft_block_size, remaining_tokens)

            draft_tokens = self._draft_block(
                accepted_ids=accepted_ids,
                verifier_reference_hidden=verifier_reference_hidden,
                verifier_past_key_values=verifier_past_key_values,
                verifier_cache_len=verifier_cache_len,
                block_size=block_size,
            )

            drafted_tokens += draft_tokens.size(1)

            accepted_len_before_draft = accepted_ids.size(1)
            candidate_ids = torch.cat([accepted_ids, draft_tokens], dim=1)

            # 3. Run verifier on the suffix after the cached prefix.
            # KV-CACHE HANDLING START
            verifier_input_start = verifier_cache_len
            verifier_input_ids = candidate_ids[:, verifier_input_start:]
            verifier = self.bridged.run_verifier(
                input_ids=verifier_input_ids,
                attention_mask=torch.ones_like(candidate_ids),
                past_key_values=verifier_past_key_values,
            )
            # KV-CACHE HANDLING END
            verifier_calls += 1

            # KV-CACHE HANDLING START
            verifier_reference_hidden_full = torch.cat(
                [
                    verifier_reference_hidden[:, :verifier_input_start, :],
                    verifier.reference_hidden,
                ],
                dim=1,
            )
            # KV-CACHE HANDLING END

            # 4. Compare verifier predictions against the draft.
            #
            # The first drafted token is checked by the verifier logits at the
            # previous accepted token position.
            # KV-CACHE HANDLING START
            verifier_logits_start = accepted_len_before_draft - 1 - verifier_input_start
            # KV-CACHE HANDLING END
            verifier_draft_tokens = verifier.logits[
                :,
                verifier_logits_start : verifier_logits_start + draft_tokens.size(1),
                :,
            ].argmax(dim=-1)

            matches = verifier_draft_tokens == draft_tokens
            mismatch_positions = (~matches[0]).nonzero(as_tuple=False)

            # 5. Stop where draft and verifier no longer match.
            if mismatch_positions.numel() == 0:
                num_accepted = draft_tokens.size(1)
            else:
                num_accepted = int(mismatch_positions[0, 0].item())

            draft_block_index += 1

            for token_offset, token_id in enumerate(_token_ids_1d(draft_tokens)):
                token_trace.append(
                    TokenData(
                        token_id=token_id,
                        type="draft",
                        status="accepted" if token_offset < num_accepted else "rejected",
                        draft_block_index=draft_block_index,
                    )
                )

            if num_accepted > 0:
                accepted_ids = torch.cat(
                    [accepted_ids, draft_tokens[:, :num_accepted]],
                    dim=1,
                )
                accepted_draft_tokens += num_accepted
                generated_tokens += num_accepted

                if generated_tokens >= max_new_tokens:
                    break

            # 6. Add the verifier token.
            #
            # If there was a mismatch, this is the verifier correction token.
            # If all draft tokens matched, this is the verifier bonus token.
            if num_accepted < draft_tokens.size(1):
                verifier_token = verifier_draft_tokens[
                    :,
                    num_accepted : num_accepted + 1,
                ]

                # The next loop needs verifier reference hidden for the prefix
                # before this verifier-produced token.
                next_reference_len = accepted_len_before_draft + num_accepted
                verifier_reference_hidden = verifier_reference_hidden_full[
                    :,
                    :next_reference_len,
                    :,
                ]

                # KV-CACHE HANDLING START
                verifier_past_key_values = crop_past_key_values(
                    verifier.past_key_values,
                    max_length=next_reference_len,
                )
                verifier_cache_len = next_reference_len
                # KV-CACHE HANDLING END
            else:
                verifier_token = verifier.logits[:, -1, :].argmax(
                    dim=-1,
                    keepdim=True,
                )

                # The bonus token is predicted from the full candidate prefix.
                verifier_reference_hidden = verifier_reference_hidden_full
                # KV-CACHE HANDLING START
                verifier_past_key_values = verifier.past_key_values
                verifier_cache_len = candidate_ids.size(1)
                # KV-CACHE HANDLING END

            token_trace.append(
                TokenData(
                    token_id=_one_token_id(verifier_token),
                    type="bonus",
                    status=None,
                    draft_block_index=None,
                )
            )

            accepted_ids = torch.cat([accepted_ids, verifier_token], dim=1)
            generated_tokens += 1

            # Chat templates can contain EOS/end-of-turn tokens in the prompt.
            # Only generated tokens should trigger stopping.
            if stop_on_eos:
                eos_token_id = self.tokenizer.eos_token_id
                if isinstance(eos_token_id, int):
                    generated_suffix = accepted_ids[0, prompt_len:]
                    eos_hits = (generated_suffix == eos_token_id).nonzero(as_tuple=False)
                    if eos_hits.numel() > 0:
                        first_eos = prompt_len + int(eos_hits[0, 0].item())
                        accepted_ids = accepted_ids[:, : first_eos + 1]
                        break

        text = self.tokenizer.decode(
            accepted_ids[0],
            skip_special_tokens=True,
        )

        saved_trace_json_path: Path | None = None

        if trace_json_path is not None:
            saved_trace_json_path = save_token_trace_json(
                tokens=token_trace,
                path=trace_json_path,
                tokenizer_name=self.bridged.config.model_name,
            )

        return SelfSpecResult(
            text=text,
            output_ids=accepted_ids.detach().cpu(),
            verifier_calls=verifier_calls,
            drafted_tokens=drafted_tokens,
            accepted_draft_tokens=accepted_draft_tokens,
            trace_json_path=saved_trace_json_path
        )
    

    @torch.inference_mode()
    def _draft_block(
        self,
        *,
        accepted_ids: torch.Tensor,
        verifier_reference_hidden: torch.Tensor,
        verifier_past_key_values: Any | None,
        verifier_cache_len: int,
        block_size: int,
    ) -> torch.Tensor:
        """
        Design B drafter block.

        accepted_ids already includes the verifier-produced next token.

        If the verifier ran on:

            [tok0, tok1, tok2, tok3, tok4]

        and produced:

            tok5

        then accepted_ids is:

            [tok0, tok1, tok2, tok3, tok4, tok5]

        while verifier_reference_hidden covers only positions:

            [0, 1, 2, 3, 4]

        Therefore:

            len(accepted_ids) == len(verifier_reference_hidden) + 1

        The bridge prev-reference tensor becomes:

            [zero, ver0, ver1, ver2, ver3, ver4]
        """
        if accepted_ids.size(0) != 1:
            raise ValueError("This minimal self-spec test only supports batch size 1.")

        accepted_len = accepted_ids.size(1)
        verifier_len = verifier_reference_hidden.size(1)

        if accepted_len != verifier_len + 1:
            raise ValueError(
                f"Design B expects accepted_ids to include one verifier-produced token. "
                f"Got accepted_len={accepted_len}, verifier_len={verifier_len}."
            )

        prev_reference_hidden = verifier_reference_hidden.new_zeros(
            verifier_reference_hidden.size(0),
            accepted_len,
            verifier_reference_hidden.size(2),
        )
        prev_reference_hidden[:, 1:, :] = verifier_reference_hidden

        draft_tokens: list[torch.Tensor] = []

        # KV-CACHE HANDLING START
        if verifier_past_key_values is None:
            raise RuntimeError("Verifier KV-cache is required for cached self-spec drafting.")

        if verifier_cache_len != verifier_len:
            raise ValueError(
                f"Expected verifier cache to match verifier reference hidden length. "
                f"Got verifier_cache_len={verifier_cache_len}, "
                f"verifier_len={verifier_len}."
            )

        current_ids = accepted_ids[:, verifier_cache_len:]
        current_prev_reference_hidden = prev_reference_hidden[:, verifier_cache_len:, :]
        current_cache_len = verifier_cache_len
        # KV-CACHE HANDLING END

        try:
            for _ in range(block_size):
                # KV-CACHE HANDLING START
                drafter = self.bridged.run_drafter(
                    input_ids=current_ids,
                    attention_mask=torch.ones(
                        current_ids.size(0),
                        current_cache_len + current_ids.size(1),
                        device=current_ids.device,
                        dtype=current_ids.dtype,
                    ),
                    prev_reference_hidden=current_prev_reference_hidden,
                    past_key_values=verifier_past_key_values,
                    use_cache=True,
                )
                # KV-CACHE HANDLING END

                next_token = drafter.logits[:, -1, :].argmax(
                    dim=-1,
                    keepdim=True,
                )

                draft_tokens.append(next_token)

                # KV-CACHE HANDLING START
                current_cache_len += current_ids.size(1)
                # KV-CACHE HANDLING END
                current_ids = next_token
                current_prev_reference_hidden = drafter.reference_hidden[
                    :,
                    -1:,
                    :,
                ].detach()
        finally:
            # KV-CACHE HANDLING START
            crop_past_key_values(
                verifier_past_key_values,
                max_length=verifier_cache_len,
            )
            # KV-CACHE HANDLING END

        return torch.cat(draft_tokens, dim=1)
    
    
def load_bridge_self_speculator(
    *,
    bridge_checkpoint_path: str | Path,
) -> BridgeSelfSpeculator:
    bridged = BridgedGapModel.load_from_checkpoint(
        bridge_checkpoint_path=bridge_checkpoint_path,
    )

    return BridgeSelfSpeculator(
        bridged_model=bridged,
    )


def self_spec_inference_test(
    *,
    bridge_checkpoint_path: str | Path,
    prompt: str,
    max_new_tokens: int,
    draft_block_size: int = 4,
    use_chat_template: bool = True,
    enable_thinking: bool = False,
) -> SelfSpecResult:
    speculator = load_bridge_self_speculator(
        bridge_checkpoint_path=bridge_checkpoint_path,
    )

    trace_json_path = make_trace_json_path(
        bridge_checkpoint_path=bridge_checkpoint_path,
        draft_block_size=draft_block_size,
        max_new_tokens=max_new_tokens,
    )

    return speculator.generate(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        draft_block_size=draft_block_size,
        use_chat_template=use_chat_template,
        enable_thinking=enable_thinking,
        trace_json_path=trace_json_path
    )


def format_user_chat_prompt(
    *,
    tokenizer: object,
    prompt: str,
    enable_thinking: bool = False,
) -> str:
    return cast(
        str,
        tokenizer.apply_chat_template(  # type: ignore[attr-defined]
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        ),
    )


def _token_ids_1d(tokens: torch.Tensor) -> list[int]:
    return [int(x) for x in tokens.detach().cpu().reshape(-1).tolist()]


def _one_token_id(token: torch.Tensor) -> int:
    ids = _token_ids_1d(token)
    if len(ids) != 1:
        raise ValueError(f"Expected one token, got {len(ids)}.")
    return ids[0]


# KV-CACHE HANDLING START
def crop_past_key_values(
    past_key_values: Any | None,
    *,
    max_length: int,
) -> Any | None:
    if past_key_values is None:
        return None

    crop = getattr(past_key_values, "crop", None)
    if callable(crop):
        crop(max_length)
        return past_key_values

    raise TypeError(
        f"Expected past_key_values to expose a callable crop(max_length), "
        f"got {type(past_key_values).__name__}."
    )
# KV-CACHE HANDLING END


def save_token_trace_json(
    *,
    tokens: list[TokenData],
    path: str | Path,
    tokenizer_name: str,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "schema_version": 1,
        "tokenizer_name": tokenizer_name,
        "tokens": [asdict(token) for token in tokens],
    }

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print("Saved speculation result at:", path)

    return path


def make_trace_json_path(
    *,
    bridge_checkpoint_path: str | Path,
    draft_block_size: int,
    max_new_tokens: int,
) -> Path:
    checkpoint_stem = Path(bridge_checkpoint_path).stem
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    return Path("speculation_traces") / (
        f"self_spec_trace__{checkpoint_stem}__"
        f"draft_{draft_block_size}__"
        f"max_new_{max_new_tokens}__"
        f"{timestamp}.json"
    )
