from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Any, cast

import torch

from skip_search_spec.helpers.tooling import TokenData, add_tokens_to_trace
from skip_search_spec.helpers.shared_decoding_tools import crop_past_key_values
from skip_search_spec.training.bridged_gap_model import BridgedGapModel
from skip_search_spec.training.flashhead.next_token_adapter import FlashHeadModule


@dataclass(slots=True)
class SelfSpecTimings:
    total_seconds: float = 0.0

    verifier_seconds: float = 0.0
    drafter_total_seconds: float = 0.0
    drafter_body_seconds: float = 0.0

    dense_head_seconds: float = 0.0
    flashhead_seconds: float = 0.0

    drafter_registration_seconds: float = 0.0
    drafter_teardown_seconds: float = 0.0

    @property
    def head_seconds(self) -> float:
        return self.dense_head_seconds + self.flashhead_seconds

    @property
    def drafter_overhead_seconds(self) -> float:
        return max(
            0.0,
            self.drafter_total_seconds
            - self.drafter_body_seconds
            - self.head_seconds,
        )

    @property
    def overhead_seconds(self) -> float:
        measured = (
            self.verifier_seconds
            + self.drafter_body_seconds
            + self.head_seconds
        )
        return max(0.0, self.total_seconds - measured)


@dataclass(slots=True)
class SelfSpecResult:
    text: str
    output_ids: torch.Tensor
    num_generated_tokens: int
    verifier_calls: int
    drafted_tokens: int
    accepted_draft_tokens: int
    timings: SelfSpecTimings
    model_name: str
    trace_json_path: Path | None = None


def sync_device_for_timing(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        return

    if device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def elapsed_seconds_since(
    start_time: float,
    *,
    device: torch.device,
) -> float:
    sync_device_for_timing(device)
    return time.perf_counter() - start_time


@dataclass(slots=True)
class KVCacheHandler:
    past_key_values: Any
    verifier_cache_len: int

    @classmethod
    def from_verifier_output_after_prompt(
        cls,
        *,
        verifier_output: Any,
        prompt_len: int,
    ) -> "KVCacheHandler":
        past_key_values = verifier_output.past_key_values
        if past_key_values is None:
            raise RuntimeError("Verifier did not return past_key_values.")

        return cls(
            past_key_values=past_key_values,
            verifier_cache_len=prompt_len,
        )

    def verifier_logits_start_for_draft_check(
        self,
        *,
        accepted_len_before_draft: int,
    ) -> int:
        return accepted_len_before_draft - 1 - self.verifier_cache_len

    def require_verifier_cache_len_to_match_reference_hidden(
        self,
        reference_hidden: torch.Tensor,
    ) -> None:
        verifier_len = reference_hidden.size(1)
        if self.verifier_cache_len != verifier_len:
            raise ValueError(
                f"Expected verifier cache to match verifier reference hidden length. "
                f"Got verifier_cache_len={self.verifier_cache_len}, "
                f"verifier_len={verifier_len}."
            )

    def crop_to_accepted_prefix_from_verifier_output(
        self,
        *,
        verifier_output: Any,
        accepted_prefix_len: int,
    ) -> None:
        if verifier_output.past_key_values is None:
            raise RuntimeError("Verifier did not return past_key_values.")

        self.past_key_values = crop_past_key_values(
            verifier_output.past_key_values,
            max_length=accepted_prefix_len,
        )
        self.verifier_cache_len = accepted_prefix_len

    def adopt_full_verifier_cache(
        self,
        *,
        verifier_output: Any,
        new_verifier_cache_len: int,
    ) -> None:
        past_key_values = verifier_output.past_key_values
        if past_key_values is None:
            raise RuntimeError("Verifier did not return past_key_values.")

        self.past_key_values = past_key_values
        self.verifier_cache_len = new_verifier_cache_len

    def crop_back_to_verifier_cache_after_temporary_drafting(self) -> None:
        crop_past_key_values(
            self.past_key_values,
            max_length=self.verifier_cache_len,
        )


def argmax_debug_first_tie(
    logits: torch.Tensor,
    *,
    name: str,
    debug: bool,
    printed_tie: list[bool],
    generated_index_start: int,
) -> torch.Tensor:
    tokens = logits.argmax(dim=-1)

    if not debug or printed_tie[0]:
        return tokens

    max_vals = logits.max(dim=-1, keepdim=True).values
    is_max = logits == max_vals
    num_max = is_max.sum(dim=-1)

    tie_pos = (num_max > 1).nonzero(as_tuple=False)
    if tie_pos.numel() == 0:
        return tokens

    pos = tie_pos[0]
    pos_tuple = tuple(pos.tolist())

    # For [1, vocab], pos is (0,)
    # For [1, seq, vocab], pos is (0, seq_offset)
    seq_offset = int(pos[-1].item()) if logits.ndim == 3 else 0
    generated_index = generated_index_start + seq_offset

    tied_ids = is_max[pos_tuple].nonzero(as_tuple=False).flatten()

    print(f"[ARGMAX TIE] {name}")
    print(f"  generated_index={generated_index}")
    print(f"  chosen_id={int(tokens[pos_tuple].item())}")
    print(f"  tied_ids={tied_ids.detach().cpu().tolist()}")

    printed_tie[0] = True
    return tokens


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
        flashhead_path: str | Path | None = None,
        flashhead_top_k_clusters: int,
    ) -> None:
        self.bridged = bridged_model
        self.model = bridged_model.model
        self.tokenizer = bridged_model.tokenizer
        self.device = bridged_model.device
        self.flashhead = (
            None
            if flashhead_path is None
            else FlashHeadModule.from_model(
                model=self.model,
                flashhead_path=flashhead_path,
                top_k_clusters=flashhead_top_k_clusters,
            )
        )

        self.bridged.eval_all()
        if self.flashhead is not None:
            self.flashhead.eval()

    def _run_verifier_timed(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        past_key_values: Any | None = None,
        timings: SelfSpecTimings,
        measure_internal_timings: bool,
    ) -> Any:
        if measure_internal_timings:
            sync_device_for_timing(self.device)
            start_time = time.perf_counter()

        verifier = self.bridged.run_verifier(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
        )

        if measure_internal_timings:
            timings.verifier_seconds += elapsed_seconds_since(
                start_time,
                device=self.device,
            )

        return verifier

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
        build_token_trace: bool = True,
        measure_internal_timings: bool = True,
        debug_argmax_ties: bool = False,
    ) -> SelfSpecResult:
        timings = SelfSpecTimings()
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
        prompt_attention_mask = torch.ones_like(input_ids)
        prompt_len = input_ids.size(1)

        if input_ids.size(0) != 1:
            raise ValueError("This minimal self-spec test only supports batch size 1.")

        # Persistent generation state.
        accepted_ids = input_ids
        verifier_calls = 0
        drafted_tokens = 0
        accepted_draft_tokens = 0
        generated_tokens = 0
        draft_block_index = 0
        token_trace: list[TokenData] | None = [] if build_token_trace else None
        saved_trace_json_path: Path | None = None
        printed_tie = [False]

        add_tokens_to_trace(token_trace, input_ids, token_type="prompt")
        eos_token_id = self.tokenizer.eos_token_id
        should_stop_on_eos = stop_on_eos and isinstance(eos_token_id, int)

        sync_device_for_timing(self.device)
        total_start_time = time.perf_counter()

        # 1. Verify the prompt once to create the base.
        verifier = self._run_verifier_timed(
            input_ids=input_ids,
            attention_mask=prompt_attention_mask,
            timings=timings,
            measure_internal_timings=measure_internal_timings,
        )
        verifier_calls += 1
        kv_cache = KVCacheHandler.from_verifier_output_after_prompt(
            verifier_output=verifier,
            prompt_len=input_ids.size(1),
        )

        bonus_token = argmax_debug_first_tie(
            verifier.logits[:, -1, :],
            name="initial_prompt_bonus",
            debug=debug_argmax_ties,
            printed_tie=printed_tie,
            generated_index_start=generated_tokens,
        ).view(1, 1)

        accepted_ids = torch.cat([accepted_ids, bonus_token], dim=1)
        verifier_reference_hidden = verifier.reference_hidden
        generated_tokens += 1

        add_tokens_to_trace(token_trace, bonus_token, token_type="bonus")

        while generated_tokens < max_new_tokens:

            # 2. Draft from the current accepted prefix.
            remaining_tokens = max_new_tokens - generated_tokens
            block_size = min(draft_block_size, remaining_tokens)

            if measure_internal_timings:
                sync_device_for_timing(self.device)
                drafter_total_start_time = time.perf_counter()

            draft_tokens = self._draft_block(
                accepted_ids=accepted_ids,
                verifier_reference_hidden=verifier_reference_hidden,
                kv_cache=kv_cache,
                block_size=block_size,
                timings=timings,
                measure_internal_timings=measure_internal_timings,
            )

            if measure_internal_timings:
                timings.drafter_total_seconds += elapsed_seconds_since(
                    drafter_total_start_time,
                    device=self.device,
                )

            drafted_tokens += draft_tokens.size(1)

            accepted_len_before_draft = accepted_ids.size(1)
            candidate_ids = torch.cat([accepted_ids, draft_tokens], dim=1)

            # 3. Run verifier on the suffix after the verifier-cached prefix.
            verifier_cache_len_before_verifier_call = kv_cache.verifier_cache_len
            verifier_input_ids = candidate_ids[:, kv_cache.verifier_cache_len:]
            verifier = self._run_verifier_timed(
                input_ids=verifier_input_ids,
                attention_mask=torch.ones_like(candidate_ids),
                past_key_values=kv_cache.past_key_values,
                timings=timings,
                measure_internal_timings=measure_internal_timings,
            )
            verifier_calls += 1

            verifier_reference_hidden_full = torch.cat(
                [verifier_reference_hidden[:, :verifier_cache_len_before_verifier_call, :], verifier.reference_hidden],
                dim=1,
            )

            # 4. Compare verifier predictions against the draft.
            #
            # The first drafted token is checked by the verifier logits at the
            # previous accepted token position.
            verifier_logits_start = kv_cache.verifier_logits_start_for_draft_check(
                accepted_len_before_draft=accepted_len_before_draft,
            )
            verifier_draft_tokens = argmax_debug_first_tie(
                verifier.logits[
                    :,
                    verifier_logits_start : verifier_logits_start + draft_tokens.size(1),
                    :,
                ],
                name=f"block_{draft_block_index + 1}_verifier_draft_tokens",
                debug=debug_argmax_ties,
                printed_tie=printed_tie,
                generated_index_start=generated_tokens,
            )
            bonus_token = argmax_debug_first_tie(
                verifier.logits[:, -1, :],
                name=f"block_{draft_block_index + 1}_bonus",
                debug=debug_argmax_ties,
                printed_tie=printed_tie,
                generated_index_start=generated_tokens + draft_tokens.size(1),
            ).view(1, 1)

            draft_block_token_count = draft_tokens.size(1)
            decision_token_ids = torch.cat(
                [
                    draft_tokens[0],
                    verifier_draft_tokens[0],
                    bonus_token[0],
                ],
                dim=0,
            ).detach().cpu().tolist()
            draft_token_ids = decision_token_ids[:draft_block_token_count]
            verifier_draft_token_ids = decision_token_ids[
                draft_block_token_count : 2 * draft_block_token_count
            ]
            bonus_token_id = decision_token_ids[-1]

            # 5. Stop where draft and verifier no longer match.
            num_accepted = 0
            generated_should_stop = False
            for draft_token_id, verifier_draft_token_id in zip(
                draft_token_ids,
                verifier_draft_token_ids,
            ):
                if draft_token_id != verifier_draft_token_id:
                    break

                num_accepted += 1
                if should_stop_on_eos and draft_token_id == eos_token_id:
                    generated_should_stop = True
                    break

            draft_block_index += 1
            num_accepted_to_append = num_accepted

            if token_trace is not None:
                add_tokens_to_trace(
                    token_trace,
                    draft_tokens,
                    token_type="draft",
                    num_accepted=num_accepted_to_append,
                    draft_block_index=draft_block_index,
                )

            if num_accepted_to_append > 0:
                accepted_ids = torch.cat(
                    [accepted_ids, draft_tokens[:, :num_accepted_to_append]],
                    dim=1,
                )
                accepted_draft_tokens += num_accepted_to_append
                generated_tokens += num_accepted_to_append

                if generated_should_stop or generated_tokens >= max_new_tokens:
                    break

            # 6. Add the verifier token.
            #
            # If there was a mismatch, this is the verifier correction token.
            # If all draft tokens matched, this is the verifier bonus token.
            if num_accepted < draft_block_token_count:
                verifier_token = verifier_draft_tokens[
                    :,
                    num_accepted : num_accepted + 1,
                ]
                verifier_token_id = verifier_draft_token_ids[num_accepted]

                # The next loop needs verifier reference hidden for the prefix
                # before this verifier-produced token.
                next_reference_len = accepted_len_before_draft + num_accepted
                verifier_reference_hidden = verifier_reference_hidden_full[
                    :,
                    :next_reference_len,
                    :,
                ]

                kv_cache.crop_to_accepted_prefix_from_verifier_output(
                    verifier_output=verifier,
                    accepted_prefix_len=next_reference_len,
                )
            else:
                verifier_token = bonus_token
                verifier_token_id = bonus_token_id

                # The bonus token is predicted from the full candidate prefix.
                verifier_reference_hidden = verifier_reference_hidden_full
                kv_cache.adopt_full_verifier_cache(
                    verifier_output=verifier,
                    new_verifier_cache_len=candidate_ids.size(1),
                )

            add_tokens_to_trace(token_trace, verifier_token, token_type="bonus")

            accepted_ids = torch.cat([accepted_ids, verifier_token], dim=1)
            generated_tokens += 1

            if should_stop_on_eos and verifier_token_id == eos_token_id:
                break


        timings.total_seconds = elapsed_seconds_since(
            total_start_time,
            device=self.device,
        )

        text = self.tokenizer.decode(
            accepted_ids[0],
            skip_special_tokens=True,
        )

        if trace_json_path is not None and token_trace is not None:
            saved_trace_json_path = save_token_trace_json(
                tokens=token_trace,
                path=trace_json_path,
                tokenizer_name=self.bridged.config.model_name,
            )

        num_generated_tokens = int(accepted_ids.size(1) - prompt_len)

        return SelfSpecResult(
            text=text,
            output_ids=accepted_ids.detach().cpu(),
            verifier_calls=verifier_calls,
            drafted_tokens=drafted_tokens,
            accepted_draft_tokens=accepted_draft_tokens,
            timings=timings,
            trace_json_path=saved_trace_json_path,
            model_name=self.bridged.config.model_name,
            num_generated_tokens=num_generated_tokens
        )
    

    @torch.inference_mode()
    def _draft_block(
        self,
        *,
        accepted_ids: torch.Tensor,
        verifier_reference_hidden: torch.Tensor,
        kv_cache: KVCacheHandler,
        block_size: int,
        timings: SelfSpecTimings,
        measure_internal_timings: bool,
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

        kv_cache.require_verifier_cache_len_to_match_reference_hidden(
            verifier_reference_hidden,
        )
        current_ids = accepted_ids[:, kv_cache.verifier_cache_len:]
        current_prev_reference_hidden = prev_reference_hidden[
            :,
            kv_cache.verifier_cache_len:,
            :,
        ]
        current_cache_len = kv_cache.verifier_cache_len

        try:
            for _ in range(block_size):
                drafter = self.bridged.run_drafter(
                    input_ids=current_ids,
                    attention_mask=torch.ones(
                        current_ids.size(0),
                        current_cache_len + current_ids.size(1),
                        device=current_ids.device,
                        dtype=current_ids.dtype,
                    ),
                    prev_reference_hidden=current_prev_reference_hidden,
                    past_key_values=kv_cache.past_key_values,
                    use_cache=True,
                    compute_logits=False,
                    timings=timings,
                    measure_internal_timings=measure_internal_timings,
                )

                if self.flashhead is None:
                    if measure_internal_timings:
                        sync_device_for_timing(self.device)
                        head_start_time = time.perf_counter()

                    dense_logits = self.model.lm_head(
                        drafter.lm_head_input_hidden[:, -1:, :]
                    )
                    next_token = dense_logits[:, -1, :].argmax(
                        dim=-1,
                        keepdim=True,
                    )

                    if measure_internal_timings:
                        timings.dense_head_seconds += elapsed_seconds_since(
                            head_start_time,
                            device=self.device,
                        )
                else:
                    if measure_internal_timings:
                        sync_device_for_timing(self.device)
                        head_start_time = time.perf_counter()

                    next_token = self.flashhead.find_token(
                        drafter.lm_head_input_hidden[0, -1, :]
                    ).view(1, 1)

                    if measure_internal_timings:
                        timings.flashhead_seconds += elapsed_seconds_since(
                            head_start_time,
                            device=self.device,
                        )

                draft_tokens.append(next_token)

                current_cache_len += current_ids.size(1)
                current_ids = next_token
                current_prev_reference_hidden = drafter.reference_hidden[
                    :,
                    -1:,
                    :,
                ].detach()
        finally:
            kv_cache.crop_back_to_verifier_cache_after_temporary_drafting()

        return torch.cat(draft_tokens, dim=1)


def self_spec_inference_test(
    *,
    bridge_checkpoint_path: str | Path,
    prompt: str,
    max_new_tokens: int,
    draft_block_size: int = 4,
    use_chat_template: bool = True,
    enable_thinking: bool = False,
    flashhead_path: str | Path | None = None,
    flashhead_top_k_clusters: int = 50,
    build_token_trace: bool = True,
    measure_internal_timings: bool = True,
) -> SelfSpecResult:

    bridged = BridgedGapModel.load_from_checkpoint(
        bridge_checkpoint_path=bridge_checkpoint_path,
        bridge_dtype="model",
    )

    speculator = BridgeSelfSpeculator(
        bridged_model=bridged,
        flashhead_path=flashhead_path,
        flashhead_top_k_clusters=flashhead_top_k_clusters,
    )

    trace_json_path = (
        make_trace_json_path(
            bridge_checkpoint_path=bridge_checkpoint_path,
            draft_block_size=draft_block_size,
            max_new_tokens=max_new_tokens,
        )
        if build_token_trace
        else None
    )

    result = speculator.generate(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        draft_block_size=draft_block_size,
        use_chat_template=use_chat_template,
        enable_thinking=enable_thinking,
        trace_json_path=trace_json_path,
        build_token_trace=build_token_trace,
        measure_internal_timings=measure_internal_timings,
    )

    return result


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
