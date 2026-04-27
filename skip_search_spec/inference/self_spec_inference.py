from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch

from skip_search_spec.training.bridged_gap_model import BridgedGapModel


@dataclass(slots=True)
class SelfSpecResult:
    text: str
    output_ids: torch.Tensor
    verifier_calls: int
    drafted_tokens: int
    accepted_draft_tokens: int


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
    ) -> SelfSpecResult:
        encoded_prompt = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )

        input_ids = cast(torch.Tensor, encoded_prompt["input_ids"]).to(self.device)

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

        bonus_token = verifier.logits[:, -1, :].argmax(
            dim=-1,
            keepdim=True,
        )

        accepted_ids = torch.cat([input_ids, bonus_token], dim=1)
        verifier_reference_hidden = verifier.reference_hidden
        generated_tokens += 1

        while generated_tokens < max_new_tokens:

            # 2. Draft from the current accepted prefix.
            remaining_tokens = max_new_tokens - generated_tokens
            block_size = min(draft_block_size, remaining_tokens)

            draft_tokens = self._draft_block(
                accepted_ids=accepted_ids,
                verifier_reference_hidden=verifier_reference_hidden,
                block_size=block_size,
            )

            drafted_tokens += draft_tokens.size(1)

            accepted_len_before_draft = accepted_ids.size(1)
            candidate_ids = torch.cat([accepted_ids, draft_tokens], dim=1)

            # 3. Run verifier on the new entire prefix.
            verifier = self.bridged.run_verifier(
                input_ids=candidate_ids,
                attention_mask=torch.ones_like(candidate_ids),
            )
            verifier_calls += 1

            # 4. Compare verifier predictions against the draft.
            #
            # The first drafted token is checked by the verifier logits at the
            # previous accepted token position.
            verifier_draft_tokens = verifier.logits[
                :,
                accepted_len_before_draft - 1 : accepted_len_before_draft - 1 + draft_tokens.size(1),
                :,
            ].argmax(dim=-1)

            matches = verifier_draft_tokens == draft_tokens
            mismatch_positions = (~matches[0]).nonzero(as_tuple=False)

            # 5. Stop where draft and verifier no longer match.
            if mismatch_positions.numel() == 0:
                num_accepted = draft_tokens.size(1)
            else:
                num_accepted = int(mismatch_positions[0, 0].item())

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
                verifier_reference_hidden = verifier.reference_hidden[
                    :,
                    :next_reference_len,
                    :,
                ]
            else:
                verifier_token = verifier.logits[:, -1, :].argmax(
                    dim=-1,
                    keepdim=True,
                )

                # The bonus token is predicted from the full candidate prefix.
                verifier_reference_hidden = verifier.reference_hidden

            accepted_ids = torch.cat([accepted_ids, verifier_token], dim=1)
            generated_tokens += 1

            # Check if now contains EOS, if yes, then return prefix up to and including EOS:
            if stop_on_eos:
                eos_token_id = self.tokenizer.eos_token_id
                if isinstance(eos_token_id, int):
                    eos_hits = (accepted_ids[0] == eos_token_id).nonzero(as_tuple=False)
                    if eos_hits.numel() > 0:
                        first_eos = int(eos_hits[0, 0].item())
                        accepted_ids = accepted_ids[:, : first_eos + 1]
                        break

        text = self.tokenizer.decode(
            accepted_ids[0],
            skip_special_tokens=True,
        )

        return SelfSpecResult(
            text=text,
            output_ids=accepted_ids.detach().cpu(),
            verifier_calls=verifier_calls,
            drafted_tokens=drafted_tokens,
            accepted_draft_tokens=accepted_draft_tokens,
        )
    

    @torch.inference_mode()
    def _draft_block(
        self,
        *,
        accepted_ids: torch.Tensor,
        verifier_reference_hidden: torch.Tensor,
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

        current_ids = accepted_ids

        prev_reference_hidden = verifier_reference_hidden.new_zeros(
            verifier_reference_hidden.size(0),
            accepted_len,
            verifier_reference_hidden.size(2),
        )
        prev_reference_hidden[:, 1:, :] = verifier_reference_hidden

        draft_tokens: list[torch.Tensor] = []

        for _ in range(block_size):
            drafter = self.bridged.run_drafter(
                input_ids=current_ids,
                attention_mask=torch.ones_like(current_ids),
                prev_reference_hidden=prev_reference_hidden,
            )

            next_token = drafter.logits[:, -1, :].argmax(
                dim=-1,
                keepdim=True,
            )

            draft_tokens.append(next_token)

            current_ids = torch.cat([current_ids, next_token], dim=1)

            prev_reference_hidden = torch.cat(
                [
                    prev_reference_hidden,
                    drafter.reference_hidden[:, -1:, :].detach(),
                ],
                dim=1,
            )

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
) -> SelfSpecResult:
    speculator = load_bridge_self_speculator(
        bridge_checkpoint_path=bridge_checkpoint_path,
    )

    return speculator.generate(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        draft_block_size=draft_block_size,
    )