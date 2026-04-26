from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn as nn

from skip_search_spec.helpers.tooling import (
    get_preferred_device,
    get_preferred_float_dtype,
    load_model_and_tokenizer,
)
from skip_search_spec.protocols.windows import ModelAndTokenizer

from skip_search_spec.helpers.shared_decoding_tools import (
    GapSpec,
    forward_model_logits,
    get_decoder_layers,
    get_first_hidden_from_inputs,
    get_hidden_size,
    get_reentry_module_for_gap,
    make_identity_skip_hook,
    validate_gap,
)

# Change this import path to wherever your training file lives.
# Important: do NOT redeclare GapBridge here.
from skip_search_spec.training.train_gap_bridge_optimized import GapBridge


@dataclass(slots=True)
class SelfSpecResult:
    text: str
    output_ids: torch.Tensor
    verifier_calls: int
    drafted_tokens: int
    accepted_draft_tokens: int


class BridgeSelfSpeculator:
    """
    Minimal greedy self-speculation core.

    Verifier:
      normal full model.

    Drafter:
      same base model, but with skipped layers and bridge injection installed
      only inside the drafter forward.

    The bridge expects:
      gap_hidden[t]
      previous-position reference hidden[t]

    During draft-block rollout:
      - for the already accepted prefix, use teacher re-entry hidden
      - for newly drafted positions, use the drafter's own bridged hidden
    """

    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        bridge: nn.Module,
        gap: GapSpec,
        device: torch.device,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.bridge = bridge
        self.gap = gap
        self.device = device

        self.model.eval()
        self.bridge.eval()

        for p in self.model.parameters():
            p.requires_grad_(False)

        for p in self.bridge.parameters():
            p.requires_grad_(False)

        num_layers = len(get_decoder_layers(self.model))
        validate_gap(gap=self.gap, num_layers=num_layers)

    @torch.inference_mode()
    def generate(
        self,
        *,
        prompt: str,
        max_new_tokens: int,
        draft_block_size: int = 4,
        stop_on_eos: bool = True,
    ) -> SelfSpecResult:
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )

        input_ids = cast(torch.Tensor, encoded["input_ids"]).to(self.device)

        if input_ids.size(0) != 1:
            raise ValueError("This minimal test only supports batch size 1.")

        prompt_len = input_ids.size(1)

        verifier_calls = 0
        drafted_tokens = 0
        accepted_draft_tokens = 0

        while input_ids.size(1) - prompt_len < max_new_tokens:
            remaining = max_new_tokens - (input_ids.size(1) - prompt_len)
            block_size = min(draft_block_size, remaining)

            # Teacher hidden is available for the currently accepted prefix.
            teacher_reentry_hidden = self._run_verifier_and_capture_reentry(
                input_ids=input_ids,
            )
            verifier_calls += 1

            draft_ids = self._draft_block(
                prefix_ids=input_ids,
                teacher_reentry_hidden=teacher_reentry_hidden,
                block_size=block_size,
            )
            drafted_tokens += draft_ids.size(1)

            verify_ids = torch.cat([input_ids, draft_ids], dim=1)

            verifier_logits = forward_model_logits(
                model=self.model,
                input_ids=verify_ids,
                attention_mask=torch.ones_like(verify_ids),
            )
            verifier_calls += 1

            old_len = input_ids.size(1)
            all_accepted = True

            for i in range(draft_ids.size(1)):
                if input_ids.size(1) - prompt_len >= max_new_tokens:
                    all_accepted = False
                    break

                # Teacher prediction for draft token i.
                #
                # draft_ids[:, i] is at absolute position old_len + i.
                # It is predicted by logits at position old_len + i - 1.
                teacher_next = verifier_logits[:, old_len + i - 1, :].argmax(
                    dim=-1,
                    keepdim=True,
                )

                draft_next = draft_ids[:, i : i + 1]

                if torch.equal(teacher_next, draft_next):
                    input_ids = torch.cat([input_ids, draft_next], dim=1)
                    accepted_draft_tokens += 1
                else:
                    input_ids = torch.cat([input_ids, teacher_next], dim=1)
                    all_accepted = False
                    break

                if self._should_stop(input_ids=input_ids, stop_on_eos=stop_on_eos):
                    all_accepted = False
                    break

            # If every draft token matched, append the verifier's bonus token.
            if (
                all_accepted
                and input_ids.size(1) - prompt_len < max_new_tokens
            ):
                bonus = verifier_logits[:, old_len + draft_ids.size(1) - 1, :].argmax(
                    dim=-1,
                    keepdim=True,
                )
                input_ids = torch.cat([input_ids, bonus], dim=1)

            if self._should_stop(input_ids=input_ids, stop_on_eos=stop_on_eos):
                break

        text = self.tokenizer.decode(
            input_ids[0],
            skip_special_tokens=True,
        )

        return SelfSpecResult(
            text=text,
            output_ids=input_ids.detach().cpu(),
            verifier_calls=verifier_calls,
            drafted_tokens=drafted_tokens,
            accepted_draft_tokens=accepted_draft_tokens,
        )

    def _should_stop(
        self,
        *,
        input_ids: torch.Tensor,
        stop_on_eos: bool,
    ) -> bool:
        if not stop_on_eos:
            return False

        eos_token_id = self.tokenizer.eos_token_id
        if not isinstance(eos_token_id, int):
            return False

        return int(input_ids[0, -1].item()) == eos_token_id

    @torch.inference_mode()
    def _run_verifier_and_capture_reentry(
        self,
        *,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Full model forward.

        Captures teacher re-entry hidden at the same location that the bridge
        was trained to predict.
        """
        reentry_module = get_reentry_module_for_gap(
            model=self.model,
            gap=self.gap,
        )

        teacher_reentry_hidden: torch.Tensor | None = None

        def reentry_prehook(module: Any, inputs: tuple[Any, ...]) -> None:
            nonlocal teacher_reentry_hidden
            teacher_reentry_hidden = get_first_hidden_from_inputs(inputs).detach()

        with ExitStack() as stack:
            handle = reentry_module.register_forward_pre_hook(reentry_prehook)
            stack.callback(handle.remove)

            _ = forward_model_logits(
                model=self.model,
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
            )

        if teacher_reentry_hidden is None:
            raise RuntimeError("Failed to capture teacher re-entry hidden.")

        return teacher_reentry_hidden

    @torch.inference_mode()
    def _draft_block(
        self,
        *,
        prefix_ids: torch.Tensor,
        teacher_reentry_hidden: torch.Tensor,
        block_size: int,
    ) -> torch.Tensor:
        """
        Greedily draft `block_size` tokens.

        For the accepted prefix:
          use teacher_reentry_hidden.

        For tokens generated inside this draft block:
          use the drafter's own bridged hidden from the previous drafter pass.
        """
        if prefix_ids.size(0) != 1:
            raise ValueError("This minimal test only supports batch size 1.")

        current_ids = prefix_ids
        prefix_len = prefix_ids.size(1)

        # reference_hidden_by_position[pos] is the hidden vector that should be
        # used as the "previous-position hidden" for position pos + 1.
        #
        # Prefix positions start from teacher hidden.
        reference_hidden_by_position = teacher_reentry_hidden.detach()

        draft_tokens: list[torch.Tensor] = []

        for _ in range(block_size):
            cur_len = current_ids.size(1)
            hidden_size = reference_hidden_by_position.size(-1)

            # If current_ids contains a newly drafted last token whose hidden has
            # not been computed yet, pad a placeholder. It will not be used as
            # prev hidden for itself; prev_state_hidden[:, t] uses hidden[t - 1].
            if reference_hidden_by_position.size(1) < cur_len:
                missing = cur_len - reference_hidden_by_position.size(1)
                pad = reference_hidden_by_position.new_zeros(
                    (1, missing, hidden_size)
                )
                reference_hidden_by_position = torch.cat(
                    [reference_hidden_by_position, pad],
                    dim=1,
                )

            prev_state_hidden = reference_hidden_by_position.new_zeros(
                (1, cur_len, hidden_size)
            )
            prev_state_hidden[:, 1:, :] = reference_hidden_by_position[
                :, : cur_len - 1, :
            ]

            logits, bridged_hidden = self._run_drafter_gap_bridge(
                input_ids=current_ids,
                prev_state_hidden=prev_state_hidden,
            )

            # Keep teacher hidden for the accepted prefix.
            # Use drafter bridged hidden for generated positions inside this block.
            updated_reference = reference_hidden_by_position[:, :cur_len, :].clone()

            if cur_len > prefix_len:
                updated_reference[:, prefix_len:cur_len, :] = bridged_hidden[
                    :, prefix_len:cur_len, :
                ].detach()

            reference_hidden_by_position = updated_reference

            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)

            draft_tokens.append(next_token)
            current_ids = torch.cat([current_ids, next_token], dim=1)

        return torch.cat(draft_tokens, dim=1)

    @torch.inference_mode()
    def _run_drafter_gap_bridge(
        self,
        *,
        input_ids: torch.Tensor,
        prev_state_hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Drafters forward:
          - capture hidden entering skipped gap
          - make skipped layers identity
          - inject bridge output at re-entry
        """
        layers = get_decoder_layers(self.model)
        reentry_module = get_reentry_module_for_gap(
            model=self.model,
            gap=self.gap,
        )

        gap_input_hidden: torch.Tensor | None = None
        bridged_hidden: torch.Tensor | None = None

        def capture_gap_input_prehook(
            module: Any,
            inputs: tuple[Any, ...],
        ) -> None:
            nonlocal gap_input_hidden
            gap_input_hidden = get_first_hidden_from_inputs(inputs)

        def inject_bridge_prehook(
            module: Any,
            inputs: tuple[Any, ...],
        ) -> tuple[Any, ...]:
            nonlocal bridged_hidden

            if gap_input_hidden is None:
                raise RuntimeError("Gap input hidden was not captured.")

            if prev_state_hidden.shape != gap_input_hidden.shape:
                raise ValueError(
                    f"prev_state_hidden.shape {prev_state_hidden.shape} "
                    f"!= gap_input_hidden.shape {gap_input_hidden.shape}"
                )

            bridged = self.bridge(
                gap_input_hidden,
                prev_state_hidden,
            )

            bridged = bridged.to(dtype=gap_input_hidden.dtype)
            bridged_hidden = bridged

            if len(inputs) == 0:
                raise RuntimeError("Re-entry module received empty inputs.")

            return (bridged, *inputs[1:])

        with ExitStack() as stack:
            handle = layers[self.gap.start].register_forward_pre_hook(
                capture_gap_input_prehook
            )
            stack.callback(handle.remove)

            for layer_idx in range(self.gap.start, self.gap.end):
                handle = layers[layer_idx].register_forward_hook(
                    make_identity_skip_hook()
                )
                stack.callback(handle.remove)

            handle = reentry_module.register_forward_pre_hook(
                inject_bridge_prehook
            )
            stack.callback(handle.remove)

            logits = forward_model_logits(
                model=self.model,
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
            )

        if bridged_hidden is None:
            raise RuntimeError("Failed to capture bridged hidden.")

        return logits, bridged_hidden.detach()

def load_bridge_self_speculator(
    *,
    bridge_checkpoint_path: str | Path,
) -> BridgeSelfSpeculator:
    device = get_preferred_device()
    compute_dtype = get_preferred_float_dtype(device)

    checkpoint = torch.load(
        bridge_checkpoint_path,
        map_location="cpu",
    )

    checkpoint_model_name = checkpoint.get("model_name")
    if not isinstance(checkpoint_model_name, str):
        raise ValueError(
            "Bridge checkpoint must contain a valid string field: 'model_name'."
        )

    model_name = checkpoint_model_name

    model_and_tokenizer: ModelAndTokenizer = load_model_and_tokenizer(
        model_name,
        model_kwargs={
            "torch_dtype": compute_dtype,
        },
    )

    model = cast(Any, model_and_tokenizer.model)
    tokenizer = model_and_tokenizer.tokenizer

    model.to(device=device)
    model.eval()

    for param in model.parameters():
        param.requires_grad_(False)

    gap = GapSpec(
        start=int(checkpoint["gap_start"]),
        length=int(checkpoint["gap_length"]),
    )

    hidden_size = int(checkpoint.get("hidden_size", get_hidden_size(model)))

    bridge = GapBridge(
        hidden_size=hidden_size,
    )

    bridge.load_state_dict(checkpoint["bridge_state_dict"])
    bridge.to(device=device, dtype=torch.float32)
    bridge.eval()

    for param in bridge.parameters():
        param.requires_grad_(False)

    return BridgeSelfSpeculator(
        model=model,
        tokenizer=tokenizer,
        bridge=bridge,
        gap=gap,
        device=device,
    )


def run_minimal_bridge_self_spec_test(
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