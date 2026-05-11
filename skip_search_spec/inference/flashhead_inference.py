from __future__ import annotations

from dataclasses import dataclass
import time
from pathlib import Path
from typing import Any, Literal, cast

import torch

from skip_search_spec.helpers.tooling import (
    get_preferred_device,
    get_preferred_float_dtype,
    load_model_and_tokenizer,
)
from skip_search_spec.training.flashhead.next_token_adapter import FlashHeadModule


@dataclass(frozen=True, slots=True)
class FlashHeadInferenceResult:
    text: str
    inference_seconds: float
    head_seconds: float
    num_generated_tokens: int
    flashhead_profile: dict[str, Any] | None = None


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


def get_causal_lm_backbone(model: Any) -> Any:
    for attribute_name in ("model", "transformer", "gpt_neox", "backbone"):
        backbone = getattr(model, attribute_name, None)
        if backbone is not None and backbone is not model and callable(backbone):
            return backbone

    raise ValueError(
        "Could not find a callable transformer backbone on the model. "
        "Add its attribute name to get_causal_lm_backbone()."
    )


def get_last_hidden_state(outputs: Any) -> torch.Tensor:
    last_hidden_state = getattr(outputs, "last_hidden_state", None)
    if isinstance(last_hidden_state, torch.Tensor):
        return last_hidden_state

    if isinstance(outputs, tuple) and isinstance(outputs[0], torch.Tensor):
        return outputs[0]

    raise ValueError("Backbone output did not expose last_hidden_state.")


def get_backbone_hidden_and_cache(outputs: Any) -> tuple[torch.Tensor, Any | None]:
    if isinstance(outputs, tuple):
        past_key_values = outputs[1] if len(outputs) > 1 else None
        return outputs[0], past_key_values

    return get_last_hidden_state(outputs), getattr(outputs, "past_key_values", None)


def get_output_lm_head(model: Any) -> Any:
    lm_head = getattr(model, "lm_head", None)
    if lm_head is not None and callable(lm_head):
        return lm_head

    output_embeddings = model.get_output_embeddings()
    if output_embeddings is not None and callable(output_embeddings):
        return output_embeddings

    raise ValueError("Could not find a callable LM head on the model.")


def generate_with_flashhead(
    model_name_or_path: str | None = None,
    prompt: str = "",
    *,
    flashhead_path: str | Path | None = None,
    flashhead: FlashHeadModule | None = None,
    flashhead_top_k_clusters: int = 50,
    max_new_tokens: int = 100,
    tokenizer_name_or_path: str | None = None,
    use_chat_template: bool = True,
    enable_thinking: bool = False,
    stop_on_eos: bool = True,
    measure_internal_timings: bool = True,
    profile_flashhead: bool | None = None,
    profile_print_every: int | None = None,
    profile_sync_device: bool | None = None,
    triton_stage2: bool | None = None,
    head_mode: Literal["flashhead", "lm_head"] = "flashhead",
    model: Any | None = None,
    tokenizer: Any | None = None,
    device: torch.device | None = None,
) -> FlashHeadInferenceResult:
    if (model is None) != (tokenizer is None):
        raise ValueError("model and tokenizer must be provided together.")

    if model is None or tokenizer is None:
        if model_name_or_path is None:
            raise ValueError(
                "model_name_or_path is required when no loaded model is provided."
            )

        device = get_preferred_device()
        dtype = get_preferred_float_dtype(device)

        mt = load_model_and_tokenizer(
            model_name_or_path,
            tokenizer_name_or_path=tokenizer_name_or_path,
            model_kwargs={"torch_dtype": dtype},
        )

        model = mt.model
        tokenizer = mt.tokenizer
    elif device is None:
        device = next(model.parameters()).device

    model.to(device)  # type: ignore[attr-defined]
    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if flashhead is None and head_mode == "flashhead":
        if flashhead_path is None:
            raise ValueError("flashhead_path is required when no flashhead is provided.")

        flashhead = FlashHeadModule.from_model(
            model=model,
            flashhead_path=flashhead_path,
            top_k_clusters=flashhead_top_k_clusters,
        )

    if flashhead is not None:
        if (
            profile_flashhead is not None
            or profile_print_every is not None
            or profile_sync_device is not None
        ):
            flashhead.set_profile_enabled(
                profile_flashhead
                if profile_flashhead is not None
                else flashhead.profile_enabled,
                print_every=profile_print_every,
                sync_device=profile_sync_device,
            )
        if triton_stage2 is not None:
            flashhead.set_triton_stage2_enabled(triton_stage2)
        if flashhead.profile_enabled:
            flashhead.reset_profile()
        flashhead.eval()
    backbone = get_causal_lm_backbone(model)
    lm_head = get_output_lm_head(model)

    model_prompt = prompt
    if use_chat_template:
        model_prompt = cast(
            str,
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            ),
        )

    inputs = tokenizer(
        model_prompt,
        return_tensors="pt",
        add_special_tokens=False,
    )
    input_ids = cast(torch.Tensor, inputs["input_ids"]).to(device)

    if input_ids.size(0) != 1:
        raise ValueError("This simple flashhead inference only supports batch size 1.")

    current_ids = input_ids
    past_key_values: Any | None = None
    generated_token_pieces: list[torch.Tensor] = []
    head_seconds = 0.0

    sync_device_for_timing(device)
    start_time = time.perf_counter()
    backbone_call = cast(Any, backbone)
    append_generated_token = generated_token_pieces.append

    with torch.inference_mode():
        for _ in range(max_new_tokens):
            outputs = backbone_call(
                input_ids=current_ids,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=False,
            )

            last_hidden_state, past_key_values = get_backbone_hidden_and_cache(
                outputs
            )
            lm_head_input_hidden = last_hidden_state[:, -1:, :]

            if measure_internal_timings:
                sync_device_for_timing(device)
                head_start_time = time.perf_counter()

            if head_mode == "flashhead":
                if flashhead is None:
                    raise RuntimeError("flashhead is required when head_mode='flashhead'.")
                next_token = flashhead.find_token(
                    lm_head_input_hidden[0, -1, :]
                ).view(1, 1)
            elif head_mode == "lm_head":
                logits = lm_head(lm_head_input_hidden)
                next_token = logits[:, -1, :].argmax(
                    dim=-1,
                    keepdim=True,
                )
            else:
                raise ValueError(f"Unknown head_mode: {head_mode}")

            if measure_internal_timings:
                head_seconds += elapsed_seconds_since(
                    head_start_time,
                    device=device,
                )

            append_generated_token(next_token)
            current_ids = next_token

            if stop_on_eos and tokenizer.eos_token_id is not None:
                if int(next_token.item()) == int(tokenizer.eos_token_id):
                    break

    inference_seconds = elapsed_seconds_since(start_time, device=device)
    accepted_ids = (
        torch.cat([input_ids, *generated_token_pieces], dim=1)
        if generated_token_pieces
        else input_ids
    )

    text = cast(str, tokenizer.decode(accepted_ids[0], skip_special_tokens=True))
    flashhead_profile = (
        flashhead.profile_summary()
        if flashhead is not None
        and getattr(flashhead, "profile_call_count", 0) > 0
        else None
    )

    return FlashHeadInferenceResult(
        text=text,
        inference_seconds=inference_seconds,
        head_seconds=head_seconds,
        num_generated_tokens=int(accepted_ids.size(1) - input_ids.size(1)),
        flashhead_profile=flashhead_profile,
    )
