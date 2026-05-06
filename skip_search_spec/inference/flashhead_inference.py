from __future__ import annotations

from dataclasses import dataclass
import time
from pathlib import Path
from typing import Any, cast

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
    flashhead_seconds: float
    num_generated_tokens: int


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
    use_cache: bool = True,
    stop_on_eos: bool = True,
    measure_internal_timings: bool = True,
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

    if flashhead is None:
        if flashhead_path is None:
            raise ValueError("flashhead_path is required when no flashhead is provided.")

        flashhead = FlashHeadModule.from_model(
            model=model,
            flashhead_path=flashhead_path,
            top_k_clusters=flashhead_top_k_clusters,
        )

    flashhead.eval()
    backbone = get_causal_lm_backbone(model)

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
    flashhead_seconds = 0.0

    sync_device_for_timing(device)
    start_time = time.perf_counter()

    backbone_call = cast(Any, backbone)
    find_token = flashhead.find_token
    append_generated_token = generated_token_pieces.append

    with torch.inference_mode():
        if use_cache:
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
                hidden_vector = last_hidden_state[0, -1, :]

                if measure_internal_timings:
                    sync_device_for_timing(device)
                    flashhead_start_time = time.perf_counter()

                next_token = find_token(hidden_vector).view(1, 1)

                if measure_internal_timings:
                    flashhead_seconds += elapsed_seconds_since(
                        flashhead_start_time,
                        device=device,
                    )

                append_generated_token(next_token)
                current_ids = next_token
        else:
            for _ in range(max_new_tokens):
                outputs = backbone_call(
                    input_ids=current_ids,
                    use_cache=False,
                    return_dict=False,
                )

                last_hidden_state, _ = get_backbone_hidden_and_cache(outputs)
                hidden_vector = last_hidden_state[0, -1, :]

                if measure_internal_timings:
                    sync_device_for_timing(device)
                    flashhead_start_time = time.perf_counter()

                next_token = find_token(hidden_vector).view(1, 1)

                if measure_internal_timings:
                    flashhead_seconds += elapsed_seconds_since(
                        flashhead_start_time,
                        device=device,
                    )

                append_generated_token(next_token)
                accepted_ids = torch.cat(
                    [input_ids, *generated_token_pieces],
                    dim=1,
                )
                current_ids = accepted_ids

    inference_seconds = elapsed_seconds_since(start_time, device=device)
    accepted_ids = (
        torch.cat([input_ids, *generated_token_pieces], dim=1)
        if generated_token_pieces
        else input_ids
    )

    if stop_on_eos and tokenizer.eos_token_id is not None:
        generated_suffix = accepted_ids[0, input_ids.size(1):]
        eos_hits = (generated_suffix == int(tokenizer.eos_token_id)).nonzero(as_tuple=False)
        if eos_hits.numel() > 0:
            first_eos = input_ids.size(1) + int(eos_hits[0, 0].item())
            accepted_ids = accepted_ids[:, : first_eos + 1]

    text = cast(str, tokenizer.decode(accepted_ids[0], skip_special_tokens=True))

    return FlashHeadInferenceResult(
        text=text,
        inference_seconds=inference_seconds,
        flashhead_seconds=flashhead_seconds,
        num_generated_tokens=int(accepted_ids.size(1) - input_ids.size(1)),
    )
