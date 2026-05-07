from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, cast

import torch

from skip_search_spec.helpers.tooling import (
    get_preferred_device,
    get_preferred_float_dtype,
    load_model_and_tokenizer,
)


@dataclass(frozen=True, slots=True)
class TimedNormalInferenceOptions:
    measure_body: bool = False
    measure_lm_head: bool = False
    measure_token_selection: bool = False


@dataclass(slots=True)
class TimedNormalInferenceTimings:
    total_seconds: float = 0.0
    body_seconds: float = 0.0
    lm_head_seconds: float = 0.0
    token_selection_seconds: float = 0.0
    body_calls: int = 0
    lm_head_calls: int = 0
    token_selection_calls: int = 0


@dataclass(frozen=True, slots=True)
class TimedNormalInferenceResult:
    text: str
    inference_seconds: float
    num_generated_tokens: int
    timings: TimedNormalInferenceTimings


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


def get_backbone_hidden_and_cache(outputs: Any) -> tuple[torch.Tensor, Any | None]:
    if isinstance(outputs, tuple):
        past_key_values = outputs[1] if len(outputs) > 1 else None
        return outputs[0], past_key_values

    last_hidden_state = getattr(outputs, "last_hidden_state", None)
    if isinstance(last_hidden_state, torch.Tensor):
        return last_hidden_state, getattr(outputs, "past_key_values", None)

    raise ValueError("Backbone output did not expose last_hidden_state.")


def get_output_lm_head(model: Any) -> Any:
    lm_head = getattr(model, "lm_head", None)
    if lm_head is not None and callable(lm_head):
        return lm_head

    output_embeddings = model.get_output_embeddings()
    if output_embeddings is not None and callable(output_embeddings):
        return output_embeddings

    raise ValueError("Could not find a callable LM head on the model.")


def generate_timed_normal(
    model_name_or_path: str | None = None,
    prompt: str = "",
    *,
    max_new_tokens: int = 100,
    tokenizer_name_or_path: str | None = None,
    use_chat_template: bool = True,
    enable_thinking: bool = False,
    stop_on_eos: bool = True,
    timing_options: TimedNormalInferenceOptions | None = None,
    model: Any | None = None,
    tokenizer: Any | None = None,
    device: torch.device | None = None,
) -> TimedNormalInferenceResult:
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

    timing_options = timing_options or TimedNormalInferenceOptions()
    timings = TimedNormalInferenceTimings()
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
        raise ValueError("Timed normal inference only supports batch size 1.")

    current_ids = input_ids
    past_key_values: Any | None = None
    generated_token_pieces: list[torch.Tensor] = []

    sync_device_for_timing(device)
    total_start_time = time.perf_counter()

    with torch.inference_mode():
        for _ in range(max_new_tokens):
            if timing_options.measure_body:
                sync_device_for_timing(device)
                body_start_time = time.perf_counter()

            outputs = backbone(
                input_ids=current_ids,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=False,
            )

            if timing_options.measure_body:
                timings.body_seconds += elapsed_seconds_since(
                    body_start_time,
                    device=device,
                )

            timings.body_calls += 1
            last_hidden_state, past_key_values = get_backbone_hidden_and_cache(outputs)
            lm_head_input_hidden = last_hidden_state[:, -1:, :]

            if timing_options.measure_lm_head:
                sync_device_for_timing(device)
                head_start_time = time.perf_counter()

            logits = lm_head(lm_head_input_hidden)

            if timing_options.measure_lm_head:
                timings.lm_head_seconds += elapsed_seconds_since(
                    head_start_time,
                    device=device,
                )

            timings.lm_head_calls += 1

            if timing_options.measure_token_selection:
                sync_device_for_timing(device)
                selection_start_time = time.perf_counter()

            next_token = logits[:, -1, :].argmax(
                dim=-1,
                keepdim=True,
            )

            if timing_options.measure_token_selection:
                timings.token_selection_seconds += elapsed_seconds_since(
                    selection_start_time,
                    device=device,
                )

            timings.token_selection_calls += 1
            generated_token_pieces.append(next_token)
            current_ids = next_token

            if stop_on_eos and tokenizer.eos_token_id is not None:
                if int(next_token.item()) == int(tokenizer.eos_token_id):
                    break

    timings.total_seconds = elapsed_seconds_since(total_start_time, device=device)
    accepted_ids = (
        torch.cat([input_ids, *generated_token_pieces], dim=1)
        if generated_token_pieces
        else input_ids
    )

    text = cast(str, tokenizer.decode(accepted_ids[0], skip_special_tokens=True))
    num_generated_tokens = int(accepted_ids.size(1) - input_ids.size(1))

    return TimedNormalInferenceResult(
        text=text,
        inference_seconds=timings.total_seconds,
        num_generated_tokens=num_generated_tokens,
        timings=timings,
    )
