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
class NormalInferenceResult:
    text: str
    inference_seconds: float
    num_generated_tokens: int
    body_seconds: float | None = None
    head_seconds: float | None = None
    other_seconds: float | None = None


def generate_normal(
    model_name_or_path: str | None = None,
    prompt: str = "",
    *,
    max_new_tokens: int = 100,
    tokenizer_name_or_path: str | None = None,
    use_chat_template: bool = True,
    enable_thinking: bool = False,
    use_cache: bool = True,
    measure_internal_timings: bool = False,
    model: Any | None = None,
    tokenizer: Any | None = None,
    device: torch.device | None = None,
) -> NormalInferenceResult:
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
    inputs = {name: tensor.to(device) for name, tensor in inputs.items()}

    sync_device_for_timing(device)
    total_start_time = time.perf_counter()

    if not measure_internal_timings:
        return _generate_normal_hf_generate(
            model=model,
            tokenizer=tokenizer,
            inputs=inputs,
            max_new_tokens=max_new_tokens,
            use_cache=use_cache,
            device=device,
            total_start_time=total_start_time,
        )

    return _generate_normal_measured(
        model=model,
        tokenizer=tokenizer,
        inputs=inputs,
        max_new_tokens=max_new_tokens,
        use_cache=use_cache,
        device=device,
        total_start_time=total_start_time,
    )


def _generate_normal_hf_generate(
    *,
    model: Any,
    tokenizer: Any,
    inputs: dict[str, torch.Tensor],
    max_new_tokens: int,
    use_cache: bool,
    device: torch.device,
    total_start_time: float,
) -> NormalInferenceResult:
    with torch.inference_mode():
        output_ids = cast(Any, model).generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=use_cache,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.0,
            temperature=None,
            top_p=None,
            top_k=None,
        )

    inference_seconds = elapsed_seconds_since(total_start_time, device=device)

    prompt_token_count = int(inputs["input_ids"].shape[-1])
    output_token_count = int(output_ids.shape[-1])
    num_generated_tokens = output_token_count - prompt_token_count

    text = cast(str, tokenizer.decode(output_ids[0], skip_special_tokens=True))

    return NormalInferenceResult(
        text=text,
        inference_seconds=inference_seconds,
        num_generated_tokens=num_generated_tokens,
    )


def _generate_normal_measured(
    *,
    model: Any,
    tokenizer: Any,
    inputs: dict[str, torch.Tensor],
    max_new_tokens: int,
    use_cache: bool,
    device: torch.device,
    total_start_time: float,
) -> NormalInferenceResult:
    backbone = get_causal_lm_backbone(model)
    lm_head = get_output_lm_head(model)

    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")

    if input_ids.size(0) != 1:
        raise ValueError("Measured normal inference only supports batch size 1.")

    generated_ids = input_ids
    past_key_values: Any | None = None

    body_seconds = 0.0
    head_seconds = 0.0

    with torch.inference_mode():
        for _ in range(max_new_tokens):
            if use_cache and past_key_values is not None:
                step_input_ids = generated_ids[:, -1:]
                step_attention_mask = attention_mask
            else:
                step_input_ids = generated_ids
                step_attention_mask = attention_mask

            sync_device_for_timing(device)
            body_start_time = time.perf_counter()

            outputs = backbone(
                input_ids=step_input_ids,
                attention_mask=step_attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_hidden_states=False,
                return_dict=True,
            )

            body_seconds += elapsed_seconds_since(
                body_start_time,
                device=device,
            )

            last_hidden_state, past_key_values = get_backbone_hidden_and_cache(outputs)
            head_input = last_hidden_state[:, -1:, :]

            sync_device_for_timing(device)
            head_start_time = time.perf_counter()

            logits = lm_head(head_input)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)

            head_seconds += elapsed_seconds_since(
                head_start_time,
                device=device,
            )

            generated_ids = torch.cat([generated_ids, next_token], dim=-1)

            if attention_mask is not None:
                attention_mask = torch.cat(
                    [
                        attention_mask,
                        torch.ones(
                            (attention_mask.shape[0], 1),
                            dtype=attention_mask.dtype,
                            device=attention_mask.device,
                        ),
                    ],
                    dim=-1,
                )

            if tokenizer.eos_token_id is not None:
                if bool((next_token == tokenizer.eos_token_id).all().item()):
                    break

    inference_seconds = elapsed_seconds_since(
        total_start_time,
        device=device,
    )

    prompt_token_count = int(input_ids.shape[-1])
    output_token_count = int(generated_ids.shape[-1])
    num_generated_tokens = output_token_count - prompt_token_count

    text = cast(str, tokenizer.decode(generated_ids[0], skip_special_tokens=True))

    other_seconds = max(
        0.0,
        inference_seconds - body_seconds - head_seconds,
    )

    return NormalInferenceResult(
        text=text,
        inference_seconds=inference_seconds,
        num_generated_tokens=num_generated_tokens,
        body_seconds=body_seconds,
        head_seconds=head_seconds,
        other_seconds=other_seconds,
    )


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


def get_output_lm_head(model: Any) -> Any:
    lm_head = getattr(model, "lm_head", None)
    if lm_head is not None and callable(lm_head):
        return lm_head

    if hasattr(model, "get_output_embeddings"):
        output_embeddings = model.get_output_embeddings()
        if output_embeddings is not None and callable(output_embeddings):
            return output_embeddings

    raise ValueError("Could not find a callable LM head on the model.")


def get_backbone_hidden_and_cache(outputs: Any) -> tuple[torch.Tensor, Any | None]:
    if isinstance(outputs, tuple):
        if not outputs or not isinstance(outputs[0], torch.Tensor):
            raise ValueError("Backbone tuple output did not contain hidden states.")

        past_key_values = outputs[1] if len(outputs) > 1 else None
        return outputs[0], past_key_values

    last_hidden_state = getattr(outputs, "last_hidden_state", None)
    if isinstance(last_hidden_state, torch.Tensor):
        return last_hidden_state, getattr(outputs, "past_key_values", None)

    raise ValueError("Backbone output did not expose last_hidden_state.")