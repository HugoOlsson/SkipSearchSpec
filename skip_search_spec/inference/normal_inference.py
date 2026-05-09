from __future__ import annotations

from dataclasses import dataclass
import time

import torch

from skip_search_spec.helpers.tooling import get_preferred_device, get_preferred_float_dtype, load_model_and_tokenizer

from typing import Any, cast


@dataclass(frozen=True, slots=True)
class NormalInferenceResult:
    text: str
    inference_seconds: float
    num_generated_tokens: int


def generate_normal(
    model_name_or_path: str | None = None,
    prompt: str = "",
    *,
    max_new_tokens: int = 100,
    tokenizer_name_or_path: str | None = None,
    use_chat_template: bool = True,
    enable_thinking: bool = False,
    use_cache: bool = True,
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

    start_time = time.perf_counter()

    model_prompt = prompt
    if use_chat_template:
        # Important: Qwen thinking control is in the chat template.
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

    with torch.inference_mode():
        output_ids = cast(Any, model).generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=use_cache,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            # Self-speculation verifies with raw argmax logits. Some instruct
            # models, notably Qwen2.5, ship non-neutral generation_config
            # values like repetition_penalty=1.1; do_sample=False does not
            # disable those logits processors.
            repetition_penalty=1.0,
            temperature=None,
            top_p=None,
            top_k=None,
        )

    prompt_token_count = int(inputs["input_ids"].shape[-1])
    output_token_count = int(output_ids.shape[-1])
    num_generated_tokens = output_token_count - prompt_token_count

    text = cast(str, tokenizer.decode(output_ids[0], skip_special_tokens=True))
    inference_seconds = time.perf_counter() - start_time

    return NormalInferenceResult(
        text=text,
        inference_seconds=inference_seconds,
        num_generated_tokens=num_generated_tokens
    )
