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
    model_name_or_path: str,
    prompt: str,
    *,
    max_new_tokens: int = 100,
    tokenizer_name_or_path: str | None = None,
    use_chat_template: bool = True,
    enable_thinking: bool = False,
    use_cache: bool = True,
) -> NormalInferenceResult:
    device = get_preferred_device()
    dtype = get_preferred_float_dtype(device)

    mt = load_model_and_tokenizer(
        model_name_or_path,
        tokenizer_name_or_path=tokenizer_name_or_path,
        model_kwargs={"torch_dtype": dtype},
    )

    model = mt.model
    tokenizer = mt.tokenizer

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
