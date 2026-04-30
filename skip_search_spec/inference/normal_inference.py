from __future__ import annotations

import torch

from skip_search_spec.helpers.tooling import get_preferred_device, get_preferred_float_dtype, load_model_and_tokenizer

from typing import Any, cast


def generate_from_plain_prompt(
    model_name_or_path: str,
    prompt: str,
    *,
    max_new_tokens: int = 100,
    tokenizer_name_or_path: str | None = None,
) -> str:
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

    # Important: Qwen thinking control is in the chat template.
    chat_prompt = cast(
        str,
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        ),
    )

    inputs = tokenizer(
        chat_prompt,
        return_tensors="pt",
        add_special_tokens=False,
    )
    inputs = {name: tensor.to(device) for name, tensor in inputs.items()}

    with torch.inference_mode():
        output_ids = cast(Any, model).generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    return cast(str, tokenizer.decode(output_ids[0], skip_special_tokens=True))