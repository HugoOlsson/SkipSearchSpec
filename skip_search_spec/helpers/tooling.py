from typing import Any, Mapping
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase
from pathlib import Path
from datasets import Dataset, load_dataset as hf_load_dataset
from skip_search_spec.protocols.windows import DatasetSpec, ModelAndTokenizer



def get_preferred_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def get_preferred_float_dtype(device: torch.device) -> torch.dtype:
    if device.type in {"cuda", "mps"}:
        return torch.bfloat16

    return torch.float32


def check_nan(name, tensor):
    if torch.isnan(tensor).any():
        print(f"NaN detected in {name}, shape={tensor.shape}")
        return True
    if torch.isinf(tensor).any():
        print(f"Inf detected in {name}, shape={tensor.shape}")
        return True
    return False


def load_dataset(dataset: DatasetSpec, **load_kwargs: Any) -> Dataset:
    """Load a Hugging Face dataset using the defaults stored in ``dataset``."""

    resolved_kwargs = dict(load_kwargs)
    resolved_kwargs.setdefault("split", dataset.split)
    return hf_load_dataset(
        dataset.huggingface_path,
        dataset.config_name,
        **resolved_kwargs,
    )


def load_model_and_tokenizer(
    model_name_or_path: str,
    *,
    tokenizer_name_or_path: str | None = None,
    model_kwargs: Mapping[str, Any] | None = None,
    tokenizer_kwargs: Mapping[str, Any] | None = None,
) -> ModelAndTokenizer:
    """Load a Hugging Face model and tokenizer with minimal standard defaults."""

    resolved_model_kwargs = dict(model_kwargs or {})
    resolved_tokenizer_kwargs = dict(tokenizer_kwargs or {})
    resolved_tokenizer_kwargs.setdefault("use_fast", True)

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path or model_name_or_path,
        **resolved_tokenizer_kwargs,
    )
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **resolved_model_kwargs)

    return ModelAndTokenizer(model=model, tokenizer=tokenizer)
