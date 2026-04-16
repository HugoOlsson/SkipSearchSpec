from typing import Any, Mapping
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase
from pathlib import Path
from datasets import Dataset, load_dataset as hf_load_dataset
from skip_search_spec.protocols.windows import DatasetSpec, ModelAndTokenizer
import torch.nn.functional as F


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


def assert_same_tokenizer(
    tok_a: PreTrainedTokenizerBase,
    tok_b: PreTrainedTokenizerBase,
) -> None:
    """Raise if two tokenizers differ in vocab or special tokens."""
    assert tok_a.vocab_size == tok_b.vocab_size, (
        f"Vocab size mismatch: {tok_a.vocab_size} vs {tok_b.vocab_size}"
    )
    assert tok_a.get_vocab() == tok_b.get_vocab(), "Vocabularies differ"
    assert tok_a.all_special_tokens == tok_b.all_special_tokens, "Special tokens differ"



def distribution_similarity_metrics(
    shift_logits_mid: torch.Tensor,
    shift_logits_full: torch.Tensor,
) -> dict[str, torch.Tensor]:
    shift_logits_mid = shift_logits_mid.detach().float()
    shift_logits_full = shift_logits_full.detach().float()

    log_p_mid = F.log_softmax(shift_logits_mid, dim=-1)
    log_p_full = F.log_softmax(shift_logits_full, dim=-1)

    p_mid = log_p_mid.exp()
    p_full = log_p_full.exp()

    kl_full_to_mid = (p_full * (log_p_full - log_p_mid)).sum(dim=-1).mean()
    kl_mid_to_full = (p_mid * (log_p_mid - log_p_full)).sum(dim=-1).mean()

    m = 0.5 * (p_full + p_mid)
    log_m = torch.log(m.clamp_min(1e-12))
    js = 0.5 * (
        (p_full * (log_p_full - log_m)).sum(dim=-1) +
        (p_mid * (log_p_mid - log_m)).sum(dim=-1)
    ).mean()

    top1_mid = shift_logits_mid.argmax(dim=-1)
    top1_full = shift_logits_full.argmax(dim=-1)
    top1_agreement = (top1_mid == top1_full).float().mean()

    full_argmax = top1_full.unsqueeze(-1)
    p_mid_on_full_argmax = p_mid.gather(dim=-1, index=full_argmax).squeeze(-1).mean()

    overlap = torch.minimum(p_mid, p_full).sum(dim=-1).mean()

    return {
        "kl_full_to_mid": kl_full_to_mid,
        "kl_mid_to_full": kl_mid_to_full,
        "js": js,
        "top1_agreement": top1_agreement,
        "p_mid_on_full_argmax": p_mid_on_full_argmax,
        "overlap": overlap,
    }
