from dataclasses import dataclass
from typing import Any, Literal, Mapping
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase
from pathlib import Path
from datasets import Dataset, load_dataset as hf_load_dataset
from skip_search_spec.protocols.windows import DatasetSpec, ModelAndTokenizer
import torch.nn.functional as F


TokenTraceType = Literal["prompt", "draft", "bonus"]
TokenTraceStatus = Literal["accepted", "rejected"]


@dataclass(frozen=True, slots=True)
class TokenData:
    token_id: int
    type: TokenTraceType
    status: TokenTraceStatus | None = None
    draft_block_index: int | None = None


def add_tokens_to_trace(
    token_trace: list[TokenData] | None,
    tokens: torch.Tensor,
    *,
    token_type: TokenTraceType,
    status: TokenTraceStatus | None = None,
    num_accepted: int | None = None,
    draft_block_index: int | None = None,
) -> None:
    if token_trace is None:
        return

    token_ids = [int(x) for x in tokens.detach().cpu().reshape(-1).tolist()]

    for token_offset, token_id in enumerate(token_ids):
        token_status = status
        if num_accepted is not None:
            token_status = (
                "accepted"
                if token_offset < num_accepted
                else "rejected"
            )

        token_trace.append(
            TokenData(
                token_id=token_id,
                type=token_type,
                status=token_status,
                draft_block_index=draft_block_index,
            )
        )


def get_preferred_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def get_preferred_float_dtype(device: torch.device) -> torch.dtype:
    dtype_name = os.environ.get("SKIP_SEARCH_TORCH_DTYPE")

    if dtype_name:
        dtype_name = dtype_name.lower().strip()

        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }

        if dtype_name not in dtype_map:
            raise ValueError(
                f"Unsupported SKIP_SEARCH_TORCH_DTYPE={dtype_name!r}. "
                f"Expected one of: {', '.join(dtype_map)}"
            )
        
        print("Did manually set dtype:", str(dtype_map[dtype_name]))
        return dtype_map[dtype_name]

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

    attn_implementation = os.environ.get("SKIP_SEARCH_ATTN_IMPLEMENTATION")
    if attn_implementation:
        resolved_model_kwargs.setdefault("attn_implementation", attn_implementation)

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
    shift_logits_drafter: torch.Tensor,
    shift_logits_verifier: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    shift_logits_drafter = shift_logits_drafter.detach().float()
    shift_logits_verifier = shift_logits_verifier.detach().float()

    def mean_metric(values: torch.Tensor) -> torch.Tensor:
        if mask is None:
            return values.mean()

        if values.shape != mask.shape:
            raise ValueError(f"values.shape {values.shape} != mask.shape {mask.shape}")

        mask_f = mask.to(device=values.device, dtype=values.dtype)
        return (values * mask_f).sum() / mask_f.sum().clamp_min(1.0)

    log_p_drafter = F.log_softmax(shift_logits_drafter, dim=-1)
    log_p_verifier = F.log_softmax(shift_logits_verifier, dim=-1)

    p_drafter = log_p_drafter.exp()
    p_verifier = log_p_verifier.exp()

    kl_verifier_to_drafter = mean_metric(
        (p_verifier * (log_p_verifier - log_p_drafter)).sum(dim=-1)
    )
    kl_drafter_to_verifier = mean_metric(
        (p_drafter * (log_p_drafter - log_p_verifier)).sum(dim=-1)
    )

    m = 0.5 * (p_verifier + p_drafter)
    log_m = torch.log(m.clamp_min(1e-12))
    js_verifier_drafter = mean_metric(0.5 * (
        (p_verifier * (log_p_verifier - log_m)).sum(dim=-1)
        + (p_drafter * (log_p_drafter - log_m)).sum(dim=-1)
    ))

    top1_drafter = shift_logits_drafter.argmax(dim=-1)
    top1_verifier = shift_logits_verifier.argmax(dim=-1)
    top1_drafter_matches_verifier = mean_metric(
        (top1_drafter == top1_verifier).float()
    )

    verifier_argmax = top1_verifier.unsqueeze(-1)
    p_drafter_on_verifier_top1 = mean_metric(
        p_drafter.gather(dim=-1, index=verifier_argmax).squeeze(-1)
    )

    prob_mass_overlap_verifier_drafter = mean_metric(
        torch.minimum(p_drafter, p_verifier).sum(dim=-1)
    )

    return {
        "kl_verifier_to_drafter": kl_verifier_to_drafter,
        "kl_drafter_to_verifier": kl_drafter_to_verifier,
        "js_verifier_drafter": js_verifier_drafter,
        "top1_drafter_matches_verifier": top1_drafter_matches_verifier,
        "p_drafter_on_verifier_top1": p_drafter_on_verifier_top1,
        "prob_mass_overlap_verifier_drafter": prob_mass_overlap_verifier_drafter,
    }
