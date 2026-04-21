from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast
from pathlib import Path
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from datasets import Dataset

from skip_search_spec.helpers.tooling import (
    distribution_similarity_metrics,
    get_preferred_device,
    get_preferred_float_dtype,
    load_dataset,
    load_model_and_tokenizer,
)
from skip_search_spec.helpers.window_building import (
    WindowDataset,
    build_all_training_windows,
    collate_windows,
    tokenize_dataset_to_examples,
)
from skip_search_spec.protocols.windows import (
    DatasetSpec,
    ModelAndTokenizer,
    WindowSettings,
)


@dataclass(slots=True)
class TrainFutureHiddenHeadsOutput:
    future_heads: nn.Module
    history: list[dict[str, float]]
    checkpoint_path: Path | None


def _stage(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def _get_backbone(model: Any) -> Any:
    if hasattr(model, "model") and hasattr(model.model, "layers") and hasattr(model.model, "norm"):
        return model.model

    raise TypeError(
        "Unsupported model structure. Expected a decoder-only HF model with "
        "`model.layers` and `model.norm`."
    )


def _get_hidden_size(model: Any) -> int:
    hidden_size = getattr(getattr(model, "config", None), "hidden_size", None)
    if isinstance(hidden_size, int) and hidden_size > 0:
        return hidden_size

    backbone = _get_backbone(model)
    if hasattr(backbone, "embed_tokens") and hasattr(backbone.embed_tokens, "embedding_dim"):
        return int(backbone.embed_tokens.embedding_dim)

    raise ValueError("Could not infer hidden size from model.")


def _get_lm_head(model: Any) -> nn.Module:
    lm_head = getattr(model, "lm_head", None)
    if isinstance(lm_head, nn.Module):
        return lm_head
    raise TypeError("Unsupported model structure. Expected model.lm_head.")


class FutureHiddenHead(nn.Module):
    """
    Minimal residual MLP head:
        h_t -> h_t + down/up(LN(h_t))

    The output stays in hidden-state space so we can:
    1. compare against true future hidden states
    2. feed it through the frozen shared lm_head
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        bottleneck_dim: int,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.down = nn.Linear(hidden_size, bottleneck_dim, bias=bias)
        self.up = nn.Linear(bottleneck_dim, hidden_size, bias=bias)

        with torch.no_grad():
            nn.init.normal_(self.down.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.up.weight)
            if self.down.bias is not None:
                nn.init.zeros_(self.down.bias)
            if self.up.bias is not None:
                nn.init.zeros_(self.up.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x_fp32 = hidden_states.float()
        delta_fp32 = self.up(F.silu(self.down(self.norm(x_fp32))))
        return x_fp32 + delta_fp32


class FutureHiddenHeads(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        num_future_steps: int,
        bottleneck_dim: int,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if num_future_steps <= 0:
            raise ValueError(f"num_future_steps must be > 0, got {num_future_steps}")

        self.num_future_steps = num_future_steps
        self.heads = nn.ModuleList(
            [
                FutureHiddenHead(
                    hidden_size=hidden_size,
                    bottleneck_dim=bottleneck_dim,
                    bias=bias,
                )
                for _ in range(num_future_steps)
            ]
        )

    def forward(self, hidden_states: torch.Tensor) -> list[torch.Tensor]:
        return [head(hidden_states) for head in self.heads]


@torch.no_grad()
def _run_model_and_capture_last_hidden(
    *,
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )

    hidden_states = getattr(outputs, "hidden_states", None)
    if not isinstance(hidden_states, tuple) or len(hidden_states) == 0:
        raise RuntimeError("Model did not return hidden_states.")

    final_hidden = hidden_states[-1]
    if not isinstance(final_hidden, torch.Tensor):
        raise RuntimeError("Final hidden state is not a tensor.")

    logits = cast(torch.Tensor, outputs.logits)
    return logits.detach(), final_hidden.detach()


def _future_valid_mask(
    *,
    attention_mask: torch.Tensor | None,
    offset: int,
) -> torch.Tensor | None:
    if attention_mask is None:
        return None

    if offset <= 0:
        raise ValueError(f"offset must be > 0, got {offset}")

    valid = attention_mask[:, :-offset] * attention_mask[:, offset:]
    return valid


def _masked_mse(
    *,
    predicted: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    diff_sq = (predicted.float() - target.float()).pow(2)

    if mask is None:
        return diff_sq.mean()

    mask_f = mask.unsqueeze(-1).to(diff_sq.dtype)
    denom = (mask_f.sum() * diff_sq.size(-1)).clamp_min(1.0)
    return (diff_sq * mask_f).sum() / denom


def _future_cross_entropy_from_hidden(
    *,
    predicted_hidden: torch.Tensor,
    labels: torch.Tensor,
    lm_head: nn.Module,
    offset: int,
    attention_mask: torch.Tensor | None,
) -> torch.Tensor:
    lm_head_dtype = next(lm_head.parameters()).dtype
    logits = cast(torch.Tensor, lm_head(predicted_hidden.to(dtype=lm_head_dtype)))

    usable_logits = logits[:, :-offset, :].contiguous()
    target_labels = labels[:, offset:].contiguous()

    flat_loss = F.cross_entropy(
        usable_logits.view(-1, usable_logits.size(-1)).float(),
        target_labels.view(-1),
        reduction="none",
    )

    flat_loss = flat_loss.view_as(target_labels)

    if attention_mask is None:
        return flat_loss.mean()

    valid_mask = _future_valid_mask(attention_mask=attention_mask, offset=offset)
    if valid_mask is None:
        return flat_loss.mean()

    valid_mask_f = valid_mask.to(flat_loss.dtype)
    denom = valid_mask_f.sum().clamp_min(1.0)
    return (flat_loss * valid_mask_f).sum() / denom


def train_future_hidden_heads(
    *,
    model_name: str,
    dataset_spec: DatasetSpec,
    context_len: int,
    max_examples: int,
    num_windows_to_use: int = 256,
    batch_size: int = 2,
    num_future_steps: int = 2,
    bottleneck_dim: int | None = None,
    num_epochs: int = 1,
    max_steps: int | None = None,
    lr: float = 1e-4,
    weight_decay: float = 0.0,
    max_grad_norm: float = 1.0,
    hidden_loss_weight: float = 1.0,
    ce_loss_weight: float = 0.0,
    model_kwargs: dict[str, Any] | None = None,
    checkpoint_dir: str | Path | None = "future_hidden_heads_checkpoints",
) -> TrainFutureHiddenHeadsOutput:
    _stage("train_future_hidden_heads: start")

    if context_len <= num_future_steps:
        raise ValueError(
            f"context_len must be > num_future_steps, got context_len={context_len}, "
            f"num_future_steps={num_future_steps}"
        )

    device = get_preferred_device()
    compute_dtype = get_preferred_float_dtype(device)

    model_and_tokenizer: ModelAndTokenizer = load_model_and_tokenizer(
        model_name,
        model_kwargs={
            "torch_dtype": compute_dtype,
            **(model_kwargs or {}),
        },
    )

    model = cast(Any, model_and_tokenizer.model)
    model.to(device=device)
    model.eval()

    for param in model.parameters():
        param.requires_grad_(False)

    hidden_size = _get_hidden_size(model)
    lm_head = _get_lm_head(model)
    lm_head.eval()

    if bottleneck_dim is None:
        bottleneck_dim = max(32, hidden_size // 4)

    future_heads = FutureHiddenHeads(
        hidden_size=hidden_size,
        num_future_steps=num_future_steps,
        bottleneck_dim=bottleneck_dim,
        bias=False,
    ).to(device=device, dtype=torch.float32)
    future_heads.train()

    optimizer = torch.optim.AdamW(
        future_heads.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    dataset: Dataset = load_dataset(dataset_spec)
    window_settings = WindowSettings(C1=context_len)

    _stage("tokenizing dataset")
    tokenized_examples = tokenize_dataset_to_examples(
        dataset,
        model_and_tokenizer.tokenizer,
        dataset_spec,
        max_examples=max_examples,
    )

    _stage("building windows")
    all_windows = build_all_training_windows(
        tokenized_examples,
        window_settings,
        dataset_spec,
    )

    if len(all_windows) < num_windows_to_use:
        raise ValueError(
            f"Requested {num_windows_to_use} windows, but only built {len(all_windows)}."
        )

    selected_windows = all_windows[:num_windows_to_use]
    window_dataset = WindowDataset(selected_windows)

    dataloader = DataLoader(
        window_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_windows,
        pin_memory=(device.type == "cuda"),
    )

    history: list[dict[str, float]] = []
    step = 0

    _stage(
        f"training future hidden heads with hidden_size={hidden_size}, "
        f"num_future_steps={num_future_steps}, bottleneck_dim={bottleneck_dim}"
    )

    for epoch_idx in range(num_epochs):
        for batch_idx, (input_ids, attention_mask) in enumerate(dataloader, start=1):
            if max_steps is not None and step >= max_steps:
                break

            step += 1
            compute_distribution_metrics = (step % 50 == 0)

            input_ids = input_ids.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)
            labels = input_ids

            optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                base_logits, final_hidden = _run_model_and_capture_last_hidden(
                    model=model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

            predicted_future_hidden_list = future_heads(final_hidden)

            hidden_losses: list[torch.Tensor] = []
            ce_losses: list[torch.Tensor] = []
            per_offset_metrics: dict[str, float] = {}

            for offset, predicted_hidden in enumerate(predicted_future_hidden_list, start=1):
                predicted_trimmed = predicted_hidden[:, :-offset, :].contiguous()
                target_trimmed = final_hidden[:, offset:, :].contiguous()
                valid_mask = _future_valid_mask(attention_mask=attention_mask, offset=offset)

                loss_hidden_k = _masked_mse(
                    predicted=predicted_trimmed,
                    target=target_trimmed,
                    mask=valid_mask,
                )
                hidden_losses.append(loss_hidden_k)
                per_offset_metrics[f"hidden_mse_t_plus_{offset}"] = loss_hidden_k.item()

                

                if compute_distribution_metrics:
                    lm_head_dtype = next(lm_head.parameters()).dtype

                    predicted_logits_k = cast(
                        torch.Tensor,
                        lm_head(predicted_trimmed.to(dtype=lm_head_dtype)),
                    )
                    target_logits_k = cast(
                        torch.Tensor,
                        lm_head(target_trimmed.to(dtype=lm_head_dtype)),
                    )

                    sim_k = distribution_similarity_metrics(
                        shift_logits_mid=predicted_logits_k,
                        shift_logits_full=target_logits_k,
                    )

                    per_offset_metrics[f"kl_real_to_pred_t_plus_{offset}"] = sim_k["kl_full_to_mid"].item()
                    per_offset_metrics[f"kl_pred_to_real_t_plus_{offset}"] = sim_k["kl_mid_to_full"].item()
                    per_offset_metrics[f"js_t_plus_{offset}"] = sim_k["js"].item()
                    per_offset_metrics[f"top1_t_plus_{offset}"] = sim_k["top1_agreement"].item()
                    per_offset_metrics[f"overlap_t_plus_{offset}"] = sim_k["overlap"].item()
                    per_offset_metrics[f"p_pred_on_real_argmax_t_plus_{offset}"] = sim_k["p_mid_on_full_argmax"].item()
                else:
                    per_offset_metrics[f"kl_real_to_pred_t_plus_{offset}"] = float("nan")
                    per_offset_metrics[f"kl_pred_to_real_t_plus_{offset}"] = float("nan")
                    per_offset_metrics[f"js_t_plus_{offset}"] = float("nan")
                    per_offset_metrics[f"top1_t_plus_{offset}"] = float("nan")
                    per_offset_metrics[f"overlap_t_plus_{offset}"] = float("nan")
                    per_offset_metrics[f"p_pred_on_real_argmax_t_plus_{offset}"] = float("nan")

                if ce_loss_weight > 0.0:
                    loss_ce_k = _future_cross_entropy_from_hidden(
                        predicted_hidden=predicted_hidden,
                        labels=labels,
                        lm_head=lm_head,
                        offset=offset,
                        attention_mask=attention_mask,
                    )
                else:
                    loss_ce_k = predicted_hidden.new_zeros(())

                ce_losses.append(loss_ce_k)
                per_offset_metrics[f"ce_t_plus_{offset}"] = loss_ce_k.item()

            loss_hidden = torch.stack(hidden_losses).mean()
            loss_ce = torch.stack(ce_losses).mean() if len(ce_losses) > 0 else final_hidden.new_zeros(())
            loss = hidden_loss_weight * loss_hidden + ce_loss_weight * loss_ce

            loss.backward()

            if max_grad_norm is not None and max_grad_norm > 0:
                clip_grad_norm_(future_heads.parameters(), max_grad_norm)

            optimizer.step()

            with torch.no_grad():
                next_token_logits_from_base_hidden = cast(torch.Tensor, lm_head(final_hidden))
                next_token_ce_base = F.cross_entropy(
                    next_token_logits_from_base_hidden[:, :-1, :].contiguous().view(-1, next_token_logits_from_base_hidden.size(-1)).float(),
                    labels[:, 1:].contiguous().view(-1),
                )

            row = {
                "step": float(step),
                "epoch": float(epoch_idx + 1),
                "loss": loss.item(),
                "loss_hidden": loss_hidden.item(),
                "loss_ce": loss_ce.item(),
                "base_next_token_ce": next_token_ce_base.item(),
                **per_offset_metrics,
            }
            history.append(row)

            if compute_distribution_metrics:
                metrics_str = " ".join(
                    f"h+{offset}={row[f'hidden_mse_t_plus_{offset}']:.2f} "
                    f"kl+{offset}={row[f'kl_real_to_pred_t_plus_{offset}']:.2f} "
                    f"top1+{offset}={row[f'top1_t_plus_{offset}']:.3f}"
                    for offset in range(1, num_future_steps + 1)
                )
            else:
                metrics_str = " ".join(
                    f"h+{offset}={row[f'hidden_mse_t_plus_{offset}']:.2f}"
                    for offset in range(1, num_future_steps + 1)
                )

            print(
                f"[step {step:>5}] "
                f"epoch={epoch_idx + 1}/{num_epochs} "
                f"batch={batch_idx}/{len(dataloader)} "
                f"loss={row['loss']:.4f} "
                f"hidden={row['loss_hidden']:.4f} "
                f"ce={row['loss_ce']:.4f} "
                f"base_ce={row['base_next_token_ce']:.4f} "
                f"{metrics_str}"
            )

        if max_steps is not None and step >= max_steps:
            break

    checkpoint_path: Path | None = None
    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        checkpoint_path = checkpoint_dir / (
            f"future_hidden_heads__{model_name.replace('/', '_')}__"
            f"future_{num_future_steps}__{timestamp}.pt"
        )

        torch.save(
            {
                "model_name": model_name,
                "context_len": context_len,
                "num_future_steps": num_future_steps,
                "hidden_size": hidden_size,
                "bottleneck_dim": bottleneck_dim,
                "future_heads_state_dict": future_heads.state_dict(),
                "history": history,
            },
            checkpoint_path,
        )

        _stage(f"saved future hidden heads checkpoint to {checkpoint_path}")

    _stage("train_future_hidden_heads: finished")

    return TrainFutureHiddenHeadsOutput(
        future_heads=future_heads,
        history=history,
        checkpoint_path=checkpoint_path,
    )


# Example entrypoint branch to add in your main():
#
# elif mode == "train_future_hidden_heads":
#     from skip_search_spec.training.train_future_hidden_heads import train_future_hidden_heads
#
#     DATASET_SPEC = DatasetSpec(
#         name="TinyStories",
#         huggingface_path="roneneldan/TinyStories",
#         config_name="default",
#         split="train",
#         text_field="text",
#     )
#
#     out = train_future_hidden_heads(
#         model_name="Qwen/Qwen2.5-0.5B",
#         dataset_spec=DATASET_SPEC,
#         context_len=256,
#         max_examples=10000,
#         num_windows_to_use=4000,
#         batch_size=8,
#         num_future_steps=2,
#         bottleneck_dim=256,
#         num_epochs=1,
#         max_steps=2000,
#         lr=1e-4,
#         hidden_loss_weight=1.0,
#         ce_loss_weight=0.0,
#     )
