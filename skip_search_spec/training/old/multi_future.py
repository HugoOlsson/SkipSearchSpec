
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

def _build_stacked_recent_hidden(
    final_hidden: torch.Tensor,
    num_input_states: int = 5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      stacked: [B, T, num_input_states * H] containing
               [h_{t-(k-1)}, ..., h_{t-1}, h_t]
      valid_mask: [B, T] indicating positions where all inputs are real
    """
    if num_input_states <= 0:
        raise ValueError(f"num_input_states must be > 0, got {num_input_states}")

    B, T, H = final_hidden.shape
    k = num_input_states

    zeros = torch.zeros(B, k - 1, H, device=final_hidden.device, dtype=final_hidden.dtype)
    padded = torch.cat([zeros, final_hidden], dim=1)  # [B, T + k - 1, H]

    parts = [
        padded[:, i:i + T, :]
        for i in range(k)
    ]
    stacked = torch.cat(parts, dim=-1)  # [B, T, kH]

    valid_mask = torch.zeros(B, T, device=final_hidden.device, dtype=torch.bool)
    valid_mask[:, k - 1:] = True

    return stacked, valid_mask

class FutureHiddenHead(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        num_input_states: int,
        extra_input_states: int = 0,
        hidden_dim: int | None = None,
        bias: bool = False,
    ) -> None:
        super().__init__()

        input_dim = hidden_size * (num_input_states + 1 + extra_input_states)
        if hidden_dim is None:
            hidden_dim = hidden_size

        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.fc2 = nn.Linear(hidden_dim, hidden_size, bias=bias)

        with torch.no_grad():
            nn.init.normal_(self.fc1.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.fc2.weight)
            if self.fc1.bias is not None:
                nn.init.zeros_(self.fc1.bias)
            if self.fc2.bias is not None:
                nn.init.zeros_(self.fc2.bias)

    def forward(
        self,
        stacked_hidden_states: torch.Tensor,
        current_hidden_state: torch.Tensor,
        extra_hidden_states: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        pieces = [
            stacked_hidden_states.float(),
            current_hidden_state.float(),
        ]

        if extra_hidden_states is not None:
            pieces.extend(x.float() for x in extra_hidden_states)

        x = torch.cat(pieces, dim=-1)
        delta = self.fc2(F.silu(self.fc1(x)))
        return current_hidden_state.float() + delta


class FutureHiddenHeads(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        num_future_steps: int,
        num_input_states: int = 5,
        hidden_dim: int | None = None,
        bias: bool = False,
        extra_input_indices_per_head: list[tuple[int, ...]] | None = None,
    ) -> None:
        super().__init__()
        if num_future_steps <= 0:
            raise ValueError(f"num_future_steps must be > 0, got {num_future_steps}")

        self.num_future_steps = num_future_steps

        if extra_input_indices_per_head is None:
            # Default:
            # t+1 -> []
            # t+2 -> [t+1]
            # t+3 -> [t+2]
            # t+4 -> [t+3]
            # ...
            extra_input_indices_per_head = [
                (() if step_idx == 0 else (step_idx - 1,))
                for step_idx in range(num_future_steps)
            ]

        if len(extra_input_indices_per_head) != num_future_steps:
            raise ValueError(
                "extra_input_indices_per_head must have length equal to num_future_steps"
            )

        for step_idx, dep_indices in enumerate(extra_input_indices_per_head):
            for dep_idx in dep_indices:
                if dep_idx < 0 or dep_idx >= step_idx:
                    raise ValueError(
                        f"Invalid dependency for head {step_idx}: {dep_idx}. "
                        "Each head may only depend on earlier predictions."
                    )

        self.extra_input_indices_per_head = extra_input_indices_per_head

        heads: list[FutureHiddenHead] = []
        for dep_indices in self.extra_input_indices_per_head:
            heads.append(
                FutureHiddenHead(
                    hidden_size=hidden_size,
                    num_input_states=num_input_states,
                    extra_input_states=len(dep_indices),
                    hidden_dim=hidden_dim,
                    bias=bias,
                )
            )

        self.heads = nn.ModuleList(heads)

    def forward(
        self,
        stacked_hidden_states: torch.Tensor,
        current_hidden_states: torch.Tensor,
    ) -> list[torch.Tensor]:
        preds: list[torch.Tensor] = []

        for head, dep_indices in zip(self.heads, self.extra_input_indices_per_head):
            extra_hidden_states = [preds[j] for j in dep_indices]

            pred = head(
                stacked_hidden_states,
                current_hidden_states,
                extra_hidden_states=extra_hidden_states if len(extra_hidden_states) > 0 else None,
            )
            preds.append(pred)

        return preds


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

def _masked_cosine_loss(
    *,
    predicted: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None,
    eps: float = 1e-8,
) -> torch.Tensor:
    pred_f = predicted.float()
    tgt_f = target.float()

    pred_norm = pred_f / pred_f.norm(dim=-1, keepdim=True).clamp_min(eps)
    tgt_norm = tgt_f / tgt_f.norm(dim=-1, keepdim=True).clamp_min(eps)

    cos_sim = (pred_norm * tgt_norm).sum(dim=-1)  # [B, T]
    loss_per_pos = 1.0 - cos_sim

    if mask is None:
        return loss_per_pos.mean()

    mask_f = mask.to(loss_per_pos.dtype)
    denom = mask_f.sum().clamp_min(1.0)
    return (loss_per_pos * mask_f).sum() / denom

def _masked_kl_from_logits(
    *,
    predicted_logits: torch.Tensor,
    target_logits: torch.Tensor,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    pred_log_probs = F.log_softmax(predicted_logits.float(), dim=-1)
    target_probs = F.softmax(target_logits.float(), dim=-1)

    kl_per_vocab = F.kl_div(
        pred_log_probs,
        target_probs,
        reduction="none",
        log_target=False,
    )  # [B, T, V]

    kl_per_pos = kl_per_vocab.sum(dim=-1)  # [B, T]

    if mask is None:
        return kl_per_pos.mean()

    mask_f = mask.to(kl_per_pos.dtype)
    denom = mask_f.sum().clamp_min(1.0)
    return (kl_per_pos * mask_f).sum() / denom

def _masked_cross_entropy_from_logits(
    *,
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    flat_loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)).float(),
        targets.reshape(-1),
        reduction="none",
    ).view_as(targets)

    if mask is None:
        return flat_loss.mean()

    mask_f = mask.to(flat_loss.dtype)
    denom = mask_f.sum().clamp_min(1.0)
    return (flat_loss * mask_f).sum() / denom


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
    num_epochs: int = 1,
    num_input_states:int = 4,
    max_steps: int | None = None,
    lr: float = 1e-4,
    weight_decay: float = 0.0,
    max_grad_norm: float = 1.0,
    hidden_loss_weight: float = 1.0,
    cosine_loss_weight: float = 1.0,
    kl_loss_weight: float = 1.0,
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


    
    
    future_heads = FutureHiddenHeads(
        hidden_size=hidden_size,
        num_future_steps=num_future_steps,
        num_input_states=num_input_states,
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
        f"num_future_steps={num_future_steps}"
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

            stacked_recent_hidden, recent_valid_mask = _build_stacked_recent_hidden(final_hidden,num_input_states=num_input_states)
            predicted_future_hidden_list = future_heads(stacked_recent_hidden, final_hidden)

            hidden_losses: list[torch.Tensor] = []
            cosine_losses: list[torch.Tensor] = []
            ce_losses: list[torch.Tensor] = []
            kl_losses: list[torch.Tensor] = []
            per_offset_metrics: dict[str, float] = {}

            for offset, predicted_hidden in enumerate(predicted_future_hidden_list, start=1):
                predicted_trimmed = predicted_hidden[:, :-offset, :].contiguous()
                target_trimmed = final_hidden[:, offset:, :].contiguous()

                recent_valid_trimmed = recent_valid_mask[:, :-offset]
                future_valid = _future_valid_mask(attention_mask=attention_mask, offset=offset)

                if future_valid is None:
                    valid_mask = recent_valid_trimmed
                else:
                    valid_mask = recent_valid_trimmed & future_valid.bool()

                loss_hidden_k = _masked_mse(
                    predicted=predicted_trimmed,
                    target=target_trimmed,
                    mask=valid_mask,
                )
                hidden_losses.append(loss_hidden_k)
                per_offset_metrics[f"hidden_mse_t_plus_{offset}"] = loss_hidden_k.item()

                loss_cosine_k = _masked_cosine_loss(
                    predicted=predicted_trimmed,
                    target=target_trimmed,
                    mask=valid_mask,
                )
                cosine_losses.append(loss_cosine_k)
                per_offset_metrics[f"cosine_loss_t_plus_{offset}"] = loss_cosine_k.item()

                lm_head_dtype = next(lm_head.parameters()).dtype

                predicted_logits_k = cast(
                    torch.Tensor,
                    lm_head(predicted_trimmed.to(dtype=lm_head_dtype)),
                )
                target_logits_k = cast(
                    torch.Tensor,
                    lm_head(target_trimmed.to(dtype=lm_head_dtype)),
                )

                loss_kl_k = _masked_kl_from_logits(
                    predicted_logits=predicted_logits_k,
                    target_logits=target_logits_k,
                    mask=valid_mask,
                )

                kl_losses.append(loss_kl_k)
                per_offset_metrics[f"kl_loss_t_plus_{offset}"] = loss_kl_k.item()

                if compute_distribution_metrics:
                    valid_positions = valid_mask.reshape(-1)

                    pred_flat = predicted_logits_k.reshape(-1, predicted_logits_k.size(-1))
                    tgt_flat = target_logits_k.reshape(-1, target_logits_k.size(-1))

                    pred_flat = pred_flat[valid_positions]
                    tgt_flat = tgt_flat[valid_positions]

                    sim_k = distribution_similarity_metrics(
                        shift_logits_mid=pred_flat,
                        shift_logits_full=tgt_flat,
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
                    usable_logits = predicted_logits_k
                    teacher_targets = target_logits_k.argmax(dim=-1)

                    loss_ce_k = _masked_cross_entropy_from_logits(
                        logits=usable_logits,
                        targets=teacher_targets,
                        mask=valid_mask,
                    )
                else:
                    loss_ce_k = predicted_hidden.new_zeros(())

                ce_losses.append(loss_ce_k)
                per_offset_metrics[f"ce_t_plus_{offset}"] = loss_ce_k.item()

            loss_hidden = torch.stack(hidden_losses).mean()
            loss_ce = torch.stack(ce_losses).mean() if len(ce_losses) > 0 else final_hidden.new_zeros(())
            loss_cosine = torch.stack(cosine_losses).mean() if len(cosine_losses) > 0 else final_hidden.new_zeros(())
            loss_kl = torch.stack(kl_losses).mean() if len(kl_losses) > 0 else final_hidden.new_zeros(())
            loss = (
                hidden_loss_weight * loss_hidden
                + cosine_loss_weight * loss_cosine
                + kl_loss_weight * loss_kl
                + ce_loss_weight * loss_ce
            )

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
                "loss_cosine": loss_cosine.item(),
                "loss_kl": loss_kl.item(),
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
                f"cos={row['loss_cosine']:.4f} "
                f"kl_loss={row['loss_kl']:.4f} "
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
#         num_epochs=1,
#         max_steps=2000,
#         lr=1e-4,
#         hidden_loss_weight=1.0,
#         ce_loss_weight=0.0,
#     )
