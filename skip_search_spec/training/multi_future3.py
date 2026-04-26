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
    build_window_index,
    collate_windows,
    tokenize_dataset_to_examples,
)
from skip_search_spec.protocols.windows import (
    DatasetSpec,
    ModelAndTokenizer,
    WindowSettings,
)


@dataclass(slots=True)
class TrainNextHiddenPredictorOutput:
    predictor: nn.Module
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


def _get_input_embedding_module(model: Any) -> nn.Module:
    backbone = _get_backbone(model)
    embed_tokens = getattr(backbone, "embed_tokens", None)
    if isinstance(embed_tokens, nn.Module):
        return embed_tokens
    raise TypeError("Unsupported model structure. Expected model.model.embed_tokens.")


def _build_recent_stack(
    sequence: torch.Tensor,
    num_items: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Builds a newest-first rolling stack from a sequence.

    Args:
        sequence: [B, T, D]
        num_items: number of items in the stack

    Returns:
        stacked: [B, T, num_items, D], where stacked[:, t] is
                 [x_t, x_{t-1}, ..., x_{t-num_items+1}]
        valid_mask: [B, T], true where all stack items are real
    """
    if num_items <= 0:
        raise ValueError(f"num_items must be > 0, got {num_items}")

    B, T, D = sequence.shape

    if num_items == 1:
        valid_mask = torch.ones(B, T, device=sequence.device, dtype=torch.bool)
        return sequence.unsqueeze(2), valid_mask

    zeros = torch.zeros(
        B,
        num_items - 1,
        D,
        device=sequence.device,
        dtype=sequence.dtype,
    )
    padded = torch.cat([zeros, sequence], dim=1)  # [B, T + num_items - 1, D]

    parts = []
    for i in range(num_items):
        start = (num_items - 1) - i
        parts.append(padded[:, start:start + T, :])

    stacked = torch.stack(parts, dim=2)  # [B, T, num_items, D]

    valid_mask = torch.zeros(B, T, device=sequence.device, dtype=torch.bool)
    valid_mask[:, num_items - 1:] = True

    return stacked, valid_mask


def _build_teacher_forced_token_stack(
    real_token_embeddings: torch.Tensor,
    num_input_states: int,
) -> torch.Tensor:
    """
    Returns token stack [B, T, K, H] with newest-first layout:

        [T_{t+1}, T_t, T_{t-1}, ..., T_{t-K+2}]

    where T_{t+1} is the *dataset* next token embedding.
    """
    if num_input_states <= 0:
        raise ValueError(f"num_input_states must be > 0, got {num_input_states}")

    B, T, H = real_token_embeddings.shape

    next_token_embeddings = torch.zeros_like(real_token_embeddings)
    next_token_embeddings[:, :-1, :] = real_token_embeddings[:, 1:, :]

    if num_input_states == 1:
        return next_token_embeddings.unsqueeze(2)

    past_token_stack, _ = _build_recent_stack(
        real_token_embeddings,
        num_input_states - 1,
    )  # [B, T, K-1, H] = [T_t, T_{t-1}, ...]

    token_stack = torch.cat(
        [
            next_token_embeddings.unsqueeze(2),  # [B, T, 1, H]
            past_token_stack,                    # [B, T, K-1, H]
        ],
        dim=2,
    )  # [B, T, K, H]

    return token_stack


class NextHiddenPredictor(nn.Module):
    """
    Vanilla residual MLP predictor for h_{t+1}.

    Uses raw inputs:
      - h_t
      - h_{t-1}
      - h_{t-2}
      - T_{t+1}
      - T_t
      - T_{t-1}

    Residual form:
      output = h_t + delta

    With hidden_size=3584 and bottleneck_dim=384:
      about 9.65M trainable params
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        num_input_states: int,
        bottleneck_dim: int = 384,
        bias: bool = True,
    ) -> None:
        super().__init__()

        if num_input_states <= 0 or num_input_states > 3:
            raise ValueError(f"Expected 1 <= num_input_states <= 3, got {num_input_states}")

        self.hidden_size = hidden_size
        self.num_input_states = num_input_states

        self.hidden_norm = nn.LayerNorm(hidden_size)
        self.token_norm = nn.LayerNorm(hidden_size)

        # 6 raw feature blocks of size H:
        # [h_t, h_{t-1}, h_{t-2}, T_{t+1}, T_t, T_{t-1}]
        input_dim = hidden_size * 6

        self.fc1 = nn.Linear(input_dim, bottleneck_dim, bias=bias)
        self.fc2 = nn.Linear(bottleneck_dim, hidden_size, bias=False)

        with torch.no_grad():
            nn.init.normal_(self.fc1.weight, mean=0.0, std=0.02)
            if self.fc1.bias is not None:
                nn.init.zeros_(self.fc1.bias)

            # start as identity: output = h_t + 0
            nn.init.zeros_(self.fc2.weight)

    def forward(
        self,
        hidden_stack: torch.Tensor,  # [B, T, K, H]
        token_stack: torch.Tensor,   # [B, T, K, H]
    ) -> torch.Tensor:
        if hidden_stack.shape != token_stack.shape:
            raise ValueError(
                f"hidden_stack.shape {hidden_stack.shape} != token_stack.shape {token_stack.shape}"
            )

        B, T, K, H = hidden_stack.shape

        if H != self.hidden_size:
            raise ValueError(f"Expected hidden size {self.hidden_size}, got {H}")
        if K != self.num_input_states:
            raise ValueError(f"Expected num_input_states={self.num_input_states}, got {K}")

        zeros = torch.zeros(B, T, H, device=hidden_stack.device, dtype=torch.float32)

        h0_raw = hidden_stack[:, :, 0, :].float()
        h0 = self.hidden_norm(h0_raw)
        h1 = self.hidden_norm(hidden_stack[:, :, 1, :].float()) if K >= 2 else zeros
        h2 = self.hidden_norm(hidden_stack[:, :, 2, :].float()) if K >= 3 else zeros

        t0 = self.token_norm(token_stack[:, :, 0, :].float())  # T_{t+1}
        t1 = self.token_norm(token_stack[:, :, 1, :].float()) if K >= 2 else zeros
        t2 = self.token_norm(token_stack[:, :, 2, :].float()) if K >= 3 else zeros

        feats = torch.cat([h0, h1, h2, t0, t1, t2], dim=-1)  # [B, T, 6H]

        delta = self.fc2(F.silu(self.fc1(feats)))
        return h0_raw + delta


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
    return valid.bool()


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
    )

    kl_per_pos = kl_per_vocab.sum(dim=-1)

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

def count_trainable_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

def train_next_hidden_teacher_forced(
    *,
    model_name: str,
    dataset_spec: DatasetSpec,
    context_len: int,
    max_examples: int,
    num_windows_to_use: int = 256,
    batch_size: int = 2,
    num_epochs: int = 1,
    num_input_states: int = 3,
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
) -> TrainNextHiddenPredictorOutput:
    _stage("train_next_hidden_teacher_forced: start")

    if context_len < 2:
        raise ValueError(f"context_len must be >= 2, got {context_len}")

    if num_input_states <= 0:
        raise ValueError(f"num_input_states must be > 0, got {num_input_states}")

    if context_len <= num_input_states:
        raise ValueError(
            f"context_len must be > num_input_states so at least one valid h_t -> h_t+1 "
            f"training position exists, got context_len={context_len}, "
            f"num_input_states={num_input_states}"
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
    embed_tokens = _get_input_embedding_module(model)

    lm_head.eval()
    embed_tokens.eval()

    predictor = NextHiddenPredictor(
        hidden_size=hidden_size,
        num_input_states=num_input_states,
    ).to(device=device, dtype=torch.float32)
    predictor.train()

    num_params = count_trainable_parameters(predictor)
    print(f"predictor trainable params: {num_params:,}")

    optimizer = torch.optim.AdamW(
        predictor.parameters(),
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

    _stage("building window index")
    window_index = build_window_index(
        tokenized_examples,
        window_settings,
    )

    if len(window_index) < num_windows_to_use:
        raise ValueError(
            f"Requested {num_windows_to_use} windows, but only built {len(window_index)}."
        )


    selected_window_index = window_index[:num_windows_to_use]

    window_dataset = WindowDataset(
        tokenized_examples=tokenized_examples,
        window_index=selected_window_index,
        window_settings=window_settings,
    )


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
        f"training teacher-forced next-hidden predictor with hidden_size={hidden_size}, "
        f"num_input_states={num_input_states}"
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

                real_token_embeddings = cast(torch.Tensor, embed_tokens(input_ids))

            # hidden_stack[:, t] = [h_t, h_{t-1}, ...]
            hidden_stack, hidden_history_valid_mask = _build_recent_stack(
                final_hidden,
                num_input_states,
            )

            # token_stack[:, t] = [T_{t+1}, T_t, T_{t-1}, ...]
            token_stack = _build_teacher_forced_token_stack(
                real_token_embeddings,
                num_input_states,
            )

            predicted_next_hidden = predictor(hidden_stack, token_stack)

            # Align prediction at position t with target h_{t+1}
            predicted_trimmed = predicted_next_hidden[:, :-1, :].contiguous()
            target_trimmed = final_hidden[:, 1:, :].contiguous()

            valid_mask = hidden_history_valid_mask[:, :-1]
            future_valid = _future_valid_mask(attention_mask=attention_mask, offset=1)
            if future_valid is not None:
                valid_mask = valid_mask & future_valid

            loss_hidden = _masked_mse(
                predicted=predicted_trimmed,
                target=target_trimmed,
                mask=valid_mask,
            )

            loss_cosine = _masked_cosine_loss(
                predicted=predicted_trimmed,
                target=target_trimmed,
                mask=valid_mask,
            )

            lm_head_dtype = next(lm_head.parameters()).dtype

            predicted_logits = cast(
                torch.Tensor,
                lm_head(predicted_trimmed.to(dtype=lm_head_dtype)),
            )
            target_logits = cast(
                torch.Tensor,
                lm_head(target_trimmed.to(dtype=lm_head_dtype)),
            )

            loss_kl = _masked_kl_from_logits(
                predicted_logits=predicted_logits,
                target_logits=target_logits,
                mask=valid_mask,
            )


            if ce_loss_weight > 0.0:
               teacher_targets = target_logits.argmax(dim=-1)  # [B, T-1]

               loss_ce = _masked_cross_entropy_from_logits(
                    logits=predicted_logits,   # [B, T-1, V]
                    targets=teacher_targets,   # [B, T-1]
                    mask=valid_mask,           # [B, T-1]
                )
            else:
                loss_ce = predicted_trimmed.new_zeros(())

            loss = (
                hidden_loss_weight * loss_hidden
                + cosine_loss_weight * loss_cosine
                + kl_loss_weight * loss_kl
                + ce_loss_weight * loss_ce
            )

            loss.backward()

            if max_grad_norm is not None and max_grad_norm > 0:
                clip_grad_norm_(predictor.parameters(), max_grad_norm)

            optimizer.step()

            per_metrics: dict[str, float] = {
                "hidden_mse_t_plus_1": loss_hidden.item(),
                "cosine_loss_t_plus_1": loss_cosine.item(),
                "kl_loss_t_plus_1": loss_kl.item(),
                "ce_t_plus_1": loss_ce.item(),
            }

            if compute_distribution_metrics:
                with torch.no_grad():
                    valid_positions = valid_mask.reshape(-1)

                    pred_flat = predicted_logits.reshape(-1, predicted_logits.size(-1))
                    tgt_flat = target_logits.reshape(-1, target_logits.size(-1))

                    pred_flat = pred_flat[valid_positions]
                    tgt_flat = tgt_flat[valid_positions]

                    if pred_flat.size(0) > 0:
                        sim = distribution_similarity_metrics(
                            shift_logits_mid=pred_flat,
                            shift_logits_full=tgt_flat,
                        )

                        per_metrics["kl_real_to_pred_t_plus_1"] = sim["kl_full_to_mid"].item()
                        per_metrics["kl_pred_to_real_t_plus_1"] = sim["kl_mid_to_full"].item()
                        per_metrics["js_t_plus_1"] = sim["js"].item()
                        per_metrics["top1_t_plus_1"] = sim["top1_agreement"].item()
                        per_metrics["overlap_t_plus_1"] = sim["overlap"].item()
                        per_metrics["p_pred_on_real_argmax_t_plus_1"] = sim["p_mid_on_full_argmax"].item()
                    else:
                        per_metrics["kl_real_to_pred_t_plus_1"] = float("nan")
                        per_metrics["kl_pred_to_real_t_plus_1"] = float("nan")
                        per_metrics["js_t_plus_1"] = float("nan")
                        per_metrics["top1_t_plus_1"] = float("nan")
                        per_metrics["overlap_t_plus_1"] = float("nan")
                        per_metrics["p_pred_on_real_argmax_t_plus_1"] = float("nan")
            else:
                per_metrics["kl_real_to_pred_t_plus_1"] = float("nan")
                per_metrics["kl_pred_to_real_t_plus_1"] = float("nan")
                per_metrics["js_t_plus_1"] = float("nan")
                per_metrics["top1_t_plus_1"] = float("nan")
                per_metrics["overlap_t_plus_1"] = float("nan")
                per_metrics["p_pred_on_real_argmax_t_plus_1"] = float("nan")

            with torch.no_grad():
                base_next_token_ce = F.cross_entropy(
                    base_logits[:, :-1, :].contiguous().view(-1, base_logits.size(-1)).float(),
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
                "base_next_token_ce": base_next_token_ce.item(),
                **per_metrics,
            }
            history.append(row)

            if compute_distribution_metrics:
                metrics_str = (
                    f"h+1={row['hidden_mse_t_plus_1']:.2f} "
                    f"kl+1={row['kl_real_to_pred_t_plus_1']:.2f} "
                    f"top1+1={row['top1_t_plus_1']:.3f}"
                )
            else:
                metrics_str = f"h+1={row['hidden_mse_t_plus_1']:.2f}"

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
            f"next_hidden_teacher_forced__{model_name.replace('/', '_')}__{timestamp}.pt"
        )

        torch.save(
            {
                "model_name": model_name,
                "context_len": context_len,
                "num_input_states": num_input_states,
                "hidden_size": hidden_size,
                "predictor_state_dict": predictor.state_dict(),
                "history": history,
            },
            checkpoint_path,
        )

        _stage(f"saved checkpoint to {checkpoint_path}")

    _stage("train_next_hidden_teacher_forced: finished")

    return TrainNextHiddenPredictorOutput(
        predictor=predictor,
        history=history,
        checkpoint_path=checkpoint_path,
    )


# Example entrypoint branch to add in your main():
#
# elif mode == "train_next_hidden_teacher_forced":
#     from skip_search_spec.training.train_next_hidden_teacher_forced import (
#         train_next_hidden_teacher_forced,
#     )
#
#     DATASET_SPEC = DatasetSpec(
#         name="TinyStories",
#         huggingface_path="roneneldan/TinyStories",
#         config_name="default",
#         split="train",
#         text_field="text",
#     )
#
#     out = train_next_hidden_teacher_forced(
#         model_name="Qwen/Qwen2.5-0.5B",
#         dataset_spec=DATASET_SPEC,
#         context_len=256,
#         max_examples=10000,
#         num_windows_to_use=4000,
#         batch_size=8,
#         num_epochs=1,
#         num_input_states=3,
#         max_steps=2000,
#         lr=1e-4,
#         hidden_loss_weight=0.25,
#         cosine_loss_weight=0.25,
#         kl_loss_weight=1.0,
#         ce_loss_weight=1.0,
#         branch_hidden_dim=None,
#         fusion_hidden_dim=None,
#     )