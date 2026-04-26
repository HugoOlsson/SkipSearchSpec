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


def _get_token_embedding_module(model: Any) -> nn.Embedding:
    backbone = _get_backbone(model)
    embed_tokens = getattr(backbone, "embed_tokens", None)
    if isinstance(embed_tokens, nn.Embedding):
        return embed_tokens
    raise TypeError("Unsupported model structure. Expected backbone.embed_tokens to be nn.Embedding.")


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


def _build_history_stack(
    sequence: torch.Tensor,
    history_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
      sequence: [B, T, H]
      history_len: number of states/tokens in the rolling history

    Returns:
      history: [B, T, history_len, H], ordered oldest -> newest
      valid_mask: [B, T], true where the full history is real
    """
    if history_len <= 0:
        raise ValueError(f"history_len must be > 0, got {history_len}")

    B, T, H = sequence.shape
    k = history_len

    zeros = torch.zeros(B, k - 1, H, device=sequence.device, dtype=sequence.dtype)
    padded = torch.cat([zeros, sequence], dim=1)  # [B, T + k - 1, H]

    parts = [padded[:, i:i + T, :].unsqueeze(2) for i in range(k)]
    history = torch.cat(parts, dim=2)  # [B, T, k, H]

    valid_mask = torch.zeros(B, T, device=sequence.device, dtype=torch.bool)
    valid_mask[:, k - 1:] = True

    return history, valid_mask


def _roll_history_stack(
    history: torch.Tensor,
    new_item: torch.Tensor,
) -> torch.Tensor:
    """
    Args:
      history: [B, T, K, H]
      new_item: [B, T, H]

    Returns:
      updated history: [B, T, K, H]
    """
    return torch.cat([history[:, :, 1:, :], new_item.unsqueeze(2)], dim=2)


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

    cos_sim = (pred_norm * tgt_norm).sum(dim=-1)
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


class _TwoLayerMLP(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        bias: bool = False,
        zero_init_output: bool = False,
    ) -> None:
        super().__init__()

        self.norm = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias=bias)

        with torch.no_grad():
            nn.init.normal_(self.fc1.weight, mean=0.0, std=0.02)
            if zero_init_output:
                nn.init.zeros_(self.fc2.weight)
            else:
                nn.init.normal_(self.fc2.weight, mean=0.0, std=0.02)

            if self.fc1.bias is not None:
                nn.init.zeros_(self.fc1.bias)
            if self.fc2.bias is not None:
                nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x.float())
        return self.fc2(F.silu(self.fc1(x)))


def _soft_or_topk_token_embedding(
    *,
    logits: torch.Tensor,
    embedding_weight: torch.Tensor,
    temperature: float = 1.0,
    topk: int | None = 8,
) -> torch.Tensor:
    """
    Args:
      logits: [B, T, V]
      embedding_weight: [V, H]

    Returns:
      expected embedding: [B, T, H]
    """
    if temperature <= 0.0:
        raise ValueError(f"temperature must be > 0, got {temperature}")

    vocab_size = logits.size(-1)

    if topk is None or topk <= 0 or topk >= vocab_size:
        probs = F.softmax(logits.float() / temperature, dim=-1).to(dtype=embedding_weight.dtype)
        return torch.matmul(probs, embedding_weight)

    topk_logits, topk_indices = torch.topk(logits.float(), k=topk, dim=-1)
    topk_probs = F.softmax(topk_logits / temperature, dim=-1).to(dtype=embedding_weight.dtype)
    topk_embeddings = F.embedding(topk_indices, embedding_weight)  # [B, T, K, H]
    return torch.sum(topk_probs.unsqueeze(-1) * topk_embeddings, dim=2)


class FutureHiddenTokenRollout(nn.Module):
    """
    Recurrent rollout module:

    1. Encode hidden-history branch
    2. Encode token-history branch
    3. Predict next-token logits
    4. Convert logits -> predicted next-token embedding
    5. Use hidden branch + token-history branch + predicted-next-token branch
       to predict the next hidden state
    6. Roll the history window and repeat
    """
    def __init__(
        self,
        *,
        hidden_size: int,
        num_input_states: int,
        branch_hidden_dim: int | None = None,
        part_dim: int | None = None,
        fuse_hidden_dim: int | None = None,
        bias: bool = False,
        token_embedding_topk: int | None = 8,
        token_embedding_temperature: float = 1.0,
    ) -> None:
        super().__init__()

        if num_input_states <= 0:
            raise ValueError(f"num_input_states must be > 0, got {num_input_states}")

        self.hidden_size = hidden_size
        self.num_input_states = num_input_states
        self.token_embedding_topk = token_embedding_topk
        self.token_embedding_temperature = token_embedding_temperature

        history_dim = num_input_states * hidden_size
        branch_hidden_dim = branch_hidden_dim or hidden_size
        part_dim = part_dim or hidden_size
        fuse_hidden_dim = fuse_hidden_dim or hidden_size

        self.hidden_history_encoder = _TwoLayerMLP(
            input_dim=history_dim,
            hidden_dim=branch_hidden_dim,
            output_dim=part_dim,
            bias=bias,
            zero_init_output=False,
        )
        self.token_history_encoder = _TwoLayerMLP(
            input_dim=history_dim,
            hidden_dim=branch_hidden_dim,
            output_dim=part_dim,
            bias=bias,
            zero_init_output=False,
        )

        self.token_predictor = _TwoLayerMLP(
            input_dim=(part_dim + part_dim + hidden_size),
            hidden_dim=fuse_hidden_dim,
            output_dim=hidden_size,
            bias=bias,
            zero_init_output=True,
        )

        self.next_token_encoder = _TwoLayerMLP(
            input_dim=hidden_size,
            hidden_dim=branch_hidden_dim,
            output_dim=part_dim,
            bias=bias,
            zero_init_output=False,
        )

        self.hidden_predictor = _TwoLayerMLP(
            input_dim=(part_dim + part_dim + part_dim + hidden_size),
            hidden_dim=fuse_hidden_dim,
            output_dim=hidden_size,
            bias=bias,
            zero_init_output=True,
        )

    def _predict_one_step(
        self,
        *,
        hidden_history: torch.Tensor,   # [B, T, K, H]
        token_history: torch.Tensor,    # [B, T, K, H]
        lm_head: nn.Module,
        embedding_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, K, H = hidden_history.shape
        del B, T, K, H

        hidden_flat = hidden_history.reshape(hidden_history.size(0), hidden_history.size(1), -1)
        token_flat = token_history.reshape(token_history.size(0), token_history.size(1), -1)
        current_hidden = hidden_history[:, :, -1, :]

        h_part = self.hidden_history_encoder(hidden_flat)
        t_hist_part = self.token_history_encoder(token_flat)

        token_features = torch.cat([h_part, t_hist_part, current_hidden.float()], dim=-1)
        token_query = current_hidden.float() + self.token_predictor(token_features)

        lm_head_dtype = next(lm_head.parameters()).dtype
        next_token_logits = cast(
            torch.Tensor,
            lm_head(token_query.to(dtype=lm_head_dtype))
        ).float()

        next_token_embedding = _soft_or_topk_token_embedding(
            logits=next_token_logits,
            embedding_weight=embedding_weight,
            temperature=self.token_embedding_temperature,
            topk=self.token_embedding_topk,
        )

        t_next_part = self.next_token_encoder(next_token_embedding.float())

        hidden_features = torch.cat(
            [h_part, t_hist_part, t_next_part, current_hidden.float()],
            dim=-1,
        )
        next_hidden = current_hidden.float() + self.hidden_predictor(hidden_features)

        return next_hidden, next_token_logits, next_token_embedding

    def forward(
        self,
        *,
        hidden_history: torch.Tensor,   # [B, T, K, H]
        token_history: torch.Tensor,    # [B, T, K, H]
        lm_head: nn.Module,
        embedding_weight: torch.Tensor,
        num_future_steps: int,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        if num_future_steps <= 0:
            raise ValueError(f"num_future_steps must be > 0, got {num_future_steps}")

        rolled_hidden_history = hidden_history.float()
        rolled_token_history = token_history.float()

        predicted_hidden_list: list[torch.Tensor] = []
        predicted_token_logits_list: list[torch.Tensor] = []
        predicted_token_embedding_list: list[torch.Tensor] = []

        for _ in range(num_future_steps):
            next_hidden, next_token_logits, next_token_embedding = self._predict_one_step(
                hidden_history=rolled_hidden_history,
                token_history=rolled_token_history,
                lm_head=lm_head,
                embedding_weight=embedding_weight,
            )

            predicted_hidden_list.append(next_hidden)
            predicted_token_logits_list.append(next_token_logits)
            predicted_token_embedding_list.append(next_token_embedding)

            rolled_hidden_history = _roll_history_stack(rolled_hidden_history, next_hidden)
            rolled_token_history = _roll_history_stack(rolled_token_history, next_token_embedding)

        return predicted_hidden_list, predicted_token_logits_list, predicted_token_embedding_list


def train_future_hidden_heads2(
    *,
    model_name: str,
    dataset_spec: DatasetSpec,
    context_len: int,
    max_examples: int,
    num_windows_to_use: int = 256,
    batch_size: int = 2,
    num_future_steps: int = 2,
    num_epochs: int = 1,
    num_input_states: int = 4,
    max_steps: int | None = None,
    lr: float = 1e-4,
    weight_decay: float = 0.0,
    max_grad_norm: float = 1.0,
    hidden_loss_weight: float = 0.25,
    cosine_loss_weight: float = 0.25,
    kl_loss_weight: float = 1.0,
    ce_loss_weight: float = 0.25,
    branch_hidden_dim: int | None = None,
    part_dim: int | None = None,
    fuse_hidden_dim: int | None = None,
    token_embedding_topk: int | None = 8,
    token_embedding_temperature: float = 1.0,
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
    token_embedding_module = _get_token_embedding_module(model)

    lm_head.eval()
    token_embedding_module.eval()

    future_heads = FutureHiddenTokenRollout(
        hidden_size=hidden_size,
        num_input_states=num_input_states,
        branch_hidden_dim=branch_hidden_dim,
        part_dim=part_dim,
        fuse_hidden_dim=fuse_hidden_dim,
        bias=False,
        token_embedding_topk=token_embedding_topk,
        token_embedding_temperature=token_embedding_temperature,
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
        f"training future hidden/token rollout with hidden_size={hidden_size}, "
        f"num_future_steps={num_future_steps}, num_input_states={num_input_states}"
    )

    for epoch_idx in range(num_epochs):
        for batch_idx, (input_ids, attention_mask) in enumerate(dataloader, start=1):
            if max_steps is not None and step >= max_steps:
                break

            step += 1
            compute_distribution_metrics = (step % 10 == 0)

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
                real_token_embeddings = token_embedding_module(input_ids)

            hidden_history, history_valid_mask = _build_history_stack(
                final_hidden,
                history_len=num_input_states,
            )
            token_history, token_history_valid_mask = _build_history_stack(
                real_token_embeddings,
                history_len=num_input_states,
            )

            if not torch.equal(history_valid_mask, token_history_valid_mask):
                raise RuntimeError("Hidden-history and token-history valid masks do not match.")

            embedding_weight = token_embedding_module.weight.detach()

            predicted_future_hidden_list, predicted_future_token_logits_list, _ = future_heads(
                hidden_history=hidden_history,
                token_history=token_history,
                lm_head=lm_head,
                embedding_weight=embedding_weight,
                num_future_steps=num_future_steps,
            )

            hidden_losses: list[torch.Tensor] = []
            cosine_losses: list[torch.Tensor] = []
            kl_losses: list[torch.Tensor] = []
            ce_losses: list[torch.Tensor] = []
            per_offset_metrics: dict[str, float] = {}

            T = input_ids.size(1)

            for offset, (predicted_hidden, predicted_token_logits) in enumerate(
                zip(predicted_future_hidden_list, predicted_future_token_logits_list),
                start=1,
            ):
                if T - offset <= 0:
                    break

                # Hidden target: h_{t+offset}
                predicted_hidden_trimmed = predicted_hidden[:, :-offset, :].contiguous()
                target_hidden_trimmed = final_hidden[:, offset:, :].contiguous()

                # Token target: token at position t+offset
                predicted_token_logits_trimmed = predicted_token_logits[:, :-offset, :].contiguous()
                target_token_logits_trimmed = base_logits[:, offset - 1:-1, :].contiguous()
                target_token_ids_trimmed = labels[:, offset:].contiguous()

                recent_valid_trimmed = history_valid_mask[:, :-offset]
                future_valid = _future_valid_mask(attention_mask=attention_mask, offset=offset)

                if future_valid is None:
                    valid_mask = recent_valid_trimmed
                else:
                    valid_mask = recent_valid_trimmed & future_valid

                loss_hidden_k = _masked_mse(
                    predicted=predicted_hidden_trimmed,
                    target=target_hidden_trimmed,
                    mask=valid_mask,
                )
                hidden_losses.append(loss_hidden_k)
                per_offset_metrics[f"hidden_mse_t_plus_{offset}"] = loss_hidden_k.item()

                loss_cosine_k = _masked_cosine_loss(
                    predicted=predicted_hidden_trimmed,
                    target=target_hidden_trimmed,
                    mask=valid_mask,
                )
                cosine_losses.append(loss_cosine_k)
                per_offset_metrics[f"cosine_loss_t_plus_{offset}"] = loss_cosine_k.item()

                loss_kl_k = _masked_kl_from_logits(
                    predicted_logits=predicted_token_logits_trimmed,
                    target_logits=target_token_logits_trimmed,
                    mask=valid_mask,
                )
                kl_losses.append(loss_kl_k)
                per_offset_metrics[f"kl_loss_t_plus_{offset}"] = loss_kl_k.item()

                if ce_loss_weight > 0.0:
                    loss_ce_k = _masked_cross_entropy_from_logits(
                        logits=predicted_token_logits_trimmed,
                        targets=target_token_ids_trimmed,
                        mask=valid_mask,
                    )
                else:
                    loss_ce_k = predicted_hidden.new_zeros(())
                ce_losses.append(loss_ce_k)
                per_offset_metrics[f"ce_t_plus_{offset}"] = loss_ce_k.item()

                if compute_distribution_metrics:
                    valid_positions = valid_mask.reshape(-1)

                    pred_flat = predicted_token_logits_trimmed.reshape(
                        -1,
                        predicted_token_logits_trimmed.size(-1),
                    )
                    tgt_flat = target_token_logits_trimmed.reshape(
                        -1,
                        target_token_logits_trimmed.size(-1),
                    )

                    pred_flat = pred_flat[valid_positions]
                    tgt_flat = tgt_flat[valid_positions]

                    if pred_flat.numel() > 0 and tgt_flat.numel() > 0:
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

                        real_next_tokens_flat = target_token_ids_trimmed.reshape(-1)[valid_positions]
                        pred_argmax_flat = pred_flat.argmax(dim=-1)
                        top1_vs_real = (pred_argmax_flat == real_next_tokens_flat).float().mean()

                        per_offset_metrics[f"top1_vs_real_token_t_plus_{offset}"] = top1_vs_real.item()
                    else:
                        per_offset_metrics[f"kl_real_to_pred_t_plus_{offset}"] = float("nan")
                        per_offset_metrics[f"kl_pred_to_real_t_plus_{offset}"] = float("nan")
                        per_offset_metrics[f"js_t_plus_{offset}"] = float("nan")
                        per_offset_metrics[f"top1_t_plus_{offset}"] = float("nan")
                        per_offset_metrics[f"overlap_t_plus_{offset}"] = float("nan")
                        per_offset_metrics[f"p_pred_on_real_argmax_t_plus_{offset}"] = float("nan")
                        per_offset_metrics[f"top1_vs_real_token_t_plus_{offset}"] = float("nan")
                else:
                    per_offset_metrics[f"kl_real_to_pred_t_plus_{offset}"] = float("nan")
                    per_offset_metrics[f"kl_pred_to_real_t_plus_{offset}"] = float("nan")
                    per_offset_metrics[f"js_t_plus_{offset}"] = float("nan")
                    per_offset_metrics[f"top1_t_plus_{offset}"] = float("nan")
                    per_offset_metrics[f"overlap_t_plus_{offset}"] = float("nan")
                    per_offset_metrics[f"p_pred_on_real_argmax_t_plus_{offset}"] = float("nan")
                    per_offset_metrics[f"top1_vs_real_token_t_plus_{offset}"] = float("nan")

            loss_hidden = (
                torch.stack(hidden_losses).mean()
                if len(hidden_losses) > 0
                else final_hidden.new_zeros(())
            )
            loss_cosine = (
                torch.stack(cosine_losses).mean()
                if len(cosine_losses) > 0
                else final_hidden.new_zeros(())
            )
            loss_kl = (
                torch.stack(kl_losses).mean()
                if len(kl_losses) > 0
                else final_hidden.new_zeros(())
            )
            loss_ce = (
                torch.stack(ce_losses).mean()
                if len(ce_losses) > 0
                else final_hidden.new_zeros(())
            )

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
                **per_offset_metrics,
            }
            history.append(row)

            if compute_distribution_metrics:
                metrics_str = " ".join(
                    f"h+{offset}={row[f'hidden_mse_t_plus_{offset}']:.2f} "
                    f"kl+{offset}={row[f'kl_real_to_pred_t_plus_{offset}']:.2f} "
                    f"tok1+{offset}={row[f'top1_vs_real_token_t_plus_{offset}']:.3f}"
                    f"top1+{offset}={row[f'top1_t_plus_{offset}']:.3f}"
                    for offset in range(1, num_future_steps + 1)
                    if f"hidden_mse_t_plus_{offset}" in row
                )
            else:
                metrics_str = " ".join(
                    f"h+{offset}={row[f'hidden_mse_t_plus_{offset}']:.2f}"
                    for offset in range(1, num_future_steps + 1)
                    if f"hidden_mse_t_plus_{offset}" in row
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
                "num_input_states": num_input_states,
                "future_heads_state_dict": future_heads.state_dict(),
                "history": history,
                "branch_hidden_dim": branch_hidden_dim,
                "part_dim": part_dim,
                "fuse_hidden_dim": fuse_hidden_dim,
                "token_embedding_topk": token_embedding_topk,
                "token_embedding_temperature": token_embedding_temperature,
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
#         num_input_states=4,
#         max_steps=2000,
#         lr=1e-4,
#         hidden_loss_weight=0.25,
#         cosine_loss_weight=0.25,
#         kl_loss_weight=1.0,
#         ce_loss_weight=0.25,
#         token_embedding_topk=8,
#     )