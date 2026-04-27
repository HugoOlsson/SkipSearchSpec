from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast
from contextlib import ExitStack
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


@dataclass(frozen=True, slots=True)
class GapSpec:
    start: int
    length: int

    @property
    def end(self) -> int:
        return self.start + self.length


@dataclass(slots=True)
class TrainGapBridgeOutput:
    bridge: nn.Module
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


def _get_decoder_layers(model: Any) -> Any:
    return _get_backbone(model).layers


def _get_hidden_size(model: Any) -> int:
    hidden_size = getattr(getattr(model, "config", None), "hidden_size", None)
    if isinstance(hidden_size, int) and hidden_size > 0:
        return hidden_size

    backbone = _get_backbone(model)
    if hasattr(backbone, "embed_tokens") and hasattr(backbone.embed_tokens, "embedding_dim"):
        return int(backbone.embed_tokens.embedding_dim)

    raise ValueError("Could not infer hidden size from model.")


def _validate_gap(*, gap: GapSpec, num_layers: int) -> None:
    if gap.length <= 0:
        raise ValueError(f"gap.length must be > 0, got {gap.length}")

    if gap.start < 0:
        raise ValueError(f"gap.start must be >= 0, got {gap.start}")

    if gap.start >= num_layers:
        raise ValueError(
            f"gap.start must be < num_layers, got gap.start={gap.start}, num_layers={num_layers}"
        )

    if gap.end > num_layers:
        raise ValueError(
            f"gap.end must be <= num_layers, got gap.end={gap.end}, num_layers={num_layers}"
        )

def _get_effective_mode(*, gap: GapSpec, num_layers: int) -> str:
    if gap.start == 0:
        return "LATE-BEGIN"
    if gap.end == num_layers:
        return "EARLY-EXIT"
    return "GAP"


def _get_reentry_module_for_gap(*, model: Any, gap: GapSpec) -> Any:
    backbone = _get_backbone(model)
    layers = backbone.layers
    num_layers = len(layers)

    if gap.end < num_layers:
        return layers[gap.end]

    if gap.end == num_layers:
        return backbone.norm

    raise ValueError(f"Invalid gap.end={gap.end} for num_layers={num_layers}")


class GapBridge(nn.Module):
    """
    Predict re-entry hidden from:
      - current hidden entering the gap
      - previous token's final-layer hidden

    For position t, inputs are:
      x_t       = hidden entering the gap at position t
      p_{t-1}   = teacher final hidden from position t-1 (shifted right)

    Output:
      bridged_t = x_t + delta_t
    """

    def __init__(
        self,
        hidden_size: int,
        bottleneck_dim: int = 384,
        bias: bool = True,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size

        self.gap_norm = nn.LayerNorm(hidden_size)
        self.prev_norm = nn.LayerNorm(hidden_size)

        # features: [x_t, p_{t-1}, x_t - p_{t-1}]
        input_dim = hidden_size * 3

        self.fc1 = nn.Linear(input_dim, bottleneck_dim, bias=bias)
        self.fc2 = nn.Linear(bottleneck_dim, hidden_size, bias=False)

        with torch.no_grad():
            nn.init.normal_(self.fc1.weight, mean=0.0, std=0.02)
            if self.fc1.bias is not None:
                nn.init.zeros_(self.fc1.bias)

            # start as identity: output = x_t + 0
            nn.init.zeros_(self.fc2.weight)

    def forward(
        self,
        gap_hidden: torch.Tensor,        # [B, T, H]
        prev_final_hidden: torch.Tensor, # [B, T, H]
    ) -> torch.Tensor:
        if gap_hidden.shape != prev_final_hidden.shape:
            raise ValueError(
                f"gap_hidden.shape {gap_hidden.shape} != prev_final_hidden.shape {prev_final_hidden.shape}"
            )

        x = gap_hidden.float()
        p = prev_final_hidden.float()

        x_n = self.gap_norm(x)
        p_n = self.prev_norm(p)

        feats = torch.cat([x_n, p_n, x_n - p_n], dim=-1)
        delta = self.fc2(F.silu(self.fc1(feats)))

        return x + delta
    
def _build_prev_final_hidden(final_hidden: torch.Tensor) -> torch.Tensor:
    """
    Build shifted previous-token final hidden.

    For position t:
      prev_final_hidden[:, t, :] = final_hidden[:, t-1, :]

    Position 0 gets zeros.
    """
    prev = torch.zeros_like(final_hidden)
    prev[:, 1:, :] = final_hidden[:, :-1, :]
    return prev


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


def _kl_teacher_to_student_next_token(
    *,
    logits_teacher: torch.Tensor,
    logits_student: torch.Tensor,
) -> torch.Tensor:
    shift_teacher = logits_teacher[:, :-1, :].contiguous()
    shift_student = logits_student[:, :-1, :].contiguous()

    log_p_student = F.log_softmax(shift_student.float(), dim=-1)
    log_p_teacher = F.log_softmax(shift_teacher.float(), dim=-1)

    kl_per_token = F.kl_div(
        log_p_student,
        log_p_teacher,
        reduction="none",
        log_target=True,
    ).sum(dim=-1)

    return kl_per_token.mean()


def _masked_hidden_mse_with_first_token_dropped(
    *,
    predicted: torch.Tensor,
    target: torch.Tensor,
    attention_mask: torch.Tensor | None,
) -> torch.Tensor:
    diff_sq = (predicted.float() - target.float()).pow(2)

    if attention_mask is None:
        mask = torch.ones(
            predicted.shape[:2],
            device=predicted.device,
            dtype=torch.bool,
        )
    else:
        mask = attention_mask.bool()

    mask = mask.clone()
    mask[:, 0] = False

    mask_f = mask.unsqueeze(-1).to(diff_sq.dtype)
    denom = (mask_f.sum() * diff_sq.size(-1)).clamp_min(1.0)
    return (diff_sq * mask_f).sum() / denom


@dataclass(slots=True)
class _TeacherCapture:
    reentry_hidden: torch.Tensor | None = None


@dataclass(slots=True)
class _StudentCapture:
    gap_input_hidden: torch.Tensor | None = None
    bridged_hidden: torch.Tensor | None = None


@torch.no_grad()
def _run_teacher_full_and_capture_reentry(
    *,
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    gap: GapSpec,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    reentry_module = _get_reentry_module_for_gap(model=model, gap=gap)
    capture = _TeacherCapture()

    def prehook(module: Any, inputs: tuple[Any, ...]) -> None:
        if len(inputs) == 0 or not isinstance(inputs[0], torch.Tensor):
            raise TypeError("Expected first layer input to be a hidden-state tensor.")
        capture.reentry_hidden = inputs[0].detach()

    with ExitStack() as stack:
        handle = reentry_module.register_forward_pre_hook(prehook)
        stack.callback(handle.remove)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )

    if capture.reentry_hidden is None:
        raise RuntimeError("Failed to capture teacher re-entry hidden state.")

    hidden_states = getattr(outputs, "hidden_states", None)
    if not isinstance(hidden_states, tuple) or len(hidden_states) == 0:
        raise RuntimeError("Model did not return hidden_states.")

    final_hidden = hidden_states[-1]
    if not isinstance(final_hidden, torch.Tensor):
        raise RuntimeError("Final hidden state is not a tensor.")

    return (
        cast(torch.Tensor, outputs.logits).detach(),
        capture.reentry_hidden,
        final_hidden.detach(),
    )

def _run_student_gap_bridge(
    *,
    model: Any,
    bridge: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    gap: GapSpec,
    prev_final_hidden: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Student:
    - capture hidden entering the skipped block
    - skip all layers in [gap.start, gap.end)
    - inject bridge output into layer gap.end, or into final norm if gap.end == num_layers
    """
    layers = _get_decoder_layers(model)
    reentry_module = _get_reentry_module_for_gap(model=model, gap=gap)
    capture = _StudentCapture()

    def capture_gap_input_prehook(module: Any, inputs: tuple[Any, ...]) -> None:
        if len(inputs) == 0 or not isinstance(inputs[0], torch.Tensor):
            raise TypeError("Expected first layer input to be a hidden-state tensor.")
        capture.gap_input_hidden = inputs[0]

    def skip_layer_hook(module: Any, inputs: tuple[Any, ...], output: Any) -> Any:
        if len(inputs) == 0 or not isinstance(inputs[0], torch.Tensor):
            raise TypeError("Expected first layer input to be a hidden-state tensor.")

        hidden_in = inputs[0]

        if isinstance(output, torch.Tensor):
            return hidden_in

        if isinstance(output, tuple) and len(output) > 0:
            return (hidden_in, *output[1:])

        raise TypeError(f"Unexpected layer output type while skipping: {type(output)}")

    def inject_bridge_prehook(module: Any, inputs: tuple[Any, ...]) -> tuple[Any, ...]:
        if capture.gap_input_hidden is None:
            raise RuntimeError("Gap input hidden state was not captured.")

        if prev_final_hidden.shape != capture.gap_input_hidden.shape:
            raise ValueError(
                f"prev_final_hidden.shape {prev_final_hidden.shape} "
                f"!= gap_input_hidden.shape {capture.gap_input_hidden.shape}"
            )

        bridged = bridge(capture.gap_input_hidden, prev_final_hidden)
        bridged = bridged.to(dtype=capture.gap_input_hidden.dtype)
        capture.bridged_hidden = bridged

        if len(inputs) == 0:
            raise RuntimeError("Re-entry module received empty inputs.")

        return (bridged, *inputs[1:])

    with ExitStack() as stack:
        handle = layers[gap.start].register_forward_pre_hook(capture_gap_input_prehook)
        stack.callback(handle.remove)

        for layer_idx in range(gap.start, gap.end):
            handle = layers[layer_idx].register_forward_hook(skip_layer_hook)
            stack.callback(handle.remove)

        handle = reentry_module.register_forward_pre_hook(inject_bridge_prehook)
        stack.callback(handle.remove)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            use_cache=False,
            return_dict=True,
        )

    if capture.bridged_hidden is None:
        raise RuntimeError("Failed to capture bridged hidden state.")

    return cast(torch.Tensor, outputs.logits), capture.bridged_hidden


def train_gap_bridge(
    *,
    model_name: str,
    dataset_spec: DatasetSpec,
    context_len: int,
    max_examples: int,
    num_windows_to_use: int = 256,
    batch_size: int = 2,
    gap_start: int,
    gap_length: int,
    num_epochs: int = 1,
    max_steps: int | None = None,
    lr: float = 1e-4,
    weight_decay: float = 0.0,
    max_grad_norm: float = 1.0,
    kl_loss_weight: float = 1.0,
    hidden_loss_weight: float = 1.0,
    ce_loss_weight: float = 0.0,
    model_kwargs: dict[str, Any] | None = None,
    checkpoint_dir: str | Path | None = "gap_bridge_checkpoints",
) -> TrainGapBridgeOutput:
    _stage("train_gap_bridge: start")

    device = get_preferred_device()
    compute_dtype = get_preferred_float_dtype(device)

    model_and_tokenizer: ModelAndTokenizer = load_model_and_tokenizer(
        model_name,
        model_kwargs={
            # HF normally expects torch_dtype, not dtype
            "torch_dtype": compute_dtype,
            **(model_kwargs or {}),
        },
    )

    model = cast(Any, model_and_tokenizer.model)
    model.to(device=device)
    model.eval()  # keep frozen teacher/student deterministic

    for param in model.parameters():
        param.requires_grad_(False)

    num_layers = len(_get_decoder_layers(model))
    gap = GapSpec(start=gap_start, length=gap_length)
    _validate_gap(gap=gap, num_layers=num_layers)

    effective_mode = _get_effective_mode(gap=gap, num_layers=num_layers)
    _stage(f"{effective_mode}")

    hidden_size = _get_hidden_size(model)
    bridge = GapBridge(
        hidden_size=hidden_size,
        bottleneck_dim=384,
        bias=True,
    ).to(
        device=device,
        dtype=torch.float32,
    )
    bridge.train()

    optimizer = torch.optim.AdamW(
        bridge.parameters(),
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

    if effective_mode == "LATE-BEGIN":
        _stage(
            f"training bridge for LATE-BEGIN skipping prefix [0, {gap.end}) "
            f"and re-entering at layer {gap.end if gap.end < num_layers else 'final_norm'} "
            f"on model with {num_layers} layers and hidden_size={hidden_size}"
        )
    elif effective_mode == "EARLY-EXIT":
        _stage(
            f"training bridge for EARLY-EXIT keep_prefix=[0, {gap.start}) "
            f"(skipping [{gap.start}, {gap.end})) "
            f"on model with {num_layers} layers and hidden_size={hidden_size}"
        )
    else:
        _stage(
            f"training bridge for GAP [{gap.start}, {gap.end}) "
            f"on model with {num_layers} layers and hidden_size={hidden_size}"
        )

    for epoch_idx in range(num_epochs):
        for batch_idx, (input_ids, attention_mask) in enumerate(dataloader, start=1):
            if max_steps is not None and step >= max_steps:
                break

            step += 1

            input_ids = input_ids.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                teacher_logits, teacher_reentry_hidden, teacher_final_hidden = _run_teacher_full_and_capture_reentry(
                    model=model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    gap=gap,
                )

                prev_final_hidden = _build_prev_final_hidden(teacher_final_hidden)

            student_logits, student_reentry_hidden = _run_student_gap_bridge(
                model=model,
                bridge=bridge,
                input_ids=input_ids,
                attention_mask=attention_mask,
                gap=gap,
                prev_final_hidden=prev_final_hidden,
            )

            loss_kl = _kl_teacher_to_student_next_token(
                logits_teacher=teacher_logits,
                logits_student=student_logits,
            )

            loss_hidden = _masked_hidden_mse_with_first_token_dropped(
                predicted=student_reentry_hidden,
                target=teacher_reentry_hidden,
                attention_mask=attention_mask,
            )

            if ce_loss_weight > 0.0:
                teacher_targets = teacher_logits[:, :-1, :].argmax(dim=-1)
                student_shift_logits = student_logits[:, :-1, :].contiguous()

                ce_mask = attention_mask[:, 1:].bool() if attention_mask is not None else None

                loss_ce_teacher = _masked_cross_entropy_from_logits(
                    logits=student_shift_logits,
                    targets=teacher_targets,
                    mask=ce_mask,
                )
            else:
                loss_ce_teacher = student_logits.new_zeros(())

            loss = (
                kl_loss_weight * loss_kl
                + hidden_loss_weight * loss_hidden
                + ce_loss_weight * loss_ce_teacher
            )

            loss.backward()

            if max_grad_norm is not None and max_grad_norm > 0:
                clip_grad_norm_(bridge.parameters(), max_grad_norm)

            optimizer.step()

            with torch.no_grad():
                sim = distribution_similarity_metrics(
                    shift_logits_mid=student_logits[:, :-1, :].contiguous(),
                    shift_logits_full=teacher_logits[:, :-1, :].contiguous(),
                )

            row = {
                "step": float(step),
                "epoch": float(epoch_idx + 1),
                "loss": loss.item(),
                "loss_kl": loss_kl.item(),
                "loss_hidden": loss_hidden.item(),
                "loss_ce": loss_ce_teacher.item(),
                "js": sim["js"].item(),
                "top1_agreement": sim["top1_agreement"].item(),
                "overlap": sim["overlap"].item(),
                "p_student_on_teacher_argmax": sim["p_mid_on_full_argmax"].item(),
            }
            history.append(row)

            print(
                f"[step {step:>5}] "
                f"mode={effective_mode} "
                f"epoch={epoch_idx + 1}/{num_epochs} "
                f"batch={batch_idx}/{len(dataloader)} "
                f"loss={row['loss']:.4f} "
                f"kl={row['loss_kl']:.4f} "
                f"hidden={row['loss_hidden']:.4f} "
                f"ce={row['loss_ce']:.4f} "
                f"js={row['js']:.4f} "
                f"top1={row['top1_agreement']:.4f} "
                f"overlap={row['overlap']:.4f}"
            )

        if max_steps is not None and step >= max_steps:
            break

    checkpoint_path: Path | None = None
    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        checkpoint_path = checkpoint_dir / (
            f"gap_bridge__{model_name.replace('/', '_')}__"
            f"start_{gap.start}__len_{gap.length}__{timestamp}.pt"
        )

        torch.save(
            {
                "model_name": model_name,
                "gap_start": gap.start,
                "gap_length": gap.length,
                "gap_end": gap.end,
                "num_layers": num_layers,
                "hidden_size": hidden_size,
                "effective_mode": effective_mode,
                "bridge_state_dict": bridge.state_dict(),
                "history": history,
            },
            checkpoint_path,
        )

        _stage(f"saved bridge checkpoint to {checkpoint_path}")

    _stage("train_gap_bridge: finished")

    return TrainGapBridgeOutput(
        bridge=bridge,
        history=history,
        checkpoint_path=checkpoint_path,
    )