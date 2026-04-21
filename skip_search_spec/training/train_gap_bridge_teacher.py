


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

    if gap.start <= 0:
        raise ValueError(
            f"gap.start must be > 0 so there is at least one pre-gap layer, got {gap.start}"
        )

    if gap.end > num_layers:
        raise ValueError(
            f"gap.end must be <= num_layers, got gap.end={gap.end}, num_layers={num_layers}"
        )


def _is_effective_early_exit(*, gap: GapSpec, num_layers: int) -> bool:
    return gap.end == num_layers


def _get_reentry_module_for_gap(*, model: Any, gap: GapSpec) -> Any:
    backbone = _get_backbone(model)
    layers = backbone.layers
    num_layers = len(layers)

    if gap.end < num_layers:
        return layers[gap.end]

    if gap.end == num_layers:
        return backbone.norm

    raise ValueError(f"Invalid gap.end={gap.end} for num_layers={num_layers}")


def _get_trainable_pre_gap_layer_indices(
    *,
    gap: GapSpec,
    num_layers: int,
    num_trainable_pre_gap_layers: int,
) -> list[int]:
    if num_trainable_pre_gap_layers <= 0:
        return []

    _validate_gap(gap=gap, num_layers=num_layers)

    start_idx = max(0, gap.start - num_trainable_pre_gap_layers)
    return list(range(start_idx, gap.start))


class GapBridge(nn.Module):
    def __init__(self, hidden_size: int, bias: bool = False) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size, bias=bias)

        with torch.no_grad():
            nn.init.zeros_(self.proj.weight)
            if self.proj.bias is not None:
                nn.init.zeros_(self.proj.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states + self.proj(hidden_states)


def _cross_entropy_next_token(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )


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


def _masked_hidden_mse(
    *,
    predicted: torch.Tensor,
    target: torch.Tensor,
    attention_mask: torch.Tensor | None,
) -> torch.Tensor:
    diff_sq = (predicted.float() - target.float()).pow(2)

    if attention_mask is None:
        return diff_sq.mean()

    mask = attention_mask.unsqueeze(-1).to(diff_sq.dtype)
    denom = (mask.sum() * diff_sq.size(-1)).clamp_min(1.0)
    return (diff_sq * mask).sum() / denom


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
) -> tuple[torch.Tensor, torch.Tensor]:
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
            output_hidden_states=False,
            use_cache=False,
            return_dict=True,
        )

    if capture.reentry_hidden is None:
        raise RuntimeError("Failed to capture teacher re-entry hidden state.")

    return cast(torch.Tensor, outputs.logits).detach(), capture.reentry_hidden


def _run_student_gap_bridge(
    *,
    model: Any,
    bridge: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    gap: GapSpec,
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

        bridged = bridge(capture.gap_input_hidden)
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


def train_gap_bridge_teacher(
    *,
    model_name: str,
    dataset_spec: DatasetSpec,
    context_len: int,
    max_examples: int,
    num_windows_to_use: int = 256,
    batch_size: int = 2,
    gap_start: int,
    gap_length: int,
    num_trainable_pre_gap_layers: int = 0,
    num_epochs: int = 1,
    max_steps: int | None = None,
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

    teacher_model_and_tokenizer: ModelAndTokenizer = load_model_and_tokenizer(
        model_name,
        model_kwargs={
            "torch_dtype": compute_dtype,
            **(model_kwargs or {}),
        },
    )

    student_model_and_tokenizer: ModelAndTokenizer = load_model_and_tokenizer(
        model_name,
        model_kwargs={
            "torch_dtype": compute_dtype,
            **(model_kwargs or {}),
        },
    )

    teacher_model = cast(Any, teacher_model_and_tokenizer.model)
    student_model = cast(Any, student_model_and_tokenizer.model)

    teacher_model.to(device=device)
    student_model.to(device=device)

    teacher_model.eval()
    student_model.eval()

    for param in teacher_model.parameters():
        param.requires_grad_(False)

    for param in student_model.parameters():
        param.requires_grad_(False)

    num_layers = len(_get_decoder_layers(student_model))
    gap = GapSpec(start=gap_start, length=gap_length)
    _validate_gap(gap=gap, num_layers=num_layers)

    effective_mode = "EARLY-EXIT" if _is_effective_early_exit(gap=gap, num_layers=num_layers) else "GAP"
    _stage(effective_mode)

    trainable_layer_indices = _get_trainable_pre_gap_layer_indices(
        gap=gap,
        num_layers=num_layers,
        num_trainable_pre_gap_layers=num_trainable_pre_gap_layers,
    )

    student_layers = _get_decoder_layers(student_model)

    for layer_idx in trainable_layer_indices:
        for param in student_layers[layer_idx].parameters():
            param.requires_grad_(True)

    hidden_size = _get_hidden_size(student_model)
    bridge = GapBridge(hidden_size=hidden_size, bias=False).to(
        device=device,
        dtype=compute_dtype,
    )
    bridge.train()

    trainable_student_layer_params: list[nn.Parameter] = []
    for layer_idx in trainable_layer_indices:
        for param in student_layers[layer_idx].parameters():
            if param.requires_grad:
                trainable_student_layer_params.append(param)

    bridge_params = list(bridge.parameters())


    optimizer = torch.optim.AdamW(
        [
            {
                "params": bridge_params,
                "lr": 1e-4,
                "weight_decay": 0.0,
            },
            {
                "params": trainable_student_layer_params,
                "lr": 1e-5,
                "weight_decay": weight_decay,
            },
        ],
    )

    dataset: Dataset = load_dataset(dataset_spec)
    window_settings = WindowSettings(C1=context_len)

    _stage("tokenizing dataset")
    tokenized_examples = tokenize_dataset_to_examples(
        dataset,
        student_model_and_tokenizer.tokenizer,
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

    if effective_mode == "EARLY-EXIT":
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

    _stage(f"trainable student pre-gap layers: {trainable_layer_indices}")

    for epoch_idx in range(num_epochs):
        for batch_idx, (input_ids, attention_mask) in enumerate(dataloader, start=1):
            if max_steps is not None and step >= max_steps:
                break

            step += 1

            input_ids = input_ids.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)
            labels = input_ids

            optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                teacher_logits, teacher_reentry_hidden = _run_teacher_full_and_capture_reentry(
                    model=teacher_model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    gap=gap,
                )

            student_logits, student_reentry_hidden = _run_student_gap_bridge(
                model=student_model,
                bridge=bridge,
                input_ids=input_ids,
                attention_mask=attention_mask,
                gap=gap,
            )

            loss_kl = _kl_teacher_to_student_next_token(
                logits_teacher=teacher_logits,
                logits_student=student_logits,
            )

            loss_hidden = _masked_hidden_mse(
                predicted=student_reentry_hidden,
                target=teacher_reentry_hidden,
                attention_mask=attention_mask,
            )

            if ce_loss_weight > 0.0:
                loss_ce = _cross_entropy_next_token(student_logits, labels)
            else:
                loss_ce = student_logits.new_zeros(())

            loss = (
                kl_loss_weight * loss_kl
                + hidden_loss_weight * loss_hidden
                + ce_loss_weight * loss_ce
            )

            loss.backward()

            if max_grad_norm is not None and max_grad_norm > 0:
                clip_grad_norm_(bridge_params, max_grad_norm)
                if len(trainable_student_layer_params) > 0:
                    clip_grad_norm_(trainable_student_layer_params, max_grad_norm)

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
                "loss_ce": loss_ce.item(),
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
            f"start_{gap.start}__len_{gap.length}__"
            f"trainable_pre_{num_trainable_pre_gap_layers}__{timestamp}.pt"
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
                "num_trainable_pre_gap_layers": num_trainable_pre_gap_layers,
                "trainable_layer_indices": trainable_layer_indices,
                "bridge_state_dict": bridge.state_dict(),
                "student_trainable_layer_state_dict": {
                    str(idx): student_layers[idx].state_dict()
                    for idx in trainable_layer_indices
                },
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
