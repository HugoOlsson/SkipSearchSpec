from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast
from contextlib import ExitStack
from pathlib import Path
import time

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

from skip_search_spec.helpers.tooling import (
    distribution_similarity_metrics,
    get_preferred_device,
    get_preferred_float_dtype,
    load_model_and_tokenizer,
)

from skip_search_spec.protocols.windows import (
    DatasetSpec,
    ModelAndTokenizer,
)

# Rename this import path to wherever you placed the shared helper file.
from skip_search_spec.helpers.shared_decoding_tools import (
    GapSpec,
    build_fixed_window_dataloader,
    build_mixed_fixed_window_dataloader,
    build_prev_hidden,
    forward_model_logits,
    get_backbone,
    get_decoder_layers,
    get_effective_gap_mode,
    get_first_hidden_from_inputs,
    get_hidden_size,
    get_reentry_module_for_gap,
    kl_teacher_to_student_next_token,
    make_identity_skip_hook,
    make_layer_pattern,
    masked_cross_entropy_from_logits,
    masked_hidden_mse_with_first_token_dropped,
    shift_next_token_logits,
    shift_next_token_mask,
    stage,
    validate_gap,
)


@dataclass(slots=True)
class TrainGapBridgeOutput:
    bridge: nn.Module
    history: list[dict[str, float]]
    checkpoint_path: Path | None


class GapBridge(nn.Module):
    """
    Linear residual bridge that predicts re-entry hidden from:
      - current hidden entering the gap
      - previous token's reference hidden

    For position t:
      x_t     = hidden entering the gap at position t
      p_{t-1} = previous-position reference hidden

    Output:
      bridged_t = x_t + delta_t
    """

    def __init__(
        self,
        hidden_size: int,
        bias: bool = False,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size

        self.gap_norm = nn.LayerNorm(hidden_size)
        self.prev_norm = nn.LayerNorm(hidden_size)

        input_dim = hidden_size * 2
        self.proj = nn.Linear(input_dim, hidden_size, bias=bias)

        with torch.no_grad():
            nn.init.zeros_(self.proj.weight)
            if self.proj.bias is not None:
                nn.init.zeros_(self.proj.bias)

    def forward(
        self,
        gap_hidden: torch.Tensor,
        prev_state_hidden: torch.Tensor,
    ) -> torch.Tensor:
        if gap_hidden.shape != prev_state_hidden.shape:
            raise ValueError(
                f"gap_hidden.shape {gap_hidden.shape} "
                f"!= prev_state_hidden.shape {prev_state_hidden.shape}"
            )

        x = gap_hidden.float()
        p = prev_state_hidden.float()

        x_n = self.gap_norm(x)
        p_n = self.prev_norm(p)

        feats = torch.cat([x_n, p_n], dim=-1)
        delta = self.proj(feats)

        return x + delta


@dataclass(slots=True)
class _StudentCapture:
    gap_input_hidden: torch.Tensor | None = None
    bridged_hidden: torch.Tensor | None = None


@dataclass(slots=True)
class _TeacherCapture:
    reentry_hidden: torch.Tensor | None = None
    final_hidden: torch.Tensor | None = None


@torch.no_grad()
def _run_teacher_full_and_capture_reentry(
    *,
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    gap: GapSpec,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    backbone = get_backbone(model)
    reentry_module = get_reentry_module_for_gap(model=model, gap=gap)
    capture = _TeacherCapture()

    def reentry_prehook(module: Any, inputs: tuple[Any, ...]) -> None:
        capture.reentry_hidden = get_first_hidden_from_inputs(inputs).detach()

    def final_norm_hook(module: Any, inputs: tuple[Any, ...], output: Any) -> Any:
        if not isinstance(output, torch.Tensor):
            raise TypeError("Expected final norm output to be a tensor.")

        capture.final_hidden = output.detach()
        return output

    with ExitStack() as stack:
        handle = reentry_module.register_forward_pre_hook(reentry_prehook)
        stack.callback(handle.remove)

        handle = backbone.norm.register_forward_hook(final_norm_hook)
        stack.callback(handle.remove)

        logits = forward_model_logits(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    if capture.reentry_hidden is None:
        raise RuntimeError("Failed to capture teacher re-entry hidden state.")

    if capture.final_hidden is None:
        raise RuntimeError("Failed to capture teacher final hidden state.")

    return (
        logits.detach(),
        capture.reentry_hidden,
        capture.final_hidden,
    )


def _run_student_gap_bridge(
    *,
    model: Any,
    bridge: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    gap: GapSpec,
    prev_state_hidden: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Student path:
      - capture hidden entering the skipped block
      - make layers [gap.start, gap.end) act like identity
      - inject bridge output into layer gap.end, or final norm if gap.end == num_layers
    """
    layers = get_decoder_layers(model)
    reentry_module = get_reentry_module_for_gap(model=model, gap=gap)
    capture = _StudentCapture()

    def capture_gap_input_prehook(module: Any, inputs: tuple[Any, ...]) -> None:
        capture.gap_input_hidden = get_first_hidden_from_inputs(inputs)

    def inject_bridge_prehook(module: Any, inputs: tuple[Any, ...]) -> tuple[Any, ...]:
        if capture.gap_input_hidden is None:
            raise RuntimeError("Gap input hidden state was not captured.")

        if prev_state_hidden.shape != capture.gap_input_hidden.shape:
            raise ValueError(
                f"prev_state_hidden.shape {prev_state_hidden.shape} "
                f"!= gap_input_hidden.shape {capture.gap_input_hidden.shape}"
            )

        bridged = bridge(capture.gap_input_hidden, prev_state_hidden)
        bridged = bridged.to(dtype=capture.gap_input_hidden.dtype)

        capture.bridged_hidden = bridged

        if len(inputs) == 0:
            raise RuntimeError("Re-entry module received empty inputs.")

        return (bridged, *inputs[1:])

    with ExitStack() as stack:
        handle = layers[gap.start].register_forward_pre_hook(capture_gap_input_prehook)
        stack.callback(handle.remove)

        for layer_idx in range(gap.start, gap.end):
            handle = layers[layer_idx].register_forward_hook(make_identity_skip_hook())
            stack.callback(handle.remove)

        handle = reentry_module.register_forward_pre_hook(inject_bridge_prehook)
        stack.callback(handle.remove)

        logits = forward_model_logits(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    if capture.bridged_hidden is None:
        raise RuntimeError("Failed to capture bridged hidden state.")

    return logits, capture.bridged_hidden


def train_skipping_layers(
    *,
    model_name: str,
    dataset_mix: list[tuple[DatasetSpec, float]],
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
    max_grad_norm: float | None = None,
    kl_loss_weight: float = 1.0,
    hidden_loss_weight: float = 0.0,
    ce_loss_weight: float = 1.0,
    teacher_temperature: float = 1.0,
    model_kwargs: dict[str, Any] | None = None,
    checkpoint_dir: str | Path | None = "gap_bridge_checkpoints",
    checkpoint_every_steps: int | None = 500,
) -> TrainGapBridgeOutput:
    stage("train_gap_bridge: start")

    device = get_preferred_device()
    compute_dtype = get_preferred_float_dtype(device)

    print("DEVICE:", device)
    print("COMPUTE_DTYPE:", compute_dtype)

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

    num_layers = len(get_decoder_layers(model))

    gap = GapSpec(
        start=gap_start,
        length=gap_length,
    )
    validate_gap(gap=gap, num_layers=num_layers)

    effective_mode = get_effective_gap_mode(
        gap=gap,
        num_layers=num_layers,
    )

    active_layer_indices = tuple(
        i for i in range(num_layers)
        if not (gap.start <= i < gap.end)
    )
    layer_pattern = make_layer_pattern(
        num_layers=num_layers,
        active_layer_indices=active_layer_indices,
    )

    stage(f"mode={effective_mode}")
    stage(f"gap=[{gap.start}, {gap.end}) length={gap.length}")
    stage(f"active_layers={layer_pattern.active_layer_indices}")
    stage(f"skipped_layers={layer_pattern.skipped_layer_indices}")
    stage(f"visual_mask={layer_pattern.visual_mask}")
    stage(f"binary_mask={layer_pattern.binary_string}")

    hidden_size = get_hidden_size(model)

    bridge = GapBridge(
        hidden_size=hidden_size,
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

    stage("building dataloader")
    dataloader = build_mixed_fixed_window_dataloader(
        dataset_mix=dataset_mix,
        model_and_tokenizer=model_and_tokenizer,
        context_len=context_len,
        max_examples=max_examples,
        num_windows_to_use=num_windows_to_use,
        batch_size=batch_size,
        device=device,
        shuffle=True,
    )

    history: list[dict[str, float]] = []
    step = 0
    checkpoint_path: Path | None = None

    checkpoint_dir_path: Path | None = None
    run_timestamp = time.strftime("%Y%m%d_%H%M%S")
    safe_model_name = model_name.replace("/", "_")

    if checkpoint_dir is not None:
        checkpoint_dir_path = Path(checkpoint_dir)
        checkpoint_dir_path.mkdir(parents=True, exist_ok=True)


    def save_checkpoint(*, tag: str) -> Path | None:
        if checkpoint_dir_path is None:
            return None

        path = checkpoint_dir_path / (
            f"gap_bridge__{safe_model_name}__"
            f"start_{gap.start}__len_{gap.length}__"
            f"{run_timestamp}__{tag}.pt"
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
                "active_layer_indices": layer_pattern.active_layer_indices,
                "skipped_layer_indices": layer_pattern.skipped_layer_indices,
                "binary_mask": layer_pattern.binary_mask,
                "visual_mask": layer_pattern.visual_mask,
                "step": step,
                "bridge_state_dict": bridge.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "history": history,
                "train_config": {
                    "context_len": context_len,
                    "max_examples": max_examples,
                    "num_windows_to_use": num_windows_to_use,
                    "batch_size": batch_size,
                    "num_epochs": num_epochs,
                    "max_steps": max_steps,
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "max_grad_norm": max_grad_norm,
                    "kl_loss_weight": kl_loss_weight,
                    "hidden_loss_weight": hidden_loss_weight,
                    "ce_loss_weight": ce_loss_weight,
                    "teacher_temperature": teacher_temperature,
                },
            },
            path,
        )

        return path

    if effective_mode == "LATE-BEGIN":
        stage(
            f"training bridge for LATE-BEGIN skipping prefix [0, {gap.end}) "
            f"and re-entering at layer {gap.end if gap.end < num_layers else 'final_norm'} "
            f"on model with {num_layers} layers and hidden_size={hidden_size}"
        )
    elif effective_mode == "EARLY-EXIT":
        stage(
            f"training bridge for EARLY-EXIT keep_prefix=[0, {gap.start}) "
            f"and skipping [{gap.start}, {gap.end}) "
            f"on model with {num_layers} layers and hidden_size={hidden_size}"
        )
    else:
        stage(
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
                (
                    teacher_logits,
                    teacher_reentry_hidden,
                    teacher_final_hidden,
                ) = _run_teacher_full_and_capture_reentry(
                    model=model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    gap=gap,
                )

                # Current choice:
                #   previous token's teacher re-entry hidden.
                #
                # Alternative:
                #   use build_prev_hidden(teacher_final_hidden)
                #   if you want previous final-layer hidden instead.
                prev_hidden = build_prev_hidden(teacher_reentry_hidden)

            student_logits, student_reentry_hidden = _run_student_gap_bridge(
                model=model,
                bridge=bridge,
                input_ids=input_ids,
                attention_mask=attention_mask,
                gap=gap,
                prev_state_hidden=prev_hidden,
            )

            loss_kl = kl_teacher_to_student_next_token(
                logits_teacher=teacher_logits,
                logits_student=student_logits,
                teacher_temperature=teacher_temperature,
                attention_mask=attention_mask,
            )

            loss_hidden = masked_hidden_mse_with_first_token_dropped(
                predicted=student_reentry_hidden,
                target=teacher_reentry_hidden,
                attention_mask=attention_mask,
            )

            if ce_loss_weight > 0.0:
                teacher_targets = shift_next_token_logits(teacher_logits).argmax(dim=-1)
                student_next_logits = shift_next_token_logits(student_logits)
                ce_mask = shift_next_token_mask(attention_mask)

                loss_ce_teacher = masked_cross_entropy_from_logits(
                    logits=student_next_logits,
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

            if (
                checkpoint_every_steps is not None
                and checkpoint_every_steps > 0
                and step % checkpoint_every_steps == 0
            ):
                checkpoint_path = save_checkpoint(tag=f"step_{step:06d}")

                if checkpoint_path is not None:
                    stage(f"saved periodic bridge checkpoint to {checkpoint_path}")

            log_every = 50

            if step == 1 or step % log_every == 0:
                with torch.no_grad():
                    sim = distribution_similarity_metrics(
                        shift_logits_mid=shift_next_token_logits(student_logits),
                        shift_logits_full=shift_next_token_logits(teacher_logits),
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

    checkpoint_path = save_checkpoint(tag=f"final_step_{step:06d}")

    if checkpoint_path is not None:
        stage(f"saved final bridge checkpoint to {checkpoint_path}")

    stage("train_gap_bridge: finished")

    return TrainGapBridgeOutput(
        bridge=bridge,
        history=history,
        checkpoint_path=checkpoint_path,
    )