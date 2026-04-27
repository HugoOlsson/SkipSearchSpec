from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
import time

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

from skip_search_spec.helpers.tooling import distribution_similarity_metrics

from skip_search_spec.protocols.windows import (
    DatasetSpec,
    ModelAndTokenizer,
)

from skip_search_spec.helpers.shared_decoding_tools import (
    build_mixed_fixed_window_dataloader,
    get_effective_gap_mode,
    kl_teacher_to_student_next_token,
    make_layer_pattern,
    masked_cross_entropy_from_logits,
    masked_hidden_mse_with_first_token_dropped,
    shift_next_token_logits,
    shift_next_token_mask,
    stage,
)

# Change this import path to wherever you placed BridgedGapModel.
from skip_search_spec.training.bridged_gap_model import (
    BridgedGapModel,
    ReferenceHiddenSource,
    build_bridged_gap_model,
)


@dataclass(slots=True)
class TrainGapBridgeOutput:
    bridged_model: BridgedGapModel
    bridge: nn.Module
    history: list[dict[str, float]]
    checkpoint_path: Path | None


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
    reference_hidden_source: ReferenceHiddenSource = "reentry",
    model_kwargs: dict[str, Any] | None = None,
    checkpoint_dir: str | Path | None = "gap_bridge_checkpoints",
    checkpoint_every_steps: int | None = 500,
) -> TrainGapBridgeOutput:
    stage("train_gap_bridge: start")

    bridged = build_bridged_gap_model(
        model_name=model_name,
        gap_start=gap_start,
        gap_length=gap_length,
        reference_hidden_source=reference_hidden_source,
        model_kwargs=model_kwargs,
    )

    bridged.train_bridge_only()

    device = bridged.device
    bridge = bridged.bridge
    gap = bridged.gap
    num_layers = bridged.num_layers
    hidden_size = bridged.hidden_size

    effective_mode = get_effective_gap_mode(
        gap=gap,
        num_layers=num_layers,
    )

    layer_pattern = make_layer_pattern(
        num_layers=num_layers,
        active_layer_indices=bridged.active_layer_indices,
    )

    print("DEVICE:", device)
    print("REFERENCE_HIDDEN_SOURCE:", reference_hidden_source)

    stage(f"mode={effective_mode}")
    stage(f"gap=[{gap.start}, {gap.end}) length={gap.length}")
    stage(f"active_layers={layer_pattern.active_layer_indices}")
    stage(f"skipped_layers={layer_pattern.skipped_layer_indices}")
    stage(f"visual_mask={layer_pattern.visual_mask}")
    stage(f"binary_mask={layer_pattern.binary_string}")

    optimizer = torch.optim.AdamW(
        bridge.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    stage("building dataloader")

    dataloader = build_mixed_fixed_window_dataloader(
        dataset_mix=dataset_mix,
        model_and_tokenizer=cast(ModelAndTokenizer, bridged),
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

        return bridged.save_checkpoint(
            path=path,
            step=step,
            optimizer_state_dict=optimizer.state_dict(),
            history=history,
            extra={
                "effective_mode": effective_mode,
                "visual_mask": layer_pattern.visual_mask,
                "binary_mask": layer_pattern.binary_mask,
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
                    "reference_hidden_source": reference_hidden_source,
                },
            },
        )

    stage(
        f"training bridge | mode={effective_mode} | "
        f"model={model_name} | layers={num_layers} | hidden_size={hidden_size} | "
        f"reference_hidden_source={reference_hidden_source}"
    )

    for epoch_idx in range(num_epochs):
        for batch_idx, (input_ids, attention_mask) in enumerate(dataloader, start=1):
            if max_steps is not None and step >= max_steps:
                break

            step += 1

            input_ids = input_ids.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            teacher = bridged.run_verifier(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            prev_reference_hidden = bridged.build_prev_reference(
                teacher.reference_hidden,
            )

            student = bridged.run_drafter(
                input_ids=input_ids,
                attention_mask=attention_mask,
                prev_reference_hidden=prev_reference_hidden,
            )

            loss_kl = kl_teacher_to_student_next_token(
                logits_teacher=teacher.logits,
                logits_student=student.logits,
                teacher_temperature=teacher_temperature,
                attention_mask=attention_mask,
            )

            loss_hidden = masked_hidden_mse_with_first_token_dropped(
                predicted=bridged.bridge_prediction_hidden(student),
                target=bridged.bridge_target_hidden(teacher),
                attention_mask=attention_mask,
            )

            if ce_loss_weight > 0.0:
                teacher_targets = shift_next_token_logits(teacher.logits).argmax(dim=-1)
                student_next_logits = shift_next_token_logits(student.logits)
                ce_mask = shift_next_token_mask(attention_mask)

                loss_ce_teacher = masked_cross_entropy_from_logits(
                    logits=student_next_logits,
                    targets=teacher_targets,
                    mask=ce_mask,
                )
            else:
                loss_ce_teacher = student.logits.new_zeros(())

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
                        shift_logits_mid=shift_next_token_logits(student.logits),
                        shift_logits_full=shift_next_token_logits(teacher.logits),
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
                    "p_student_on_teacher_argmax": sim[
                        "p_mid_on_full_argmax"
                    ].item(),
                }

                history.append(row)

                print(
                    f"[step {step:>5}] "
                    f"mode={effective_mode} "
                    f"ref={reference_hidden_source} "
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

    bridged.eval_all()

    stage("train_gap_bridge: finished")

    return TrainGapBridgeOutput(
        bridged_model=bridged,
        bridge=bridge,
        history=history,
        checkpoint_path=checkpoint_path,
    )