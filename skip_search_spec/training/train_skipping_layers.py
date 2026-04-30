from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
import time

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

from skip_search_spec.helpers.tooling import distribution_similarity_metrics

from skip_search_spec.protocols.measurements import MeasurementRun, MetricEvent, RunContext, dataset_mix_config, dataset_mix_name, json_safe, print_metric_events_line, safe_path_part, save_at_interval
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

from skip_search_spec.training.bridged_gap_model import (
    BridgedGapModel,
    ReferenceHiddenSource,
    build_bridged_gap_model,
)


@dataclass(slots=True)
class TrainGapBridgeOutput:
    bridged_model: BridgedGapModel
    bridge: nn.Module
    checkpoint_path: Path | None


def train_skipping_layers(
    *,
    model_name: str,
    dataset_mix: list[tuple[DatasetSpec, float, int]],
    context_len: int = 256, # Always use 256 token windows to get a good balance between context and compute
    num_windows_to_use: int,
    batch_size: int = 2,
    active_start_layers: int,
    active_end_layers: int,
    num_epochs: int = 1, # Always one to maximize data exposure per compute
    max_steps: int | None = 100_000_000, # just something big
    lr: float = 1e-4,
    weight_decay: float = 0.0,
    max_grad_norm: float | None = None,
    kl_loss_weight: float = 1.0,
    ce_loss_weight: float = 1.0,
    hidden_loss_weight: float = 0.0,
    teacher_temperature: float = 1.0,
    reference_hidden_source: ReferenceHiddenSource = "reentry",
    model_kwargs: dict[str, Any] | None = None,
    checkpoint_every_steps: int | None = None,
    log_every: int = 100,
    measurement_save_interval_seconds: float = 60.0,
) -> TrainGapBridgeOutput:
    stage("train_gap_bridge: start")

    bridged = build_bridged_gap_model(
        model_name=model_name,
        active_start_layers=active_start_layers,
        active_end_layers=active_end_layers,
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
        num_windows_to_use=num_windows_to_use,
        batch_size=batch_size,
        device=device,
        shuffle=True,
    )


    step = 0
    checkpoint_path: Path | None = None

    checkpoint_dir_path: Path | None = None

    safe_model_name = model_name.replace("/", "_").replace(".", "_")

    gap_text = f"{gap.start}:{gap.length}:{num_layers - gap.end}"

    run_name = (
        f"{safe_model_name}_"
        f"{gap_text}"
    )

    run_context = RunContext.create(
        run_name=run_name,
        experiment_type="middle_gap_skip",
        model_names=(model_name,),
        dataset_name=dataset_mix_name(dataset_mix),
        run_config={
            "model_name": model_name,
            "dataset_mix": dataset_mix_config(dataset_mix),
            "device": str(device),
            "num_layers": num_layers,
            "hidden_size": hidden_size,
            "effective_mode": effective_mode,
            "gap_start": gap.start,
            "gap_end": gap.end,
            "gap_length": gap.length,
            "active_layers": list(layer_pattern.active_layer_indices),
            "skipped_layers": list(layer_pattern.skipped_layer_indices),
            "visual_mask": layer_pattern.visual_mask,
            "binary_mask": layer_pattern.binary_string,
            "context_len": context_len,
            "num_windows_to_use": num_windows_to_use,
            "batch_size": batch_size,
            "steps_per_epoch": len(dataloader),
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
            "model_kwargs": json_safe(model_kwargs or {}),
            "checkpoint_every_steps": checkpoint_every_steps,
            "log_every": log_every,
        },
    )

    run_context.print()

    metric_events: list[MetricEvent] = []

    run_placeholder = MeasurementRun(
        context=run_context,
        metric_events=metric_events,
    )

    checkpoint_dir_path = run_placeholder.default_output_dir()

    def save_checkpoint(*, checkpoint_label: str) -> Path | None:
        if checkpoint_dir_path is None:
            return None

        path = checkpoint_dir_path / f"checkpoint_{checkpoint_label}.pt"

        return bridged.save_checkpoint(
            path=path,
            step=step,
            optimizer_state_dict=optimizer.state_dict(),
            extra={
                "effective_mode": effective_mode,
                "visual_mask": layer_pattern.visual_mask,
                "binary_mask": layer_pattern.binary_mask,
                "train_config": {
                    "context_len": context_len,
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
                checkpoint_path = save_checkpoint(checkpoint_label=f"step_{step:06d}")

                if checkpoint_path is not None:
                    stage(f"saved periodic bridge checkpoint to {checkpoint_path}")

            if step == 1 or step % log_every == 0:
                with torch.no_grad():
                    sim = distribution_similarity_metrics(
                        shift_logits_drafter=shift_next_token_logits(student.logits),
                        shift_logits_verifier=shift_next_token_logits(teacher.logits),
                    )

                batch_metrics = [
                    MetricEvent.create(phase="train", name="loss_total", value=loss.item(), step=step),
                    MetricEvent.create(phase="train", name="loss_kl_verifier_to_drafter", value=loss_kl.item(), step=step),
                    MetricEvent.create(phase="train", name="loss_bridge_reentry_mse", value=loss_hidden.item(), step=step),
                    MetricEvent.create(phase="train", name="loss_ce_drafter_on_verifier_top1", value=loss_ce_teacher.item(), step=step),
                    MetricEvent.create(phase="train", name="kl_verifier_to_drafter", value=sim["kl_verifier_to_drafter"].item(), step=step),
                    MetricEvent.create(phase="train", name="kl_drafter_to_verifier", value=sim["kl_drafter_to_verifier"].item(), step=step),
                    MetricEvent.create(phase="train", name="js_verifier_drafter", value=sim["js_verifier_drafter"].item(), step=step),
                    MetricEvent.create(phase="train", name="top1_drafter_matches_verifier", value=sim["top1_drafter_matches_verifier"].item(), step=step),
                    MetricEvent.create(phase="train", name="prob_mass_overlap_verifier_drafter", value=sim["prob_mass_overlap_verifier_drafter"].item(), step=step),
                    MetricEvent.create(phase="train", name="p_drafter_on_verifier_top1", value=sim["p_drafter_on_verifier_top1"].item(), step=step),
                ]

                metric_events.extend(batch_metrics)

                row = {
                    "step": float(step),
                    "epoch": float(epoch_idx + 1),
                    "batch": float(batch_idx),
                }
                row.update({event.name: event.value for event in batch_metrics})

                print(
                    f"[step {step:>5}] "
                    f"mode={effective_mode} "
                    f"ref={reference_hidden_source} "
                    f"epoch={epoch_idx + 1}/{num_epochs} "
                    f"batch={batch_idx}/{len(dataloader)} ",
                    end="",
                )
                print_metric_events_line(batch_metrics, decimals=4)

                run = MeasurementRun(
                    context=run_context,
                    metric_events=metric_events,
                )

                save_at_interval(
                    run,
                    min_interval_seconds=measurement_save_interval_seconds,
                )

        if max_steps is not None and step >= max_steps:
            break

    checkpoint_path = save_checkpoint(checkpoint_label=f"final_step_{step:06d}")

    if checkpoint_path is not None:
        stage(f"saved final bridge checkpoint to {checkpoint_path}")

    measurement_path = MeasurementRun(
        context=run_context,
        metric_events=metric_events,
    ).save()

    stage(f"saved final measurement log to {measurement_path}")
        
    bridged.eval_all()

    stage("train_gap_bridge: finished")

    return TrainGapBridgeOutput(
        bridged_model=bridged,
        bridge=bridge,
        checkpoint_path=checkpoint_path,
    )