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
    build_window_index,
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
    student_model: nn.Module
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
    if gap.start == 0 and gap.end == num_layers:
        return "FULL-GAP"
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


def _freeze_module(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad_(False)


def _freeze_all_parameters(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad_(False)


def _dedupe_parameters(params: list[nn.Parameter]) -> list[nn.Parameter]:
    out: list[nn.Parameter] = []
    seen: set[int] = set()
    for p in params:
        pid = id(p)
        if pid not in seen:
            out.append(p)
            seen.add(pid)
    return out


def _count_parameters(params: list[nn.Parameter]) -> int:
    return sum(p.numel() for p in params)


def _get_kept_layer_indices(*, gap: GapSpec, num_layers: int) -> list[int]:
    return list(range(0, gap.start)) + list(range(gap.end, num_layers))


def _collect_trainable_student_modules(
    *,
    model: Any,
    gap: GapSpec,
    train_kept_layers: bool,
    train_final_norm: bool,
    train_embed_tokens: bool,
    train_lm_head: bool,
) -> list[tuple[str, nn.Module]]:
    backbone = _get_backbone(model)
    layers = backbone.layers
    num_layers = len(layers)

    named_modules: list[tuple[str, nn.Module]] = []

    if train_kept_layers:
        for layer_idx in _get_kept_layer_indices(gap=gap, num_layers=num_layers):
            named_modules.append((f"model.layers.{layer_idx}", cast(nn.Module, layers[layer_idx])))

    if train_final_norm:
        named_modules.append(("model.norm", cast(nn.Module, backbone.norm)))

    if train_embed_tokens and hasattr(backbone, "embed_tokens"):
        named_modules.append(("model.embed_tokens", cast(nn.Module, backbone.embed_tokens)))

    if train_lm_head and hasattr(model, "lm_head"):
        named_modules.append(("lm_head", cast(nn.Module, model.lm_head)))

    deduped: list[tuple[str, nn.Module]] = []
    seen_names: set[str] = set()
    for name, module in named_modules:
        if name not in seen_names:
            deduped.append((name, module))
            seen_names.add(name)

    return deduped


def _enable_trainable_student_modules(
    *,
    named_modules: list[tuple[str, nn.Module]],
) -> list[nn.Parameter]:
    params: list[nn.Parameter] = []
    for _, module in named_modules:
        for p in module.parameters():
            p.requires_grad_(True)
            params.append(p)
    return _dedupe_parameters(params)


def _state_dict_subset_for_prefixes(
    *,
    state_dict: dict[str, torch.Tensor],
    prefixes: list[str],
) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        if any(k == prefix or k.startswith(prefix + ".") for prefix in prefixes):
            out[k] = v.detach().cpu()
    return out


class GapBridge(nn.Module):
    """
    Linear bridge that predicts re-entry hidden from:
      - current hidden entering the gap
      - previous token's teacher re-entry hidden

    For position t, inputs are:
      x_t       = hidden entering the gap at position t
      r_{t-1}   = teacher re-entry hidden from position t-1 (shifted right)

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
        gap_hidden: torch.Tensor,        # [B, T, H]
        prev_state_hidden: torch.Tensor, # [B, T, H]
    ) -> torch.Tensor:
        if gap_hidden.shape != prev_state_hidden.shape:
            raise ValueError(
                f"gap_hidden.shape {gap_hidden.shape} != prev_state_hidden.shape {prev_state_hidden.shape}"
            )

        x = gap_hidden.float()
        p = prev_state_hidden.float()

        x_n = self.gap_norm(x)
        p_n = self.prev_norm(p)

        feats = torch.cat([x_n, p_n], dim=-1)
        delta = self.proj(feats)

        return x + delta


def _build_prev_hidden(hidden: torch.Tensor) -> torch.Tensor:
    prev = torch.zeros_like(hidden)
    prev[:, 1:, :] = hidden[:, :-1, :]
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


def _masked_kl_teacher_to_student_next_token(
    *,
    logits_teacher: torch.Tensor,
    logits_student: torch.Tensor,
    attention_mask: torch.Tensor | None,
) -> torch.Tensor:
    shift_teacher = logits_teacher[:, :-1, :].contiguous().float()
    shift_student = logits_student[:, :-1, :].contiguous().float()

    log_p_student = F.log_softmax(shift_student, dim=-1)
    log_p_teacher = F.log_softmax(shift_teacher, dim=-1)

    kl_per_token = F.kl_div(
        log_p_student,
        log_p_teacher,
        reduction="none",
        log_target=True,
    ).sum(dim=-1)

    if attention_mask is None:
        return kl_per_token.mean()

    mask = attention_mask[:, 1:].to(dtype=kl_per_token.dtype)
    denom = mask.sum().clamp_min(1.0)
    return (kl_per_token * mask).sum() / denom


def _temperature_sharpened_masked_kl_teacher_to_student_next_token(
    *,
    logits_teacher: torch.Tensor,
    logits_student: torch.Tensor,
    attention_mask: torch.Tensor | None,
    teacher_temperature: float = 1,
) -> torch.Tensor:
    shift_teacher = logits_teacher[:, :-1, :].contiguous().float()
    shift_student = logits_student[:, :-1, :].contiguous().float()

    teacher_log_probs = F.log_softmax(shift_teacher / teacher_temperature, dim=-1)
    student_log_probs = F.log_softmax(shift_student, dim=-1)

    kl_per_token = F.kl_div(
        student_log_probs,
        teacher_log_probs,
        reduction="none",
        log_target=True,
    ).sum(dim=-1)

    if attention_mask is None:
        return kl_per_token.mean()

    mask = attention_mask[:, 1:].to(dtype=kl_per_token.dtype)
    denom = mask.sum().clamp_min(1.0)
    return (kl_per_token * mask).sum() / denom


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
class _StudentCapture:
    gap_input_hidden: torch.Tensor | None = None
    bridged_hidden: torch.Tensor | None = None


@dataclass(slots=True)
class _TeacherCapture:
    reentry_hidden: torch.Tensor | None = None


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

    def reentry_prehook(module: Any, inputs: tuple[Any, ...]) -> None:
        if len(inputs) == 0 or not isinstance(inputs[0], torch.Tensor):
            raise TypeError("Expected first layer input to be a hidden-state tensor.")
        capture.reentry_hidden = inputs[0].detach()

    with ExitStack() as stack:
        h1 = reentry_module.register_forward_pre_hook(reentry_prehook)
        stack.callback(h1.remove)

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
    prev_state_hidden: torch.Tensor,
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


def train_gap_bridge_finetune(
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
    backbone_lr: float | None = None,
    weight_decay: float = 0.0,
    max_grad_norm: float = 1.0,
    kl_loss_weight: float = 1.0,
    hidden_loss_weight: float = 1.0,
    ce_loss_weight: float = 0.0,
    teacher_temperature: float = 1,
    train_kept_layers: bool = True,
    train_final_norm: bool = True,
    train_embed_tokens: bool = False,
    train_lm_head: bool = False,
    model_kwargs: dict[str, Any] | None = None,
    checkpoint_dir: str | Path | None = "gap_bridge_checkpoints",
) -> TrainGapBridgeOutput:
    _stage("train_gap_bridge: start")

    device = get_preferred_device()
    compute_dtype = get_preferred_float_dtype(device)

    print("DEVICE:", device)
    print("COMPUTE_DTYPE:", compute_dtype)

    if backbone_lr is None:
        backbone_lr = lr

    _stage("loading frozen teacher model")
    teacher_mt: ModelAndTokenizer = load_model_and_tokenizer(
        model_name,
        model_kwargs={
            "torch_dtype": compute_dtype,
            **(model_kwargs or {}),
        },
    )
    teacher_model = cast(Any, teacher_mt.model)
    teacher_model.to(device=device)
    teacher_model.eval()
    _freeze_all_parameters(cast(nn.Module, teacher_model))

    _stage("loading trainable student model")
    student_mt: ModelAndTokenizer = load_model_and_tokenizer(
        model_name,
        model_kwargs={
            "torch_dtype": compute_dtype,
            **(model_kwargs or {}),
        },
    )
    student_model = cast(Any, student_mt.model)
    student_model.to(device=device)

    # Keep dropout off for stable teacher/student matching.
    # Gradients still flow even in eval mode.
    student_model.eval()
    _freeze_all_parameters(cast(nn.Module, student_model))

    num_layers = len(_get_decoder_layers(student_model))
    gap = GapSpec(start=gap_start, length=gap_length)
    _validate_gap(gap=gap, num_layers=num_layers)

    effective_mode = _get_effective_mode(gap=gap, num_layers=num_layers)
    _stage(f"mode={effective_mode}")

    hidden_size = _get_hidden_size(student_model)
    bridge = GapBridge(
        hidden_size=hidden_size,
    ).to(
        device=device,
        dtype=torch.float32,
    )
    bridge.train()

    named_trainable_student_modules = _collect_trainable_student_modules(
        model=student_model,
        gap=gap,
        train_kept_layers=train_kept_layers,
        train_final_norm=train_final_norm,
        train_embed_tokens=train_embed_tokens,
        train_lm_head=train_lm_head,
    )

    student_trainable_params = _enable_trainable_student_modules(
        named_modules=named_trainable_student_modules,
    )
    bridge_params = _dedupe_parameters(list(bridge.parameters()))

    _stage(
        "student trainable modules: "
        + (", ".join(name for name, _ in named_trainable_student_modules) if named_trainable_student_modules else "<none>")
    )
    _stage(
        f"trainable parameter counts | bridge={_count_parameters(bridge_params):,} "
        f"| student={_count_parameters(student_trainable_params):,}"
    )

    param_groups: list[dict[str, Any]] = [
        {
            "params": bridge_params,
            "lr": lr,
            "weight_decay": weight_decay,
        }
    ]
    if student_trainable_params:
        param_groups.append(
            {
                "params": student_trainable_params,
                "lr": backbone_lr,
                "weight_decay": weight_decay,
            }
        )

    optimizer = torch.optim.AdamW(param_groups)

    dataset: Dataset = load_dataset(dataset_spec)
    window_settings = WindowSettings(C1=context_len)

    _stage("tokenizing dataset")
    tokenized_examples = tokenize_dataset_to_examples(
        dataset,
        student_mt.tokenizer,
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

    if effective_mode == "FULL-GAP":
        _stage(
            f"training bridge for FULL-GAP [0, {gap.end}) "
            f"on model with {num_layers} layers and hidden_size={hidden_size}"
        )
    elif effective_mode == "LATE-BEGIN":
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
                teacher_logits, teacher_reentry_hidden = _run_teacher_full_and_capture_reentry(
                    model=teacher_model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    gap=gap,
                )
                prev_reentry_hidden = _build_prev_hidden(teacher_reentry_hidden)

            student_logits, student_reentry_hidden = _run_student_gap_bridge(
                model=student_model,
                bridge=bridge,
                input_ids=input_ids,
                attention_mask=attention_mask,
                gap=gap,
                prev_state_hidden=prev_reentry_hidden,
            )

            loss_kl = _temperature_sharpened_masked_kl_teacher_to_student_next_token(
                logits_teacher=teacher_logits,
                logits_student=student_logits,
                attention_mask=attention_mask,
                teacher_temperature=teacher_temperature,
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
                clip_grad_norm_(bridge_params + student_trainable_params, max_grad_norm)

            optimizer.step()

            log_every = 50

            if step == 1 or step % log_every == 0:
                with torch.no_grad():
                    sim = distribution_similarity_metrics(
                        shift_logits_mid=student_logits[:, :-1, :].contiguous(),
                        shift_logits_full=teacher_logits[:, :-1, :].contiguous(),
                    )

                row = {
                    "step": float(step),
                    "epoch": float(epoch_idx + 1),
                    "loss": float(loss.item()),
                    "loss_kl": float(loss_kl.item()),
                    "loss_hidden": float(loss_hidden.item()),
                    "loss_ce": float(loss_ce_teacher.item()),
                    "js": float(sim["js"].item()),
                    "top1_agreement": float(sim["top1_agreement"].item()),
                    "overlap": float(sim["overlap"].item()),
                    "p_student_on_teacher_argmax": float(sim["p_mid_on_full_argmax"].item()),
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

        trainable_prefixes = [name for name, _ in named_trainable_student_modules]
        student_subset_state_dict = _state_dict_subset_for_prefixes(
            state_dict=cast(dict[str, torch.Tensor], student_model.state_dict()),
            prefixes=trainable_prefixes,
        )
        bridge_state_dict = {k: v.detach().cpu() for k, v in bridge.state_dict().items()}

        torch.save(
            {
                "model_name": model_name,
                "gap_start": gap.start,
                "gap_length": gap.length,
                "gap_end": gap.end,
                "num_layers": num_layers,
                "hidden_size": hidden_size,
                "effective_mode": effective_mode,
                "bridge_state_dict": bridge_state_dict,
                "student_trainable_prefixes": trainable_prefixes,
                "student_trainable_state_dict": student_subset_state_dict,
                "train_kept_layers": train_kept_layers,
                "train_final_norm": train_final_norm,
                "train_embed_tokens": train_embed_tokens,
                "train_lm_head": train_lm_head,
                "history": history,
            },
            checkpoint_path,
        )

        _stage(f"saved bridge checkpoint to {checkpoint_path}")

    _stage("train_gap_bridge: finished")

    return TrainGapBridgeOutput(
        student_model=cast(nn.Module, student_model),
        bridge=bridge,
        history=history,
        checkpoint_path=checkpoint_path,
    )