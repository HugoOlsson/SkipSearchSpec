from __future__ import annotations

import re
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from datasets.arrow_dataset import Dataset

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
from skip_search_spec.protocols.measurements import (
    MeasurementRun,
    MetricEvent,
    RunContext,
    print_metric_events_line,
    save_at_interval,
)
from skip_search_spec.protocols.windows import DatasetSpec, ModelAndTokenizer, WindowSettings


def _stage(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def _get_backbone(model: Any) -> Any:
    if hasattr(model, "model") and hasattr(model.model, "layers") and hasattr(model.model, "norm"):
        return model.model

    raise TypeError(
        "Unsupported model structure. Expected a decoder-only HF model with "
        "`model.layers` and `model.norm`."
    )


def _extract_hidden_from_layer_output(layer_output: Any) -> torch.Tensor:
    if isinstance(layer_output, torch.Tensor):
        return layer_output

    if isinstance(layer_output, tuple) and len(layer_output) > 0:
        first = layer_output[0]
        if isinstance(first, torch.Tensor):
            return first

    raise TypeError(
        f"Could not extract hidden states from layer output of type {type(layer_output)}"
    )


def _replace_hidden_in_layer_output(layer_output: Any, new_hidden: torch.Tensor) -> Any:
    if isinstance(layer_output, torch.Tensor):
        return new_hidden

    if isinstance(layer_output, tuple) and len(layer_output) > 0:
        return type(layer_output)((new_hidden, *layer_output[1:]))

    raise TypeError(
        f"Could not replace hidden states in layer output of type {type(layer_output)}"
    )


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
    shift_logits_teacher = logits_teacher[:, :-1, :].contiguous()
    shift_logits_student = logits_student[:, :-1, :].contiguous()

    log_p_student = F.log_softmax(shift_logits_student.float(), dim=-1)
    log_p_teacher = F.log_softmax(shift_logits_teacher.float(), dim=-1)

    kl_per_token = F.kl_div(
        log_p_student,
        log_p_teacher,
        reduction="none",
        log_target=True,
    ).sum(dim=-1)

    return kl_per_token.mean()


def parse_gap_from_ablation_mask_name(mask_name: str) -> tuple[int, int]:
    """
    Example:
        drop_internal_block__start_10__len_6
    -> (10, 15)
    """
    match = re.fullmatch(r"drop_internal_block__start_(\d+)__len_(\d+)", mask_name)
    if match is None:
        raise ValueError(
            f"Mask name {mask_name!r} is not a drop_internal_block mask."
        )

    start = int(match.group(1))
    length = int(match.group(2))
    if length <= 0:
        raise ValueError(f"Invalid length {length} in mask name {mask_name!r}")

    end = start + length - 1
    return start, end


def _resolve_adapter_bottleneck_dim(
    *,
    hidden_size: int,
    adapter_bottleneck_dim: int | None,
) -> int:
    if adapter_bottleneck_dim is not None:
        if adapter_bottleneck_dim <= 0:
            raise ValueError(
                f"adapter_bottleneck_dim must be positive, got {adapter_bottleneck_dim}"
            )
        return adapter_bottleneck_dim

    # Reasonable default: small but not tiny.
    return max(64, hidden_size // 16)


class DraftPathResidualAdapter(nn.Module):
    """
    Zero-init residual bottleneck adapter used only in draft mode.

    Form:
        x -> x + tanh(gate) * Up( SiLU( Down( LN(x) ) ) )
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        bottleneck_dim: int,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, eps=eps)
        self.down = nn.Linear(hidden_size, bottleneck_dim, bias=False)
        self.up = nn.Linear(bottleneck_dim, hidden_size, bias=False)
        self.gate = nn.Parameter(torch.zeros(()))

        with torch.no_grad():
            nn.init.normal_(self.down.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype

        x = hidden_states.float()
        delta = self.up(F.silu(self.down(self.norm(x))))
        out = x + torch.tanh(self.gate.float()) * delta

        return out.to(dtype=input_dtype)


class MiddleGapDraftAdapterModel(nn.Module):
    """
    Frozen full model + draft-only adapters.

    Full path:
      - original frozen model, unchanged

    Draft path:
      - same frozen model, but layers [gap_start, ..., gap_end] are skipped
      - a draft-only bridge adapter is applied before the first post-gap layer
      - optional draft-only post-gap adapters are applied after the first few
        post-gap layers

    Important:
      - the original model weights are never trained
      - only the adapter modules in this wrapper are trainable
    """

    def __init__(
        self,
        *,
        base_model: Any,
        gap_start_layer_index: int,
        gap_end_layer_index: int,
        use_bridge_adapter: bool = True,
        num_post_gap_adapters: int = 0,
        adapter_bottleneck_dim: int | None = None,
    ):
        super().__init__()
        self.base_model = base_model
        self.gap_start_layer_index = gap_start_layer_index
        self.gap_end_layer_index = gap_end_layer_index
        self.use_bridge_adapter = use_bridge_adapter
        self.num_post_gap_adapters = num_post_gap_adapters

        self.backbone = _get_backbone(self.base_model)
        self.layers = self.backbone.layers

        num_layers = len(self.layers)
        if not (0 <= gap_start_layer_index <= gap_end_layer_index < num_layers):
            raise ValueError(
                f"Invalid gap [{gap_start_layer_index}, {gap_end_layer_index}] "
                f"for model with {num_layers} layers."
            )

        hidden_size = self.base_model.config.hidden_size
        self.adapter_bottleneck_dim = _resolve_adapter_bottleneck_dim(
            hidden_size=hidden_size,
            adapter_bottleneck_dim=adapter_bottleneck_dim,
        )

        first_post_gap_layer_idx = self.gap_end_layer_index + 1

        self.bridge_adapter: DraftPathResidualAdapter | None
        if self.use_bridge_adapter and first_post_gap_layer_idx < num_layers:
            self.bridge_adapter = DraftPathResidualAdapter(
                hidden_size=hidden_size,
                bottleneck_dim=self.adapter_bottleneck_dim,
            )
        else:
            self.bridge_adapter = None

        self.post_gap_adapters = nn.ModuleDict()
        if num_post_gap_adapters > 0:
            for offset in range(num_post_gap_adapters):
                layer_idx = first_post_gap_layer_idx + offset
                if layer_idx >= num_layers:
                    break

                self.post_gap_adapters[str(layer_idx)] = DraftPathResidualAdapter(
                    hidden_size=hidden_size,
                    bottleneck_dim=self.adapter_bottleneck_dim,
                )

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    def trainable_adapter_parameters(self) -> list[nn.Parameter]:
        params: list[nn.Parameter] = []

        if self.bridge_adapter is not None:
            params.extend(list(self.bridge_adapter.parameters()))

        for adapter in self.post_gap_adapters.values():
            params.extend(list(adapter.parameters()))

        return params

    def full_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        with torch.no_grad():
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,
                use_cache=False,
                return_dict=True,
            )
        return cast(torch.Tensor, outputs.logits)

    def draft_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        def make_skip_hook() -> Any:
            def skip_hook(_module: Any, inputs: tuple[Any, ...], output: Any) -> Any:
                if len(inputs) == 0 or not isinstance(inputs[0], torch.Tensor):
                    raise TypeError(
                        f"Expected first layer input to be hidden_states tensor, got "
                        f"{type(inputs[0]) if len(inputs) > 0 else 'empty inputs'}"
                    )

                hidden_in = inputs[0]
                return _replace_hidden_in_layer_output(output, hidden_in)

            return skip_hook

        def bridge_prehook(_module: Any, inputs: tuple[Any, ...]) -> tuple[Any, ...]:
            if len(inputs) == 0 or not isinstance(inputs[0], torch.Tensor):
                raise TypeError(
                    f"Expected first layer input to be hidden_states tensor, got "
                    f"{type(inputs[0]) if len(inputs) > 0 else 'empty inputs'}"
                )

            if self.bridge_adapter is None:
                return inputs

            hidden_in = inputs[0]
            hidden_out = self.bridge_adapter(hidden_in)
            return (hidden_out, *inputs[1:])

        def make_post_gap_adapter_hook(adapter: DraftPathResidualAdapter) -> Any:
            def adapter_hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> Any:
                hidden = _extract_hidden_from_layer_output(output)
                hidden_out = adapter(hidden)
                return _replace_hidden_in_layer_output(output, hidden_out)

            return adapter_hook

        with ExitStack() as stack:
            for layer_idx in range(self.gap_start_layer_index, self.gap_end_layer_index + 1):
                handle = self.layers[layer_idx].register_forward_hook(make_skip_hook())
                stack.callback(handle.remove)

            first_post_gap_layer_idx = self.gap_end_layer_index + 1
            if self.bridge_adapter is not None and first_post_gap_layer_idx < self.num_layers:
                handle = self.layers[first_post_gap_layer_idx].register_forward_pre_hook(
                    bridge_prehook
                )
                stack.callback(handle.remove)

            for layer_idx_str, adapter_module in self.post_gap_adapters.items():
                layer_idx = int(layer_idx_str)
                adapter = cast(DraftPathResidualAdapter, adapter_module)
                handle = self.layers[layer_idx].register_forward_hook(
                    make_post_gap_adapter_hook(adapter)
                )
                stack.callback(handle.remove)

            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,
                use_cache=False,
                return_dict=True,
            )

        return cast(torch.Tensor, outputs.logits)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits_draft = self.draft_logits(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        logits_full = self.full_logits(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return logits_draft, logits_full


def save_middle_gap_checkpoint(
    *,
    model: MiddleGapDraftAdapterModel,
    checkpoint_path: str,
    base_model_name: str,
    optimizer: torch.optim.Optimizer | None = None,
    extra_config: dict[str, Any] | None = None,
) -> None:
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint: dict[str, Any] = {
        "base_model_name": base_model_name,
        "gap_start_layer_index": model.gap_start_layer_index,
        "gap_end_layer_index": model.gap_end_layer_index,
        "use_bridge_adapter": model.use_bridge_adapter,
        "num_post_gap_adapters": model.num_post_gap_adapters,
        "adapter_bottleneck_dim": model.adapter_bottleneck_dim,
        "bridge_adapter_state_dict": (
            model.bridge_adapter.state_dict() if model.bridge_adapter is not None else None
        ),
        "post_gap_adapters_state_dict": model.post_gap_adapters.state_dict(),
        "extra_config": extra_config or {},
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    torch.save(checkpoint, path)


def load_middle_gap_checkpoint(
    *,
    checkpoint_path: str,
    device: torch.device,
    compute_dtype: torch.dtype,
) -> tuple[MiddleGapDraftAdapterModel, Any, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    base_model_name = cast(str, checkpoint["base_model_name"])
    gap_start_layer_index = cast(int, checkpoint["gap_start_layer_index"])
    gap_end_layer_index = cast(int, checkpoint["gap_end_layer_index"])
    use_bridge_adapter = cast(bool, checkpoint["use_bridge_adapter"])
    num_post_gap_adapters = cast(int, checkpoint["num_post_gap_adapters"])
    adapter_bottleneck_dim = cast(int, checkpoint["adapter_bottleneck_dim"])
    extra_config = cast(dict[str, Any], checkpoint.get("extra_config", {}))

    model_and_tokenizer: ModelAndTokenizer = load_model_and_tokenizer(
        base_model_name,
        model_kwargs={"torch_dtype": compute_dtype},
    )
    base_model = cast(Any, model_and_tokenizer.model)
    base_model.to(device=device, dtype=compute_dtype)
    base_model.eval()

    for p in base_model.parameters():
        p.requires_grad = False

    model = MiddleGapDraftAdapterModel(
        base_model=base_model,
        gap_start_layer_index=gap_start_layer_index,
        gap_end_layer_index=gap_end_layer_index,
        use_bridge_adapter=use_bridge_adapter,
        num_post_gap_adapters=num_post_gap_adapters,
        adapter_bottleneck_dim=adapter_bottleneck_dim,
    )

    bridge_adapter_state_dict = checkpoint.get("bridge_adapter_state_dict")
    if model.bridge_adapter is not None and bridge_adapter_state_dict is not None:
        model.bridge_adapter.load_state_dict(bridge_adapter_state_dict)

    model.post_gap_adapters.load_state_dict(checkpoint["post_gap_adapters_state_dict"])
    model.to(device=device)

    return model, model_and_tokenizer.tokenizer, extra_config


def train_middle_gap_skip(
    *,
    model_name: str,
    gap_start_layer: int,
    gap_end_layer: int,
    dataset_spec: DatasetSpec,
    batch_size: int,
    checkpoint_path: str,
    max_examples: int,
    context_len: int,
    alpha: float,
    beta: float,
    use_bridge_adapter: bool = True,
    num_post_gap_adapters: int = 0,
    adapter_bottleneck_dim: int | None = None,
    save_optimizer: bool = False,
) -> None:
    """
    Train draft-only adapters for a middle-gap skip path.

    Loss:
        alpha * CE(draft_path, labels)
      + beta  * KL(full_path || draft_path)

    Notes:
    - The original model weights are frozen.
    - Only adapter weights are trainable.
    - alpha and beta are independent weights here, not a convex combination.
    """
    _stage("train_middle_gap_skip: start")

    if alpha < 0.0:
        raise ValueError(f"alpha must be >= 0, got {alpha}")
    if beta < 0.0:
        raise ValueError(f"beta must be >= 0, got {beta}")

    device = get_preferred_device()
    compute_dtype = get_preferred_float_dtype(device)

    _stage(f"device={device} dtype={compute_dtype}")

    _stage("loading frozen base model")
    model_and_tokenizer: ModelAndTokenizer = load_model_and_tokenizer(
        model_name,
        model_kwargs={"torch_dtype": compute_dtype},
    )
    base_model = cast(Any, model_and_tokenizer.model)
    base_model.to(device=device, dtype=compute_dtype)
    base_model.eval()

    for p in base_model.parameters():
        p.requires_grad = False

    dataset: Dataset = load_dataset(dataset_spec)
    window_settings = WindowSettings(C1=context_len)

    _stage("tokenizing dataset")
    tokenized_examples = tokenize_dataset_to_examples(
        dataset,
        model_and_tokenizer.tokenizer,
        dataset_spec,
        max_examples=max_examples,
    )

    _stage("building training windows")
    training_windows = build_all_training_windows(
        tokenized_examples,
        window_settings,
        dataset_spec,
    )

    if len(training_windows) == 0:
        raise ValueError("No training windows were produced.")

    model = MiddleGapDraftAdapterModel(
        base_model=base_model,
        gap_start_layer_index=gap_start_layer,
        gap_end_layer_index=gap_end_layer,
        use_bridge_adapter=use_bridge_adapter,
        num_post_gap_adapters=num_post_gap_adapters,
        adapter_bottleneck_dim=adapter_bottleneck_dim,
    )
    model.to(device=device)

    trainable_params = model.trainable_adapter_parameters()
    if len(trainable_params) == 0:
        raise ValueError(
            "No trainable adapter parameters were created. "
            "Enable use_bridge_adapter or set num_post_gap_adapters > 0."
        )

    _stage(
        "gap setup: "
        f"gap=[{gap_start_layer},{gap_end_layer}] "
        f"gap_len={gap_end_layer - gap_start_layer + 1} "
        f"use_bridge_adapter={use_bridge_adapter} "
        f"num_post_gap_adapters={num_post_gap_adapters} "
        f"adapter_bottleneck_dim={model.adapter_bottleneck_dim}"
    )

    # Keep the base model deterministic; train/eval only matters for the adapters.
    model.train()
    model.base_model.eval()

    optimizer = torch.optim.AdamW(
        [{"params": trainable_params, "lr": 1e-4}]
    )

    window_dataset = WindowDataset(training_windows)
    dataloader = DataLoader(
        window_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_windows,
        pin_memory=(device.type == "cuda"),
    )

    run_context = RunContext.create(
        run_id=f"middle-gap-draft-adapters-{int(time.time())}",
        experiment_type="middle_gap_skip",
        model_names=(model_name,),
        dataset_name=dataset_spec.name,
        run_config={
            "model": model_name,
            "dataset": str(dataset_spec),
            "device": str(device),
            "compute_dtype": str(compute_dtype),
            "context_len": context_len,
            "total_windows": len(training_windows),
            "batch_size": batch_size,
            "steps_per_epoch": len(dataloader),
            "gap_start_layer": gap_start_layer,
            "gap_end_layer": gap_end_layer,
            "gap_length": gap_end_layer - gap_start_layer + 1,
            "total_layers": model.num_layers,
            "alpha": alpha,
            "beta": beta,
            "use_bridge_adapter": use_bridge_adapter,
            "num_post_gap_adapters": num_post_gap_adapters,
            "adapter_bottleneck_dim": model.adapter_bottleneck_dim,
        },
    )

    run_context.print()

    metric_events: list[MetricEvent] = []
    ce_draft_last: torch.Tensor | None = None
    ce_full_last: torch.Tensor | None = None

    _stage("starting training loop")

    for step, (input_ids, attention_mask) in enumerate(dataloader):
        input_ids = input_ids.to(device, non_blocking=True)
        attention_mask = attention_mask.to(device, non_blocking=True)
        labels = input_ids

        optimizer.zero_grad(set_to_none=True)

        logits_draft, logits_full = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        shift_logits_draft = logits_draft[:, :-1, :].contiguous()
        shift_logits_full = logits_full[:, :-1, :].contiguous()

        ce_draft = _cross_entropy_next_token(logits_draft, labels)
        with torch.no_grad():
            ce_full = _cross_entropy_next_token(logits_full, labels)

        kl_full_to_draft = _kl_teacher_to_student_next_token(
            logits_teacher=logits_full,
            logits_student=logits_draft,
        )

        loss = alpha * ce_draft + beta * kl_full_to_draft

        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()

        ce_draft_last = ce_draft
        ce_full_last = ce_full

        if step % 10 == 0:
            with torch.no_grad():
                metrics = distribution_similarity_metrics(
                    shift_logits_mid=shift_logits_draft,
                    shift_logits_full=shift_logits_full,
                )

            batch_metrics = [
                MetricEvent.create(phase="train", name="loss", value=loss.item(), step=step),
                MetricEvent.create(phase="train", name="ce_draft", value=ce_draft.item(), step=step),
                MetricEvent.create(phase="train", name="ce_full", value=ce_full.item(), step=step),
                MetricEvent.create(
                    phase="train",
                    name="ce_gap",
                    value=(ce_draft.item() - ce_full.item()),
                    step=step,
                ),
                MetricEvent.create(
                    phase="train",
                    name="kl_full_to_draft",
                    value=metrics["kl_full_to_mid"].item(),
                    step=step,
                ),
                MetricEvent.create(phase="train", name="js", value=metrics["js"].item(), step=step),
                MetricEvent.create(
                    phase="train",
                    name="top1_agreement",
                    value=metrics["top1_agreement"].item(),
                    step=step,
                ),
                MetricEvent.create(phase="train", name="overlap", value=metrics["overlap"].item(), step=step),
                MetricEvent.create(
                    phase="train",
                    name="p_draft_on_full_argmax",
                    value=metrics["p_mid_on_full_argmax"].item(),
                    step=step,
                ),
            ]

            metric_events.extend(batch_metrics)

            print(f"[{step}/{len(dataloader)}] ", end="")
            print_metric_events_line(batch_metrics, decimals=4)

            run = MeasurementRun(context=run_context, metric_events=metric_events)
            save_at_interval(run)

    if ce_draft_last is not None and ce_full_last is not None:
        print("Done. Final CE gap:", (ce_draft_last.item() - ce_full_last.item()))

    save_middle_gap_checkpoint(
        model=model,
        checkpoint_path=checkpoint_path,
        base_model_name=model_name,
        optimizer=optimizer if save_optimizer else None,
        extra_config={
            "gap_start_layer": gap_start_layer,
            "gap_end_layer": gap_end_layer,
            "alpha": alpha,
            "beta": beta,
            "context_len": context_len,
            "batch_size": batch_size,
            "use_bridge_adapter": use_bridge_adapter,
            "num_post_gap_adapters": num_post_gap_adapters,
            "adapter_bottleneck_dim": model.adapter_bottleneck_dim,
        },
    )
    print(f"Saved checkpoint to: {checkpoint_path}")

    run = MeasurementRun(
        context=run_context,
        metric_events=metric_events,
    )
    saved_log_path = run.save()
    print(f"Saved measurement log to: {saved_log_path}")