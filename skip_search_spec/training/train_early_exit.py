from __future__ import annotations

from pathlib import Path
import time
from typing import Any, cast

from datasets.arrow_dataset import Dataset
import torch
from torch import nn
from torch.utils.data import Dataset as TorchDataset, DataLoader
import torch.nn.functional as F

from skip_search_spec.helpers.storage import save_early_exit_checkpoint
from skip_search_spec.helpers.tooling import distribution_similarity_metrics, get_preferred_device, get_preferred_float_dtype, load_dataset, load_model_and_tokenizer
from skip_search_spec.helpers.window_building import WindowDataset, build_all_training_windows, collate_windows, tokenize_dataset_to_examples
from skip_search_spec.protocols.measurements import  MeasurementRun, MetricEvent, RunContext, save_at_interval, print_metric_events_line
from skip_search_spec.protocols.windows import DatasetSpec, ModelAndTokenizer, WindowSettings


class _StopAtExitLayer(Exception):
    def __init__(self, hidden_states: torch.Tensor):
        super().__init__("Stopped at early-exit layer")
        self.hidden_states = hidden_states


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


class EarlyExitModel(nn.Module):
    """
    A small wrapper around a decoder-only HF model with:
    - full_logits(...) for verifier/full-model decoding
    - draft_logits(...) for true early-exit drafting
    - forward(...) for training, returning both mid and full logits

    Assumes a Qwen/LLaMA-style backbone with:
        base_model.model.layers
        base_model.model.norm
        base_model.lm_head
    """

    def __init__(
        self,
        base_model: Any,
        inner_exit_layer_index: int,
    ):
        super().__init__()
        self.base_model = base_model
        self.inner_exit_layer_index = inner_exit_layer_index

        hidden_size = self.base_model.config.hidden_size
        self.mid_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        ref_param = next(self.base_model.parameters())
        self.mid_proj.to(device=ref_param.device, dtype=ref_param.dtype)

        with torch.no_grad():
            nn.init.zeros_(self.mid_proj.weight)

        num_layers = len(self.layers)
        if not (0 <= self.inner_exit_layer_index < num_layers):
            raise ValueError(
                f"inner_exit_layer_index {self.inner_exit_layer_index} out of range "
                f"(model has {num_layers} layers)"
            )

    @property
    def backbone(self) -> Any:
        return self.base_model.model

    @property
    def layers(self) -> Any:
        return self.backbone.layers

    def _apply_mid_head(self, hidden_states: torch.Tensor) -> torch.Tensor:
        base = self.backbone.norm(hidden_states)
        mid_hidden = base + self.mid_proj(base)
        return self.base_model.lm_head(mid_hidden)

    def full_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            use_cache=False,
            return_dict=True,
        )
        return cast(torch.Tensor, outputs.logits)

    def draft_hidden(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Run the base model only up to the exit layer by registering a hook on that
        layer and aborting the forward pass as soon as its output is available.

        This keeps the code compact while still being a real early-exit path.
        """
        exit_layer = self.layers[self.inner_exit_layer_index]

        def stop_hook(_module: Any, _inputs: Any, output: Any) -> None:
            hidden = _extract_hidden_from_layer_output(output)
            raise _StopAtExitLayer(hidden)

        handle = exit_layer.register_forward_hook(stop_hook)

        hidden_mid: torch.Tensor | None = None
        try:
            _ = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,
                use_cache=False,
                return_dict=True,
            )
        except _StopAtExitLayer as exc:
            hidden_mid = exc.hidden_states
        finally:
            handle.remove()

        if hidden_mid is None:
            raise RuntimeError("Failed to capture hidden states at the early-exit layer.")

        return hidden_mid

    def draft_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_mid = self.draft_hidden(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return self._apply_mid_head(hidden_mid)

    def training_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Training path: run the full model once with hidden states so we can build
        both the early-exit logits and the full logits efficiently.
        """
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )

        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise ValueError(
                "Model did not return hidden_states. Make sure output_hidden_states=True is supported."
            )

        hidden_mid = hidden_states[self.inner_exit_layer_index + 1]
        logits_mid = self._apply_mid_head(hidden_mid)
        logits_full = cast(torch.Tensor, outputs.logits)
        return logits_mid, logits_full

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.training_logits(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )


def train_early_exit(
    *,
    model_name: str,
    early_exit_layer: int,
    dataset_spec: DatasetSpec,
    batch_size: int,
    checkpoint_path: str,
    max_examples: int,
    context_len: int,
    alpha: float,
    beta: float,
    save_optimizer: bool,
) -> None:
    def stage(message: str) -> None:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)

    stage("train_early_exit: start")
    device = get_preferred_device()
    compute_dtype = get_preferred_float_dtype(device)

  

    model_and_tokenizer: ModelAndTokenizer = load_model_and_tokenizer(
        model_name,
        model_kwargs={"torch_dtype": compute_dtype},
    )

    base_model = cast(Any, model_and_tokenizer.model)
    base_model.to(device=device, dtype=compute_dtype)

    dataset: Dataset = load_dataset(dataset_spec)
    context_parts = WindowSettings(C1=context_len)

    tokenized_examples = tokenize_dataset_to_examples(
        dataset,
        model_and_tokenizer.tokenizer,
        dataset_spec,
        max_examples=max_examples,
    )

    training_windows = build_all_training_windows(
        tokenized_examples,
        context_parts,
        dataset_spec,
    )

    num_layers = len(base_model.model.layers)

    early_exit_model = EarlyExitModel(
        base_model=base_model,
        inner_exit_layer_index=early_exit_layer,
    )
    early_exit_model.to(device=device, dtype=compute_dtype)

    for p in early_exit_model.parameters():
        p.requires_grad = False

    start = 0
    end = early_exit_layer + 1 + 1 # Include the layer after the early exit layer

    if end > num_layers:
        raise ValueError(
            f"Requested training through layer {early_exit_layer + 1}, "
            f"but model only has {num_layers} layers."
        )

    trained_blocks = list(early_exit_model.layers[start:end])

    for p in early_exit_model.mid_proj.parameters():
        p.requires_grad = True

    for block in trained_blocks:
        for p in block.parameters():
            p.requires_grad = True

    early_exit_model.train()

    inner_projection_params = list(early_exit_model.mid_proj.parameters())
    backbone_params: list[nn.Parameter] = []
    for block in trained_blocks:
        backbone_params.extend(list(block.parameters()))

    optimizer = torch.optim.AdamW(
        [
            {"params": inner_projection_params, "lr": 3e-4},
            {"params": backbone_params, "lr": 1e-5},
        ]
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
        run_id=f"early-exit-{int(time.time())}",
        experiment_type="early_exit",
        model_names=(model_name,),
        dataset_name=dataset_spec.name,
        run_config={
            "model": model_name,
            "dataset": str(dataset_spec),
            "device": str(device),
            "compute_dtype": str(compute_dtype),
            "C1": context_parts.C1,
            "total_windows": len(training_windows),
            "batch_size": batch_size,
            "steps_per_epoch": len(dataloader),
            "early_exit_layer": early_exit_layer,
            "total_layers": num_layers,
            "alpha": alpha,
            "beta": beta,
            "trained_layers": [
                i for i, layer in enumerate(early_exit_model.layers)
                if layer in trained_blocks
            ]
        },
    )

    run_context.print()

    ce_mid: torch.Tensor | None = None
    ce_full: torch.Tensor | None = None

    metric_events: list[MetricEvent] = []

    for step, (input_ids, attention_mask) in enumerate(dataloader):
        input_ids = input_ids.to(device, non_blocking=True)
        attention_mask = attention_mask.to(device, non_blocking=True)
        labels = input_ids

        optimizer.zero_grad(set_to_none=True)

        logits_mid, logits_full = early_exit_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        shift_logits_mid = logits_mid[:, :-1, :].contiguous()
        shift_logits_full = logits_full[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        log_p_mid = F.log_softmax(shift_logits_mid.float(), dim=-1)
        log_p_full = F.log_softmax(shift_logits_full.detach().float(), dim=-1)

        kl_per_token = F.kl_div(
            log_p_mid,
            log_p_full,
            reduction="none",
            log_target=True,
        ).sum(dim=-1)
        kl_full_to_mid = kl_per_token.mean()

        ce_mid = F.cross_entropy(
            shift_logits_mid.view(-1, shift_logits_mid.size(-1)),
            shift_labels.view(-1),
        )
        ce_full = F.cross_entropy(
            shift_logits_full.view(-1, shift_logits_full.size(-1)),
            shift_labels.view(-1),
        )

        loss = alpha * ce_mid + (1.0 - alpha) * ce_full + beta * kl_full_to_mid

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            inner_projection_params + backbone_params,
            max_norm=1.0,
        )
        optimizer.step()

        if step % 10 == 0:
            with torch.no_grad():
                metrics = distribution_similarity_metrics(
                    shift_logits_mid=shift_logits_mid,
                    shift_logits_full=shift_logits_full,
                )

            batch_metrics = [
                MetricEvent.create(phase="train", name="loss", value=loss.item(), step=step),
                MetricEvent.create(phase="train", name="ce_mid", value=ce_mid.item(), step=step),
                MetricEvent.create(phase="train", name="ce_full", value=ce_full.item(), step=step),
                MetricEvent.create(phase="train", name="ce_gap", value=(ce_mid.item() - ce_full.item()), step=step),
                MetricEvent.create(phase="train", name="kl_full_to_mid", value=metrics["kl_full_to_mid"].item(), step=step),
                MetricEvent.create(phase="train", name="js", value=metrics["js"].item(), step=step),
                MetricEvent.create(phase="train", name="top1_agreement", value=metrics["top1_agreement"].item(), step=step),
                MetricEvent.create(phase="train", name="overlap", value=metrics["overlap"].item(), step=step),
                MetricEvent.create(
                    phase="train",
                    name="p_mid_on_full_argmax",
                    value=metrics["p_mid_on_full_argmax"].item(),
                    step=step,
                ),
            ]

            metric_events.extend(batch_metrics)
            print(f"[{step}/{len(dataloader)}] ", end="")
            print_metric_events_line(batch_metrics, decimals=4)

            run = MeasurementRun(context=run_context, metric_events=metric_events)
            save_at_interval(run) 
        

    if ce_mid is not None and ce_full is not None:
        print("Done. Final gap:", (ce_mid.item() - ce_full.item()))

    save_early_exit_checkpoint(
        early_exit_model=early_exit_model,
        checkpoint_path=checkpoint_path,
        base_model_name=model_name,
        optimizer=optimizer if save_optimizer else None,
    )
    print(f"Saved checkpoint to: {checkpoint_path}")

    run = MeasurementRun(
        context=run_context,
        metric_events=metric_events,
    )

    saved_log_path = run.save()
    print(f"Saved measurement log to: {saved_log_path}")
