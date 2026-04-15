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
from skip_search_spec.helpers.tooling import get_preferred_device, get_preferred_float_dtype, load_dataset, load_model_and_tokenizer
from skip_search_spec.helpers.window_building import WindowDataset, build_all_training_windows, collate_windows, tokenize_dataset_to_examples
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


def distribution_similarity_metrics(
    shift_logits_mid: torch.Tensor,
    shift_logits_full: torch.Tensor,
) -> dict[str, torch.Tensor]:
    shift_logits_mid = shift_logits_mid.detach().float()
    shift_logits_full = shift_logits_full.detach().float()

    log_p_mid = F.log_softmax(shift_logits_mid, dim=-1)
    log_p_full = F.log_softmax(shift_logits_full, dim=-1)

    p_mid = log_p_mid.exp()
    p_full = log_p_full.exp()

    kl_full_to_mid = (p_full * (log_p_full - log_p_mid)).sum(dim=-1).mean()
    kl_mid_to_full = (p_mid * (log_p_mid - log_p_full)).sum(dim=-1).mean()

    m = 0.5 * (p_full + p_mid)
    log_m = torch.log(m.clamp_min(1e-12))
    js = 0.5 * (
        (p_full * (log_p_full - log_m)).sum(dim=-1) +
        (p_mid * (log_p_mid - log_m)).sum(dim=-1)
    ).mean()

    top1_mid = shift_logits_mid.argmax(dim=-1)
    top1_full = shift_logits_full.argmax(dim=-1)
    top1_agreement = (top1_mid == top1_full).float().mean()

    full_argmax = top1_full.unsqueeze(-1)
    p_mid_on_full_argmax = p_mid.gather(dim=-1, index=full_argmax).squeeze(-1).mean()

    overlap = torch.minimum(p_mid, p_full).sum(dim=-1).mean()

    return {
        "kl_full_to_mid": kl_full_to_mid,
        "kl_mid_to_full": kl_mid_to_full,
        "js": js,
        "top1_agreement": top1_agreement,
        "p_mid_on_full_argmax": p_mid_on_full_argmax,
        "overlap": overlap,
    }


def train_early_exit(
    *,
    model_name: str,
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
    stage(f"device={device}, compute_dtype={compute_dtype}")

    stage(f"loading model+tokenizer: {model_name}")
    model_and_tokenizer: ModelAndTokenizer = load_model_and_tokenizer(
        model_name,
        model_kwargs={"torch_dtype": compute_dtype},
    )
    stage("loaded model+tokenizer")

    base_model = cast(Any, model_and_tokenizer.model)
    stage("moving base model to target device")
    base_model.to(device=device, dtype=compute_dtype)
    stage("base model moved to target device")

    stage(f"loading dataset: {dataset_spec}")
    dataset: Dataset = load_dataset(dataset_spec)
    stage("dataset loaded")
    context_parts = WindowSettings(C1=context_len)

    stage("tokenizing dataset")
    tokenized_examples = tokenize_dataset_to_examples(
        dataset,
        model_and_tokenizer.tokenizer,
        dataset_spec,
        max_examples=max_examples,
    )
    stage(f"tokenization done: {len(tokenized_examples)} examples")

    stage("building training windows")
    training_windows = build_all_training_windows(
        tokenized_examples,
        context_parts,
        dataset_spec,
    )
    stage(f"window building done: {len(training_windows)} windows")

    num_layers = len(base_model.model.layers)
    inner_exit_layer_index = 15#int(num_layers // 1.2)

    early_exit_model = EarlyExitModel(
        base_model=base_model,
        inner_exit_layer_index=inner_exit_layer_index,
    )
    early_exit_model.to(device=device, dtype=compute_dtype)

    for p in early_exit_model.parameters():
        p.requires_grad = False

    start = inner_exit_layer_index - 4
    end = inner_exit_layer_index + 2

    if start < 0:
        raise ValueError(
            f"Trained layer range [{start}:{end}] clips below 0 — model too small or exit layer too early."
        )
    if end > num_layers:
        raise ValueError(
            f"Trained layer range [{start}:{end}] exceeds model depth {num_layers}."
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

    print("STARTING EARLY-EXIT TRAINING")
    print(f"  model={model_name}")
    print(f"  dataset={dataset_spec}")
    print(f"  device={device}")
    print(f"  compute_dtype={compute_dtype}")
    print(f"  context: C1={context_parts.C1}")
    print(f"  total_windows={len(training_windows)}")
    print(f"  batch_size={batch_size}")
    print(f"  steps_per_epoch={len(dataloader)}")
    print(f"  early_exit_layer={inner_exit_layer_index}, total_layers={num_layers}")
    trained_indices = [
        i for i, layer in enumerate(early_exit_model.layers)
        if layer in trained_blocks
    ]
    print(f"  trained_layers={trained_indices}")
    print("")

    ce_mid: torch.Tensor | None = None
    ce_full: torch.Tensor | None = None

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

            print(
                f"[{step}/{len(dataloader)}] "
                f"loss={loss.item():.4f}  "
                f"ce_mid={ce_mid.item():.4f}  "
                f"ce_full={ce_full.item():.4f}  "
                f"gap={(ce_mid.item() - ce_full.item()):.4f}  "
                f"kl(full||mid)={metrics['kl_full_to_mid'].item():.4f}  "
                f"js={metrics['js'].item():.4f}  "
                f"top1_agree={metrics['top1_agreement'].item():.4f}  "
                f"overlap={metrics['overlap'].item():.4f}  "
                f"p_mid@full_argmax={metrics['p_mid_on_full_argmax'].item():.4f}"
            )

    if ce_mid is not None and ce_full is not None:
        print("Done. Final gap:", (ce_mid.item() - ce_full.item()))

    save_early_exit_checkpoint(
        early_exit_model=early_exit_model,
        checkpoint_path=checkpoint_path,
        base_model_name=model_name,
        optimizer=optimizer if save_optimizer else None,
    )
    print(f"Saved checkpoint to: {checkpoint_path}")
