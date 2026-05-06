from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Callable, Literal, cast

import torch
import torch.nn as nn

from skip_search_spec.helpers.tooling import (
    get_preferred_device,
    get_preferred_float_dtype,
    load_model_and_tokenizer,
)
from skip_search_spec.protocols.windows import ModelAndTokenizer
from skip_search_spec.helpers.shared_decoding_tools import (
    GapSpec,
    forward_model,
    get_backbone,
    get_decoder_layers,
    get_first_hidden_from_inputs,
    get_hidden_size,
    get_reentry_module_for_gap,
    make_identity_skip_hook,
    validate_gap,
)


ReferenceHiddenSource = Literal["reentry", "final"]

class NoOpDecoderLayer(nn.Module):
    """
    Cheap replacement for a HF decoder layer.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        if kwargs.get("output_attentions", False):
            raise RuntimeError(
                "NoOpDecoderLayer does not support output_attentions=True. "
                "A skipped layer has no attention weights to return."
            )

        return hidden_states


@dataclass(frozen=True, slots=True)
class BridgedGapConfig:
    model_name: str
    gap_start: int
    gap_length: int
    reference_hidden_source: ReferenceHiddenSource = "reentry"

    @property
    def gap(self) -> GapSpec:
        return GapSpec(
            start=self.gap_start,
            length=self.gap_length,
        )


@dataclass(slots=True)
class VerifierBridgeOutput:
    logits: torch.Tensor
    past_key_values: Any | None

    # Hidden entering the re-entry module.
    # For internal gaps, this is input to layer gap.end.
    # For early-exit, this is input to final norm.
    reentry_hidden: torch.Tensor

    # Hidden after the last decoder layer, before final norm / lm_head.
    final_lm_layer_hidden: torch.Tensor

    # The hidden source used as the bridge's previous-position conditioning.
    # This can be reentry_hidden today, final_hidden tomorrow, etc.
    reference_hidden: torch.Tensor


@dataclass(slots=True)
class DrafterBridgeOutput:
    logits: torch.Tensor | None
    past_key_values: Any | None

    # Hidden entering the skipped gap.
    gap_input_hidden: torch.Tensor

    # Bridge output injected at the re-entry point.
    bridged_reentry_hidden: torch.Tensor

    # Hidden after the last decoder layer, before final norm / lm_head.
    final_lm_layer_hidden: torch.Tensor

    # Hidden after final norm, immediately before lm_head.
    lm_head_input_hidden: torch.Tensor

    # The hidden source to use for future drafted positions.
    reference_hidden: torch.Tensor


class LinearPrevHiddenGapBridge(nn.Module):
    """
    Default bridge module.

    It predicts re-entry hidden from:
      - hidden entering the skipped gap
      - previous-position reference hidden

    The reference hidden can be teacher re-entry hidden, final hidden, or any
    future choice controlled by BridgedGapModel.
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
        self.proj = nn.Linear(hidden_size * 2, hidden_size, bias=bias)

        with torch.no_grad():
            nn.init.zeros_(self.proj.weight)
            if self.proj.bias is not None:
                nn.init.zeros_(self.proj.bias)

    def forward(
        self,
        gap_hidden: torch.Tensor,
        prev_reference_hidden: torch.Tensor,
    ) -> torch.Tensor:
        if gap_hidden.shape != prev_reference_hidden.shape:
            raise ValueError(
                f"gap_hidden.shape {gap_hidden.shape} "
                f"!= prev_reference_hidden.shape {prev_reference_hidden.shape}"
            )

        bridge_dtype = self.proj.weight.dtype
        x = gap_hidden.to(dtype=bridge_dtype)
        p = prev_reference_hidden.to(dtype=bridge_dtype)

        x_n = self.gap_norm(x)
        p_n = self.prev_norm(p)

        delta = self.proj(torch.cat([x_n, p_n], dim=-1))
        return x + delta


class BridgedGapModel:
    """
    Central runtime object for a model with a trained bridge over a skipped gap.

    Stable public API:
      - run_verifier(...)
      - run_drafter(...)
      - build_prev_reference(...)
      - save_checkpoint(...)
      - load_from_checkpoint(...)

    The code using this class should not need to know whether the bridge is
    conditioned on previous re-entry hidden, previous final hidden, or another
    future reference source.
    """

    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        bridge: nn.Module,
        config: BridgedGapConfig,
        device: torch.device,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.bridge = bridge
        self.config = config
        self.gap = config.gap
        self.device = device

        self.model.eval()
        self.bridge.eval()

        for p in self.model.parameters():
            p.requires_grad_(False)

        num_layers = len(get_decoder_layers(self.model))
        validate_gap(gap=self.gap, num_layers=num_layers)

    @property
    def num_layers(self) -> int:
        return len(get_decoder_layers(self.model))

    @property
    def hidden_size(self) -> int:
        return get_hidden_size(self.model)

    @property
    def active_layer_indices(self) -> tuple[int, ...]:
        return tuple(
            i
            for i in range(self.num_layers)
            if not (self.gap.start <= i < self.gap.end)
        )

    @property
    def skipped_layer_indices(self) -> tuple[int, ...]:
        return tuple(range(self.gap.start, self.gap.end))

    def train_bridge_only(self) -> None:
        self.model.eval()
        self.bridge.train()

        for p in self.model.parameters():
            p.requires_grad_(False)

        for p in self.bridge.parameters():
            p.requires_grad_(True)

    def eval_all(self) -> None:
        self.model.eval()
        self.bridge.eval()

        for p in self.model.parameters():
            p.requires_grad_(False)

        for p in self.bridge.parameters():
            p.requires_grad_(False)

    def build_prev_reference(
        self,
        reference_hidden: torch.Tensor,
    ) -> torch.Tensor:
        """
        Shift reference hidden right by one position.

        If reference_hidden is:
            [h0, h1, h2, h3]

        then output is:
            [0, h0, h1, h2]

        This is the tensor passed to the bridge as previous-position reference.
        """
        prev = torch.zeros_like(reference_hidden)
        prev[:, 1:, :] = reference_hidden[:, :-1, :]
        return prev

    def select_reference_hidden(
        self,
        *,
        reentry_hidden: torch.Tensor,
        final_lm_layer_hidden: torch.Tensor,
    ) -> torch.Tensor:
        """
        Select the hidden stream used as previous-position bridge conditioning.

        "reentry":
            hidden entering the re-entry module.

        "final_lm_layer":
            hidden after the last decoder layer, before final norm / lm_head.
        """
        if self.config.reference_hidden_source == "reentry":
            return reentry_hidden

        if self.config.reference_hidden_source == "final":
            return final_lm_layer_hidden

        raise ValueError(
            f"Unknown reference_hidden_source: "
            f"{self.config.reference_hidden_source}"
        )

    @torch.no_grad()
    def run_verifier(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        past_key_values: Any | None = None,
    ) -> VerifierBridgeOutput:
        """
        Run the full model normally and capture:
          - logits
          - teacher re-entry hidden
          - teacher final hidden
          - selected reference hidden
        """
        reentry_module = get_reentry_module_for_gap(
            model=self.model,
            gap=self.gap,
        )
        backbone = get_backbone(self.model)

        reentry_hidden: torch.Tensor | None = None
        final_lm_layer_hidden: torch.Tensor | None = None

        def reentry_prehook(module: Any, inputs: tuple[Any, ...]) -> None:
            nonlocal reentry_hidden
            reentry_hidden = get_first_hidden_from_inputs(inputs).detach()

        def final_norm_prehook(
            module: Any,
            inputs: tuple[Any, ...],
        ) -> None:
            nonlocal final_lm_layer_hidden
            final_lm_layer_hidden = get_first_hidden_from_inputs(inputs).detach()

        with ExitStack() as stack:
            handle = reentry_module.register_forward_pre_hook(reentry_prehook)
            stack.callback(handle.remove)

            handle = backbone.norm.register_forward_pre_hook(final_norm_prehook)
            stack.callback(handle.remove)

            verifier_forward = forward_model(
                model=self.model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                past_key_values=past_key_values,
            )

        if reentry_hidden is None:
            raise RuntimeError("Failed to capture verifier re-entry hidden.")

        if final_lm_layer_hidden is None:
            raise RuntimeError("Failed to capture verifier final hidden.")

        reference_hidden = self.select_reference_hidden(
            reentry_hidden=reentry_hidden,
            final_lm_layer_hidden=final_lm_layer_hidden,
        )

        return VerifierBridgeOutput(
            logits=verifier_forward.logits.detach(),
            past_key_values=verifier_forward.past_key_values,
            reentry_hidden=reentry_hidden,
            final_lm_layer_hidden=final_lm_layer_hidden,
            reference_hidden=reference_hidden.detach(),
        )

    def run_drafter(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        prev_reference_hidden: torch.Tensor,
        past_key_values: Any | None = None,
        use_cache: bool = False,
        compute_logits: bool = True,
        timings: Any | None = None,
        measure_internal_timings: bool = False,
    ) -> DrafterBridgeOutput:
        """
        Run the model with:
          - gap input captured
          - skipped layers replaced by identity outputs
          - bridge output injected at re-entry

        Returns both re-entry and final hidden, but callers should usually use
        output.reference_hidden instead of caring which one is currently active.
        """
        layers = get_decoder_layers(self.model)
        reentry_module = get_reentry_module_for_gap(
            model=self.model,
            gap=self.gap,
        )
        backbone = get_backbone(self.model)

        gap_input_hidden: torch.Tensor | None = None
        bridged_reentry_hidden: torch.Tensor | None = None
        final_lm_layer_hidden: torch.Tensor | None = None
        lm_head_input_hidden: torch.Tensor | None = None

        def capture_gap_input_prehook(
            module: Any,
            inputs: tuple[Any, ...],
        ) -> None:
            nonlocal gap_input_hidden
            gap_input_hidden = get_first_hidden_from_inputs(inputs).detach()

        def inject_bridge_prehook(
            module: Any,
            inputs: tuple[Any, ...],
        ) -> tuple[Any, ...]:
            nonlocal bridged_reentry_hidden

            if gap_input_hidden is None:
                raise RuntimeError("Gap input hidden was not captured.")

            if prev_reference_hidden.shape != gap_input_hidden.shape:
                raise ValueError(
                    f"prev_reference_hidden.shape {prev_reference_hidden.shape} "
                    f"!= gap_input_hidden.shape {gap_input_hidden.shape}"
                )

            bridged = self.bridge(
                gap_input_hidden,
                prev_reference_hidden,
            )

            bridged = bridged.to(dtype=gap_input_hidden.dtype)
            bridged_reentry_hidden = bridged

            if len(inputs) == 0:
                raise RuntimeError("Re-entry module received empty inputs.")

            return (bridged, *inputs[1:])

        def final_norm_prehook(
            module: Any,
            inputs: tuple[Any, ...],
        ) -> None:
            nonlocal final_lm_layer_hidden
            final_lm_layer_hidden = get_first_hidden_from_inputs(inputs)

        def final_norm_hook(
            module: Any,
            inputs: tuple[Any, ...],
            output: Any,
        ) -> None:
            nonlocal lm_head_input_hidden
            if not isinstance(output, torch.Tensor):
                raise TypeError(
                    f"Expected final norm output to be a Tensor, got {type(output)}"
                )
            lm_head_input_hidden = output

        with ExitStack() as stack:
            registration_start_time = (
                time.perf_counter() if measure_internal_timings else None
            )
            original_layers: list[tuple[int, nn.Module]] = []

            for layer_idx in range(self.gap.start, self.gap.end):
                original_layer = layers[layer_idx]
                original_layers.append((layer_idx, original_layer))

                layers[layer_idx] = NoOpDecoderLayer()

            def restore_original_layers() -> None:
                teardown_start_time = (
                    time.perf_counter() if measure_internal_timings else None
                )
                for layer_idx, original_layer in reversed(original_layers):
                    layers[layer_idx] = original_layer
                if (
                    measure_internal_timings
                    and timings is not None
                    and teardown_start_time is not None
                ):
                    timings.drafter_teardown_seconds += (
                        time.perf_counter() - teardown_start_time
                    )

            stack.callback(restore_original_layers)

            handle = layers[self.gap.start].register_forward_pre_hook(
                capture_gap_input_prehook
            )
            stack.callback(handle.remove)

            handle = reentry_module.register_forward_pre_hook(
                inject_bridge_prehook
            )
            stack.callback(handle.remove)

            handle = backbone.norm.register_forward_pre_hook(final_norm_prehook)
            stack.callback(handle.remove)

            handle = backbone.norm.register_forward_hook(final_norm_hook)
            stack.callback(handle.remove)

            if (
                measure_internal_timings
                and timings is not None
                and registration_start_time is not None
            ):
                timings.drafter_registration_seconds += (
                    time.perf_counter() - registration_start_time
                )

            if compute_logits:
                drafter_forward = forward_model(
                    model=self.model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=use_cache,
                    past_key_values=past_key_values,
                )
                logits = drafter_forward.logits
                output_past_key_values = drafter_forward.past_key_values
            else:
                backbone_forward = backbone(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=False,
                    use_cache=use_cache,
                    past_key_values=past_key_values,
                    return_dict=True,
                )
                logits = None
                output_past_key_values = getattr(
                    backbone_forward,
                    "past_key_values",
                    None,
                )

        if gap_input_hidden is None:
            raise RuntimeError("Failed to capture gap input hidden.")

        if bridged_reentry_hidden is None:
            raise RuntimeError("Failed to capture bridged re-entry hidden.")

        if final_lm_layer_hidden is None:
            raise RuntimeError("Failed to capture drafter final hidden.")

        if lm_head_input_hidden is None:
            raise RuntimeError("Failed to capture drafter lm-head input hidden.")

        reference_hidden = self.select_reference_hidden(
            reentry_hidden=bridged_reentry_hidden,
            final_lm_layer_hidden=final_lm_layer_hidden,
        )

        return DrafterBridgeOutput(
            logits=logits,
            past_key_values=output_past_key_values,
            gap_input_hidden=gap_input_hidden,
            bridged_reentry_hidden=bridged_reentry_hidden,
            final_lm_layer_hidden=final_lm_layer_hidden,
            lm_head_input_hidden=lm_head_input_hidden,
            reference_hidden=reference_hidden,
        )

    def save_checkpoint(
        self,
        *,
        path: str | Path,
        step: int,
        optimizer_state_dict: dict[str, Any] | None = None,
        history: list[dict[str, float]] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload: dict[str, Any] = {
            "schema_version": 1,
            "model_name": self.config.model_name,
            "gap_start": self.gap.start,
            "gap_length": self.gap.length,
            "gap_end": self.gap.end,
            "num_layers": self.num_layers,
            "hidden_size": self.hidden_size,
            "reference_hidden_source": self.config.reference_hidden_source,
            "active_layer_indices": self.active_layer_indices,
            "skipped_layer_indices": self.skipped_layer_indices,
            "step": step,
            "bridge_class": type(self.bridge).__name__,
            "bridge_state_dict": self.bridge.state_dict(),
            "history": history or [],
        }

        if optimizer_state_dict is not None:
            payload["optimizer_state_dict"] = optimizer_state_dict

        if extra is not None:
            payload["extra"] = extra

        torch.save(payload, path)
        return path

    @classmethod
    def load_from_checkpoint(
        cls,
        *,
        bridge_checkpoint_path: str | Path,
        bridge_factory: Callable[[int, dict[str, Any]], nn.Module] | None = None,
        model_kwargs: dict[str, Any] | None = None,
        bridge_dtype: torch.dtype | Literal["model"] = torch.float32,
    ) -> "BridgedGapModel":
        """
        Load model + tokenizer + bridge from checkpoint.

        bridge_factory lets you change bridge architectures without changing
        training/inference code.

        bridge_dtype defaults to float32 for training/evaluation stability.
        Use "model" for inference when the bridge should follow the loaded
        model dtype.

        Signature:
            bridge_factory(hidden_size, checkpoint) -> nn.Module
        """
        device = get_preferred_device()
        compute_dtype = get_preferred_float_dtype(device)

        print("DEVICE:", device)
        print("COMPUTE_DTYPE:", compute_dtype)

        checkpoint = torch.load(
            bridge_checkpoint_path,
            map_location="cpu",
        )

        model_name = checkpoint.get("model_name")
        if not isinstance(model_name, str):
            raise ValueError("Checkpoint must contain string field 'model_name'.")

        model_and_tokenizer: ModelAndTokenizer = load_model_and_tokenizer(
            model_name,
            model_kwargs={
                "torch_dtype": compute_dtype,
                **(model_kwargs or {}),
            },
        )

        model = cast(Any, model_and_tokenizer.model)
        tokenizer = model_and_tokenizer.tokenizer

        model.to(device=device)
        model.eval()

        gap_start = int(checkpoint["gap_start"])
        gap_length = int(checkpoint["gap_length"])
        hidden_size = int(checkpoint.get("hidden_size", get_hidden_size(model)))

        reference_hidden_source = cast(
            ReferenceHiddenSource,
            checkpoint.get("reference_hidden_source", "reentry"),
        )

        config = BridgedGapConfig(
            model_name=model_name,
            gap_start=gap_start,
            gap_length=gap_length,
            reference_hidden_source=reference_hidden_source,
        )

        if bridge_factory is None:
            bridge = LinearPrevHiddenGapBridge(hidden_size=hidden_size)
        else:
            bridge = bridge_factory(hidden_size, checkpoint)

        bridge.load_state_dict(checkpoint["bridge_state_dict"])
        if bridge_dtype == "model":
            bridge_dtype = next(model.parameters()).dtype
        bridge.to(device=device, dtype=bridge_dtype)
        bridge.eval()

        return cls(
            model=model,
            tokenizer=tokenizer,
            bridge=bridge,
            config=config,
            device=device,
        )
    

    def bridge_target_hidden(
        self,
        verifier: VerifierBridgeOutput,
    ) -> torch.Tensor:
        """
        Hidden target the bridge is trained to reproduce.
        Today this is teacher re-entry hidden.

        If the bridge target changes later, update this method only.
        """
        return verifier.reentry_hidden


    def bridge_prediction_hidden(
        self,
        drafter: DrafterBridgeOutput,
    ) -> torch.Tensor:
        """
        Hidden prediction produced by the bridge.
        Today this is bridged re-entry hidden.

        If the bridge output changes later, update this method only.
        """
        return drafter.bridged_reentry_hidden



def build_bridged_gap_model(
    *,
    model_name: str,
    active_start_layers: int,
    active_end_layers: int,
    reference_hidden_source: ReferenceHiddenSource = "reentry",
    bridge_factory: Callable[[int], nn.Module] | None = None,
    model_kwargs: dict[str, Any] | None = None,
) -> BridgedGapModel:
    """
    Convenience constructor for training from scratch.
    """
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
    tokenizer = model_and_tokenizer.tokenizer

    model.to(device=device)
    model.eval()

    hidden_size = get_hidden_size(model)

    if bridge_factory is None:
        bridge = LinearPrevHiddenGapBridge(hidden_size=hidden_size)
    else:
        bridge = bridge_factory(hidden_size)

    bridge.to(device=device, dtype=torch.float32)


    num_layers = len(get_decoder_layers(model))
    gap_start = active_start_layers
    gap_length = num_layers - active_start_layers - active_end_layers

    config = BridgedGapConfig(
        model_name=model_name,
        gap_start=gap_start,
        gap_length=gap_length,
        reference_hidden_source=reference_hidden_source,
    )

    return BridgedGapModel(
        model=model,
        tokenizer=tokenizer,
        bridge=bridge,
        config=config,
        device=device,
    )
