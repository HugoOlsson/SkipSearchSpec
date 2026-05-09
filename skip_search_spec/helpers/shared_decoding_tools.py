from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, cast
from contextlib import ExitStack
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from datasets import Dataset

from skip_search_spec.helpers.dataset_formatting import maybe_format_dataset_to_text
from skip_search_spec.helpers.tooling import (
    distribution_similarity_metrics,
    load_dataset,
)
from skip_search_spec.helpers.window_building import (
    PackedWindowDataset,
    build_window_index,
    collate_windows,
    tokenize_dataset_to_examples,
)
from skip_search_spec.protocols.windows import (
    DatasetSpec,
    ModelAndTokenizer,
    WindowSettings,
)


# =============================================================================
# Logging
# =============================================================================


def stage(
    message: str,
) -> None:
    prefix_parts: list[str] = []

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


# =============================================================================
# Decoder-only model helpers
# =============================================================================


def get_backbone(model: Any) -> Any:
    """
    Return the inner decoder backbone for common HF decoder-only models.

    Expected structure:
        model.model.layers
        model.model.norm

    Works for Qwen/LLaMA-style causal language models.
    """
    if hasattr(model, "model") and hasattr(model.model, "layers") and hasattr(model.model, "norm"):
        return model.model

    raise TypeError(
        "Unsupported model structure. Expected a decoder-only HF model with "
        "`model.layers` and `model.norm`."
    )


def get_decoder_layers(model: Any) -> Any:
    return get_backbone(model).layers


def get_hidden_size(model: Any) -> int:
    hidden_size = getattr(getattr(model, "config", None), "hidden_size", None)
    if isinstance(hidden_size, int) and hidden_size > 0:
        return hidden_size

    backbone = get_backbone(model)
    if hasattr(backbone, "embed_tokens") and hasattr(backbone.embed_tokens, "embedding_dim"):
        return int(backbone.embed_tokens.embedding_dim)

    raise ValueError("Could not infer hidden size from model.")


@dataclass(frozen=True, slots=True)
class ModelForwardOutput:
    logits: torch.Tensor
    past_key_values: Any | None


def forward_model_logits(
    *,
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    use_cache: bool = False,
    past_key_values: Any | None = None,
) -> torch.Tensor:
    return forward_model(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=use_cache,
        past_key_values=past_key_values,
    ).logits


def forward_model(
    *,
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    use_cache: bool = False,
    past_key_values: Any | None = None,
) -> ModelForwardOutput:
    """
    Run a decoder-only model and optionally keep its HF KV-cache.

    The default stays uncached because the drafter path currently replaces
    skipped layers with NoOpDecoderLayer, which is not cache-aware yet.
    """
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=False,
        use_cache=use_cache,
        past_key_values=past_key_values,
        return_dict=True,
    )
    return ModelForwardOutput(
        logits=cast(torch.Tensor, outputs.logits),
        past_key_values=getattr(outputs, "past_key_values", None),
    )


def crop_past_key_values(
    past_key_values: Any | None,
    *,
    max_length: int,
) -> Any | None:
    if past_key_values is None:
        return None

    crop = getattr(past_key_values, "crop", None)
    if callable(crop):
        crop(max_length)
        return past_key_values

    raise TypeError(
        f"Expected past_key_values to expose a callable crop(max_length), "
        f"got {type(past_key_values).__name__}."
    )


# =============================================================================
# Gap helpers
# =============================================================================


@dataclass(frozen=True, slots=True)
class GapSpec:
    start: int
    length: int

    @property
    def end(self) -> int:
        return self.start + self.length


def validate_gap(*, gap: GapSpec, num_layers: int) -> None:
    if gap.length <= 0:
        raise ValueError(f"gap.length must be > 0, got {gap.length}")

    if gap.start < 0:
        raise ValueError(f"gap.start must be >= 0, got {gap.start}")

    if gap.start >= num_layers:
        raise ValueError(
            f"gap.start must be < num_layers, got gap.start={gap.start}, "
            f"num_layers={num_layers}"
        )

    if gap.end > num_layers:
        raise ValueError(
            f"gap.end must be <= num_layers, got gap.end={gap.end}, "
            f"num_layers={num_layers}"
        )


def get_effective_gap_mode(*, gap: GapSpec, num_layers: int) -> str:
    if gap.start == 0:
        return "LATE-BEGIN"

    if gap.end == num_layers:
        return "EARLY-EXIT"

    return "GAP"


def get_reentry_module_for_gap(*, model: Any, gap: GapSpec) -> Any:
    backbone = get_backbone(model)
    layers = backbone.layers
    num_layers = len(layers)

    if gap.end < num_layers:
        return layers[gap.end]

    if gap.end == num_layers:
        return backbone.norm

    raise ValueError(f"Invalid gap.end={gap.end} for num_layers={num_layers}")


# =============================================================================
# Hook helpers for layer skipping / decoder surgery
# =============================================================================


def get_first_hidden_from_inputs(inputs: tuple[Any, ...]) -> torch.Tensor:
    if len(inputs) == 0 or not isinstance(inputs[0], torch.Tensor):
        raise TypeError(
            f"Expected first layer input to be hidden_states tensor, got "
            f"{type(inputs[0]) if len(inputs) > 0 else 'empty inputs'}"
        )

    return inputs[0]


def replace_layer_output_with_hidden_input(
    *,
    inputs: tuple[Any, ...],
    output: Any,
) -> Any:
    """
    Replace a decoder layer's output with its input hidden state.

    This makes a layer act like identity while preserving extra tuple outputs,
    such as attention weights / cache entries if present.
    """
    hidden_in = get_first_hidden_from_inputs(inputs)

    if isinstance(output, torch.Tensor):
        return hidden_in

    if isinstance(output, tuple) and len(output) > 0:
        return (hidden_in, *output[1:])

    raise TypeError(f"Unexpected decoder layer output type while skipping: {type(output)}")


def make_identity_skip_hook() -> Any:
    def skip_hook(module: Any, inputs: tuple[Any, ...], output: Any) -> Any:
        return replace_layer_output_with_hidden_input(inputs=inputs, output=output)

    return skip_hook


@torch.no_grad()
def forward_with_layer_mask(
    *,
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    keep_layer_indices: set[int],
) -> torch.Tensor:
    """
    Run the model normally, but replace skipped decoder layers with identity.

    This keeps the original HF forward path intact, so the model still handles:
      - RoPE / position embeddings
      - causal masks
      - sliding-window masks
      - model-family-specific kwargs
    """
    layers = get_decoder_layers(model)

    with ExitStack() as stack:
        for layer_idx, layer in enumerate(layers):
            if layer_idx not in keep_layer_indices:
                handle = layer.register_forward_hook(make_identity_skip_hook())
                stack.callback(handle.remove)

        logits = forward_model_logits(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    return logits


def masked_hidden_mse_with_first_token_dropped(
    *,
    predicted: torch.Tensor,
    target: torch.Tensor,
    attention_mask: torch.Tensor | None,
) -> torch.Tensor:
    if predicted.shape != target.shape:
        raise ValueError(f"predicted.shape {predicted.shape} != target.shape {target.shape}")

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


# =============================================================================
# Next-token shifting helpers
# =============================================================================


def shift_next_token_logits(logits: torch.Tensor) -> torch.Tensor:
    return logits[:, :-1, :].contiguous()


def shift_next_token_labels(labels: torch.Tensor) -> torch.Tensor:
    return labels[:, 1:].contiguous()


def shift_next_token_mask(mask: torch.Tensor | None) -> torch.Tensor | None:
    if mask is None:
        return None

    return (
        mask[:, :-1].contiguous().bool()
        & mask[:, 1:].contiguous().bool()
    )


# =============================================================================
# Loss helpers
# =============================================================================


def masked_mean(
    values: torch.Tensor,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    if mask is None:
        return values.mean()

    if values.shape != mask.shape:
        raise ValueError(f"values.shape {values.shape} != mask.shape {mask.shape}")

    mask_f = mask.to(values.dtype)
    denom = mask_f.sum().clamp_min(1.0)

    return (values * mask_f).sum() / denom


def cross_entropy_next_token(
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    shift_logits = shift_next_token_logits(logits)
    shift_labels = shift_next_token_labels(labels)

    return F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)).float(),
        shift_labels.reshape(-1),
    )


def masked_cross_entropy_from_logits(
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

    return masked_mean(flat_loss, mask)


def kl_teacher_to_student_next_token(
    *,
    logits_teacher: torch.Tensor,
    logits_student: torch.Tensor,
    teacher_temperature: float = 1.0,
    student_temperature: float = 1.0,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    KL teacher -> student over next-token distributions.

    This computes:

        KL(P_teacher || P_student)

    where the teacher and student logits are shifted to next-token positions.
    """
    if teacher_temperature <= 0:
        raise ValueError(f"teacher_temperature must be > 0, got {teacher_temperature}")

    if student_temperature <= 0:
        raise ValueError(f"student_temperature must be > 0, got {student_temperature}")

    shift_teacher = shift_next_token_logits(logits_teacher).float()
    shift_student = shift_next_token_logits(logits_student).float()

    teacher_log_probs = F.log_softmax(shift_teacher / teacher_temperature, dim=-1)
    student_log_probs = F.log_softmax(shift_student / student_temperature, dim=-1)

    kl_per_token = F.kl_div(
        student_log_probs,
        teacher_log_probs,
        reduction="none",
        log_target=True,
    ).sum(dim=-1)

    shifted_mask = shift_next_token_mask(attention_mask)

    return masked_mean(kl_per_token, shifted_mask)


# =============================================================================
# Dataset/window/dataloader helper
# =============================================================================

def build_fixed_window_dataloader(
    *,
    dataset_spec: DatasetSpec,
    model_and_tokenizer: ModelAndTokenizer,
    context_len: int,
    max_examples: int,
    num_windows_to_use: int,
    batch_size: int,
    device: torch.device,
    shuffle: bool,
) -> DataLoader:
    dataset: Dataset = load_dataset(dataset_spec)
    dataset = maybe_format_dataset_to_text(dataset, dataset_spec)

    tokenized_examples = tokenize_dataset_to_examples(
        dataset,
        model_and_tokenizer.tokenizer,
        dataset_spec,
        max_examples=max_examples,
    )

    separator_token_ids: Any = model_and_tokenizer.tokenizer(
        "\n\n# New example\n\n",
        add_special_tokens=False,
    )["input_ids"]
    

    packed_windows = pack_tokenized_examples_to_windows(
        tokenized_examples,
        context_len=context_len,
        separator_token_ids=separator_token_ids,
    )

    if len(packed_windows) < num_windows_to_use:
        raise ValueError(
            f"Requested {num_windows_to_use} packed windows, "
            f"but only built {len(packed_windows)}."
        )

    window_dataset = PackedWindowDataset(
        packed_windows[:num_windows_to_use]
    )


    return DataLoader(
        window_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_windows,
        pin_memory=(device.type == "cuda"),
    )


def split_count_by_weights(
    *,
    total: int,
    weights: list[float],
) -> list[int]:
    if total <= 0:
        raise ValueError(f"total must be > 0, got {total}")

    if len(weights) == 0:
        raise ValueError("weights must not be empty")

    weight_sum = sum(weights)
    if weight_sum <= 0:
        raise ValueError("weights must sum to > 0")

    raw_counts = [total * w / weight_sum for w in weights]
    counts = [int(x) for x in raw_counts]

    # Give remaining items to datasets with largest fractional remainder.
    remaining = total - sum(counts)
    remainders = [
        (raw_counts[i] - counts[i], i)
        for i in range(len(weights))
    ]
    remainders.sort(reverse=True)

    for _, idx in remainders[:remaining]:
        counts[idx] += 1

    # Avoid accidentally giving a nonzero-weight dataset zero windows/examples.
    for i, w in enumerate(weights):
        if w > 0 and counts[i] == 0:
            counts[i] = 1

    return counts


def pack_tokenized_examples_to_windows(
    tokenized_examples: list[torch.Tensor],
    *,
    context_len: int,
    separator_token_ids: list[int] | None = None,
) -> list[torch.Tensor]:
    """
    Build fixed-length packed windows.

    Policy:
      - Every window starts at the beginning of a fresh example.
      - Additional examples are appended with a separator before them.
      - Once the buffer reaches context_len, cut the final added example.
      - Overflow from the final added example is discarded.
    """
    if context_len <= 0:
        raise ValueError(f"context_len must be > 0, got {context_len}")

    if separator_token_ids is None:
        separator_token_ids = []

    windows: list[torch.Tensor] = []
    idx = 0

    while idx < len(tokenized_examples):
        buffer: list[int] = []
        is_first_example_in_window = True

        while idx < len(tokenized_examples) and len(buffer) < context_len:
            example = tokenized_examples[idx]

            if example.ndim != 1:
                raise ValueError(
                    f"Expected each tokenized example to be 1D, "
                    f"got {tuple(example.shape)}"
                )

            ids = example.tolist()

            if is_first_example_in_window:
                ids_to_add = ids
            else:
                ids_to_add = separator_token_ids + ids

            needed = context_len - len(buffer)

            if len(ids_to_add) <= needed:
                buffer.extend(ids_to_add)
            else:
                buffer.extend(ids_to_add[:needed])

            idx += 1
            is_first_example_in_window = False

        if len(buffer) == context_len:
            windows.append(torch.tensor(buffer, dtype=torch.long))

    return windows

def build_mixed_fixed_window_dataloader(
    *,
    dataset_mix: list[tuple[DatasetSpec, float, int]],
    model_and_tokenizer: ModelAndTokenizer,
    context_len: int,
    num_windows_to_use: int,
    batch_size: int,
    device: torch.device,
    shuffle: bool = True,
) -> DataLoader[Any]:
    if len(dataset_mix) == 0:
        raise ValueError("dataset_mix must contain at least one dataset.")

    for dataset_spec, weight, max_examples in dataset_mix:
        if weight <= 0:
            raise ValueError(
                f"Dataset {dataset_spec.name} has invalid weight={weight}. "
                "Weights must be > 0."
            )

        if max_examples <= 0:
            raise ValueError(
                f"Dataset {dataset_spec.name} has invalid max_examples={max_examples}. "
                "Per-dataset max_examples must be > 0."
            )

    weights = [weight for _, weight, _ in dataset_mix]

    windows_per_dataset = split_count_by_weights(
        total=num_windows_to_use,
        weights=weights,
    )

    datasets: list[Any] = []
    collate_fn: Any | None = None

    for (dataset_spec, weight, source_max_examples), source_num_windows in zip(
        dataset_mix,
        windows_per_dataset,
    ):
        stage(
            f"building source dataloader: {dataset_spec.name} "
            f"weight={weight:.3f} "
            f"max_examples={source_max_examples} "
            f"num_windows={source_num_windows}"
        )

        source_loader = build_fixed_window_dataloader(
            dataset_spec=dataset_spec,
            model_and_tokenizer=model_and_tokenizer,
            context_len=context_len,
            max_examples=source_max_examples,
            num_windows_to_use=source_num_windows,
            batch_size=batch_size,
            device=device,
            shuffle=shuffle,
        )

        actual_windows = len(source_loader.dataset)


        stage(
            f"source built: {dataset_spec.name} "
            f"requested_windows={source_num_windows} "
            f"actual_windows={actual_windows} "
            f"actual_share_of_requested_total={actual_windows / num_windows_to_use:.6%}"
        )

        datasets.append(source_loader.dataset)

        if collate_fn is None:
            collate_fn = source_loader.collate_fn

    if collate_fn is None:
        raise RuntimeError("Failed to get collate_fn from source dataloaders.")

    mixed_dataset = ConcatDataset(datasets)

    stage(
        f"built mixed dataset with {len(mixed_dataset)} windows "
        f"from {len(datasets)} sources"
    )

    return DataLoader(
        mixed_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )


@dataclass(frozen=True, slots=True)
class LayerPattern:
    active_layer_indices: tuple[int, ...]
    skipped_layer_indices: tuple[int, ...]
    binary_mask: tuple[int, ...]
    visual_mask: str
    binary_string: str


def make_layer_pattern(
    *,
    num_layers: int,
    active_layer_indices: Iterable[int],
    active_char: str = "█",
    skipped_char: str = "·",
) -> LayerPattern:
    """
    Build a reusable layer pattern for logs/results.

    Convention:
      1 / active_char  = active layer
      0 / skipped_char = skipped layer
    """
    if num_layers <= 0:
        raise ValueError(f"num_layers must be > 0, got {num_layers}")

    active = tuple(sorted({int(i) for i in active_layer_indices}))

    invalid = [i for i in active if not (0 <= i < num_layers)]
    if invalid:
        raise ValueError(
            f"active_layer_indices contains invalid indices for "
            f"num_layers={num_layers}: {invalid}"
        )

    active_set = set(active)

    binary = tuple(1 if i in active_set else 0 for i in range(num_layers))
    skipped = tuple(i for i, is_active in enumerate(binary) if is_active == 0)

    visual = "".join(active_char if x == 1 else skipped_char for x in binary)
    binary_string = "[" + ",".join(str(x) for x in binary) + "]"

    return LayerPattern(
        active_layer_indices=active,
        skipped_layer_indices=skipped,
        binary_mask=binary,
        visual_mask=visual,
        binary_string=binary_string,
    )
