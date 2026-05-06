from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from datasets import Dataset
from torch.utils.data import Dataset as TorchDataset
from transformers import PreTrainedTokenizerBase

from skip_search_spec.protocols.windows import DatasetSpec, WindowSettings


DEFAULT_CHAT_SYSTEM_PROMPT = "You are a helpful assistant."


@dataclass(frozen=True, slots=True)
class TokenizedChatExample:
    input_ids: torch.Tensor
    loss_mask: torch.Tensor


@dataclass(frozen=True, slots=True)
class ChatWindow:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    loss_mask: torch.Tensor


def _normalize_chat_messages(
    raw_messages: Any,
    *,
    system_prompt: str | None = DEFAULT_CHAT_SYSTEM_PROMPT,
) -> list[dict[str, str]]:
    if not isinstance(raw_messages, list):
        return []

    messages: list[dict[str, str]] = []

    if system_prompt is not None and system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})

    for raw_message in raw_messages:
        if not isinstance(raw_message, dict):
            continue

        role = raw_message.get("role")
        content = raw_message.get("content")

        if (
            isinstance(role, str)
            and role in {"user", "assistant"}
            and isinstance(content, str)
            and content.strip()
        ):
            messages.append({"role": role, "content": content.strip()})

    return messages


def _chat_template_has_generation_spans(tokenizer: PreTrainedTokenizerBase) -> bool:
    chat_template = getattr(tokenizer, "chat_template", None)
    return isinstance(chat_template, str) and "{% generation" in chat_template


def _get_assistant_mask(encoded: Any) -> Any:
    if "assistant_masks" in encoded:
        return encoded["assistant_masks"]
    if "assistant_tokens_mask" in encoded:
        return encoded["assistant_tokens_mask"]

    raise KeyError(
        "Chat template did not return an assistant mask. "
        "The tokenizer chat_template probably needs {% generation %} markers."
    )


def _tokenize_chat_with_generation_spans(
    *,
    messages: list[dict[str, str]],
    tokenizer: PreTrainedTokenizerBase,
) -> tuple[list[int], list[bool]]:
    encoded = tokenizer.apply_chat_template(  # type: ignore[attr-defined]
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_assistant_tokens_mask=True,
        enable_thinking=False,
    )

    input_ids = encoded["input_ids"]
    assistant_mask = _get_assistant_mask(encoded)

    if not isinstance(input_ids, list):
        raise TypeError("Chat template output 'input_ids' must be a list.")
    if not isinstance(assistant_mask, list):
        raise TypeError("Chat template assistant mask must be a list.")
    if len(input_ids) != len(assistant_mask):
        raise ValueError(
            "Chat template produced input_ids and assistant mask with different "
            f"lengths: {len(input_ids)} != {len(assistant_mask)}"
        )

    return input_ids, [bool(x) for x in assistant_mask]


def _tokenize_chat_with_offset_fallback(
    *,
    messages: list[dict[str, str]],
    tokenizer: PreTrainedTokenizerBase,
) -> tuple[list[int], list[bool]]:
    rendered = tokenizer.apply_chat_template(  # type: ignore[attr-defined]
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )

    if not isinstance(rendered, str):
        raise TypeError("Rendered chat template must be a string.")

    tokenized = tokenizer(
        rendered,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )

    input_ids = tokenized["input_ids"]
    offsets = tokenized["offset_mapping"]

    if not isinstance(input_ids, list):
        raise TypeError("Tokenizer output 'input_ids' must be a list.")
    if not isinstance(offsets, list):
        raise TypeError("Tokenizer output 'offset_mapping' must be a list.")
    if len(input_ids) != len(offsets):
        raise ValueError(
            "Tokenizer produced input_ids and offsets with different lengths: "
            f"{len(input_ids)} != {len(offsets)}"
        )

    assistant_mask = [False] * len(input_ids)
    search_start = 0

    for message in messages:
        content = message["content"].strip()
        if not content:
            continue

        content_start = rendered.find(content, search_start)
        if content_start < 0:
            raise ValueError(
                "Could not locate trimmed message content in rendered chat template. "
                "The offset fallback assumes message content is rendered verbatim."
            )

        content_end = content_start + len(content)
        search_start = content_end

        if message["role"] != "assistant":
            continue

        for token_idx, offset in enumerate(offsets):
            if not (
                isinstance(offset, (list, tuple))
                and len(offset) == 2
                and isinstance(offset[0], int)
                and isinstance(offset[1], int)
            ):
                raise TypeError("Tokenizer offsets must be pairs of ints.")

            token_start, token_end = offset
            if token_start < content_end and token_end > content_start:
                assistant_mask[token_idx] = True

    return input_ids, assistant_mask


def _tokenize_chat_with_assistant_mask(
    *,
    messages: list[dict[str, str]],
    tokenizer: PreTrainedTokenizerBase,
) -> tuple[list[int], list[bool]]:
    if _chat_template_has_generation_spans(tokenizer):
        return _tokenize_chat_with_generation_spans(
            messages=messages,
            tokenizer=tokenizer,
        )

    return _tokenize_chat_with_offset_fallback(
        messages=messages,
        tokenizer=tokenizer,
    )


def tokenize_dataset_to_examples(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    dataset_spec: DatasetSpec,
    max_examples: int | None = None,
    batch_size: int = 256,
    print_every: int = 10_000,
    system_prompt: str | None = DEFAULT_CHAT_SYSTEM_PROMPT,
) -> list[TokenizedChatExample]:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")
    if print_every <= 0:
        raise ValueError(f"print_every must be > 0, got {print_every}")

    if not _chat_template_has_generation_spans(tokenizer):
        print(
            "Tokenizer chat_template has no {% generation %} spans; "
            "using offset-based assistant-mask fallback."
        )

    messages_field = dataset_spec.text_field
    tokenized_examples: list[TokenizedChatExample] = []
    next_print = print_every

    for row_start in range(0, len(dataset), batch_size):
        if max_examples is not None and len(tokenized_examples) >= max_examples:
            break

        row_stop = min(row_start + batch_size, len(dataset))
        batch = dataset[row_start:row_stop]
        raw_chats = batch[messages_field]

        if not isinstance(raw_chats, list):
            raise TypeError(f"Expected dataset column '{messages_field}' to be a list.")

        for raw_messages in raw_chats:
            if max_examples is not None and len(tokenized_examples) >= max_examples:
                break

            messages = _normalize_chat_messages(
                raw_messages,
                system_prompt=system_prompt,
            )

            if not any(message["role"] == "assistant" for message in messages):
                continue

            input_ids, assistant_mask = _tokenize_chat_with_assistant_mask(
                messages=messages,
                tokenizer=tokenizer,
            )

            if len(input_ids) != len(assistant_mask):
                raise ValueError(
                    f"input_ids and assistant_mask differ in length: "
                    f"{len(input_ids)} != {len(assistant_mask)}"
                )

            if not any(assistant_mask):
                raise ValueError(
                    "Assistant mask is all false for a chat containing assistant messages."
                )

            input_ids_tensor = torch.tensor(input_ids, dtype=torch.long)
            assistant_mask_tensor = torch.tensor(assistant_mask, dtype=torch.bool)

            # loss_mask[i] means: train the prediction made at position i.
            # The target for that prediction is token i + 1.
            loss_mask = torch.zeros_like(assistant_mask_tensor)
            loss_mask[:-1] = assistant_mask_tensor[1:]

            if not loss_mask.any():
                continue

            tokenized_examples.append(
                TokenizedChatExample(
                    input_ids=input_ids_tensor,
                    loss_mask=loss_mask,
                )
            )

        while len(tokenized_examples) >= next_print:
            total = str(max_examples) if max_examples is not None else "all available"
            print(f"Has tokenized {len(tokenized_examples)} examples of {total}")
            next_print += print_every

    total = str(max_examples) if max_examples is not None else "all available"
    print(f"Finished tokenizing {len(tokenized_examples)} examples of {total}")

    return tokenized_examples


def _pad_1d(
    tensor: torch.Tensor,
    *,
    length: int,
    value: int | bool,
) -> torch.Tensor:
    if tensor.ndim != 1:
        raise ValueError(f"Expected a 1D tensor, got shape {tuple(tensor.shape)}")

    n = int(tensor.numel())
    if n > length:
        raise ValueError(f"Tensor length {n} exceeds target length {length}")
    if n == length:
        return tensor

    pad = torch.full(
        (length - n,),
        value,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    return torch.cat([tensor, pad], dim=0)


def _has_trainable_loss(
    *,
    loss_mask: torch.Tensor,
    start: int,
    end: int,
    train_start_offset: int,
) -> bool:
    """
    Check whether the candidate window has at least one trainable token in the
    region that the trainer actually uses.

    The final position is excluded because it has no next-token target inside
    the returned window.
    """
    train_start = min(start + train_start_offset, end)
    train_end = max(train_start, end - 1)
    return bool(loss_mask[train_start:train_end].any())


def _make_window(
    *,
    example: TokenizedChatExample,
    start: int,
    c1: int,
    pad_token_id: int,
) -> ChatWindow:
    input_ids = example.input_ids[start : start + c1]
    loss_mask = example.loss_mask[start : start + c1].clone()

    real_len = int(input_ids.numel())
    if real_len == 0:
        raise ValueError("Cannot create an empty chat window.")

    # The final real token has no target inside this fixed window.
    loss_mask[real_len - 1] = False

    attention_mask = torch.ones(real_len, dtype=torch.long)

    return ChatWindow(
        input_ids=_pad_1d(input_ids, length=c1, value=pad_token_id),
        attention_mask=_pad_1d(attention_mask, length=c1, value=0),
        loss_mask=_pad_1d(loss_mask, length=c1, value=False),
    )


def build_chat_windows(
    *,
    tokenized_examples: list[TokenizedChatExample],
    window_settings: WindowSettings,
    pad_token_id: int,
    one_window_per_example: bool = True,
    train_start_offset: int = 0,
) -> list[ChatWindow]:
    """
    Build fixed-length chat windows.

    Policy:
      - short examples are kept and padded if they contain trainable assistant loss
      - long examples contribute the first non-overlapping window with trainable loss
        when one_window_per_example=True
      - long examples contribute all non-overlapping windows with trainable loss
        when one_window_per_example=False
    """
    c1 = window_settings.C1

    if c1 <= 0:
        raise ValueError(f"window_settings.C1 must be > 0, got {c1}")
    if train_start_offset < 0 or train_start_offset >= c1:
        raise ValueError(
            f"train_start_offset must be in [0, {c1}), got {train_start_offset}"
        )

    windows: list[ChatWindow] = []

    for example_idx, example in enumerate(tokenized_examples):
        if example.input_ids.ndim != 1:
            raise ValueError(
                f"Example {example_idx}: expected input_ids to be 1D, "
                f"got shape {tuple(example.input_ids.shape)}"
            )
        if example.loss_mask.shape != example.input_ids.shape:
            raise ValueError(
                f"Example {example_idx}: loss_mask shape {tuple(example.loss_mask.shape)} "
                f"must match input_ids shape {tuple(example.input_ids.shape)}"
            )

        n = int(example.input_ids.numel())
        if n == 0:
            continue

        if n <= c1:
            if _has_trainable_loss(
                loss_mask=example.loss_mask,
                start=0,
                end=n,
                train_start_offset=train_start_offset,
            ):
                windows.append(
                    _make_window(
                        example=example,
                        start=0,
                        c1=c1,
                        pad_token_id=pad_token_id,
                    )
                )
            continue

        for start in range(0, n - c1 + 1, c1):
            end = start + c1

            if not _has_trainable_loss(
                loss_mask=example.loss_mask,
                start=start,
                end=end,
                train_start_offset=train_start_offset,
            ):
                continue

            windows.append(
                _make_window(
                    example=example,
                    start=start,
                    c1=c1,
                    pad_token_id=pad_token_id,
                )
            )

            if one_window_per_example:
                break

    return windows


class ChatWindowDataset(TorchDataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(self, windows: list[ChatWindow]) -> None:
        self.windows = windows

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        window = self.windows[idx]
        return window.input_ids, window.attention_mask, window.loss_mask


def collate_windows(
    batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not batch:
        raise ValueError("collate_windows received an empty batch.")

    input_ids = torch.stack([item[0] for item in batch], dim=0)
    attention_mask = torch.stack([item[1] for item in batch], dim=0)
    loss_mask = torch.stack([item[2] for item in batch], dim=0)

    return input_ids, attention_mask, loss_mask