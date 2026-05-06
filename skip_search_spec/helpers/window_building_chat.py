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


def _has_generation_spans(tokenizer: PreTrainedTokenizerBase) -> bool:
    template = getattr(tokenizer, "chat_template", None)
    return isinstance(template, str) and "{% generation" in template


def _normalize_messages(
    raw_messages: Any,
    system_prompt: str | None,
) -> list[dict[str, str]]:
    if not isinstance(raw_messages, list):
        return []

    messages: list[dict[str, str]] = []
    if system_prompt is not None and system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})

    for raw in raw_messages:
        if not isinstance(raw, dict):
            continue
        role = raw.get("role")
        content = raw.get("content")
        if (
            role in {"user", "assistant"}
            and isinstance(content, str)
            and content.strip()
        ):
            messages.append({"role": role, "content": content.strip()})

    return messages


def _assistant_mask_from_template(
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
    mask = encoded.get("assistant_masks", encoded.get("assistant_tokens_mask"))

    if not isinstance(input_ids, list) or not isinstance(mask, list):
        raise TypeError("Expected list input_ids and assistant mask from chat template.")
    if len(input_ids) != len(mask):
        raise ValueError(f"input_ids/mask length mismatch: {len(input_ids)} != {len(mask)}")

    return input_ids, [bool(x) for x in mask]


def _assistant_mask_from_offsets(
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

    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    input_ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"]

    if not isinstance(input_ids, list) or not isinstance(offsets, list):
        raise TypeError("Expected list input_ids and offset_mapping.")
    if len(input_ids) != len(offsets):
        raise ValueError(f"input_ids/offset length mismatch: {len(input_ids)} != {len(offsets)}")

    mask = [False] * len(input_ids)
    search_start = 0

    for message in messages:
        content = message["content"]
        start = rendered.find(content, search_start)
        if start < 0:
            raise ValueError("Could not locate message content in rendered chat.")
        end = start + len(content)
        search_start = end

        if message["role"] != "assistant":
            continue

        for i, offset in enumerate(offsets):
            token_start, token_end = offset
            if token_start < end and token_end > start:
                mask[i] = True

    return input_ids, mask


def _tokenize_chat(
    messages: list[dict[str, str]],
    tokenizer: PreTrainedTokenizerBase,
) -> tuple[list[int], list[bool]]:
    if _has_generation_spans(tokenizer):
        return _assistant_mask_from_template(messages, tokenizer)
    return _assistant_mask_from_offsets(messages, tokenizer)


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
        raise ValueError("batch_size must be > 0")
    if print_every <= 0:
        raise ValueError("print_every must be > 0")

    if not _has_generation_spans(tokenizer):
        print("Tokenizer has no generation spans; using offset assistant-mask fallback.")

    examples: list[TokenizedChatExample] = []
    field = dataset_spec.text_field
    next_print = print_every

    for row_start in range(0, len(dataset), batch_size):
        if max_examples is not None and len(examples) >= max_examples:
            break

        batch = dataset[row_start : min(row_start + batch_size, len(dataset))]
        raw_chats = batch[field]
        if not isinstance(raw_chats, list):
            raise TypeError(f"Expected dataset column {field!r} to be a list.")

        for raw_messages in raw_chats:
            if max_examples is not None and len(examples) >= max_examples:
                break

            messages = _normalize_messages(raw_messages, system_prompt)
            if not any(m["role"] == "assistant" for m in messages):
                continue

            input_ids, assistant_mask = _tokenize_chat(messages, tokenizer)
            if len(input_ids) != len(assistant_mask):
                raise ValueError("input_ids and assistant_mask length mismatch.")
            if not any(assistant_mask):
                raise ValueError("Assistant mask is empty for chat with assistant messages.")

            input_ids_tensor = torch.tensor(input_ids, dtype=torch.long)
            assistant_mask_tensor = torch.tensor(assistant_mask, dtype=torch.bool)

            loss_mask = torch.zeros_like(assistant_mask_tensor)
            loss_mask[:-1] = assistant_mask_tensor[1:]

            if loss_mask.any():
                examples.append(TokenizedChatExample(input_ids_tensor, loss_mask))

        while len(examples) >= next_print:
            total = max_examples if max_examples is not None else "all available"
            print(f"Has tokenized {len(examples)} examples of {total}")
            next_print += print_every

    total = max_examples if max_examples is not None else "all available"
    print(f"Finished tokenizing {len(examples)} examples of {total}")
    return examples


def _pad(x: torch.Tensor, length: int, value: int | bool) -> torch.Tensor:
    if x.numel() > length:
        raise ValueError(f"Cannot pad tensor of length {x.numel()} to {length}.")
    if x.numel() == length:
        return x
    return torch.cat(
        [x, torch.full((length - x.numel(),), value, dtype=x.dtype)],
        dim=0,
    )


def _make_window(
    example: TokenizedChatExample,
    start: int,
    c1: int,
    pad_token_id: int,
) -> ChatWindow:
    input_ids = example.input_ids[start : start + c1]
    loss_mask = example.loss_mask[start : start + c1].clone()
    real_len = int(input_ids.numel())

    if real_len == 0:
        raise ValueError("Cannot create empty window.")

    loss_mask[real_len - 1] = False
    attention_mask = torch.ones(real_len, dtype=torch.long)

    return ChatWindow(
        input_ids=_pad(input_ids, c1, pad_token_id),
        attention_mask=_pad(attention_mask, c1, 0),
        loss_mask=_pad(loss_mask, c1, False),
    )


def build_chat_windows(
    tokenized_examples: list[TokenizedChatExample],
    window_settings: WindowSettings,
    pad_token_id: int,
    one_window_per_example: bool = True,
) -> list[ChatWindow]:
    c1 = window_settings.C1
    if c1 <= 0:
        raise ValueError(f"window_settings.C1 must be > 0, got {c1}")

    windows: list[ChatWindow] = []

    for example in tokenized_examples:
        n = int(example.input_ids.numel())
        if n == 0:
            continue

        if one_window_per_example:
            # Chat mode: only keep prefix windows.
            # This guarantees every training window starts at the real chat start:
            # BOS/system/user/... not a mid-conversation fragment.
            window = _make_window(example, start=0, c1=c1, pad_token_id=pad_token_id)

            if window.loss_mask.any():
                windows.append(window)

            continue

        # Non-chat-continuation mode: allow later chunks.
        # These are language-model chunks, not clean chat-prefix windows.
        for start in range(0, n, c1):
            window = _make_window(example, start=start, c1=c1, pad_token_id=pad_token_id)

            if window.loss_mask.any():
                windows.append(window)

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

    return (
        torch.stack([x[0] for x in batch]),
        torch.stack([x[1] for x in batch]),
        torch.stack([x[2] for x in batch]),
    )