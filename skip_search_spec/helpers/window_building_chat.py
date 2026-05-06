from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import torch
from datasets import Dataset
from transformers import PreTrainedTokenizerBase
from torch.utils.data import Dataset as TorchDataset

from skip_search_spec.protocols.windows import DatasetSpec, WindowSettings


DEFAULT_CHAT_SYSTEM_PROMPT = "You are a helpful assistant."


@dataclass(frozen=True, slots=True)
class TokenizedChatExample:
    input_ids: torch.Tensor
    loss_mask: torch.Tensor


def _normalize_chat_messages(
    raw_messages: Any,
    *,
    system_prompt: str | None = DEFAULT_CHAT_SYSTEM_PROMPT,
) -> list[dict[str, str]]:
    if not isinstance(raw_messages, list):
        return []

    messages: list[dict[str, str]] = []
    if system_prompt is not None and len(system_prompt.strip()) > 0:
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
            and len(content.strip()) > 0
        ):
            messages.append({"role": role, "content": content})

    return messages


def _get_assistant_mask(encoded: Any) -> Any:
    if "assistant_masks" in encoded:
        return encoded["assistant_masks"]
    if "assistant_tokens_mask" in encoded:
        return encoded["assistant_tokens_mask"]
    raise KeyError(
        "Chat template did not return an assistant mask. "
        "The tokenizer chat_template probably needs {% generation %} markers."
    )


def _chat_template_has_generation_spans(tokenizer: PreTrainedTokenizerBase) -> bool:
    chat_template = getattr(tokenizer, "chat_template", None)
    return isinstance(chat_template, str) and "{% generation" in chat_template


def _tokenize_chat_with_assistant_mask(
    *,
    messages: list[dict[str, str]],
    tokenizer: PreTrainedTokenizerBase,
) -> tuple[list[int], list[bool]]:
    has_generation_spans = _chat_template_has_generation_spans(tokenizer)
    if has_generation_spans:
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
                "Chat template produced input_ids and assistant mask with "
                f"different lengths: {len(input_ids)} != {len(assistant_mask)}"
            )
        if not any(bool(x) for x in assistant_mask):
            raise ValueError(
                "Chat template contains {% generation %} spans but returned an "
                "all-zero assistant mask for a chat with assistant messages."
            )
        return input_ids, [bool(x) for x in assistant_mask]

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

    assistant_mask = [False for _ in input_ids]
    search_start = 0
    for message in messages:
        # Offset fallback assumes the chat template renders trimmed message
        # content verbatim. This matches Llama-3.2-Instruct's template.
        content = message["content"].strip()
        if not content:
            continue

        content_start = rendered.find(content, search_start)
        if content_start < 0:
            raise ValueError(
                "Could not locate trimmed message content in rendered chat "
                "template. The offset fallback assumes message content is "
                "rendered verbatim."
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


def tokenize_dataset_to_examples(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    dataset_spec: DatasetSpec,
    max_examples: int | None = None,
    batch_size: int = 256,
    print_every: int = 10_000,
    system_prompt: str | None = DEFAULT_CHAT_SYSTEM_PROMPT,
) -> list[TokenizedChatExample]:
    """
    Tokenize chat dataset examples into token ids and assistant-only loss masks.

    Each kept example:
      - reads dataset_spec.text_field as a list of {role, content} messages
      - ignores dataset system messages and prepends system_prompt instead
      - uses tokenizer.apply_chat_template(..., enable_thinking=False)
      - is stored as input_ids and a shifted loss_mask, both shape [T]

    Empty chats and chats without assistant tokens are skipped.

    max_examples counts kept examples, matching the behavior of the original code.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")
    if print_every <= 0:
        raise ValueError(f"print_every must be > 0, got {print_every}")
    
    using_offset_fallback = not _chat_template_has_generation_spans(tokenizer)
    if using_offset_fallback:
        print(
            "Tokenizer chat_template has no {% generation %} spans; "
            "using offset-based assistant-mask fallback."
        )

    messages_field = dataset_spec.text_field
    num_rows = len(dataset)

    tokenized_examples: list[TokenizedChatExample] = []
    row_start = 0
    next_print = print_every

    while row_start < num_rows:
        if max_examples is not None and len(tokenized_examples) >= max_examples:
            break

        row_stop = min(row_start + batch_size, num_rows)
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
            if not any(bool(x) for x in assistant_mask):
                raise ValueError(
                    "Chat template returned an all-zero assistant mask for a chat "
                    "that contains assistant messages. The tokenizer chat_template "
                    "probably needs {% generation %} markers."
                )

            input_ids_tensor = torch.tensor(input_ids, dtype=torch.long)
            assistant_mask_tensor = torch.tensor(assistant_mask, dtype=torch.bool)
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
            total_str = str(max_examples) if max_examples is not None else "all available"
            print(f"Has tokenized {len(tokenized_examples)} examples of {total_str}")
            next_print += print_every

        row_start = row_stop

    total_str = str(max_examples) if max_examples is not None else "all available"
    print(f"Finished tokenizing {len(tokenized_examples)} examples of {total_str}")

    return tokenized_examples


def build_window_index(
    tokenized_examples: list[TokenizedChatExample],
    window_settings: WindowSettings,
    one_window_per_example: bool = True,
) -> list[tuple[int, int]]:
    c1 = window_settings.C1
    if c1 <= 0:
        raise ValueError(f"window_settings.C1 must be > 0, got {c1}")

    window_index: list[tuple[int, int]] = []

    for example_idx, example in enumerate(tokenized_examples):
        input_ids = example.input_ids
        loss_mask = example.loss_mask

        if input_ids.ndim != 1:
            raise ValueError(
                f"Expected each tokenized example to be 1D, got shape {tuple(input_ids.shape)}"
            )
        if loss_mask.shape != input_ids.shape:
            raise ValueError(
                f"Expected loss_mask shape {tuple(loss_mask.shape)} to match "
                f"input_ids shape {tuple(input_ids.shape)}"
            )

        n = int(input_ids.numel())

        # Short examples are valid. They will be padded in WindowDataset.
        if n <= c1:
            if loss_mask[:-1].any() if n > 1 else loss_mask.any():
                window_index.append((example_idx, 0))
            continue

        if one_window_per_example:
            # Keep the first C1-token window that actually contains trainable loss.
            for start in range(0, n - c1 + 1, c1):
                if loss_mask[start : start + c1 - 1].any():
                    window_index.append((example_idx, start))
                    break
            continue

        # Keep all non-overlapping windows that contain trainable loss.
        for start in range(0, n - c1 + 1, c1):
            if loss_mask[start : start + c1 - 1].any():
                window_index.append((example_idx, start))

        if len(window_index) > 0 and len(window_index) % 10000 == 0:
            print("Has built", len(window_index), "windows")

    return window_index


class WindowDataset(TorchDataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """
    Lazy window dataset backed by:
      - tokenized_examples: list of token ids plus loss masks
      - window_index: list of (example_idx, start) pairs

    __getitem__ returns:
      - input_ids: [C1]
      - attention_mask: [C1]
      - loss_mask: [C1]
    """

    def __init__(
        self,
        tokenized_examples: list[TokenizedChatExample],
        window_index: list[tuple[int, int]],
        window_settings: WindowSettings,
        pad_token_id: int,
    ) -> None:
        self.tokenized_examples = tokenized_examples
        self.window_index = window_index
        self.c1 = window_settings.C1
        self.pad_token_id = pad_token_id

        if self.c1 <= 0:
            raise ValueError(f"window_settings.C1 must be > 0, got {self.c1}")

    def __len__(self) -> int:
        return len(self.window_index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        example_idx, start = self.window_index[idx]
        example = self.tokenized_examples[example_idx]

        input_ids_window = example.input_ids[start : start + self.c1]
        loss_mask_window = example.loss_mask[start : start + self.c1].clone()

        n = int(input_ids_window.numel())

        if n > self.c1:
            raise RuntimeError(
                f"Window at idx={idx} has length {n} but expected at most {self.c1}."
            )

        # Last real position cannot predict a target inside this returned window.
        if n > 0:
            loss_mask_window[-1] = False

        attention_mask_window = torch.ones_like(input_ids_window, dtype=torch.long)

        if n < self.c1:
            pad_len = self.c1 - n

            input_ids_window = torch.cat(
                [
                    input_ids_window,
                    torch.full(
                        (pad_len,),
                        self.pad_token_id,
                        dtype=input_ids_window.dtype,
                    ),
                ],
                dim=0,
            )

            attention_mask_window = torch.cat(
                [
                    attention_mask_window,
                    torch.zeros(
                        (pad_len,),
                        dtype=torch.long,
                    ),
                ],
                dim=0,
            )

            loss_mask_window = torch.cat(
                [
                    loss_mask_window,
                    torch.zeros(
                        (pad_len,),
                        dtype=torch.bool,
                    ),
                ],
                dim=0,
            )

        if input_ids_window.numel() != self.c1:
            raise RuntimeError(
                f"Window at idx={idx} has length {input_ids_window.numel()} but expected {self.c1}."
            )
        if attention_mask_window.numel() != self.c1:
            raise RuntimeError(
                f"Attention mask at idx={idx} has length {attention_mask_window.numel()} but expected {self.c1}."
            )
        if loss_mask_window.numel() != self.c1:
            raise RuntimeError(
                f"Loss mask at idx={idx} has length {loss_mask_window.numel()} but expected {self.c1}."
            )

        return input_ids_window, attention_mask_window, loss_mask_window


def collate_windows(
    batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Stack fixed-length windows into:
      - input_ids: [B, C1]
      - attention_mask: [B, C1]
      - loss_mask: [B, C1], true where next-token loss should be trained
    """
    if not batch:
        raise ValueError("collate_windows received an empty batch.")

    input_ids = torch.stack([item[0] for item in batch], dim=0)
    attention_mask = torch.stack([item[1] for item in batch], dim=0)
    loss_mask = torch.stack([item[2] for item in batch], dim=0)

    return input_ids, attention_mask, loss_mask



def debug_raw_dataset_message_lengths(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    dataset_spec: DatasetSpec,
    *,
    num_examples: int = 10_000,
) -> None:
    """
    Debug helper that avoids the chat-template/tokenization pipeline as much as possible.

    It directly inspects the first num_examples raw dataset rows:
      - reads dataset_spec.text_field
      - checks raw message roles/content
      - joins raw content strings without chat template
      - tokenizes that plain text
      - prints length statistics

    This does NOT call:
      - _normalize_chat_messages()
      - tokenizer.apply_chat_template()
      - _tokenize_chat_with_assistant_mask()
      - tokenize_dataset_to_examples()
    """
    messages_field = dataset_spec.text_field
    limit = min(num_examples, len(dataset))

    raw_char_lengths: list[int] = []
    raw_token_lengths: list[int] = []
    user_char_lengths: list[int] = []
    assistant_char_lengths: list[int] = []
    user_token_lengths: list[int] = []
    assistant_token_lengths: list[int] = []

    skipped_non_list = 0
    skipped_no_assistant = 0
    skipped_no_valid_content = 0

    debug_printed = 0

    for row_idx in range(limit):
        row = dataset[row_idx]
        raw_messages = row[messages_field]

        if not isinstance(raw_messages, list):
            skipped_non_list += 1
            continue

        user_parts: list[str] = []
        assistant_parts: list[str] = []
        all_parts: list[str] = []

        for raw_message in raw_messages:
            if not isinstance(raw_message, dict):
                continue

            role = raw_message.get("role")
            content = raw_message.get("content")

            if not isinstance(role, str) or not isinstance(content, str):
                continue

            content = content.strip()
            if not content:
                continue

            if role == "user":
                user_parts.append(content)
                all_parts.append(content)
            elif role == "assistant":
                assistant_parts.append(content)
                all_parts.append(content)
            elif role == "system":
                all_parts.append(content)

        if not all_parts:
            skipped_no_valid_content += 1
            continue

        if not assistant_parts:
            skipped_no_assistant += 1
            continue

        raw_text = "\n\n".join(all_parts)
        user_text = "\n\n".join(user_parts)
        assistant_text = "\n\n".join(assistant_parts)

        raw_ids = tokenizer(raw_text, add_special_tokens=False)["input_ids"]
        user_ids = tokenizer(user_text, add_special_tokens=False)["input_ids"]
        assistant_ids = tokenizer(assistant_text, add_special_tokens=False)["input_ids"]

        raw_char_lengths.append(len(raw_text))
        raw_token_lengths.append(len(raw_ids))
        user_char_lengths.append(len(user_text))
        assistant_char_lengths.append(len(assistant_text))
        user_token_lengths.append(len(user_ids))
        assistant_token_lengths.append(len(assistant_ids))

        if debug_printed < 5 and len(raw_ids) < 256:
            debug_printed += 1
            print("\nDEBUG RAW EXAMPLE UNDER 256 TOKENS")
            print("row_idx:", row_idx)
            print("raw_token_length:", len(raw_ids))
            print("user_token_length:", len(user_ids))
            print("assistant_token_length:", len(assistant_ids))
            print("raw_char_length:", len(raw_text))
            print("num_messages:", len(raw_messages))
            print("roles:", [
                m.get("role") if isinstance(m, dict) else None
                for m in raw_messages
            ])
            print("RAW TEXT:")
            print(raw_text[:3000])
            print("END DEBUG RAW EXAMPLE UNDER 256 TOKENS\n")

    if not raw_token_lengths:
        print("DEBUG RAW LENGTHS: no valid examples found.")
        return

    def percentile(xs: list[int], q: float) -> int:
        xs_sorted = sorted(xs)
        index = int((len(xs_sorted) - 1) * q)
        return xs_sorted[index]

    def print_stats(name: str, xs: list[int]) -> None:
        xs_sorted = sorted(xs)
        print(
            f"{name} min/p10/p25/median/p75/p90/max:",
            xs_sorted[0],
            percentile(xs_sorted, 0.10),
            percentile(xs_sorted, 0.25),
            percentile(xs_sorted, 0.50),
            percentile(xs_sorted, 0.75),
            percentile(xs_sorted, 0.90),
            xs_sorted[-1],
        )

    print("\nDEBUG RAW DATASET MESSAGE LENGTHS")
    print("requested_examples:", num_examples)
    print("inspected_rows:", limit)
    print("valid_examples_with_assistant:", len(raw_token_lengths))
    print("skipped_non_list:", skipped_non_list)
    print("skipped_no_valid_content:", skipped_no_valid_content)
    print("skipped_no_assistant:", skipped_no_assistant)

    print_stats("raw_token_length", raw_token_lengths)
    print_stats("user_token_length", user_token_lengths)
    print_stats("assistant_token_length", assistant_token_lengths)
    print_stats("raw_char_length", raw_char_lengths)
    print_stats("user_char_length", user_char_lengths)
    print_stats("assistant_char_length", assistant_char_lengths)

    print(
        "fraction_raw_token_length_lt_256:",
        sum(x < 256 for x in raw_token_lengths) / len(raw_token_lengths),
    )
    print(
        "fraction_raw_token_length_ge_256:",
        sum(x >= 256 for x in raw_token_lengths) / len(raw_token_lengths),
    )
    print("END DEBUG RAW DATASET MESSAGE LENGTHS\n")