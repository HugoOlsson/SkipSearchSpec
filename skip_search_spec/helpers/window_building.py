from __future__ import annotations

from typing import Any
import torch
from datasets import Dataset
from transformers import PreTrainedTokenizerBase
from torch.utils.data import Dataset as TorchDataset

from skip_search_spec.protocols.windows import DatasetSpec, WindowSettings


def tokenize_dataset_to_examples(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    dataset_spec: DatasetSpec,
    max_examples: int | None = None,
    batch_size: int = 256,
    print_every: int = 10_000,
) -> list[torch.Tensor]:
    """
    Tokenize dataset examples into a list of 1D torch.LongTensor objects.

    Each kept example:
      - uses tokenizer(..., add_special_tokens=False)
      - appends exactly one EOS token
      - is stored as a 1D tensor of shape [T]

    Empty / whitespace-only texts are skipped.

    max_examples counts kept examples, matching the behavior of the original code.
    """
    eos_token_id = tokenizer.eos_token_id
    if not isinstance(eos_token_id, int):
        raise ValueError("tokenizer.eos_token_id must be an int.")

    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")
    if print_every <= 0:
        raise ValueError(f"print_every must be > 0, got {print_every}")

    text_field = dataset_spec.text_field
    num_rows = len(dataset)

    tokenized_examples: list[torch.Tensor] = []
    row_start = 0
    next_print = print_every

    while row_start < num_rows:
        if max_examples is not None and len(tokenized_examples) >= max_examples:
            break

        row_stop = min(row_start + batch_size, num_rows)
        batch = dataset[row_start:row_stop]

        raw_texts = batch[text_field]
        if not isinstance(raw_texts, list):
            raise TypeError(f"Expected dataset column '{text_field}' to be a list.")

        kept_texts: list[str] = []
        for text in raw_texts:
            if isinstance(text, str) and len(text.strip()) > 0:
                kept_texts.append(text)

        if kept_texts:
            encoded = tokenizer(
                kept_texts,
                add_special_tokens=False,
                padding=False,
                truncation=False,
            )

            input_ids_batch = encoded["input_ids"]
            if not isinstance(input_ids_batch, list):
                raise TypeError("Tokenizer output 'input_ids' must be a list.")

            remaining = None
            if max_examples is not None:
                remaining = max_examples - len(tokenized_examples)

            for ids in input_ids_batch[:remaining]:
                if not isinstance(ids, list):
                    raise TypeError("Each tokenized example must be a list of token ids.")

                ids_with_eos = ids + [eos_token_id]
                tokenized_examples.append(torch.tensor(ids_with_eos, dtype=torch.long))

            while len(tokenized_examples) >= next_print:
                total_str = str(max_examples) if max_examples is not None else "all available"
                print(f"Has tokenized {len(tokenized_examples)} examples of {total_str}")
                next_print += print_every

        row_start = row_stop

    total_str = str(max_examples) if max_examples is not None else "all available"
    print(f"Finished tokenizing {len(tokenized_examples)} examples of {total_str}")

    return tokenized_examples


def build_window_index(
    tokenized_examples: list[torch.Tensor],
    window_settings: WindowSettings,
) -> list[tuple[int, int]]:
    """
    Build a lightweight index of windows.

    Returns a list of (example_idx, start) pairs, where each window is:
        tokenized_examples[example_idx][start : start + C1]

    Semantics match the original code:
      - skip examples shorter than C1
      - use non-overlapping stride C1
      - do not cross example boundaries
    """
    c1 = window_settings.C1
    if c1 <= 0:
        raise ValueError(f"window_settings.C1 must be > 0, got {c1}")

    window_index: list[tuple[int, int]] = []

    for example_idx, example_token_ids in enumerate(tokenized_examples):
        if example_token_ids.ndim != 1:
            raise ValueError(
                f"Expected each tokenized example to be 1D, got shape {tuple(example_token_ids.shape)}"
            )

        n = int(example_token_ids.numel())
        if n < c1:
            continue

        for start in range(0, n - c1 + 1, c1):
            window_index.append((example_idx, start))

        if len(window_index) % 10000 == 0:
            print("Has built", len(window_index), "windows")

    return window_index


class WindowDataset(TorchDataset[torch.Tensor]):
    """
    Lazy window dataset backed by:
      - tokenized_examples: list of 1D LongTensor examples
      - window_index: list of (example_idx, start) pairs

    __getitem__ returns a 1D LongTensor of shape [C1].
    """

    def __init__(
        self,
        tokenized_examples: list[torch.Tensor],
        window_index: list[tuple[int, int]],
        window_settings: WindowSettings,
    ) -> None:
        self.tokenized_examples = tokenized_examples
        self.window_index = window_index
        self.c1 = window_settings.C1

        if self.c1 <= 0:
            raise ValueError(f"window_settings.C1 must be > 0, got {self.c1}")

    def __len__(self) -> int:
        return len(self.window_index)

    def __getitem__(self, idx: int) -> torch.Tensor:
        example_idx, start = self.window_index[idx]
        example_tokens = self.tokenized_examples[example_idx]
        window = example_tokens[start : start + self.c1]

        if window.numel() != self.c1:
            raise RuntimeError(
                f"Window at idx={idx} has length {window.numel()} but expected {self.c1}."
            )

        return window


def collate_windows(batch: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Stack fixed-length windows into:
      - input_ids: [B, C1]
      - attention_mask: [B, C1] of ones
    """
    if not batch:
        raise ValueError("collate_windows received an empty batch.")

    input_ids = torch.stack(batch, dim=0)
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask