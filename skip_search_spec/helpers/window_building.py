from datasets import Dataset
import torch
from transformers import PreTrainedTokenizerBase
from typing import Any, Mapping, cast
from skip_search_spec.protocols.windows import DatasetSpec, TokenizedWindow, WindowSettings
from torch.utils.data import Dataset as TorchDataset

def tokenize_dataset_to_examples(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    dataset_spec: DatasetSpec,
    max_examples: int | None = None,
) -> list[list[int]]:
    eos_token_id = tokenizer.eos_token_id
    if not isinstance(eos_token_id, int):
        raise ValueError("tokenizer.eos_token_id must be an int.")

    tokenized_examples: list[list[int]] = []
    num_tokenized_examples = 0

    for raw_example in dataset:
        if max_examples is not None and num_tokenized_examples >= max_examples:
            break

        example = cast(Mapping[str, object], raw_example)
        text = example.get(dataset_spec.text_field)

        if isinstance(text, str) and len(text.strip()) > 0:
            token_ids = cast(list[int], tokenizer.encode(text, add_special_tokens=False))
            token_ids.append(eos_token_id)

            tokenized_examples.append(token_ids)
            num_tokenized_examples += 1

            if num_tokenized_examples%1000 == 0:
                print("Has tokenized", num_tokenized_examples, "examples of ", max_examples)

    return tokenized_examples


def build_all_training_windows(
    tokenized_examples: list[list[int]],
    window_settings: WindowSettings,
    dataset_spec: DatasetSpec,
) -> list[TokenizedWindow]:

    C1 = window_settings.C1

    windows: list[TokenizedWindow] = []

    for example_token_ids in tokenized_examples:
        if len(example_token_ids) < C1:
            continue

        for start in range(0, len(example_token_ids) - C1 + 1, C1):
            window_token_ids = example_token_ids[start : start + C1]
            windows.append(
                TokenizedWindow(
                    token_ids=window_token_ids,
                    dataset_spec=dataset_spec,
                )
            )

            windows_list_len = len(windows)
            if windows_list_len%1000 == 0:
                print("Has built", windows_list_len, "windows")

    return windows


class WindowDataset(TorchDataset):
    def __init__(self, training_windows: list[Any]):
        self.training_windows = training_windows

    def __len__(self) -> int:
        return len(self.training_windows)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.tensor(self.training_windows[idx].token_ids, dtype=torch.long)


def collate_windows(batch: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids = torch.stack(batch, dim=0)
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask

