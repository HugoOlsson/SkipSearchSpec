



from pathlib import Path
from typing import Any, cast

from datasets.arrow_dataset import Dataset
import torch
from torch import nn
from torch.utils.data import Dataset as TorchDataset, DataLoader
import torch.nn.functional as F

from skip_search_spec.helpers.storage import save_early_exit_checkpoint
from skip_search_spec.helpers.tooling import assert_same_tokenizer, get_preferred_device, get_preferred_float_dtype, load_dataset, load_model_and_tokenizer
from skip_search_spec.helpers.window_building import WindowDataset, build_all_training_windows, collate_windows, tokenize_dataset_to_examples
from skip_search_spec.protocols.windows import DatasetSpec, ModelAndTokenizer, WindowSettings




def train_drafter_for_verifier(draft_model_name: str, 
                               verifier_model_name: str, 
                               dataset_spec: DatasetSpec, 
                               window_size: int, 
                               max_examples: int = 10_000):

    device = get_preferred_device()
    compute_dtype = get_preferred_float_dtype(device)

    draft_model_and_tokenizer: ModelAndTokenizer = load_model_and_tokenizer(
        draft_model_name,
        model_kwargs={"torch_dtype": compute_dtype},
    )

    verifier_model_and_tokenizer: ModelAndTokenizer = load_model_and_tokenizer(
        verifier_model_name,
        model_kwargs={"torch_dtype": compute_dtype},
    )

    assert_same_tokenizer(draft_model_and_tokenizer.tokenizer, verifier_model_and_tokenizer.tokenizer)

    window_tokenization_tokenizer = draft_model_and_tokenizer.tokenizer
    dataset: Dataset = load_dataset(dataset_spec)
    context_parts = WindowSettings(C1=window_size)

    tokenized_examples = tokenize_dataset_to_examples(
        dataset,
        window_tokenization_tokenizer,
        dataset_spec,
        max_examples=max_examples,
    )

    training_windows = build_all_training_windows(
        tokenized_examples,
        context_parts,
        dataset_spec,
    )