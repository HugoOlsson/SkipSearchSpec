from __future__ import annotations

from typing import Any

from datasets import load_dataset as hf_load_dataset
from skip_search_spec.protocols.windows import DatasetSpec


# -----------------------------------------------------------------------------
# Dataset specs
# -----------------------------------------------------------------------------

DATASET_SPEC_FINEWEB_EDU_1B = DatasetSpec(
    name="FineWeb-Edu-1B",
    huggingface_path="codelion/fineweb-edu-1B",
    config_name="default",
    split="train",
    text_field="text",
)

DATASET_SPEC_COSMOPEDIA_100K = DatasetSpec(
    name="Cosmopedia-100k",
    huggingface_path="HuggingFaceTB/cosmopedia-100k",
    config_name=None,
    split="train",
    text_field="text",
)

DATASET_SPEC_FINEMATH_4PLUS = DatasetSpec(
    name="FineMath-4plus",
    huggingface_path="HuggingFaceTB/finemath",
    config_name="finemath-4plus",
    split="train",
    text_field="text",
)

DATASET_SPEC_PYTHON_CODES_25K = DatasetSpec(
    name="Python-Codes-25k",
    huggingface_path="flytech/python-codes-25k",
    config_name=None,
    split="train",
    text_field="text",
)

# IMPORTANT:
# Do not load intfloat/multilingual_cc_news directly with:
#   huggingface_path="intfloat/multilingual_cc_news"
#
# That repo uses a dataset script, and newer versions of `datasets` error with:
#   RuntimeError: Dataset scripts are no longer supported
#
# Instead, use the generic parquet loader and point directly to selected Parquet
# files from the repo.
DATASET_SPEC_MULTILINGUAL_CC_NEWS = DatasetSpec(
    name="Multilingual-CC-News-en-es-fr-de-sv",
    huggingface_path="parquet",
    config_name=None,
    split="train",
    # The dataset has fields like: title, maintext, url, date_publish.
    # Use maintext for plain LM training.
    text_field="maintext",
)


# -----------------------------------------------------------------------------
# Extra Hugging Face load kwargs
# -----------------------------------------------------------------------------
# DatasetSpec does not appear to have a place for custom load_dataset kwargs,
# so keep them in a side table keyed by DatasetSpec.name.
#
# Languages:
#   en = English
#   es = Spanish
#   fr = French
#   de = German
#   sv = Swedish
#
# We load only the first shard for each language to keep the non-streaming
# download smaller and avoid the custom dataset script.
DATASET_LOAD_KWARGS_BY_NAME: dict[str, dict[str, Any]] = {
    "Multilingual-CC-News-en-es-fr-de-sv": {
        "data_files": {
            "train": [
                "hf://datasets/intfloat/multilingual_cc_news/en/train/0000.parquet",
                "hf://datasets/intfloat/multilingual_cc_news/es/train/0000.parquet",
                "hf://datasets/intfloat/multilingual_cc_news/fr/train/0000.parquet",
                "hf://datasets/intfloat/multilingual_cc_news/de/train/0000.parquet",
                "hf://datasets/intfloat/multilingual_cc_news/sv/train/0000.parquet",
            ],
        },
    },
}


# -----------------------------------------------------------------------------
# Loader patch
# -----------------------------------------------------------------------------
# Replace your current load_dataset helper with this version if this file owns
# dataset loading. If your project imports load_dataset from
# skip_search_spec.helpers.tooling instead, apply this same change there too.
def load_dataset(dataset: DatasetSpec, **load_kwargs: Any):
    """
    Load a Hugging Face dataset using DatasetSpec defaults.

    Supports extra per-dataset kwargs such as:
      load_dataset("parquet", data_files={...})
    """
    resolved_kwargs: dict[str, Any] = {}

    resolved_kwargs.update(DATASET_LOAD_KWARGS_BY_NAME.get(dataset.name, {}))
    resolved_kwargs.update(load_kwargs)

    resolved_kwargs.setdefault("split", dataset.split)

    return hf_load_dataset(
        dataset.huggingface_path,
        dataset.config_name,
        **resolved_kwargs,
    )


# -----------------------------------------------------------------------------
# Mix helpers
# -----------------------------------------------------------------------------

def _max_examples_for_source(
    num_windows: int,
    fraction: float,
    examples_per_window: float,
) -> int:
    return max(1, int(num_windows * fraction * examples_per_window))


def get_dataset_mix(num_windows: int = 10_000) -> list[tuple[DatasetSpec, float, int]]:
    """
    Non-streaming, English-first mix with a small multilingual component.

    Suggested purpose:
      - FineWeb-Edu-1B: broad educational prose
      - Cosmopedia-100k: synthetic textbook/story/guide prose
      - FineMath-4plus: math and solution-like reasoning
      - Python-Codes-25k: instruction-style Python code examples
      - Multilingual CC News: small multilingual prose exposure
    """
    mix: list[tuple[DatasetSpec, float, float]] = [
        # Broad educational prose.
        (DATASET_SPEC_FINEWEB_EDU_1B, 0.38, 1.3),

        # Synthetic educational prose, stories, and guides.
        (DATASET_SPEC_COSMOPEDIA_100K, 0.32, 1.5),

        # Math/reasoning text.
        (DATASET_SPEC_FINEMATH_4PLUS, 0.10, 2.0),

        # Python instruction/code examples.
        (DATASET_SPEC_PYTHON_CODES_25K, 0.10, 4.8),

        # Multilingual article/news prose.
        (DATASET_SPEC_MULTILINGUAL_CC_NEWS, 0.10, 1.5),
    ]

    return [
        (
            spec,
            fraction,
            _max_examples_for_source(
                num_windows=num_windows,
                fraction=fraction,
                examples_per_window=examples_per_window,
            ),
        )
        for spec, fraction, examples_per_window in mix
    ]


def get_dataset_mix_no_multilingual(
    num_windows: int = 10_000,
) -> list[tuple[DatasetSpec, float, int]]:
    """
    Backup mix if multilingual_cc_news is too large or slow to download.

    This stays closer to your original English/math/code benchmark.
    """
    mix: list[tuple[DatasetSpec, float, float]] = [
        (DATASET_SPEC_FINEWEB_EDU_1B, 0.33, 1.3),
        (DATASET_SPEC_COSMOPEDIA_100K, 0.40, 1.5),
        (DATASET_SPEC_FINEMATH_4PLUS, 0.22, 2.0),
        (DATASET_SPEC_PYTHON_CODES_25K, 0.15, 3.0),
    ]

    return [
        (
            spec,
            fraction,
            _max_examples_for_source(
                num_windows=num_windows,
                fraction=fraction,
                examples_per_window=examples_per_window,
            ),
        )
        for spec, fraction, examples_per_window in mix
    ]