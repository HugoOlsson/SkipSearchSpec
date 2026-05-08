from __future__ import annotations

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

DATASET_SPEC_DOLLY_15K_FORMATTED = DatasetSpec(
    name="Dolly-15k-formatted",
    huggingface_path="databricks/databricks-dolly-15k",
    config_name=None,
    split="train",
    # Created by maybe_format_dataset_to_text(...)
    text_field="text",
)

DATASET_SPEC_METAMATHQA_40K_FORMATTED = DatasetSpec(
    name="MetaMathQA-40K-formatted",
    huggingface_path="meta-math/MetaMathQA-40K",
    config_name=None,
    split="train",
    # Created by maybe_format_dataset_to_text(...)
    text_field="text",
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
    Non-streaming English/math/code/instruction mix.

    This version adds formatted Q/A data to reduce:
      - over-continuation into extra examples
      - weak chat-style completions
      - unstable answer formatting
    """
    mix: list[tuple[DatasetSpec, float, float]] = [
        # Broad educational prose.
        (DATASET_SPEC_FINEWEB_EDU_1B, 0.27, 1.3),

        # Synthetic educational prose, stories, and guides.
        (DATASET_SPEC_COSMOPEDIA_100K, 0.30, 1.5),

        # General math/reasoning text.
        (DATASET_SPEC_FINEMATH_4PLUS, 0.10, 2.0),

        # Python instruction/code examples.
        (DATASET_SPEC_PYTHON_CODES_25K, 0.15, 4.8),

        # General instruction / Q&A formatting.
        (DATASET_SPEC_DOLLY_15K_FORMATTED, 0.08, 3.0),

        # Math Q&A / solution formatting.
        (DATASET_SPEC_METAMATHQA_40K_FORMATTED, 0.10, 3.0),
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


def get_dataset_mix_more_math_code(
    num_windows: int = 10_000,
) -> list[tuple[DatasetSpec, float, int]]:
    """
    More aggressive version if math/code eval prompts matter more.
    """
    mix: list[tuple[DatasetSpec, float, float]] = [
        (DATASET_SPEC_FINEWEB_EDU_1B, 0.30, 1.3),
        (DATASET_SPEC_COSMOPEDIA_100K, 0.35, 1.5),
        (DATASET_SPEC_FINEMATH_4PLUS, 0.17, 2.2),
        (DATASET_SPEC_PYTHON_CODES_25K, 0.18, 5.0),
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