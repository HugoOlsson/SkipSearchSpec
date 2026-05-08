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
    Non-streaming English/math/code mix.

    Suggested purpose:
      - FineWeb-Edu-1B: broad educational prose
      - Cosmopedia-100k: synthetic textbook/story/guide prose
      - FineMath-4plus: math and solution-like reasoning
      - Python-Codes-25k: instruction-style Python code examples
    """
    mix: list[tuple[DatasetSpec, float, float]] = [
        # Broad educational prose.
        (DATASET_SPEC_FINEWEB_EDU_1B, 0.33, 1.3),

        # Synthetic educational prose, stories, and guides.
        (DATASET_SPEC_COSMOPEDIA_100K, 0.40, 1.5),

        # Math/reasoning text.
        (DATASET_SPEC_FINEMATH_4PLUS, 0.12, 2.0),

        # Python instruction/code examples.
        (DATASET_SPEC_PYTHON_CODES_25K, 0.15, 4.8),
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