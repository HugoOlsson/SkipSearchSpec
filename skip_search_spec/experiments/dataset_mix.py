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

DATASET_SPEC_ALPACA_FORMATTED = DatasetSpec(
    name="Alpaca-52k-formatted",
    huggingface_path="tatsu-lab/alpaca",
    config_name=None,
    split="train",
    text_field="text",
)

DATASET_SPEC_SCIQ_FORMATTED = DatasetSpec(
    name="SciQ-formatted",
    huggingface_path="allenai/sciq",
    config_name=None,
    split="train",
    text_field="text",
)

DATASET_SPEC_SQUAD_FORMATTED = DatasetSpec(
    name="SQuAD-formatted",
    huggingface_path="rajpurkar/squad",
    config_name=None,
    split="train",
    text_field="text",
)

DATASET_SPEC_GSM8K_FORMATTED = DatasetSpec(
    name="GSM8K-formatted",
    huggingface_path="openai/gsm8k",
    config_name="main",
    split="train",
    text_field="text",
)

DATASET_SPEC_MBPP_FORMATTED = DatasetSpec(
    name="MBPP-formatted",
    huggingface_path="google-research-datasets/mbpp",
    config_name=None,
    split="train",
    text_field="text",
)

DATASET_SPEC_DIALOGSUM_FORMATTED = DatasetSpec(
    name="DialogSum-formatted",
    huggingface_path="knkarthick/dialogsum",
    config_name=None,
    split="train",
    text_field="text",
)

DATASET_SPEC_TINYSTORIES = DatasetSpec(
    name="TinyStories",
    huggingface_path="roneneldan/TinyStories",
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



def get_dataset_mix_question_answer_only(
    num_windows: int = 10_000,
) -> list[tuple[DatasetSpec, float, int]]:
    """
    Q/A-only non-streaming mix.

    Suggested purpose:
      - Dolly-15k-formatted: general instruction / answer formatting
      - MetaMathQA-40K-formatted: math problem / solution formatting

    Useful for debugging whether formatted Q/A windows are being built correctly,
    or for training a short run focused only on answer-style continuations.
    """
    mix: list[tuple[DatasetSpec, float, float]] = [
        # General instruction / Q&A formatting.
        (DATASET_SPEC_DOLLY_15K_FORMATTED, 0.15, 3.0),

        # Math Q&A / solution formatting.
        (DATASET_SPEC_METAMATHQA_40K_FORMATTED, 0.85, 3.0),
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


def get_dataset_mix_prompt_aligned(
    num_windows: int = 10_000,
) -> list[tuple[DatasetSpec, float, int]]:
    """
    Prompt-aligned mix for eval prompts that include:
      - QA
      - summarization
      - math word problems
      - code
      - rewriting
      - stories
      - article/procedure/comparison continuations
    """
    mix: list[tuple[DatasetSpec, float, float]] = [
        # Article/prose/procedure/comparison continuation.
        (DATASET_SPEC_COSMOPEDIA_100K, 0.20, 1.5),
        (DATASET_SPEC_FINEWEB_EDU_1B, 0.15, 1.3),

        # General instruction / answer formatting.
        (DATASET_SPEC_ALPACA_FORMATTED, 0.15, 2.5),
        (DATASET_SPEC_DOLLY_15K_FORMATTED, 0.10, 3.0),

        # Science / factual QA.
        (DATASET_SPEC_SCIQ_FORMATTED, 0.08, 4.0),
        (DATASET_SPEC_SQUAD_FORMATTED, 0.07, 4.0),

        # Math solution formatting.
        (DATASET_SPEC_GSM8K_FORMATTED, 0.08, 4.0),
        (DATASET_SPEC_METAMATHQA_40K_FORMATTED, 0.07, 3.0),

        # Code completion / small Python tasks.
        (DATASET_SPEC_MBPP_FORMATTED, 0.05, 5.0),
        (DATASET_SPEC_PYTHON_CODES_25K, 0.05, 4.8),

        # Summarization and story style.
        (DATASET_SPEC_DIALOGSUM_FORMATTED, 0.05, 3.0),
        (DATASET_SPEC_TINYSTORIES, 0.05, 2.0),
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