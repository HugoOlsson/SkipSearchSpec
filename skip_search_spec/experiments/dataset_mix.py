
from skip_search_spec.protocols.windows import DatasetSpec


DATASET_SPEC_STORIES = DatasetSpec(
    name="TinyStories",
    huggingface_path="roneneldan/TinyStories",
    config_name="default",
    split="train",
    text_field="text",
)

DATASET_SPEC_FINEPDFS = DatasetSpec(
    name="FinePDFs-1B",
    huggingface_path="codelion/finepdfs-1B",
    config_name="default",
    split="train",
    text_field="text",
)

DATASET_SPEC_DCLM = DatasetSpec(
    name="DCLM-Baseline-1B",
    huggingface_path="codelion/dclm-baseline-1B",
    config_name="default",
    split="train",
    text_field="text",
)

DATASET_SPEC_EDU = DatasetSpec(
    name="FineWeb-Edu-1B",
    huggingface_path="codelion/fineweb-edu-1B",
    config_name="default",
    split="train",
    text_field="text",
)


def _max_examples_for_source(
    num_windows: int,
    fraction: float,
    examples_per_window: float,
) -> int:
    return max(1, int(num_windows * fraction * examples_per_window))


def get_dataset_mix(num_windows: int = 10_000) -> list[tuple[DatasetSpec, float, int]]:
    fraction_finepdfs = 0.30
    fraction_dclm = 0.30
    fraction_edu = 0.35
    fraction_tiny = 0.05

    return [
        (
            DATASET_SPEC_FINEPDFS,
            fraction_finepdfs,
            _max_examples_for_source(num_windows, fraction_finepdfs, 1.1),
        ),
        (
            DATASET_SPEC_DCLM,
            fraction_dclm,
            _max_examples_for_source(num_windows, fraction_dclm, 1.5),
        ),
        (
            DATASET_SPEC_EDU,
            fraction_edu,
            _max_examples_for_source(num_windows, fraction_edu, 1.5),
        ),
        (
            DATASET_SPEC_STORIES,
            fraction_tiny,
            _max_examples_for_source(num_windows, fraction_tiny, 7),
        ),
    ]
