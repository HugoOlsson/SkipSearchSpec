
from skip_search_spec.protocols.windows import DatasetSpec




DATASET_SPEC_EDU = DatasetSpec(
    name="FineWeb-Edu-1B",
    huggingface_path="codelion/fineweb-edu-1B",
    config_name="default",
    split="train",
    text_field="text",
)

DATASET_SPEC_COSMOPEDIA_100K = DatasetSpec(
    name="Cosmopedia-100k",
    huggingface_path="HuggingFaceTB/cosmopedia-100k",
    config_name=None,  # or "default" if your loader requires a string
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
    fineweb_fraction = 0.5
    cosmopedia_fraction = 0.5

    return [
        (
            DATASET_SPEC_EDU,
            fineweb_fraction,
            _max_examples_for_source(num_windows, fineweb_fraction, 1.2),
        ),
        (
            DATASET_SPEC_COSMOPEDIA_100K,
            cosmopedia_fraction,
            _max_examples_for_source(num_windows, cosmopedia_fraction, 1.2),
        ),
    ]
