

from skip_search_spec.protocols.windows import DatasetSpec


def get_dataset_mix(num_windows: int = 10_000) -> list[tuple[DatasetSpec, float, int]]:
    DATASET_SPEC_STORIES= DatasetSpec(
        name="TinyStories",
        huggingface_path="roneneldan/TinyStories",
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

    number_of_windows = num_windows
    num_epochs = 1 # Ensure never get scores on data it has seen
    fraction_tiny = 0.3
    fraction_edu = 0.7
     
    return [
        (DATASET_SPEC_STORIES, fraction_tiny, int(number_of_windows*fraction_tiny*7)),
        (DATASET_SPEC_EDU, fraction_edu, int(number_of_windows*fraction_edu*1.5)),
    ]


