
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

DATASET_SPEC_TULU_CHAT = DatasetSpec(
    name="Tulu-3-SFT-Mixture-English",
    huggingface_path="yimingzhang/tulu-3-sft-mixture-english",
    split="train",
    text_field="messages",
)


def _max_examples_for_source(
    num_windows: int,
    fraction: float,
    examples_per_window: float,
) -> int:
    return max(1, int(num_windows * fraction * examples_per_window))


def get_dataset_mix(num_windows: int = 10_000) -> list[tuple[DatasetSpec, float, int]]:
    return [
        (
            DATASET_SPEC_EDU,
            1.0,
            _max_examples_for_source(num_windows, 1.0, 1.5),
        ),
    ]


def get_chat_dataset_mix(num_windows: int = 10_000) -> list[tuple[DatasetSpec, float, int]]:
    return [
        (
            DATASET_SPEC_TULU_CHAT,
            1.0,
            _max_examples_for_source(num_windows, 1.0, 3),
        ),
    ]
