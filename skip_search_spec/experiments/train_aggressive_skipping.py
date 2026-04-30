


from skip_search_spec.protocols.windows import DatasetSpec
from skip_search_spec.training.train_skipping_layers import train_skipping_layers


# This is the experiment to 

def train_aggressive_skipping_ablations():

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

        number_of_windows = 10_000
        num_epochs = 1 # Ensure never get scores on data it has seen
        fraction_tiny = 0.3
        fraction_edu = 0.7

        models = ["Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-4B", "meta-llama/Llama-3.2-1B"]
        active_start_end_lengths = [(1,1), (2, 0), (4, 4), (8, 0), (6, 6), (12, 0)]

        for active_start_layers, active_end_layers in active_start_end_lengths: 

            for model in models:

                train_skipping_layers(
                    model_name=model,
                    dataset_mix=[
                        (DATASET_SPEC_STORIES, fraction_tiny, int(number_of_windows*fraction_tiny*7)),
                        (DATASET_SPEC_EDU, fraction_edu, int(number_of_windows*fraction_edu*1.5)),
                    ],
                    num_windows_to_use=number_of_windows,
                    batch_size=4,
                    active_start_layers=active_start_layers, 
                    active_end_layers=active_end_layers,
                )