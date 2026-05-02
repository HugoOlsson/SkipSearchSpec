
from skip_search_spec.experiments.dataset_mix import get_dataset_mix
from skip_search_spec.training.train_skipping_layers import train_skipping_layers


# This is the experiment to 

def train_aggressive_skipping_ablations():
        number_of_windows = 10_000

        models = ["Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-4B", "meta-llama/Llama-3.2-1B"]
        active_start_end_lengths = [(1,1), (2, 0), (4, 4), (8, 0), (6, 6), (12, 0)]

        for active_start_layers, active_end_layers in active_start_end_lengths: 

            for model in models:

                train_skipping_layers(
                    model_name=model,
                    dataset_mix=get_dataset_mix(number_of_windows),
                    num_windows_to_use=number_of_windows,
                    batch_size=4,
                    active_start_layers=active_start_layers, 
                    active_end_layers=active_end_layers,
                )
