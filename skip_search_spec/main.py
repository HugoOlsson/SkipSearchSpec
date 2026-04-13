"""Project entrypoint."""

from __future__ import annotations

import torch
import sys

from skip_search_spec.helpers.tooling import load_model_and_tokenizer
from skip_search_spec.protocols.windows import DatasetSpec, WindowSettings
from skip_search_spec.training.flashhead.flashhead_research import  build_flashhead_head, evaluate_flashhead
from skip_search_spec.training.train_early_exit import train_early_exit






CONTEXT_PARTS = WindowSettings(C1=250)

STORE_PATH_FLASH_HEAD = "checkpoints/flashhead_qwen_0_5b.pt"
MODEL_NAME_FLASH_HEAD = "Qwen/Qwen2.5-0.5B"


def main() -> None:
    if len(sys.argv) <= 1:
        raise ValueError("Must provide a run mode")
    mode = sys.argv[1]

   
    if mode == "train_early_exit":

        DATASET_SPEC_EARLY_EXIT = DatasetSpec(
            name="FineWeb-Edu-1B",
            huggingface_path="codelion/fineweb-edu-1B",
            config_name="default",
            split="train",
            text_field="text",
        )

        MODEL_NAME_EARLY_EXIT = "Qwen/Qwen2.5-3B"

        train_early_exit(
            model_name=MODEL_NAME_EARLY_EXIT,
            dataset_spec=DATASET_SPEC_EARLY_EXIT,
            batch_size=2,
            checkpoint_path="checkpoints/early_exit_model15B.pt",
            max_examples=10_000,
            context_len=300,
            alpha=0.8,
            beta=2.0,
            save_optimizer=False,
        )

    elif mode == "build_flashhead":
        build_flashhead_head(STORE_PATH_FLASH_HEAD, MODEL_NAME_FLASH_HEAD)

    elif mode == "evaluate_flashhead":
        evaluate_flashhead(STORE_PATH_FLASH_HEAD, MODEL_NAME_FLASH_HEAD)

    
    # elif mode == "run_early_exit":
    #     run_self_speculation()

    # elif mode == "benchmark_self_speculation":
    #     benchmark_self_speculation()
    

    else:
        raise ValueError(f"Unknown mode: {mode}")

if __name__ == "__main__":
    main()