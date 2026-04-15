"""Project entrypoint."""

from __future__ import annotations
import sys

from skip_search_spec.protocols.windows import DatasetSpec

STORE_PATH_FLASH_HEAD = "checkpoints/flashhead_qwen_0_5b.pt"
MODEL_NAME_FLASH_HEAD = "Qwen/Qwen2.5-0.5B"


def main() -> None:
    if len(sys.argv) <= 1:
        raise ValueError("Must provide a run mode")
    mode = sys.argv[1]
    print(f"[entry] mode={mode}", flush=True)

   
    if mode == "train_early_exit":
        print("[entry] importing train_early_exit", flush=True)
        from skip_search_spec.training.train_early_exit import train_early_exit
        print("[entry] imported train_early_exit", flush=True)

        DATASET_SPEC_EARLY_EXIT = DatasetSpec(
            name="FineWeb-Edu-1B",
            huggingface_path="codelion/fineweb-edu-1B",
            config_name="default",
            split="train",
            text_field="text",
        )

        MODEL_NAME_EARLY_EXIT = "Qwen/Qwen2.5-1.5B"

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
    
    elif mode == "train_drafter_for_verifier":
        from skip_search_spec.training.train_drafter_ability import train_drafter_for_verifier

        DATASET_SPEC_DRAFT_FOR_VERIFIER = DatasetSpec(
            name="FineWeb-Edu-1B",
            huggingface_path="codelion/fineweb-edu-1B",
            config_name="default",
            split="train",
            text_field="text",
        )

        DRAFT_MODEL_NAME = "Qwen/Qwen2.5-0.5B"

        VERIFIER_MODEL_NAME = "Qwen/Qwen2.5-1.5B"

        train_drafter_for_verifier(
          draft_model_name=DRAFT_MODEL_NAME,
          verifier_model_name=VERIFIER_MODEL_NAME,
          dataset_spec=DATASET_SPEC_DRAFT_FOR_VERIFIER,
          window_size=500,
          number_of_layers_allowed_to_change=10,
          batch_size=4,
          max_examples=1_000
        )


    elif mode == "build_flashhead":
        from skip_search_spec.training.flashhead.flashhead_research import build_flashhead_head
        build_flashhead_head(STORE_PATH_FLASH_HEAD, MODEL_NAME_FLASH_HEAD)

    elif mode == "evaluate_flashhead":
        from skip_search_spec.training.flashhead.flashhead_research import evaluate_flashhead
        evaluate_flashhead(STORE_PATH_FLASH_HEAD, MODEL_NAME_FLASH_HEAD)

    
    # elif mode == "run_early_exit":
    #     run_self_speculation()

    # elif mode == "benchmark_self_speculation":
    #     benchmark_self_speculation()
    

    else:
        raise ValueError(f"Unknown mode: {mode}")

if __name__ == "__main__":
    main()
