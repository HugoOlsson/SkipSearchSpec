"""Project entrypoint."""

from __future__ import annotations
import sys
from skip_search_spec.inference.evaluate_gap_bridge_real_inference import run_minimal_bridge_self_spec_test
from skip_search_spec.protocols.windows import DatasetSpec



STORE_PATH_FLASH_HEAD = "checkpoints/flashhead_qwen_0_5b.pt"
MODEL_NAME_FLASH_HEAD = "Qwen/Qwen2.5-0.5B"


def main() -> None:
    if len(sys.argv) <= 1:
        raise ValueError("Must provide a run mode")
    mode = sys.argv[1]
    print(f"[entry] mode={mode}", flush=True)

   
    # if mode == "train_early_exit":
    #     print("[entry] importing train_early_exit", flush=True)
    #     from skip_search_spec.training.train_early_exit import train_early_exit
    #     print("[entry] imported train_early_exit", flush=True)

    #     DATASET_SPEC_EARLY_EXIT = DatasetSpec(
    #         name="FineWeb-Edu-1B",
    #         huggingface_path="codelion/fineweb-edu-1B",
    #         config_name="default",
    #         split="train",
    #         text_field="text",
    #     )

    #     MODEL_NAME_EARLY_EXIT = "Qwen/Qwen2.5-1.5B"

    #     early_exit_layers = [2,4,8,12,16,20]

    #     for early_exit_layer in early_exit_layers:
    #         train_early_exit(
    #             model_name=MODEL_NAME_EARLY_EXIT,
    #             dataset_spec=DATASET_SPEC_EARLY_EXIT,
    #             early_exit_layer=early_exit_layer,
    #             batch_size=2,
    #             checkpoint_path="checkpoints/early_exit_model15B.pt",
    #             max_examples=100,
    #             context_len=300,
    #             alpha=0.8,
    #             beta=2.0,
    #             save_optimizer=False,
    #         )

    # elif mode == "train_middle_gap_skip":
    #     from skip_search_spec.training.old.train_middle_gap_skip import train_middle_gap_skip

    #     DATASET_SPEC= DatasetSpec(
    #         name="FineWeb-Edu-1B",
    #         huggingface_path="codelion/fineweb-edu-1B",
    #         config_name="default",
    #         split="train",
    #         text_field="text",
    #     )

    #     MODEL_NAME= "Qwen/Qwen2.5-0.5B"


    #     train_middle_gap_skip(
    #         model_name=MODEL_NAME,
    #         gap_start_layer=9,
    #         gap_end_layer=13,
    #         dataset_spec=DATASET_SPEC,
    #         batch_size=2,
    #         checkpoint_path="checkpoints/middle_gap_skip.pt",
    #         max_examples=3_000,
    #         context_len=512,
    #         alpha=1.0,
    #         beta=1.0,
    #         save_optimizer=False,
    #     )

    if mode == "train_gap_bridge":
        from skip_search_spec.training.train_gap_bridge_optimized import train_gap_bridge2

        DATASET_SPEC= DatasetSpec(
            name="TinyStories",
            huggingface_path="roneneldan/TinyStories",
            config_name="default",
            split="train",
            text_field="text",
        )

        print("Version: 2.9")
        train_gap_bridge2(
            model_name="Qwen/Qwen2.5-0.5B",
            dataset_spec=DATASET_SPEC,
            context_len=256,
            max_examples=500_000,
            num_windows_to_use=90_000,
            batch_size=20,
            gap_start=5,
            gap_length=14, 
            num_epochs=2,
            lr=2e-4,
            max_steps=100000,
            kl_loss_weight=1.0,
            hidden_loss_weight=0.0,
            ce_loss_weight=1.0,
        )

    elif mode == "train_gap_bridge_finetune":
        from skip_search_spec.training.train_gap_bridge_finetune import train_gap_bridge_finetune

        DATASET_SPEC= DatasetSpec(
            name="TinyStories",
            huggingface_path="roneneldan/TinyStories",
            config_name="default",
            split="train",
            text_field="text",
        )

        print("Version: 2.8")
        train_gap_bridge_finetune(
            model_name="Qwen/Qwen2.5-0.5B",
            dataset_spec=DATASET_SPEC,
            context_len=256,
            max_examples=100_000,
            num_windows_to_use=10_000,
            batch_size=2,
            gap_start=1,
            gap_length=22, 
            num_epochs=2,
            lr=2e-4,
            max_steps=100000,
            kl_loss_weight=1.0,
            hidden_loss_weight=0.0,
            ce_loss_weight=1.0,
        )

    elif mode == "train_future_hidden_heads":
        from skip_search_spec.training.multi_future3 import train_next_hidden_teacher_forced

        DATASET_SPEC = DatasetSpec(
            name="TinyStories",
            huggingface_path="roneneldan/TinyStories",
            config_name="default",
            split="train",
            text_field="text",
        )

        print("Version V4.3")
        out = train_next_hidden_teacher_forced(
            model_name="Qwen/Qwen2.5-7B",
            dataset_spec=DATASET_SPEC,
            context_len=256,
            max_examples=200_000,
            num_windows_to_use=10_000,
            batch_size=10,
            num_epochs=2,
            max_steps=2000000,
            lr=2e-4,
            hidden_loss_weight=0.0,
            cosine_loss_weight=0.0,
            kl_loss_weight=1.0,
            ce_loss_weight=1.0,
            num_input_states=3,
        )

    # elif mode == "train_gap_bridge_teacher":
    #     from skip_search_spec.training.old.train_gap_bridge_teacher import train_gap_bridge_teacher

    #     DATASET_SPEC= DatasetSpec(
    #         name="FineWeb-Edu-1B",
    #         huggingface_path="codelion/fineweb-edu-1B",
    #         config_name="default",
    #         split="train",
    #         text_field="text",
    #     )

    #     out = train_gap_bridge_teacher(
    #         model_name="Qwen/Qwen2.5-7B",
    #         dataset_spec=DATASET_SPEC,
    #         context_len=512,
    #         max_examples=20000,
    #         num_windows_to_use=30000,
    #         batch_size=2,
    #         num_trainable_pre_gap_layers=10,
    #         gap_start=7,
    #         gap_length=14, 
    #         num_epochs=2,
    #         max_steps=100000,
    #         kl_loss_weight=1.0,
    #         hidden_loss_weight=1.0,
    #         ce_loss_weight=0.0,
    #     )
    
    # elif mode == "train_drafter_for_verifier":
    #     from skip_search_spec.training.old.train_drafter_ability import train_drafter_for_verifier

    #     DATASET_SPEC_DRAFT_FOR_VERIFIER = DatasetSpec(
    #         name="FineWeb-Edu-1B",
    #         huggingface_path="codelion/fineweb-edu-1B",
    #         config_name="default",
    #         split="train",
    #         text_field="text",
    #     )

    #     DRAFT_MODEL_NAME = "Qwen/Qwen2.5-1.5B"

    #     VERIFIER_MODEL_NAME = "Qwen/Qwen2.5-3B"

    #     train_drafter_for_verifier(
    #       draft_model_name=DRAFT_MODEL_NAME,
    #       verifier_model_name=VERIFIER_MODEL_NAME,
    #       dataset_spec=DATASET_SPEC_DRAFT_FOR_VERIFIER,
    #       window_size=500,
    #       number_of_layers_allowed_to_change=None,
    #       batch_size=2,
    #       max_examples=50_000
    #     )


    elif mode == "build_flashhead":
        from skip_search_spec.training.flashhead.flashhead_research import build_flashhead_head
        build_flashhead_head(STORE_PATH_FLASH_HEAD, MODEL_NAME_FLASH_HEAD)

    elif mode == "evaluate_flashhead":
        from skip_search_spec.training.flashhead.flashhead_research import evaluate_flashhead
        evaluate_flashhead(STORE_PATH_FLASH_HEAD, MODEL_NAME_FLASH_HEAD)


    elif mode == "evaluate_layer_ablations":
        from skip_search_spec.training.evaluate_layer_ablations import evaluate_layer_ablations

        DATASET_SPEC = DatasetSpec(
            name="FineWeb-Edu-1B",
            huggingface_path="codelion/fineweb-edu-1B",
            config_name="default",
            split="train",
            text_field="text",
        )


        results = evaluate_layer_ablations(
            model_name="Qwen/Qwen2.5-14B",
            dataset_spec=DATASET_SPEC,
            context_len=256,
            max_examples=100,
            num_windows_to_use=20,
            batch_size=4,
        )

    elif mode == "plot_layer_ablation_results":
        from skip_search_spec.analysis.plot_ablations_results import plot_ablation_json


        plot_ablation_json(
            "ablation_results/layer_ablations_Qwen_Qwen2.5-14B_20260421_132035.json",
            metric="kl_per_removed_layer",
            top_k=None,   # or e.g. 50
        )

        # plot_ablation_heatmap_from_json( "ablation_results/layer_ablations_Qwen_Qwen2.5-0.5B_20260418_122333.json")

    elif mode == "test_self_spec_with_bridge":
        result = run_minimal_bridge_self_spec_test(
            bridge_checkpoint_path="gap_bridge_checkpoints/gap_bridge__Qwen_Qwen2.5-0.5B__start_5__len_14__20260425_153308.pt",
            prompt="Once upon a time there was a bird called",
            max_new_tokens=80,
            draft_block_size=1,
        )

        print(result.text)
        print(
            {
                "verifier_calls": result.verifier_calls,
                "drafted_tokens": result.drafted_tokens,
                "accepted_draft_tokens": result.accepted_draft_tokens,
                "accept_rate": result.accepted_draft_tokens / max(result.drafted_tokens, 1),
            }
        )

    
    # elif mode == "run_early_exit":
    #     run_self_speculation()

    # elif mode == "benchmark_self_speculation":
    #     benchmark_self_speculation()
    

    else:
        raise ValueError(f"Unknown mode: {mode}")

if __name__ == "__main__":
    main()
