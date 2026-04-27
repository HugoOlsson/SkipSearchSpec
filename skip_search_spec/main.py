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


    if mode == "train_skipping_layers":
        from skip_search_spec.training.train_skipping_layers import train_skipping_layers

        DATASET_SPEC= DatasetSpec(
            name="TinyStories",
            huggingface_path="roneneldan/TinyStories",
            config_name="default",
            split="train",
            text_field="text",
        )

        DATASET_SPEC2 = DatasetSpec(
            name="FineWeb-Edu-1B",
            huggingface_path="codelion/fineweb-edu-1B",
            config_name="default",
            split="train",
            text_field="text",
        )

        print("Version: 3.9")
        train_skipping_layers(
            model_name="Qwen/Qwen2.5-0.5B",
            dataset_mix=[
                (DATASET_SPEC, 0.8),
                (DATASET_SPEC2, 0.2),
            ],
            context_len=256,
            max_examples=50_000,
            num_windows_to_use=5_000,
            batch_size=2,
            gap_start=1,
            gap_length=22, 
            num_epochs=3,
            lr=1e-4,
            max_steps=100000,
            kl_loss_weight=1.0,
            hidden_loss_weight=0,
            ce_loss_weight=1.0,
        )

    elif mode == "build_flashhead":
        from skip_search_spec.training.flashhead.flashhead_research import build_flashhead_head
        build_flashhead_head(STORE_PATH_FLASH_HEAD, MODEL_NAME_FLASH_HEAD)

    elif mode == "evaluate_flashhead":
        from skip_search_spec.training.flashhead.flashhead_research import evaluate_flashhead
        evaluate_flashhead(STORE_PATH_FLASH_HEAD, MODEL_NAME_FLASH_HEAD)


    elif mode == "evaluate_layer_ablations":
        from skip_search_spec.training.evaluate_layer_skip_ablations import evaluate_layer_skip_ablations

        DATASET_SPEC = DatasetSpec(
            name="FineWeb-Edu-1B",
            huggingface_path="codelion/fineweb-edu-1B",
            config_name="default",
            split="train",
            text_field="text",
        )


        results = evaluate_layer_skip_ablations(
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

    elif mode == "test_self_spec":
        if len(sys.argv) <= 3:
            raise ValueError(
                "Must provide 3 arguments"
            )
        
        from skip_search_spec.inference.self_spec_inference import self_spec_inference_test
        
        draft_block_size = sys.argv[2]

        bridge_checkpoint_path = sys.argv[3]


        result = self_spec_inference_test(
            bridge_checkpoint_path=bridge_checkpoint_path,
            prompt="Once upon a time there was a girl called Lilly. Lilly liked to play in the sun with her friends.",
            max_new_tokens=150,
            draft_block_size=int(draft_block_size),
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

    else:
        raise ValueError(f"Unknown mode: {mode}")

if __name__ == "__main__":
    main()




    # elif mode == "train_future_hidden_heads":
    #     from skip_search_spec.training.old.multi_future3 import train_next_hidden_teacher_forced

    #     DATASET_SPEC = DatasetSpec(
    #         name="TinyStories",
    #         huggingface_path="roneneldan/TinyStories",
    #         config_name="default",
    #         split="train",
    #         text_field="text",
    #     )

    #     print("Version V4.3")
    #     out = train_next_hidden_teacher_forced(
    #         model_name="Qwen/Qwen2.5-7B",
    #         dataset_spec=DATASET_SPEC,
    #         context_len=256,
    #         max_examples=200_000,
    #         num_windows_to_use=10_000,
    #         batch_size=10,
    #         num_epochs=2,
    #         max_steps=2000000,
    #         lr=2e-4,
    #         hidden_loss_weight=0.0,
    #         cosine_loss_weight=0.0,
    #         kl_loss_weight=1.0,
    #         ce_loss_weight=1.0,
    #         num_input_states=3,
    #     )

