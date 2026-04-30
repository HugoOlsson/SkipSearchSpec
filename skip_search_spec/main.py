"""Project entrypoint."""

from __future__ import annotations

import os



os.environ.setdefault("FLA_TILELANG", "0")
# os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")

import sys

from skip_search_spec.protocols.windows import DatasetSpec



STORE_PATH_FLASH_HEAD = "checkpoints/flashhead_qwen_0_5b.pt"
MODEL_NAME_FLASH_HEAD = "Qwen/Qwen3.5-4B"


def main() -> None:
    if len(sys.argv) <= 1:
        raise ValueError("Must provide a run mode")
    mode = sys.argv[1]
    print(f"[entry] mode={mode}", flush=True)


    if mode == "train_skipping_layers":
        from skip_search_spec.training.train_skipping_layers import train_skipping_layers

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

        number_of_windows = 30_000
        num_epochs = 1 # Ensure never get scores on data it has seen
        fraction_tiny = 0.3
        fraction_edu = 0.7

        models = ["meta-llama/Llama-3.2-1B"]
        active_start_end_lengths = [(2, 2)]

        # SINGLE LAYER AT START
        print("Version: 1.4")

        for active_start_layers, active_end_layers in active_start_end_lengths: 

            for model in models:

                train_skipping_layers(
                    model_name=model,
                    dataset_mix=[
                        (DATASET_SPEC_STORIES, fraction_tiny, int(number_of_windows*fraction_tiny*7)),
                        (DATASET_SPEC_EDU, fraction_edu, int(number_of_windows*fraction_edu*1.5)),
                    ],
                    context_len=256,
                    num_windows_to_use=number_of_windows,
                    batch_size=4,
                    active_start_layers=active_start_layers, 
                    active_end_layers=active_end_layers,
                    num_epochs=num_epochs,
                    lr=1e-4,
                    max_steps=1000000, #just something big
                    kl_loss_weight=1.0,
                    hidden_loss_weight=0,
                    ce_loss_weight=1.0,
                    checkpoint_every_steps=2000
                )

       
        # SINGLE LAYER AT START

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


        # results = evaluate_layer_skip_ablations(
        #     model_name="Qwen/Qwen2.5-14B",
        #     dataset_spec=DATASET_SPEC,
        #     context_len=256,
        #     max_examples=100,
        #     num_windows_to_use=20,
        #     batch_size=10,
        # )

        # results = evaluate_layer_skip_ablations(
        #     model_name="Qwen/Qwen2.5-7B",
        #     dataset_spec=DATASET_SPEC,
        #     context_len=256,
        #     max_examples=100,
        #     num_windows_to_use=20,
        #     batch_size=10,
        # )

        # results = evaluate_layer_skip_ablations(
        #     model_name="Qwen/Qwen2.5-3B",
        #     dataset_spec=DATASET_SPEC,
        #     context_len=256,
        #     max_examples=100,
        #     num_windows_to_use=20,
        #     batch_size=10,
        # )


        # results = evaluate_layer_skip_ablations(
        #     model_name="Qwen/Qwen2.5-1.5B",
        #     dataset_spec=DATASET_SPEC,
        #     context_len=256,
        #     max_examples=100,
        #     num_windows_to_use=20,
        #     batch_size=10,
        # )


        # results = evaluate_layer_skip_ablations(
        #     model_name="Qwen/Qwen2.5-0.5B",
        #     dataset_spec=DATASET_SPEC,
        #     context_len=256,
        #     max_examples=100,
        #     num_windows_to_use=20,
        #     batch_size=10,
        # )


        # results = evaluate_layer_skip_ablations(
        #     model_name="Qwen/Qwen3.5-9B",
        #     dataset_spec=DATASET_SPEC,
        #     context_len=256,
        #     max_examples=100,
        #     num_windows_to_use=20,
        #     batch_size=10,
        # )

        # results = evaluate_layer_skip_ablations(
        #     model_name="Qwen/Qwen3.5-4B",
        #     dataset_spec=DATASET_SPEC,
        #     context_len=256,
        #     max_examples=100,
        #     num_windows_to_use=20,
        #     batch_size=10,
        # )

        # results = evaluate_layer_skip_ablations(
        #     model_name="Qwen/Qwen3.5-2B",
        #     dataset_spec=DATASET_SPEC,
        #     context_len=256,
        #     max_examples=100,
        #     num_windows_to_use=20,
        #     batch_size=10,
        # )

        # results = evaluate_layer_skip_ablations(
        #     model_name="Qwen/Qwen3.5-0.8B",
        #     dataset_spec=DATASET_SPEC,
        #     context_len=256,
        #     max_examples=100,
        #     num_windows_to_use=20,
        #     batch_size=10,
        # )


        results = evaluate_layer_skip_ablations(
            model_name="meta-llama/Llama-3.2-1B",
            dataset_spec=DATASET_SPEC,
            context_len=256,
            max_examples=100,
            num_windows_to_use=20,
            batch_size=10,
        )

        results = evaluate_layer_skip_ablations(
            model_name="meta-llama/Llama-3.2-3B",
            dataset_spec=DATASET_SPEC,
            context_len=256,
            max_examples=100,
            num_windows_to_use=20,
            batch_size=10,
        )

        results = evaluate_layer_skip_ablations(
            model_name="meta-llama/Llama-3.1-8B",
            dataset_spec=DATASET_SPEC,
            context_len=256,
            max_examples=100,
            num_windows_to_use=20,
            batch_size=10,
        )

    elif mode == "plot_layer_ablation_results":
        from skip_search_spec.analysis.plot_ablations_results import plot_ablation_json

        file_path = sys.argv[2]

        plot_ablation_json(
            file_path,
            metric="mean_top1_agreement",
            top_k=None,   # or e.g. 50
        )

        # plot_ablation_heatmap_from_json( "ablation_results/layer_ablations_Qwen_Qwen2.5-0.5B_20260418_122333.json")


    elif mode == "plot_spec_results":
        from skip_search_spec.analysis.render_speculation_trace_html import render_speculation_trace_html

        file_path = sys.argv[2]

        render_speculation_trace_html(trace_json_path=file_path, output_html_path="out.html")


    elif mode == "plot_training_metric":
        from skip_search_spec.analysis.plot_training_metrics import plot_training_metric_jsons

        file_paths = [
            "measurements/2026-04-29-3b686b/middle_gap_skip/19080074_AP29__Qwen_Qwen2_5-3B_1_34_1/run.json",
            "measurements/2026-04-29-3b686b/middle_gap_skip/19124208_AP29__Qwen_Qwen3_5-4B_1_30_1/run.json",
            "measurements/2026-04-29-3b686b/middle_gap_skip/19200197_AP29__Qwen_Qwen2_5-3B_2_34_0/run.json",
            "measurements/2026-04-29-3b686b/middle_gap_skip/19244111_AP29__Qwen_Qwen3_5-4B_2_30_0/run.json",
            "measurements/2026-04-29-3b686b/middle_gap_skip/19315436_AP29__Qwen_Qwen2_5-3B_4_28_4/run.json",
            "measurements/2026-04-29-3b686b/middle_gap_skip/19370067_AP29__Qwen_Qwen3_5-4B_4_24_4/run.json",
            "measurements/2026-04-29-3b686b/middle_gap_skip/19450569_AP29__Qwen_Qwen2_5-3B_8_28_0/run.json",
            "measurements/2026-04-29-3b686b/middle_gap_skip/19495741_AP29__Qwen_Qwen3_5-4B_8_24_0/run.json",
        ]

        plot_training_metric_jsons(
            file_paths,
            metric_name="kl_verifier_to_drafter",
            phase="train",
            output_dir="measurements/training_metric_plots3",
        )


    elif mode == "plot_training_metric_bars":
        from skip_search_spec.analysis.plot_training_metric_bars import plot_training_metric_average_bars_jsons

        file_paths = [
            "measurements/2026-04-29-ac289b/middle_gap_skip/14241917_AP29__Qwen_Qwen2_5-3B_4_28_4/run.json",
            "measurements/2026-04-29-ac289b/middle_gap_skip/14132321_AP29__Qwen_Qwen3_5-4B_2_30_0/run.json",
            "measurements/2026-04-29-ac289b/middle_gap_skip/14061160_AP29__Qwen_Qwen2_5-3B_2_34_0/run.json",
            "measurements/2026-04-29-ac289b/middle_gap_skip/13550030_AP29__Qwen_Qwen3_5-4B_1_30_1/run.json",
            "measurements/2026-04-29-ac289b/middle_gap_skip/13473853_AP29__Qwen_Qwen2_5-3B_1_34_1/run.json",
        ]

        plot_training_metric_average_bars_jsons(
            file_paths,
            metric_name="kl_verifier_to_drafter",
            phase="train",
            output_dir="measurements/training_metric_plots_bars",
        )



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
            prompt="Tell me a story about a dog called Bob.",
            max_new_tokens=300,
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

    elif mode == "test_normal_inference":
        from skip_search_spec.inference.normal_inference import generate_from_plain_prompt

        text = generate_from_plain_prompt(
            model_name_or_path="Qwen/Qwen3.5-4B",
            prompt="Tell me a story about a dog called Bob.",
            max_new_tokens=300,
        )


        print(text)


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

