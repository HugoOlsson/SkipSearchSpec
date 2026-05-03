"""Project entrypoint."""

from __future__ import annotations

import os



os.environ.setdefault("FLA_TILELANG", "0")
# os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")

import sys

from skip_search_spec.protocols.windows import DatasetSpec



STORE_PATH_FLASH_HEAD = "checkpoints/flashhead_llama32_3b.pt"
MODEL_NAME_FLASH_HEAD = "meta-llama/Llama-3.2-3B"
INFERENCE_TEST_MAX_NEW_TOKENS = 200
INFERENCE_TEST_PROMPTS = [
    (
        "Recent U.S. presidents list",
        "The 10 latest presidents of the USA is: 1. Donald Trump, ",
    ),
    (
        "Talking about Paris",
        "The capital of France is quite large and its name is",
    ),
    (
        "Story about Bob",
        (
            "There once was a man named Bob that lived in the state Texas. "
            "He liked to drive his pickup truck"
        ),
    ),
]


def main() -> None:
    if len(sys.argv) <= 1:
        raise ValueError("Must provide a run mode")
    mode = sys.argv[1]
    print(f"[entry] mode={mode}", flush=True)


    if mode == "train_skipping_layers":
        from skip_search_spec.experiments.dataset_mix import get_dataset_mix
        from skip_search_spec.training.train_skipping_layers import train_skipping_layers

        number_of_windows = 100_000
        num_epochs = 1 # Ensure never get scores on data it has seen

        models = ["meta-llama/Llama-3.2-1B"]
        active_start_end_lengths = [(4, 4)]

        # SINGLE LAYER AT START
        print("Version: 1.7")

        for active_start_layers, active_end_layers in active_start_end_lengths: 

            for model in models:

                train_skipping_layers(
                    model_name=model,
                    dataset_mix=get_dataset_mix(number_of_windows),
                    context_len=256,
                    num_windows_to_use=number_of_windows,
                    batch_size=30,
                    active_start_layers=active_start_layers, 
                    active_end_layers=active_end_layers,
                    num_epochs=num_epochs,
                    lr=1e-4,
                    max_steps=1000000, #just something big
                    kl_loss_weight=1.0,
                    hidden_loss_weight=0.0,
                    ce_loss_weight=1.0,
                    checkpoint_every_steps=1000,
                    log_every=100,
                    num_draft_sections=4,
                    reference_hidden_source="final"
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
        flashhead_path = sys.argv[4] if len(sys.argv) > 4 else None

        total_inference_seconds = 0.0

        for test_idx, (test_name, prompt) in enumerate(INFERENCE_TEST_PROMPTS, start=1):
            print()
            print(f"Test {test_idx}: {test_name}")
            print()
            print("Prompt:")
            print(prompt)
            print()

            result = self_spec_inference_test(
                bridge_checkpoint_path=bridge_checkpoint_path,
                prompt=prompt,
                max_new_tokens=INFERENCE_TEST_MAX_NEW_TOKENS,
                draft_block_size=int(draft_block_size),
                use_chat_template=False,
                flashhead_path=flashhead_path,
                measure_internal_timings=True
            )
            timings = result.timings
            total_inference_seconds += timings.total_seconds

            print(result.text)
            print(
                {
                    "verifier_calls": result.verifier_calls,
                    "drafted_tokens": result.drafted_tokens,
                    "accepted_draft_tokens": result.accepted_draft_tokens,
                    "accept_rate": result.accepted_draft_tokens / max(result.drafted_tokens, 1),
                    "total_seconds": timings.total_seconds,
                    "dense_head_seconds": timings.dense_head_seconds,
                    "flashhead_seconds": timings.flashhead_seconds,
                }
            )

        print()
        print({"total_inference_seconds": total_inference_seconds})



    elif mode == "test_normal_inference":
        from skip_search_spec.inference.normal_inference import generate_from_plain_prompt

        total_inference_seconds = 0.0

        for test_idx, (test_name, prompt) in enumerate(INFERENCE_TEST_PROMPTS, start=1):
            print()
            print(f"Test {test_idx}: {test_name}")
            print()
            print("Prompt:")
            print(prompt)
            print()

            result = generate_from_plain_prompt(
                model_name_or_path="Qwen/Qwen3-4B",
                prompt=prompt,
                max_new_tokens=INFERENCE_TEST_MAX_NEW_TOKENS,
                use_chat_template=False,
                use_cache=True,
            )
            total_inference_seconds += result.inference_seconds

            print(result.text)
            print({"inference_seconds": result.inference_seconds})

        print()
        print({"total_inference_seconds": total_inference_seconds})


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
