"""Project entrypoint."""

from __future__ import annotations

import os
from pprint import pprint

import torch

from skip_search_spec.experiments.dataset_mix import get_dataset_mix_v2, get_dataset_mix_openorca, get_dataset_mix_prompt_aligned, get_dataset_mix_question_answer_only
from skip_search_spec.experiments.inference_prompts import (
    CHAT_TEST_PROMPTS,
    INFERENCE_TEST_PROMPTS_CONCRETE,
    INFERENCE_TEST_PROMPTS_CONCRETE_SWEDISH,
    INFERENCE_TEST_PROMPTS_EASY,
    INFERENCE_TEST_PROMPTS_HARD,
)



os.environ.setdefault("FLA_TILELANG", "0")
# os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")

import sys

from skip_search_spec.protocols.windows import DatasetSpec



STORE_PATH_FLASH_HEAD = "checkpoints/flashhead_llama32_3b_16032c.pt"
MODEL_NAME_FLASH_HEAD = "meta-llama/Llama-3.2-3B-Instruct"

INFERENCE_TEST_MAX_NEW_TOKENS = 200


def main() -> None:
    if len(sys.argv) <= 1:
        raise ValueError("Must provide a run mode")
    mode = sys.argv[1]
    print(f"[entry] mode={mode}", flush=True)


    if mode == "train_skipping_layers":
        import argparse

        from skip_search_spec.experiments.dataset_mix import get_dataset_mix
        from skip_search_spec.training.train_skipping_layers import train_skipping_layers

        parser = argparse.ArgumentParser(prog=f"{sys.argv[0]} train_skipping_layers")
        parser.add_argument(
            "--name-comment",
            default=None,
            help="Optional comment to prefix the measurement run directory name.",
        )
        args = parser.parse_args(sys.argv[2:])

        number_of_windows = 100_000
        num_epochs = 1 # Ensure never get scores on data it has seen

        models = ["meta-llama/Llama-3.1-8B-Instruct"]
        active_start_end_lengths = [(1, 1)]

        for active_start_layers, active_end_layers in active_start_end_lengths: 

            for model in models:

                train_skipping_layers(
                    model_name=model,
                    dataset_mix=get_dataset_mix_v2(number_of_windows),
                    context_len=256,
                    num_windows_to_use=number_of_windows,
                    batch_size=10,
                    active_start_layers=active_start_layers, 
                    active_end_layers=active_end_layers,
                    num_epochs=num_epochs,
                    lr=1e-4,
                    max_steps=1000000, #just something big
                    kl_loss_weight=1.0,
                    hidden_loss_weight=0.0,
                    ce_loss_weight=1.0,
                    checkpoint_every_steps=None,
                    log_every=100,
                    num_draft_sections=5,
                    reference_hidden_source="final",
                    name_comment=args.name_comment,
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
        #     model_name="mistralai/Mistral-7B-Instruct-v0.3",
        #     dataset_spec=DATASET_SPEC,
        #     context_len=256,
        #     max_examples=100,
        #     num_windows_to_use=20,
        #     batch_size=1,
        # )

        results = evaluate_layer_skip_ablations(
            model_name="meta-llama/Llama-3.2-1B-Instruct",
            dataset_spec=DATASET_SPEC,
            context_len=256,
            max_examples=100,
            num_windows_to_use=20,
            batch_size=4,
        )

        # results = evaluate_layer_skip_ablations(
        #     model_name="meta-llama/Llama-3.2-3B",
        #     dataset_spec=DATASET_SPEC,
        #     context_len=256,
        #     max_examples=100,
        #     num_windows_to_use=20,
        #     batch_size=2,
        # )

        # results = evaluate_layer_skip_ablations(
        #     model_name="meta-llama/Llama-3.1-8B",
        #     dataset_spec=DATASET_SPEC,
        #     context_len=256,
        #     max_examples=100,
        #     num_windows_to_use=20,
        #     batch_size=10,
        # )

    elif mode == "plot_layer_ablation_results":
        from skip_search_spec.analysis.plot_ablations_results import plot_ablation_json

        file_path = sys.argv[2]

        plot_ablation_json(
            file_path,
            metric="mean_kl_full_to_masked",
            ascending=True,
            top_k=None,   # or e.g. 50
        )

        plot_ablation_json(
            file_path,
            metric="mean_ce_gap",
            ascending=True,
            top_k=None,   # or e.g. 50
        )

        # plot_ablation_json(
        #     file_path,
        #     metric="kl_per_removed_layer",
        #     ascending=True,
        #     top_k=None,   # or e.g. 50
        # )

        plot_ablation_json(
            file_path,
            metric="mean_top1_agreement",
            ascending=False,
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
        from skip_search_spec.inference.self_spec_inference import BridgeSelfSpeculator
        from skip_search_spec.inference.normal_inference import generate_normal
        from skip_search_spec.training.bridged_gap_model import BridgedGapModel
        import argparse
        import matplotlib.pyplot as plt

        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument(
            "--compare-to-normal",
            action="store_true",
            help="Compare self-spec output to normal generation.",
        )
        parser.add_argument(
            "--prompt-set",
            choices=(
                "completion-style",
                "chat-style",
                "concrete-completion-style",
                "swedish-concrete-completion-style",
            ),
            default="completion-style",
            help="Prompt set to run. completion-style preserves the old behavior.",
        )

        args, remaining_argv = parser.parse_known_args(sys.argv[2:])
        if len(remaining_argv) < 2:
            raise ValueError(
                "Must provide draft_block_size and bridge_checkpoint_path."
            )

        draft_block_size = remaining_argv[0]
        bridge_checkpoint_path = remaining_argv[1]
        flashhead_path = remaining_argv[2] if len(remaining_argv) > 2 else None
        prompt_sets = {
            "chat-style": (CHAT_TEST_PROMPTS, True),
            "completion-style": (INFERENCE_TEST_PROMPTS_EASY, False),
            "concrete-completion-style": (INFERENCE_TEST_PROMPTS_CONCRETE, False),
            "swedish-concrete-completion-style": (
                INFERENCE_TEST_PROMPTS_CONCRETE_SWEDISH,
                False,
            ),
        }
        test_prompts, use_chat_template = prompt_sets[args.prompt_set]

        total_inference_seconds = 0.0
        total_accept_rate = 0.0
        total_speedup_per_token = 0.0
        total_number_of_examples_ran = 0
        total_tokens_produced_self_spec = 0
        total_tokens_produced_normal = 0
        speedups_per_token = []
        number_exact_matches_between_self_spec_and_normal = 0

        bridged = BridgedGapModel.load_from_checkpoint(
            bridge_checkpoint_path=bridge_checkpoint_path,
            bridge_dtype=torch.float32,
        )
        speculator = BridgeSelfSpeculator(
            bridged_model=bridged,
            flashhead_path=flashhead_path,
            flashhead_top_k_clusters=100,
        )

        for test_idx, (test_name, prompt) in enumerate(test_prompts, start=1):
            print()
            print(f"Test {test_idx}: {test_name}")
            print()
            print("Prompt:")
            print(prompt)
            print()

            result = speculator.generate(
                prompt=prompt,
                max_new_tokens=INFERENCE_TEST_MAX_NEW_TOKENS,
                draft_block_size=int(draft_block_size),
                use_chat_template=use_chat_template,
                build_token_trace=False,
                measure_internal_timings=False
            )
            total_number_of_examples_ran += 1
            timings = result.timings
            total_inference_seconds += timings.total_seconds
            accept_rate = result.accepted_draft_tokens / max(result.drafted_tokens, 1)
            total_accept_rate += accept_rate
            total_tokens_produced_self_spec += result.num_generated_tokens

            print(result.text)
            print(
                {
                    "verifier_calls": result.verifier_calls,
                    "drafted_tokens": result.drafted_tokens,
                    "accepted_draft_tokens": result.accepted_draft_tokens,
                    "accept_rate": accept_rate,
                    "total_seconds": timings.total_seconds,
                    "dense_head_seconds": timings.dense_head_seconds,
                    "flashhead_seconds": timings.flashhead_seconds,
                    "drafter_registration_seconds": timings.drafter_registration_seconds,
                    "drafter_teardown_seconds": timings.drafter_teardown_seconds
                }
            )

            if args.compare_to_normal:
                print("Runs normal inference")
                normal_run_result = generate_normal(
                    prompt=prompt,
                    max_new_tokens=INFERENCE_TEST_MAX_NEW_TOKENS,
                    use_chat_template=use_chat_template,
                    use_cache=True,
                    model=bridged.model,
                    tokenizer=bridged.tokenizer,
                    device=bridged.device,
                )
                total_tokens_produced_normal += normal_run_result.num_generated_tokens
                did_match = result.text == normal_run_result.text
                if did_match:
                    number_exact_matches_between_self_spec_and_normal += 1
                print("Did match normal:", did_match)

                if not did_match:
                    first_mismatch_idx = next(
                        (
                            i
                            for i, (a, b) in enumerate(zip(result.text, normal_run_result.text))
                            if a != b
                        ),
                        min(len(result.text), len(normal_run_result.text)),
                    )

                    print("First text mismatch index:", first_mismatch_idx)
                    print("Self-spec from mismatch:", repr(result.text[first_mismatch_idx:first_mismatch_idx + 120]))
                    print("Normal from mismatch:", repr(normal_run_result.text[first_mismatch_idx:first_mismatch_idx + 120]))
                print("Speedup with self-spec:", normal_run_result.inference_seconds/timings.total_seconds)
               
                normal_tps = (
                    normal_run_result.num_generated_tokens
                    / normal_run_result.inference_seconds
                )

                self_spec_tps = (
                    result.num_generated_tokens
                    / result.timings.total_seconds
                )

                print("Normal tokens/sec:", normal_tps)
                print("Self-spec tokens/sec:", self_spec_tps)

                speedup_per_gen_token = self_spec_tps / normal_tps
                total_speedup_per_token += speedup_per_gen_token
                speedups_per_token.append(speedup_per_gen_token)
                print("Speedup per generated token:", speedup_per_gen_token)
               
        print()
        print({"total_inference_seconds": total_inference_seconds})
        print("Per example average acceptance rate:", total_accept_rate/total_number_of_examples_ran)
        print("Per example average speedup per token:", total_speedup_per_token/total_number_of_examples_ran)
        print("Total tokens produced by self-spec:", total_tokens_produced_self_spec)
        print("Total tokens produced by normal:", total_tokens_produced_normal)
        print("Ratio exact matches:", number_exact_matches_between_self_spec_and_normal/total_number_of_examples_ran)
        if speedups_per_token:
            plt.figure()
            plt.hist(speedups_per_token, bins=min(20, len(speedups_per_token)))
            plt.xlabel("Speedup per generated token")
            plt.ylabel("Number of prompts")
            plt.title("Distribution of self-spec speedups")
            plt.savefig("speedups_per_token_histogram.png", dpi=200, bbox_inches="tight")
            plt.close()

            print("Saved speedup histogram to speedups_per_token_histogram.png")



    elif mode == "test_normal_inference":
        from skip_search_spec.inference.normal_inference import generate_normal

        total_inference_seconds = 0.0

        for test_idx, (test_name, prompt) in enumerate(INFERENCE_TEST_PROMPTS_HARD, start=1):
            print()
            print(f"Test {test_idx}: {test_name}")
            print()
            print("Prompt:")
            print(prompt)
            print()

            result = generate_normal(
                model_name_or_path="meta-llama/Llama-3.2-3B",
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

    elif mode == "test_flashhead_inference":
        import argparse

        from skip_search_spec.inference.flashhead_inference import generate_with_flashhead
        from skip_search_spec.helpers.tooling import get_preferred_device, get_preferred_float_dtype, load_model_and_tokenizer
        from skip_search_spec.training.flashhead.next_token_adapter import FlashHeadModule

        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument(
            "--compare-to-normal",
            action="store_true",
            help="Compare flashhead to the same custom backbone loop with the dense LM head.",
        )

        args, remaining_argv = parser.parse_known_args(sys.argv[2:])

        model_name = remaining_argv[0] if len(remaining_argv) > 0 else MODEL_NAME_FLASH_HEAD
        flashhead_path = remaining_argv[1] if len(remaining_argv) > 1 else STORE_PATH_FLASH_HEAD
        flashhead_top_k_clusters = int(remaining_argv[2]) if len(remaining_argv) > 2 else 50

        device = get_preferred_device()
        dtype = get_preferred_float_dtype(device)
        mt = load_model_and_tokenizer(
            model_name,
            model_kwargs={"torch_dtype": dtype},
        )
        model = mt.model
        tokenizer = mt.tokenizer
        model.to(device)
        model.eval()

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        flashhead = FlashHeadModule.from_model(
            model=model,
            flashhead_path=flashhead_path,
            top_k_clusters=flashhead_top_k_clusters,
        )
        flashhead.eval()

        total_inference_seconds = 0.0
        total_flashhead_head_seconds = 0.0
        total_speedup_per_token = 0.0
        total_number_of_examples_ran = 0
        total_tokens_produced_flashhead = 0
        total_tokens_produced_dense_head = 0
        number_exact_matches_between_flashhead_and_dense_head = 0

        for test_idx, (test_name, prompt) in enumerate(CHAT_TEST_PROMPTS, start=1):
            print()
            print(f"Test {test_idx}: {test_name}")
            print()
            print("Prompt:")
            print(prompt)
            print()

            result = generate_with_flashhead(
                prompt=prompt,
                flashhead=flashhead,
                flashhead_top_k_clusters=flashhead_top_k_clusters,
                max_new_tokens=INFERENCE_TEST_MAX_NEW_TOKENS,
                use_chat_template=True,
                measure_internal_timings=True,
                head_mode="flashhead",
                model=model,
                tokenizer=tokenizer,
                device=device,
            )
            total_inference_seconds += result.inference_seconds
            total_flashhead_head_seconds += result.head_seconds
            total_number_of_examples_ran += 1
            total_tokens_produced_flashhead += result.num_generated_tokens

            print(result.text)
            print(
                {
                    "inference_seconds": result.inference_seconds,
                    "head_seconds": result.head_seconds,
                    "num_generated_tokens": result.num_generated_tokens,
                }
            )

            if args.compare_to_normal:
                print("Runs custom dense-head inference")
                dense_head_result = generate_with_flashhead(
                    prompt=prompt,
                    max_new_tokens=INFERENCE_TEST_MAX_NEW_TOKENS,
                    use_chat_template=True,
                    measure_internal_timings=True,
                    head_mode="lm_head",
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                )
                total_tokens_produced_dense_head += dense_head_result.num_generated_tokens
                did_match = result.text == dense_head_result.text
                if did_match:
                    number_exact_matches_between_flashhead_and_dense_head += 1
                print("Did match dense head:", did_match)

                if not did_match:
                    first_mismatch_idx = next(
                        (
                            i
                            for i, (a, b) in enumerate(zip(result.text, dense_head_result.text))
                            if a != b
                        ),
                        min(len(result.text), len(dense_head_result.text)),
                    )

                    print("First text mismatch index:", first_mismatch_idx)
                    print("Flashhead from mismatch:", repr(result.text[first_mismatch_idx:first_mismatch_idx + 120]))
                    print("Dense head from mismatch:", repr(dense_head_result.text[first_mismatch_idx:first_mismatch_idx + 120]))
                print(
                    {
                        "dense_head_inference_seconds": dense_head_result.inference_seconds,
                        "dense_head_head_seconds": dense_head_result.head_seconds,
                        "dense_head_num_generated_tokens": dense_head_result.num_generated_tokens,
                        "flashhead_inference_seconds": result.inference_seconds,
                        "flashhead_head_seconds": result.head_seconds,
                        "flashhead_num_generated_tokens": result.num_generated_tokens,
                    }
                )

                dense_head_tps = (
                    dense_head_result.num_generated_tokens
                    / dense_head_result.inference_seconds
                )

                flashhead_tps = (
                    result.num_generated_tokens
                    / result.inference_seconds
                )
                wall_clock_latency_ratio = (
                    dense_head_result.inference_seconds
                    / result.inference_seconds
                )
                speedup_per_generated_token = flashhead_tps / dense_head_tps

                print("Dense-head tokens/sec:", dense_head_tps)
                print("Flashhead tokens/sec:", flashhead_tps)
                print("Wall-clock latency ratio:", wall_clock_latency_ratio)
                print("Speedup per generated token:", speedup_per_generated_token)

                total_speedup_per_token += speedup_per_generated_token

        print()
        print(
            {
                "total_inference_seconds": total_inference_seconds,
                "total_flashhead_head_seconds": total_flashhead_head_seconds,
            }
        )
        print("Total tokens produced by flashhead:", total_tokens_produced_flashhead)
        print("Total tokens produced by dense head:", total_tokens_produced_dense_head)
        if args.compare_to_normal:
            print("Per example average speedup per token:", total_speedup_per_token/total_number_of_examples_ran)
            print("Ratio exact matches:", number_exact_matches_between_flashhead_and_dense_head/total_number_of_examples_ran)

    elif mode == "test_timed_normal_inference":
        import argparse

        from skip_search_spec.helpers.tooling import get_preferred_device, get_preferred_float_dtype, load_model_and_tokenizer
        from skip_search_spec.inference.timed_normal_inference import (
            TimedNormalInferenceOptions,
            TimedNormalInferenceTimings,
            generate_timed_normal,
        )

        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument(
            "--prompt-set",
            choices=(
                "completion-style",
                "chat-style",
                "hard-completion-style",
                "concrete-completion-style",
                "swedish-concrete-completion-style",
            ),
            default="chat-style",
            help="Prompt set to run.",
        )
        parser.add_argument(
            "--max-new-tokens",
            type=int,
            default=INFERENCE_TEST_MAX_NEW_TOKENS,
            help="Maximum number of new tokens per prompt.",
        )
        parser.add_argument(
            "--limit",
            type=int,
            default=None,
            help="Optional maximum number of prompts to run.",
        )
        parser.add_argument(
            "--measure-body",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Synchronize and time backbone/body forwards.",
        )
        parser.add_argument(
            "--measure-lm-head",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Synchronize and time LM-head calls.",
        )
        parser.add_argument(
            "--measure-token-selection",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Synchronize and time greedy argmax token selection.",
        )
        parser.add_argument(
            "--print-text",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Print generated text for each prompt.",
        )

        args, remaining_argv = parser.parse_known_args(sys.argv[2:])
        model_name = remaining_argv[0] if len(remaining_argv) > 0 else MODEL_NAME_FLASH_HEAD
        prompt_sets = {
            "chat-style": (CHAT_TEST_PROMPTS, True),
            "completion-style": (INFERENCE_TEST_PROMPTS_EASY, False),
            "hard-completion-style": (INFERENCE_TEST_PROMPTS_HARD, False),
            "concrete-completion-style": (INFERENCE_TEST_PROMPTS_CONCRETE, False),
            "swedish-concrete-completion-style": (
                INFERENCE_TEST_PROMPTS_CONCRETE_SWEDISH,
                False,
            ),
        }
        test_prompts, use_chat_template = prompt_sets[args.prompt_set]
        if args.limit is not None:
            test_prompts = test_prompts[:args.limit]

        timing_options = TimedNormalInferenceOptions(
            measure_body=args.measure_body,
            measure_lm_head=args.measure_lm_head,
            measure_token_selection=args.measure_token_selection,
        )

        device = get_preferred_device()
        dtype = get_preferred_float_dtype(device)
        mt = load_model_and_tokenizer(
            model_name,
            model_kwargs={"torch_dtype": dtype},
        )
        model = mt.model
        tokenizer = mt.tokenizer
        model.to(device)
        model.eval()

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        total_timings = TimedNormalInferenceTimings()
        total_generated_tokens = 0
        total_number_of_examples_ran = 0

        def enabled_timing_measurements(
            timings: TimedNormalInferenceTimings,
        ) -> dict[str, float | int]:
            measurements: dict[str, float | int] = {}
            if timing_options.measure_body:
                measurements["body_seconds"] = timings.body_seconds
                measurements["body_calls"] = timings.body_calls
                measurements["body_seconds_per_call"] = (
                    timings.body_seconds / max(timings.body_calls, 1)
                )
                measurements["body_share_of_total"] = (
                    timings.body_seconds / max(timings.total_seconds, 1e-12)
                )
            if timing_options.measure_lm_head:
                measurements["lm_head_seconds"] = timings.lm_head_seconds
                measurements["lm_head_calls"] = timings.lm_head_calls
                measurements["lm_head_seconds_per_call"] = (
                    timings.lm_head_seconds / max(timings.lm_head_calls, 1)
                )
                measurements["lm_head_share_of_total"] = (
                    timings.lm_head_seconds / max(timings.total_seconds, 1e-12)
                )
            if timing_options.measure_token_selection:
                measurements["token_selection_seconds"] = timings.token_selection_seconds
                measurements["token_selection_calls"] = timings.token_selection_calls
                measurements["token_selection_seconds_per_call"] = (
                    timings.token_selection_seconds
                    / max(timings.token_selection_calls, 1)
                )
                measurements["token_selection_share_of_total"] = (
                    timings.token_selection_seconds
                    / max(timings.total_seconds, 1e-12)
                )
            return measurements

        for test_idx, (test_name, prompt) in enumerate(test_prompts, start=1):
            print()
            print(f"Test {test_idx}: {test_name}")
            print()
            print("Prompt:")
            print(prompt)
            print()

            result = generate_timed_normal(
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                use_chat_template=use_chat_template,
                timing_options=timing_options,
                model=model,
                tokenizer=tokenizer,
                device=device,
            )

            total_number_of_examples_ran += 1
            total_generated_tokens += result.num_generated_tokens
            total_timings.total_seconds += result.timings.total_seconds
            total_timings.body_seconds += result.timings.body_seconds
            total_timings.lm_head_seconds += result.timings.lm_head_seconds
            total_timings.token_selection_seconds += result.timings.token_selection_seconds
            total_timings.body_calls += result.timings.body_calls
            total_timings.lm_head_calls += result.timings.lm_head_calls
            total_timings.token_selection_calls += result.timings.token_selection_calls

            if args.print_text:
                print(result.text)

            print(
                {
                    "inference_seconds": result.inference_seconds,
                    "num_generated_tokens": result.num_generated_tokens,
                    "tokens_per_sec": (
                        result.num_generated_tokens
                        / max(result.inference_seconds, 1e-12)
                    ),
                    **enabled_timing_measurements(result.timings),
                }
            )

        print()
        pprint(
            {
                "total_inference_seconds": total_timings.total_seconds,
                "total_generated_tokens": total_generated_tokens,
                "overall_tokens_per_sec": (
                    total_generated_tokens
                    / max(total_timings.total_seconds, 1e-12)
                ),
                "num_examples": total_number_of_examples_ran,
                **enabled_timing_measurements(total_timings),
            },
            sort_dicts=False,
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
