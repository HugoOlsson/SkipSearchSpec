



from pathlib import Path
from typing import Any, cast

from datasets.arrow_dataset import Dataset
import torch
from torch import nn
from torch.utils.data import Dataset as TorchDataset, DataLoader
import torch.nn.functional as F
from transformers import PreTrainedModel

from skip_search_spec.helpers.storage import save_early_exit_checkpoint
from skip_search_spec.helpers.tooling import assert_same_tokenizer, distribution_similarity_metrics, get_preferred_device, get_preferred_float_dtype, load_dataset, load_model_and_tokenizer
from skip_search_spec.helpers.window_building import WindowDataset, build_all_training_windows, collate_windows, tokenize_dataset_to_examples
from skip_search_spec.protocols.windows import DatasetSpec, ModelAndTokenizer, WindowSettings




def train_drafter_for_verifier(*,draft_model_name: str, 
                               verifier_model_name: str, 
                               dataset_spec: DatasetSpec, 
                               window_size: int, 
                               number_of_layers_allowed_to_change: int | None = None,
                               batch_size: int = 4,
                               max_examples: int = 10_000):

    device = get_preferred_device()
    compute_dtype = get_preferred_float_dtype(device)

    draft_model_and_tokenizer: ModelAndTokenizer = load_model_and_tokenizer(
        draft_model_name,
        model_kwargs={"torch_dtype": compute_dtype},
    )

    verifier_model_and_tokenizer: ModelAndTokenizer = load_model_and_tokenizer(
        verifier_model_name,
        model_kwargs={"torch_dtype": compute_dtype},
    )

    assert_same_tokenizer(draft_model_and_tokenizer.tokenizer, verifier_model_and_tokenizer.tokenizer)

    window_tokenization_tokenizer = draft_model_and_tokenizer.tokenizer
    dataset: Dataset = load_dataset(dataset_spec)
    context_parts = WindowSettings(C1=window_size)

    tokenized_examples = tokenize_dataset_to_examples(
        dataset,
        window_tokenization_tokenizer,
        dataset_spec,
        max_examples=max_examples,
    )

    training_windows = build_all_training_windows(
        tokenized_examples,
        context_parts,
        dataset_spec,
    )



    total_layers = len(draft_model_and_tokenizer.model.model.layers)

    if number_of_layers_allowed_to_change is None:
        start = 0
    else:
        if number_of_layers_allowed_to_change <= 0:
            raise ValueError("number_of_layers_allowed_to_change must be positive or None.")
        if number_of_layers_allowed_to_change > total_layers:
            raise ValueError(
                f"number_of_layers_allowed_to_change={number_of_layers_allowed_to_change} "
                f"but model only has {total_layers} layers."
            )
        start = total_layers - number_of_layers_allowed_to_change

    # Freeze everything first
    for p in draft_model_and_tokenizer.model.parameters():
        p.requires_grad = False

    # Unfreeze only the last N layers
    for layer in draft_model_and_tokenizer.model.model.layers[start:]:
        for p in layer.parameters():
            p.requires_grad = True

    # Also unfreeze the LM head and final layernorm if you want the output projection to adapt
    for p in draft_model_and_tokenizer.model.lm_head.parameters():
        p.requires_grad = True
    for p in draft_model_and_tokenizer.model.model.norm.parameters():
        p.requires_grad = True


    trainable_params = [p for p in draft_model_and_tokenizer.model.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(
        [{"params": trainable_params, "lr": 5e-5}],
    )

    window_dataset = WindowDataset(training_windows)
    dataloader = DataLoader(
        window_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_windows,
        pin_memory=(device.type == "cuda"),
    )

    print("STARTING DRAFTER-FOR-VERIFIER TRAINING")
    print(f"  draft_model={draft_model_name}")
    print(f"  verifier_model={verifier_model_name}")
    print(f"  dataset={dataset_spec}")
    print(f"  device={device}")
    print(f"  compute_dtype={compute_dtype}")
    print(f"  context: C1={context_parts.C1}")
    print(f"  total_windows={len(training_windows)}")
    print(f"  batch_size={batch_size}")
    print(f"  steps_per_epoch={len(dataloader)}")
    print(f"  total_layers={total_layers}, trainable_layers={number_of_layers_allowed_to_change} (layers {start}–{total_layers-1})")


    cast(Any, draft_model_and_tokenizer.model).to(device=device, dtype=compute_dtype)
    cast(Any, verifier_model_and_tokenizer.model).to(device=device, dtype=compute_dtype)

    draft_model_and_tokenizer.model.train()
    verifier_model_and_tokenizer.model.eval()

    ce_draft: torch.Tensor | None = None
    ce_verifier: torch.Tensor | None = None

    grad_accum_steps = 8

    optimizer.zero_grad(set_to_none=True)
    
    for step, (input_ids, attention_mask) in enumerate(dataloader):
        input_ids = input_ids.to(device, non_blocking=True)
        attention_mask = attention_mask.to(device, non_blocking=True)
        labels = input_ids

        

        # Draft forward (gradients flow)
        draft_outputs = draft_model_and_tokenizer.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        logits_draft = draft_outputs.logits

        # Verifier forward (no gradients)
        with torch.no_grad():
            verifier_outputs = verifier_model_and_tokenizer.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            logits_verifier = verifier_outputs.logits

        shift_logits_draft = logits_draft[:, :-1, :].contiguous()
        shift_logits_verifier = logits_verifier[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        T = 2.0

        log_p_draft = F.log_softmax(shift_logits_draft.float() / T, dim=-1)
        log_p_verifier = F.log_softmax(shift_logits_verifier.float() / T, dim=-1)

        kl_per_token = F.kl_div(
            log_p_draft,
            log_p_verifier,
            reduction="none",
            log_target=True,
        ).sum(dim=-1)

        kl_verifier_to_draft = (T * T) * kl_per_token.mean()

        ce_draft = F.cross_entropy(
            shift_logits_draft.view(-1, shift_logits_draft.size(-1)),
            shift_labels.view(-1),
        )
        ce_verifier = F.cross_entropy(
            shift_logits_verifier.view(-1, shift_logits_verifier.size(-1)),
            shift_labels.view(-1),
        )

        loss = kl_verifier_to_draft / grad_accum_steps

        loss.backward()
        if (step + 1) % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if step % 10 == 0:
            with torch.no_grad():
                metrics = distribution_similarity_metrics(
                    shift_logits_mid=shift_logits_draft,
                    shift_logits_full=shift_logits_verifier,
                )

            print(
                f"[{step}/{len(dataloader)}] "
                f"loss={loss.item():.4f}  "
                f"ce_draft={ce_draft.item():.4f}  "
                f"ce_verifier={ce_verifier.item():.4f}  "
                f"gap={(ce_draft.item() - ce_verifier.item()):.4f}  "
                f"kl(verifier||draft)={kl_verifier_to_draft.item():.4f}  "
                f"js={metrics['js'].item():.4f}  "
                f"top1_agree={metrics['top1_agreement'].item():.4f}  "
                f"overlap={metrics['overlap'].item():.4f}  "
                f"p_draft@verifier_argmax={metrics['p_mid_on_full_argmax'].item():.4f}"
            )
