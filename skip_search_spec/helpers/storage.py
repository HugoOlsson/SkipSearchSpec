from __future__ import annotations


from pathlib import Path
from typing import TYPE_CHECKING, Any
import torch
from skip_search_spec.helpers.tooling import load_model_and_tokenizer
from skip_search_spec.protocols.windows import ModelAndTokenizer
from typing import Any, cast

if TYPE_CHECKING:
    from skip_search_spec.training.old.train_early_exit import EarlyExitModel


def save_early_exit_checkpoint(
    early_exit_model: EarlyExitModel,
    checkpoint_path: str,
    base_model_name: str,
    optimizer: torch.optim.Optimizer | None = None,
) -> None:
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint: dict[str, Any] = {
        "model_state_dict": early_exit_model.state_dict(),
        "base_model_name": base_model_name,
        "inner_exit_layer_index": early_exit_model.inner_exit_layer_index,
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    torch.save(checkpoint, path)



def load_early_exit_checkpoint(
    checkpoint_path: str,
    device: torch.device,
    compute_dtype: torch.dtype,
) -> tuple[EarlyExitModel, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    base_model_name = cast(str, checkpoint["base_model_name"])
    inner_exit_layer_index = cast(int, checkpoint["inner_exit_layer_index"])

    model_and_tokenizer: ModelAndTokenizer = load_model_and_tokenizer(
        base_model_name,
        model_kwargs={"torch_dtype": compute_dtype},
    )

    base_model = cast(Any, model_and_tokenizer.model)
    base_model.to(device=device, dtype=compute_dtype)

    early_exit_model = EarlyExitModel(
        base_model=base_model,
        inner_exit_layer_index=inner_exit_layer_index,
    )
    early_exit_model.load_state_dict(checkpoint["model_state_dict"])
    early_exit_model.to(device=device, dtype=compute_dtype)

    return early_exit_model, model_and_tokenizer.tokenizer