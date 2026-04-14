



from dataclasses import dataclass

from transformers import PreTrainedModel, PreTrainedTokenizerBase





@dataclass(frozen=True, slots=True)
class DatasetSpec:
    """Describe a dataset source and its default load settings."""

    name: str
    huggingface_path: str
    split: str = "train"
    config_name: str | None = None
    text_field: str = "text"

    def __str__(self) -> str:
        target = self.huggingface_path
        if self.config_name is not None:
            target = f"{target}/{self.config_name}"
        return f"{self.name}<{target}:{self.split}>"
    
@dataclass(frozen=True, slots=True)
class TokenizedWindow:
    token_ids: list[int]
    dataset_spec: DatasetSpec

    

@dataclass(frozen=True, slots=True)
class ModelAndTokenizer:
    """Compact holder for a loaded Hugging Face model and tokenizer."""

    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase



@dataclass(frozen=True, slots=True)
class WindowSettings:
    # C1 is the length of the window, the number of tokens it will contain.
    # The reason for why it is called C1 is because previous projects where it made sense
    C1: int
