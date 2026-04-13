



from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TokenizedWindow:
    token_ids: list[int]
    T_len: int
    dataset_spec: DatasetSpec



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