from __future__ import annotations

from typing import Any, Callable

from datasets import Dataset

from skip_search_spec.protocols.windows import DatasetSpec


DatasetRowFormatter = Callable[[dict[str, Any]], str | None]


def _clean(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    return value.strip()


def _first_nonempty(row: dict[str, Any], keys: list[str]) -> str:
    for key in keys:
        value = _clean(row.get(key))
        if value:
            return value
    return ""


def format_instruction_qa_row(row: dict[str, Any]) -> str | None:
    """
    Flexible formatter for instruction / Q&A datasets.

    Works with common schemas such as:
      - instruction, context, response
      - question, answer
      - prompt, completion
      - input, output
    """
    instruction = _first_nonempty(
        row,
        [
            "instruction",
            "question",
            "prompt",
            "query",
            "problem",
            "input",
        ],
    )
    context = _first_nonempty(
        row,
        [
            "context",
            "passage",
            "document",
            "source",
        ],
    )
    answer = _first_nonempty(
        row,
        [
            "response",
            "answer",
            "output",
            "completion",
            "solution",
        ],
    )

    if not instruction or not answer:
        return None

    if context:
        return (
            f"Instruction:\n"
            f"{instruction}\n\n"
            f"Context:\n"
            f"{context}\n\n"
            f"Answer:\n"
            f"{answer}"
        )

    return (
        f"Instruction:\n"
        f"{instruction}\n\n"
        f"Answer:\n"
        f"{answer}"
    )


def format_math_qa_row(row: dict[str, Any]) -> str | None:
    """
    Formatter for math datasets.

    Works with schemas such as:
      - query, response
      - problem, solution
      - question, answer
    """
    problem = _first_nonempty(
        row,
        [
            "problem",
            "question",
            "query",
            "instruction",
            "prompt",
        ],
    )
    solution = _first_nonempty(
        row,
        [
            "solution",
            "answer",
            "response",
            "output",
            "completion",
        ],
    )

    if not problem or not solution:
        return None

    return (
        f"Problem:\n"
        f"{problem}\n\n"
        f"Solution:\n"
        f"{solution}"
    )


DATASET_ROW_FORMATTERS_BY_NAME: dict[str, DatasetRowFormatter] = {
    "Dolly-15k-formatted": format_instruction_qa_row,
    "MetaMathQA-40K-formatted": format_math_qa_row,
}


def maybe_format_dataset_to_text(
    dataset: Dataset,
    dataset_spec: DatasetSpec,
) -> Dataset:
    """
    Convert structured datasets into a single text column when needed.

    If no formatter is registered for the dataset name, returns the dataset
    unchanged.

    The formatted column name is dataset_spec.text_field.
    """
    formatter = DATASET_ROW_FORMATTERS_BY_NAME.get(dataset_spec.name)
    if formatter is None:
        return dataset

    output_field = dataset_spec.text_field

    def format_batch(batch: dict[str, list[Any]]) -> dict[str, list[str]]:
        num_rows = len(next(iter(batch.values()))) if batch else 0
        texts: list[str] = []

        for row_idx in range(num_rows):
            row = {
                key: values[row_idx]
                for key, values in batch.items()
            }

            text = formatter(row)
            texts.append(text or "")

        return {output_field: texts}

    formatted = dataset.map(
        format_batch,
        batched=True,
        desc=f"Formatting {dataset_spec.name} into '{output_field}'",
    )

    columns_to_remove = [
        column
        for column in formatted.column_names
        if column != output_field
    ]

    if columns_to_remove:
        formatted = formatted.remove_columns(columns_to_remove)

    return formatted