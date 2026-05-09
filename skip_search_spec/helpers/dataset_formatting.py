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


# def format_instruction_qa_row(row: dict[str, Any]) -> str | None:
#     """
#     Flexible formatter for instruction / Q&A datasets.

#     Works with common schemas such as:
#       - instruction, context, response
#       - question, answer
#       - prompt, completion
#       - input, output
#     """
#     instruction = _first_nonempty(
#         row,
#         [
#             "instruction",
#             "question",
#             "prompt",
#             "query",
#             "problem",
#             "input",
#         ],
#     )
#     context = _first_nonempty(
#         row,
#         [
#             "context",
#             "passage",
#             "document",
#             "source",
#         ],
#     )
#     answer = _first_nonempty(
#         row,
#         [
#             "response",
#             "answer",
#             "output",
#             "completion",
#             "solution",
#         ],
#     )

#     if not instruction or not answer:
#         return None

#     if context:
#         return (
#             f"Instruction:\n"
#             f"{instruction}\n\n"
#             f"Context:\n"
#             f"{context}\n\n"
#             f"Answer:\n"
#             f"{answer}"
#         )

#     return (
#         f"Instruction:\n"
#         f"{instruction}\n\n"
#         f"Answer:\n"
#         f"{answer}"
#     )


# def format_math_qa_row(row: dict[str, Any]) -> str | None:
#     """
#     Formatter for math datasets.

#     Works with schemas such as:
#       - query, response
#       - problem, solution
#       - question, answer
#     """
#     problem = _first_nonempty(
#         row,
#         [
#             "problem",
#             "question",
#             "query",
#             "instruction",
#             "prompt",
#         ],
#     )
#     solution = _first_nonempty(
#         row,
#         [
#             "solution",
#             "answer",
#             "response",
#             "output",
#             "completion",
#         ],
#     )

#     if not problem or not solution:
#         return None

#     return (
#         f"Problem:\n"
#         f"{problem}\n\n"
#         f"Solution:\n"
#         f"{solution}"
#     )


# def format_sciq_row(row: dict[str, Any]) -> str | None:
#     question = _clean(row.get("question"))
#     support = _clean(row.get("support"))
#     answer = _clean(row.get("correct_answer"))

#     if not question or not answer:
#         return None

#     if support:
#         return (
#             f"Science question:\n{question}\n\n"
#             f"Context:\n{support}\n\n"
#             f"Answer:\n{answer}"
#         )

#     return f"Science question:\n{question}\n\nAnswer:\n{answer}"


# def format_squad_row(row: dict[str, Any]) -> str | None:
#     title = _clean(row.get("title"))
#     context = _clean(row.get("context"))
#     question = _clean(row.get("question"))
#     answers = row.get("answers")

#     answer = ""
#     if isinstance(answers, dict):
#         texts = answers.get("text")
#         if isinstance(texts, list) and texts:
#             answer = _clean(texts[0])

#     if not context or not question or not answer:
#         return None

#     header = f"Article: {title}\n\n" if title else ""
#     return (
#         f"{header}"
#         f"Passage:\n{context}\n\n"
#         f"Question:\n{question}\n\n"
#         f"Answer:\n{answer}"
#     )


# def format_gsm8k_row(row: dict[str, Any]) -> str | None:
#     question = _clean(row.get("question"))
#     answer = _clean(row.get("answer"))

#     if not question or not answer:
#         return None

#     return f"Problem:\n{question}\n\nSolution:\n{answer}"


# def format_mbpp_row(row: dict[str, Any]) -> str | None:
#     text = _clean(row.get("text"))
#     code = _clean(row.get("code"))
#     test_list = row.get("test_list")

#     tests = ""
#     if isinstance(test_list, list) and test_list:
#         tests = "\n".join(str(x) for x in test_list)

#     if not text or not code:
#         return None

#     if tests:
#         return (
#             f"Python task:\n{text}\n\n"
#             f"Tests:\n{tests}\n\n"
#             f"Solution:\n```python\n{code}\n```"
#         )

#     return f"Python task:\n{text}\n\nSolution:\n```python\n{code}\n```"


# def format_dialogsum_row(row: dict[str, Any]) -> str | None:
#     dialogue = _clean(row.get("dialogue"))
#     summary = _clean(row.get("summary"))

#     if not dialogue or not summary:
#         return None

#     return f"Dialogue:\n{dialogue}\n\nSummary:\n{summary}"

# def format_openorca_row(row: dict[str, Any]) -> str | None:
#     system_prompt = _clean(row.get("system_prompt"))
#     question = _clean(row.get("question"))
#     response = _clean(row.get("response"))

#     if not question or not response:
#         return None

#     if system_prompt:
#         return (
#             "### System\n"
#             f"{system_prompt}\n\n"
#             "### Instruction\n"
#             f"{question}\n\n"
#             "### Response\n"
#             f"{response}"
#         )

#     return (
#         "### Instruction\n"
#         f"{question}\n\n"
#         "### Response\n"
#         f"{response}"
#     )

def format_lamini_row(row: dict[str, Any]) -> str | None:
    instruction = _first_nonempty(
        row,
        ["instruction", "question", "prompt", "input"],
    )
    response = _first_nonempty(
        row,
        ["response", "answer", "output", "completion"],
    )

    if not instruction or not response:
        return None

    return (
        "# Instruction\n"
        f"{instruction}\n\n"
        "# Response\n"
        f"{response}"
    )



DATASET_ROW_FORMATTERS_BY_NAME: dict[str, DatasetRowFormatter] = {
    # "Dolly-15k-formatted": format_instruction_qa_row,
    # "MetaMathQA-40K-formatted": format_math_qa_row,
    # "Alpaca-52k-formatted": format_instruction_qa_row,
    # "SciQ-formatted": format_sciq_row,
    # "SQuAD-formatted": format_squad_row,
    # "GSM8K-formatted": format_gsm8k_row,
    # "MBPP-formatted": format_mbpp_row,
    # "DialogSum-formatted": format_dialogsum_row,
    # "OpenOrca-formatted": format_openorca_row,
    "LaMini-instruction-formatted": format_lamini_row,
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