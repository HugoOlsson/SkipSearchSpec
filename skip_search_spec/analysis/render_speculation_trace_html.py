from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


def _style_for_token(token: dict[str, Any]) -> str:
    token_type = token["type"]
    status = token.get("status")

    if token_type == "prompt":
        return "background: rgba(120, 120, 120, 0.18);"

    if token_type == "bonus":
        return "background: rgba(80, 140, 255, 0.20);"

    if token_type == "draft" and status == "accepted":
        return "background: rgba(50, 200, 90, 0.25);"

    if token_type == "draft" and status == "rejected":
        return (
            "background: rgba(240, 70, 70, 0.25); "
        )

    return "background: rgba(240, 200, 80, 0.25);"


def render_speculation_trace_html(
    *,
    trace_json_path: str | Path,
    output_html_path: str | Path | None = None,
) -> Path:
    trace_json_path = Path(trace_json_path)

    if output_html_path is None:
        output_html_path = trace_json_path.with_suffix(".html")

    output_html_path = Path(output_html_path)

    with trace_json_path.open("r", encoding="utf-8") as f:
        trace = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(
        trace["tokenizer_name"],
        trust_remote_code=True,
    )

    spans: list[str] = []

    for token in trace["tokens"]:
        token_id = int(token["token_id"])

        text = tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

        style = _style_for_token(token)

        title = (
            f"token_id={token_id} | "
            f"type={token.get('type')} | "
            f"status={token.get('status')} | "
            f"draft_block={token.get('draft_block_index')}"
        )

        for char in text:
            spans.append(
                f'<span style="{style}" title="{html.escape(title, quote=True)}">'
                f"{html.escape(char)}"
                f"</span>"
            )

    html_doc = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Speculation Trace</title>
  <style>
    body {{
      margin: 40px;
      font-family: system-ui, sans-serif;
      line-height: 1.7;
    }}

    p {{
      white-space: pre-wrap;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      font-size: 15px;
    }}
  </style>
</head>
<body>
  <h1>Speculation Trace</h1>

  <p>
<span style="background: rgba(120, 120, 120, 0.18);">prompt</span>
<span style="background: rgba(80, 140, 255, 0.20);">bonus/verifier</span>
<span style="background: rgba(50, 200, 90, 0.25);">accepted draft</span>
<span style="background: rgba(240, 70, 70, 0.25);">rejected draft</span>
  </p>

  <p>{"".join(spans)}</p>
</body>
</html>
"""

    output_html_path.parent.mkdir(parents=True, exist_ok=True)

    with output_html_path.open("w", encoding="utf-8") as f:
        f.write(html_doc)

    return output_html_path