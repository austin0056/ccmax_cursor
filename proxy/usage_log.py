"""Per-request dual-protocol usage logging.

Writes one JSON line per request to USAGE_LOG_PATH and prints a concise summary
to stderr. This is the artifact you reconcile billing against:

  openai_protocol     -> what the distribution layer charges (tiktoken)
  anthropic_protocol  -> what the supplier charges (upstream usage, incl. cache)
"""
import json
import sys
import time
from typing import Dict

from .config import settings
from . import tokenizer as tk


def log_usage(request_id: str, model: str, openai_usage: Dict[str, int],
              anthropic_usage: Dict[str, int], stream: bool) -> None:
    record = {
        "ts": int(time.time()),
        "request_id": request_id,
        "model": model,
        "stream": stream,
        "openai_protocol": openai_usage,
        "anthropic_protocol": {
            **anthropic_usage,
            "billable_input_total": tk.anthropic_billable_input(anthropic_usage),
        },
    }
    cost = tk.anthropic_cost_usd(anthropic_usage)
    if cost is not None:
        record["supplier_cost_usd"] = cost

    try:
        with open(settings.usage_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        pass

    au = anthropic_usage
    summary = (
        f"[usage] {model} stream={int(stream)} | "
        f"OpenAI(prompt={openai_usage['prompt_tokens']},"
        f"compl={openai_usage['completion_tokens']},"
        f"total={openai_usage['total_tokens']}) | "
        f"Anthropic(in={au['input_tokens']},out={au['output_tokens']},"
        f"cache_w={au['cache_creation_input_tokens']},"
        f"cache_r={au['cache_read_input_tokens']})"
    )
    if cost is not None:
        summary += f" | supplier=${cost}"
    print(summary, file=sys.stderr, flush=True)
