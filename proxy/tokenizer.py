"""Token accounting for BOTH protocols.

Two independent ledgers, exactly as required:

  * OpenAI protocol  (distribution layer)  -> counted here with tiktoken.
        These numbers are what we return to Cursor in the ``usage`` field.

  * Anthropic protocol (supplier)          -> taken verbatim from the upstream
        ``/v1/messages`` usage object (NOT recomputed). We only *merge* the
        usage that arrives across streaming events, because this upstream
        reports the true cost (incl. injected cached prompt) in ``message_delta``,
        not ``message_start``.
"""
import json
from typing import Any, Dict, List, Optional

import tiktoken

from .config import settings

# ---------------------------------------------------------------------------
# tiktoken encoding (lazy, with graceful fallback if the vocab can't download)
# ---------------------------------------------------------------------------
_enc = None
_enc_failed = False


def _encoding():
    global _enc, _enc_failed
    if _enc is None and not _enc_failed:
        for name in (settings.tokenizer_encoding, "o200k_base", "cl100k_base"):
            try:
                _enc = tiktoken.get_encoding(name)
                break
            except Exception:
                continue
        else:
            _enc_failed = True
    return _enc


def _count_str(s: Optional[str]) -> int:
    if not s:
        return 0
    enc = _encoding()
    if enc is None:  # offline fallback: ~4 chars/token
        return max(1, len(s) // 4)
    return len(enc.encode(s, disallowed_special=()))


# ---------------------------------------------------------------------------
# OpenAI protocol counting (distribution layer)
# ---------------------------------------------------------------------------
# Per the OpenAI cookbook for gpt-4o / gpt-4-0613+ chat models.
_TOKENS_PER_MESSAGE = 3
_TOKENS_PER_NAME = 1
_REPLY_PRIMING = 3
_PER_TOOL_CALL = 3


def _count_content(content: Any) -> int:
    if content is None:
        return 0
    if isinstance(content, str):
        return _count_str(content)
    if isinstance(content, list):
        total = 0
        for part in content:
            if not isinstance(part, dict):
                total += _count_str(str(part))
                continue
            ptype = part.get("type")
            if ptype in ("text", "input_text"):
                total += _count_str(part.get("text", ""))
            elif ptype in ("image_url", "image", "input_image"):
                total += settings.image_tokens_each
            elif "text" in part:
                total += _count_str(part.get("text", ""))
            else:
                total += _count_str(json.dumps(part, ensure_ascii=False))
        return total
    return _count_str(str(content))


def _count_tool_schema(schema: Any) -> int:
    """Approximate token cost of a JSON-Schema parameters block.

    NOTE: OpenAI does not document how function/tool schemas are tokenised; even
    OpenAI's own server count is a heuristic. This approximation is *consistent*
    (good for billing), but treat tool-schema tokens as an estimate, unlike
    message/content tokens which are exact.
    """
    if not isinstance(schema, dict):
        return _count_str(json.dumps(schema, ensure_ascii=False))
    total = 0
    for name, pdef in (schema.get("properties") or {}).items():
        total += _count_str(name) + 2
        if isinstance(pdef, dict):
            if pdef.get("type"):
                total += _count_str(str(pdef["type"])) + 2
            if pdef.get("description"):
                total += _count_str(str(pdef["description"]))
            for enum_val in pdef.get("enum") or []:
                total += _count_str(str(enum_val)) + 3
            if pdef.get("type") == "object":
                total += _count_tool_schema(pdef)
            items = pdef.get("items")
            if pdef.get("type") == "array" and isinstance(items, dict):
                total += _count_tool_schema(items)
    return total


def count_tools_tokens(tools: Optional[List[dict]]) -> int:
    if not tools:
        return 0
    total = 12  # overall wrapper (approx)
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        total += _count_str(fn.get("name", ""))
        total += _count_str(fn.get("description", ""))
        # schema may be under `parameters` (OpenAI) or `input_schema` (Cursor/Anthropic-style)
        schema = fn.get("parameters") or fn.get("input_schema") or tool.get("input_schema") or {}
        total += _count_tool_schema(schema)
        total += 12  # per-function wrapper (approx)
    return total


def count_openai_prompt_tokens(
    messages: List[dict],
    tools: Optional[List[dict]] = None,
    tool_choice: Any = None,
) -> int:
    total = 0
    for msg in messages or []:
        total += _TOKENS_PER_MESSAGE
        total += _count_str(msg.get("role", ""))
        if msg.get("name"):
            total += _count_str(msg["name"]) + _TOKENS_PER_NAME
        total += _count_content(msg.get("content"))
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function", {})
            total += _count_str(fn.get("name", ""))
            total += _count_str(fn.get("arguments", "") or "")
            total += _PER_TOOL_CALL
        if msg.get("tool_call_id"):
            total += _count_str(msg["tool_call_id"])
    total += _REPLY_PRIMING
    total += count_tools_tokens(tools)
    return total


def count_openai_completion_tokens(content: Optional[str], tool_calls: Optional[List[dict]] = None,
                                   reasoning: Optional[str] = None) -> int:
    total = _count_str(content) + _count_str(reasoning)
    for tc in tool_calls or []:
        fn = tc.get("function", {})
        total += _count_str(fn.get("name", ""))
        total += _count_str(fn.get("arguments", "") or "")
        total += _PER_TOOL_CALL
    return total


def count_text(s: Optional[str]) -> int:
    """Token count of a plain string (e.g. reasoning content), via the OpenAI tokenizer."""
    return _count_str(s)


# ---------------------------------------------------------------------------
# Anthropic protocol accounting (supplier) -- taken from upstream, only merged
# ---------------------------------------------------------------------------
_ANTHROPIC_KEYS = (
    "input_tokens",
    "output_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
)


def new_anthropic_usage() -> Dict[str, int]:
    return {k: 0 for k in _ANTHROPIC_KEYS}


def merge_anthropic_usage(acc: Dict[str, int], usage: Optional[dict]) -> Dict[str, int]:
    """Last-non-null-wins merge across streaming events.

    Anthropic streaming reports cumulative/authoritative values; this upstream in
    particular only reveals the cached-prompt cost in ``message_delta``. Taking the
    latest value for each key captures that, where ``message_start`` alone would not.
    """
    if not usage:
        return acc
    for key in _ANTHROPIC_KEYS:
        val = usage.get(key)
        if isinstance(val, (int, float)):
            acc[key] = int(val)
    return acc


def anthropic_billable_input(usage: Dict[str, int]) -> int:
    """Total input-side tokens the supplier saw (uncached + cache write + cache read)."""
    return (
        usage.get("input_tokens", 0)
        + usage.get("cache_creation_input_tokens", 0)
        + usage.get("cache_read_input_tokens", 0)
    )


def anthropic_cost_usd(usage: Dict[str, int]) -> Optional[float]:
    s = settings
    if not any((s.price_input, s.price_output, s.price_cache_write, s.price_cache_read)):
        return None
    cost = (
        usage.get("input_tokens", 0) / 1e6 * s.price_input
        + usage.get("output_tokens", 0) / 1e6 * s.price_output
        + usage.get("cache_creation_input_tokens", 0) / 1e6 * s.price_cache_write
        + usage.get("cache_read_input_tokens", 0) / 1e6 * s.price_cache_read
    )
    return round(cost, 6)
