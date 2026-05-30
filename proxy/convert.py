"""Bidirectional format conversion: OpenAI Chat Completions <-> Anthropic Messages.

  request:   OpenAI body            -> Anthropic /v1/messages body
  response:  Anthropic message      -> OpenAI chat.completion
  streaming: Anthropic SSE events   -> OpenAI chat.completion.chunk SSE
"""
import json
import uuid
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Tuple

from . import debug
from . import tokenizer as tk
from .config import settings

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def gen_id(prefix: str = "chatcmpl") -> str:
    return f"{prefix}-{uuid.uuid4().hex}"


_STOP_MAP = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
    "pause_turn": "stop",
    "refusal": "stop",
}


def map_finish_reason(anthropic_stop: Optional[str]) -> str:
    return _STOP_MAP.get(anthropic_stop or "", "stop")


# ---------------------------------------------------------------------------
# request:  OpenAI -> Anthropic
# ---------------------------------------------------------------------------

def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") in ("text", "input_text") or "text" in part:
                    out.append(part.get("text", ""))
            else:
                out.append(str(part))
        return "".join(out)
    return str(content)


def _image_block_from_url(url: Any) -> dict:
    if isinstance(url, str) and url.startswith("data:"):
        try:
            header, b64 = url.split(",", 1)
            media_type = header.split(";")[0].split(":", 1)[1]
            return {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}}
        except Exception:
            pass
    return {"type": "image", "source": {"type": "url", "url": url}}


def _openai_content_to_blocks(content: Any) -> Any:
    """User content -> Anthropic content (string kept as string; arrays -> blocks)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        blocks = []
        for part in content:
            if not isinstance(part, dict):
                blocks.append({"type": "text", "text": str(part)})
                continue
            ptype = part.get("type")
            if ptype in ("text", "input_text"):
                blocks.append({"type": "text", "text": part.get("text", "")})
            elif ptype == "image_url":
                image_url = part.get("image_url")
                url = image_url.get("url") if isinstance(image_url, dict) else image_url
                blocks.append(_image_block_from_url(url))
            elif ptype == "image" and "source" in part:
                blocks.append(part)
            else:
                blocks.append({"type": "text", "text": json.dumps(part, ensure_ascii=False)})
        return blocks or ""
    return str(content)


def _convert_tool_choice(tool_choice: Any) -> dict:
    if tool_choice is None or tool_choice == "auto":
        return {"type": "auto"}
    if tool_choice == "required":
        return {"type": "any"}
    if tool_choice == "none":
        # Anthropic has no universal "forbid"; auto is the safe best-effort.
        return {"type": "auto"}
    if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        return {"type": "tool", "name": (tool_choice.get("function") or {}).get("name")}
    return {"type": "auto"}


def openai_to_anthropic_request(body: dict, upstream_model: str = "") -> dict:
    messages = body.get("messages") or []
    system_parts: List[str] = []
    a_messages: List[dict] = []
    pending_tool_results: List[dict] = []

    def flush_tool_results():
        if pending_tool_results:
            a_messages.append({"role": "user", "content": list(pending_tool_results)})
            pending_tool_results.clear()

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")

        if role in ("system", "developer"):
            flush_tool_results()
            text = _content_to_text(content)
            if text:
                system_parts.append(text)
            continue

        if role == "tool":
            pending_tool_results.append({
                "type": "tool_result",
                "tool_use_id": msg.get("tool_call_id", ""),
                "content": _content_to_text(content),
            })
            continue

        flush_tool_results()

        if role == "assistant":
            blocks: List[dict] = []
            text = _content_to_text(content)
            if text:
                blocks.append({"type": "text", "text": text})
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function", {})
                try:
                    parsed = json.loads(fn.get("arguments") or "{}")
                except Exception:
                    parsed = {}
                blocks.append({
                    "type": "tool_use",
                    "id": tc.get("id", "") or gen_id("toolu"),
                    "name": fn.get("name", ""),
                    "input": parsed,
                })
            a_messages.append({"role": "assistant", "content": blocks or [{"type": "text", "text": ""}]})
        else:  # user (or unknown -> user)
            a_messages.append({"role": "user", "content": _openai_content_to_blocks(content)})

    flush_tool_results()

    out: Dict[str, Any] = {
        "model": upstream_model or body.get("model"),
        "messages": a_messages,
        "max_tokens": int(
            body.get("max_tokens") or body.get("max_completion_tokens") or 0
        ) or _default_max_tokens(),
    }
    system_text = "\n\n".join(p for p in system_parts if p)
    if settings.system_suffix:
        system_text = f"{system_text}\n\n{settings.system_suffix}".strip()
    if system_text:
        out["system"] = system_text
    if body.get("temperature") is not None:
        out["temperature"] = max(0.0, min(1.0, float(body["temperature"])))  # Anthropic caps at 1.0
    if body.get("top_p") is not None:
        out["top_p"] = float(body["top_p"])
    stop = body.get("stop")
    if stop:
        out["stop_sequences"] = [stop] if isinstance(stop, str) else list(stop)
    if body.get("stream"):
        out["stream"] = True
    tools = body.get("tools")
    if tools:
        out["tools"] = [
            {
                "name": (t.get("function", t) or {}).get("name"),
                "description": (t.get("function", t) or {}).get("description", "") or "",
                "input_schema": (t.get("function", t) or {}).get("parameters")
                or {"type": "object", "properties": {}},
            }
            for t in tools
        ]
        out["tool_choice"] = _convert_tool_choice(body.get("tool_choice"))
    return out


def _default_max_tokens() -> int:
    from .config import settings
    return settings.default_max_tokens


# ---------------------------------------------------------------------------
# response (non-stream):  Anthropic -> OpenAI
# ---------------------------------------------------------------------------

def extract_completion(a_resp: dict) -> Tuple[str, List[dict]]:
    """Return (text, openai_tool_calls) from an Anthropic message's content blocks."""
    text_parts: List[str] = []
    tool_calls: List[dict] = []
    for block in a_resp.get("content") or []:
        btype = block.get("type")
        if btype == "text":
            text_parts.append(block.get("text", ""))
        elif btype == "tool_use":
            tool_calls.append({
                "id": block.get("id") or gen_id("call"),
                "type": "function",
                "function": {
                    "name": block.get("name", ""),
                    "arguments": json.dumps(block.get("input") or {}, ensure_ascii=False),
                },
            })
    return "".join(text_parts), tool_calls


def anthropic_to_openai_response(a_resp: dict, model: str, openai_usage: dict, created: int) -> dict:
    text, tool_calls = extract_completion(a_resp)
    message: Dict[str, Any] = {"role": "assistant"}
    message["content"] = text if text else (None if tool_calls else "")
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": a_resp.get("id") or gen_id(),
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": map_finish_reason(a_resp.get("stop_reason")),
            "logprobs": None,
        }],
        "usage": openai_usage,
    }


# ---------------------------------------------------------------------------
# streaming:  Anthropic SSE -> OpenAI SSE
# ---------------------------------------------------------------------------

def _chunk(model: str, created: int, rid: str, delta: dict, finish: Optional[str]) -> dict:
    return {
        "id": rid,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }


def _sse(obj: dict) -> str:
    return "data: " + json.dumps(obj, ensure_ascii=False) + "\n\n"


async def translate_stream(
    line_iter: AsyncIterator[str],
    model: str,
    created: int,
    rid: str,
    prompt_tokens: int,
    include_usage: bool,
    log_cb: Callable[[str, str, dict, dict], None],
    display_model: Optional[str] = None,
) -> AsyncIterator[str]:
    """Consume Anthropic SSE lines, yield OpenAI SSE chunks.

    Chunks report ``display_model`` when given (a client-facing rebrand alias);
    otherwise they report the upstream's own model name from ``message_start``.
    ``log_cb`` always receives the real upstream model name (for cost logging),
    alongside the OpenAI usage (tiktoken) and merged supplier Anthropic usage.
    """
    anthropic_usage = tk.new_anthropic_usage()
    text_accum: List[str] = []
    tool_meta: Dict[int, dict] = {}      # anthropic block index -> {oi, id, name}
    tool_args: Dict[int, List[str]] = {}  # openai tool index -> arg fragments
    next_oi = 0
    finish_reason: Optional[str] = None
    resolved_model = model            # upstream's real model name (for logging)
    chunk_model = display_model or model  # client-facing name shown in chunks
    opened = False                    # whether the opening role chunk has been emitted yet

    async for raw in line_iter:
        line = raw.strip()
        if not line or not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        try:
            evt = json.loads(data)
        except json.JSONDecodeError:
            continue

        etype = evt.get("type")

        if etype == "message_start":
            msg = evt.get("message") or {}
            tk.merge_anthropic_usage(anthropic_usage, msg.get("usage"))
            if msg.get("model"):
                resolved_model = msg["model"]   # the upstream's real model name
                if not display_model:
                    chunk_model = msg["model"]  # no rebrand -> show upstream's name
            if not opened:
                opened = True
                yield _sse(_chunk(chunk_model, created, rid, {"role": "assistant", "content": ""}, None))
            continue

        # any other event: ensure the opening role chunk has gone out first
        if not opened:
            opened = True
            yield _sse(_chunk(chunk_model, created, rid, {"role": "assistant", "content": ""}, None))

        if etype == "content_block_start":
            idx = evt.get("index")
            block = evt.get("content_block") or {}
            if block.get("type") == "tool_use":
                oi = next_oi
                next_oi += 1
                tool_meta[idx] = {"oi": oi, "id": block.get("id"), "name": block.get("name")}
                tool_args[oi] = []
                yield _sse(_chunk(chunk_model, created, rid, {
                    "tool_calls": [{
                        "index": oi,
                        "id": block.get("id"),
                        "type": "function",
                        "function": {"name": block.get("name"), "arguments": ""},
                    }],
                }, None))
            elif block.get("type") == "text" and block.get("text"):
                text_accum.append(block["text"])
                yield _sse(_chunk(chunk_model, created, rid, {"content": block["text"]}, None))

        elif etype == "content_block_delta":
            idx = evt.get("index")
            delta = evt.get("delta") or {}
            dtype = delta.get("type")
            if dtype == "text_delta":
                piece = delta.get("text", "")
                text_accum.append(piece)
                yield _sse(_chunk(chunk_model, created, rid, {"content": piece}, None))
            elif dtype == "input_json_delta":
                meta = tool_meta.get(idx)
                if meta is not None:
                    frag = delta.get("partial_json", "")
                    tool_args[meta["oi"]].append(frag)
                    yield _sse(_chunk(chunk_model, created, rid, {
                        "tool_calls": [{"index": meta["oi"], "function": {"arguments": frag}}],
                    }, None))

        elif etype == "message_delta":
            tk.merge_anthropic_usage(anthropic_usage, evt.get("usage"))
            stop = (evt.get("delta") or {}).get("stop_reason")
            if stop:
                finish_reason = map_finish_reason(stop)

        # ping / content_block_stop / message_stop -> nothing to emit

    # ---- finalize ----
    if not opened:  # stream produced nothing usable; still emit a well-formed open
        opened = True
        yield _sse(_chunk(chunk_model, created, rid, {"role": "assistant", "content": ""}, None))

    final_tool_calls = [
        {
            "id": meta["id"],
            "type": "function",
            "function": {"name": meta["name"], "arguments": "".join(tool_args.get(meta["oi"], []))},
        }
        for _, meta in sorted(tool_meta.items(), key=lambda kv: kv[1]["oi"])
    ]
    completion_text = "".join(text_accum)
    finish_reason = finish_reason or ("tool_calls" if final_tool_calls else "stop")

    completion_tokens = tk.count_openai_completion_tokens(completion_text, final_tool_calls)
    openai_usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }
    log_cb(resolved_model, chunk_model, openai_usage, anthropic_usage)
    debug.log_response(rid, finish_reason, final_tool_calls, completion_text, stream=True)

    # closing chunk carries the finish reason
    yield _sse(_chunk(chunk_model, created, rid, {}, finish_reason))

    # optional usage chunk (OpenAI 'include_usage' convention: empty choices + usage)
    if include_usage:
        yield _sse({
            "id": rid,
            "object": "chat.completion.chunk",
            "created": created,
            "model": chunk_model,
            "choices": [],
            "usage": openai_usage,
        })

    yield "data: [DONE]\n\n"
