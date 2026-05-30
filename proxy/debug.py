"""Opt-in request/response debugging (set DEBUG_DUMP=1).

Per request it prints a compact one-line summary to stderr — safe to read in a
Zeabur/Railway log stream — and writes the FULL request + response JSON to
DEBUG_LOG_PATH (default debug.jsonl) for local inspection.

The full file can contain your prompts/code, so it is git-ignored; keep it local.
This is the tool for diagnosing client quirks like Cursor "Plan mode shows 0 todos":
the request summary reveals which tools/tool_choice the client actually sends, and
the response summary reveals whether the model called them.
"""
import json
import sys
import time
from collections import deque

from .config import settings

# In-memory ring buffer of recent records, exposed via GET /debug/recent so the
# full request/response can be inspected in a browser (no log access needed).
_recent = deque(maxlen=12)


def enabled() -> bool:
    return settings.debug_dump


def recent() -> list:
    return list(_recent)


def _tool_names(tools):
    names = []
    for t in tools or []:
        if isinstance(t, dict):
            fn = t.get("function", t)
            if isinstance(fn, dict) and fn.get("name"):
                names.append(fn["name"])
    return names


def _write(record: dict) -> None:
    _recent.append(record)
    try:
        with open(settings.debug_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except OSError:
        pass


def log_request(rid: str, body: dict, a_request: dict, stream: bool) -> None:
    if not settings.debug_dump:
        return
    print(
        f"[debug:req] {rid} model={body.get('model')} stream={int(bool(stream))} "
        f"msgs={len(body.get('messages') or [])} tools={_tool_names(body.get('tools'))} "
        f"tool_choice={body.get('tool_choice')}",
        file=sys.stderr, flush=True,
    )
    _write({
        "ts": int(time.time()), "request_id": rid, "kind": "request",
        "stream": bool(stream), "openai_body": body, "anthropic_request": a_request,
    })


def log_response(rid: str, finish_reason, tool_calls, text, stream: bool) -> None:
    if not settings.debug_dump:
        return
    names = [(tc.get("function") or {}).get("name") for tc in (tool_calls or []) if isinstance(tc, dict)]
    print(
        f"[debug:resp] {rid} stream={int(bool(stream))} finish={finish_reason} "
        f"tool_calls={names} text_len={len(text or '')}",
        file=sys.stderr, flush=True,
    )
    _write({
        "ts": int(time.time()), "request_id": rid, "kind": "response",
        "stream": bool(stream), "finish_reason": finish_reason,
        "tool_calls": tool_calls, "text": text,
    })


def log_error(rid: str, status, detail: str, stream: bool) -> None:
    if not settings.debug_dump:
        return
    detail = (detail or "")[:2000]
    print(f"[debug:err] {rid} stream={int(bool(stream))} upstream_status={status} detail={detail[:200]!r}",
          file=sys.stderr, flush=True)
    _write({
        "ts": int(time.time()), "request_id": rid, "kind": "error",
        "stream": bool(stream), "upstream_status": status, "detail": detail,
    })
