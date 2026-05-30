"""FastAPI app exposing a perfect OpenAI-protocol surface for Cursor, backed by
the Anthropic-protocol upstream, with dual token accounting on every request."""
import json
import sys
import time
from typing import Optional

import httpx
import uvicorn
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from . import convert
from . import debug
from . import tokenizer as tk
from .config import settings
from .usage_log import log_usage

app = FastAPI(title="OpenAI<->Anthropic Proxy", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


def _check_auth(authorization: Optional[str]) -> None:
    if not settings.proxy_api_key:
        return
    token = ""
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
    if token != settings.proxy_api_key:
        raise HTTPException(status_code=401, detail="Invalid API key for proxy")


def _anthropic_headers() -> dict:
    headers = {
        "x-api-key": settings.upstream_api_key,
        "anthropic-version": settings.anthropic_version,
        "content-type": "application/json",
    }
    if settings.anthropic_beta:
        headers["anthropic-beta"] = settings.anthropic_beta
    return headers


def _error_to_openai(status: int, raw: bytes) -> dict:
    message = raw.decode("utf-8", "ignore")
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict) and isinstance(parsed.get("error"), dict):
            message = parsed["error"].get("message", message)
    except Exception:
        pass
    return {"error": {"message": message or "upstream error", "type": "upstream_error", "code": status}}


@app.get("/healthz")
async def healthz():
    return {"ok": True, "upstream": settings.upstream_base_url}


@app.get("/debug/recent")
async def debug_recent(key: Optional[str] = None, authorization: Optional[str] = Header(default=None)):
    """Inspect the last few captured requests/responses in a browser (needs DEBUG_DUMP=1).

    Open /debug/recent?key=<PROXY_API_KEY> (key only required if PROXY_API_KEY is set).
    """
    if settings.proxy_api_key:
        token = key or ""
        if not token and authorization and authorization.lower().startswith("bearer "):
            token = authorization[7:].strip()
        if token != settings.proxy_api_key:
            raise HTTPException(status_code=401, detail="invalid key")
    if not settings.debug_dump:
        return {"enabled": False,
                "hint": "Set DEBUG_DUMP=1, redeploy, retry the action in Cursor, then reload this page."}
    records = debug.recent()
    return {"enabled": True, "count": len(records), "records": records}


@app.get("/debug/config")
async def debug_config(key: Optional[str] = None, authorization: Optional[str] = Header(default=None)):
    """Verify which knobs are actually active (no secrets). Confirms e.g. that SYSTEM_SUFFIX took effect."""
    if settings.proxy_api_key:
        token = key or ""
        if not token and authorization and authorization.lower().startswith("bearer "):
            token = authorization[7:].strip()
        if token != settings.proxy_api_key:
            raise HTTPException(status_code=401, detail="invalid key")
    return {
        "upstream_base_url": settings.upstream_base_url,
        "has_upstream_key": bool(settings.upstream_api_key),
        "proxy_api_key_set": bool(settings.proxy_api_key),
        "debug_dump": settings.debug_dump,
        "model_override": settings.model_override or None,
        "model_map": settings.model_map_upstream_to_display,
        "model_allow": sorted(settings.model_allow),
        "default_max_tokens": settings.default_max_tokens,
        "system_suffix_active": bool(settings.system_suffix),
        "system_suffix_len": len(settings.system_suffix),
        "system_suffix_preview": settings.system_suffix[:140],
    }


@app.get("/v1/models")
async def list_models(authorization: Optional[str] = Header(default=None)):
    _check_auth(authorization)
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(
            f"{settings.upstream_base_url}/v1/models",
            headers={"Authorization": f"Bearer {settings.upstream_api_key}"},
        )
    payload = r.json() if r.content else {"object": "list", "data": []}
    # Rebrand upstream model IDs to display names, then restrict to the catalog (if any).
    u2d = settings.model_map_upstream_to_display
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        kept = []
        for m in payload["data"]:
            if isinstance(m, dict) and m.get("id") in u2d:
                if m.get("root") == m.get("id"):
                    m["root"] = u2d[m["id"]]
                m["id"] = u2d[m["id"]]
            if isinstance(m, dict) and settings.is_model_allowed(m.get("id")):
                kept.append(m)
        payload["data"] = kept
    return JSONResponse(status_code=r.status_code, content=payload)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request, authorization: Optional[str] = Header(default=None)):
    _check_auth(authorization)
    body = await request.json()

    client_model = body.get("model")
    if not settings.is_model_allowed(client_model):
        return JSONResponse(status_code=404, content={"error": {
            "message": f"The model `{client_model}` does not exist or is not available.",
            "type": "invalid_request_error",
            "code": "model_not_found",
        }})
    upstream_model = settings.resolve_upstream_model(client_model)  # name sent to supplier
    stream_display_model = client_model if client_model in settings.model_map_display_to_upstream else None
    stream = bool(body.get("stream"))
    a_request = convert.openai_to_anthropic_request(body, upstream_model)

    # OpenAI-protocol prompt tokens (distribution layer) — counted before sending.
    prompt_tokens = tk.count_openai_prompt_tokens(
        body.get("messages") or [], body.get("tools"), body.get("tool_choice")
    )

    created = int(time.time())
    rid = convert.gen_id()
    url = f"{settings.upstream_base_url}/v1/messages"
    headers = _anthropic_headers()
    debug.log_request(rid, body, a_request, stream)

    if not stream:
        async with httpx.AsyncClient(timeout=settings.request_timeout) as client:
            r = await client.post(url, headers=headers, json=a_request)
        if r.status_code >= 400:
            return JSONResponse(status_code=r.status_code, content=_error_to_openai(r.status_code, r.content))

        a_resp = r.json()
        upstream_returned = a_resp.get("model") or upstream_model  # supplier's real model name
        display_model = settings.resolve_display_model(client_model, upstream_returned)
        anthropic_usage = tk.merge_anthropic_usage(tk.new_anthropic_usage(), a_resp.get("usage"))
        text, tool_calls = convert.extract_completion(a_resp)
        debug.log_response(rid, convert.map_finish_reason(a_resp.get("stop_reason")), tool_calls, text, stream=False)
        completion_tokens = tk.count_openai_completion_tokens(text, tool_calls)
        openai_usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
        log_usage(rid, upstream_returned, openai_usage, anthropic_usage, stream=False, client_model=display_model)
        response = convert.anthropic_to_openai_response(a_resp, display_model, openai_usage, created)
        return JSONResponse(
            content=response,
            headers={"x-upstream-anthropic-usage": json.dumps(anthropic_usage)},
        )

    include_usage = bool((body.get("stream_options") or {}).get("include_usage"))

    async def event_stream():
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(settings.request_timeout, connect=15.0)) as client:
                async with client.stream("POST", url, headers=headers, json=a_request) as r:
                    if r.status_code >= 400:
                        raw = await r.aread()
                        print(f"[stream-error] {rid}: upstream HTTP {r.status_code}: {raw[:300]!r}",
                              file=sys.stderr, flush=True)
                        yield convert._sse(_error_to_openai(r.status_code, raw))
                        yield "data: [DONE]\n\n"
                        return
                    async for chunk in convert.translate_stream(
                        r.aiter_lines(),
                        model=upstream_model,
                        created=created,
                        rid=rid,
                        prompt_tokens=prompt_tokens,
                        include_usage=include_usage,
                        log_cb=lambda um, cm, ou, au: log_usage(rid, um, ou, au, stream=True, client_model=cm),
                        display_model=stream_display_model,
                    ):
                        yield chunk
        except Exception as e:
            # Upstream drop / timeout mid-stream: end the SSE cleanly instead of a silent cutoff,
            # so the client gets a proper finish rather than a hung/interrupted connection.
            print(f"[stream-error] {rid}: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
            yield convert._sse(convert._chunk(stream_display_model or upstream_model, created, rid, {}, "stop"))
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )


def main():
    print(f"OpenAI<->Anthropic proxy on http://{settings.host}:{settings.port}  ->  {settings.upstream_base_url}")
    if not settings.upstream_api_key:
        print("WARNING: UPSTREAM_API_KEY is empty — set it in .env")
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="info")


if __name__ == "__main__":
    main()
