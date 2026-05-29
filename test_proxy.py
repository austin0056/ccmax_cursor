"""End-to-end smoke test against a RUNNING proxy.

Start the proxy first (./run.ps1 or `python -m proxy.main`), then:
    python test_proxy.py

It exercises the OpenAI surface exactly like Cursor would and prints both token
ledgers so you can eyeball that distribution-layer (OpenAI) and supplier
(Anthropic) counts are populated and sane.
"""
import json
import os
import sys

import httpx

BASE = os.environ.get("PROXY_BASE", "http://127.0.0.1:8787")
KEY = os.environ.get("PROXY_API_KEY", "")
MODEL = os.environ.get("TEST_MODEL", "claude-opus-4-8")
HEADERS = {"Authorization": f"Bearer {KEY or 'sk-anything'}", "Content-Type": "application/json"}


def line(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


def test_models():
    line("GET /v1/models")
    r = httpx.get(f"{BASE}/v1/models", headers=HEADERS, timeout=30)
    data = r.json()
    ids = [m["id"] for m in data.get("data", [])]
    print(f"status={r.status_code}  models={len(ids)}  sample={ids[:5]}")
    assert r.status_code == 200 and ids


def test_non_stream():
    line("POST /v1/chat/completions  (non-stream)")
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are concise."},
            {"role": "user", "content": "Reply with exactly: pong"},
        ],
        "max_tokens": 32,
    }
    r = httpx.post(f"{BASE}/v1/chat/completions", headers=HEADERS, json=payload, timeout=120)
    body = r.json()
    print("content :", body["choices"][0]["message"]["content"])
    print("finish  :", body["choices"][0]["finish_reason"])
    print("OpenAI usage     (distribution):", json.dumps(body["usage"]))
    print("Anthropic usage  (supplier)    :", r.headers.get("x-upstream-anthropic-usage"))
    assert r.status_code == 200
    assert body["usage"]["prompt_tokens"] > 0 and body["usage"]["completion_tokens"] > 0


def test_stream():
    line("POST /v1/chat/completions  (stream + include_usage)")
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Count from 1 to 5."}],
        "max_tokens": 48,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    text, usage = "", None
    with httpx.stream("POST", f"{BASE}/v1/chat/completions", headers=HEADERS, json=payload, timeout=120) as r:
        for raw in r.iter_lines():
            if not raw or not raw.startswith("data:"):
                continue
            data = raw[5:].strip()
            if data == "[DONE]":
                break
            evt = json.loads(data)
            if evt.get("usage"):
                usage = evt["usage"]
            for ch in evt.get("choices", []):
                text += (ch.get("delta") or {}).get("content") or ""
    print("streamed content:", repr(text))
    print("final OpenAI usage chunk:", json.dumps(usage))
    assert text.strip()
    assert usage and usage["completion_tokens"] > 0


def test_tools():
    line("POST /v1/chat/completions  (tool calling -> OpenAI tool_calls)")
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "What's the weather in Paris? Use the tool."}],
        "max_tokens": 256,
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string", "description": "City name"}},
                    "required": ["city"],
                },
            },
        }],
        "tool_choice": "auto",
    }
    r = httpx.post(f"{BASE}/v1/chat/completions", headers=HEADERS, json=payload, timeout=120)
    body = r.json()
    msg = body["choices"][0]["message"]
    print("finish    :", body["choices"][0]["finish_reason"])
    print("tool_calls:", json.dumps(msg.get("tool_calls"), ensure_ascii=False))
    print("OpenAI usage    :", json.dumps(body["usage"]))
    print("Anthropic usage :", r.headers.get("x-upstream-anthropic-usage"))
    assert msg.get("tool_calls"), "expected a tool call"
    assert body["choices"][0]["finish_reason"] == "tool_calls"


if __name__ == "__main__":
    try:
        test_models()
        test_non_stream()
        test_stream()
        test_tools()
    except (httpx.ConnectError, httpx.ReadTimeout) as e:
        print(f"\nERROR: cannot reach proxy at {BASE} — is it running? ({e})")
        sys.exit(1)
    print("\nAll smoke tests passed. [OK]")
