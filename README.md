# OpenAI ↔ Anthropic Proxy for Cursor (dual token accounting)

A local proxy that gives **Cursor a perfect OpenAI-protocol endpoint** while talking
to your supplier (`api.aigclaw.ai`) over the **Anthropic protocol**. Every request is
metered twice:

```
  Cursor ──OpenAI /v1/chat/completions──▶  [ this proxy ]  ──Anthropic /v1/messages──▶  supplier
   (distribution layer)                   convert + meter                              (supplier)
        ▲                                       │
        └──────── OpenAI response + usage ◀──────┘
```

| Layer | Protocol | How tokens are counted |
|-------|----------|------------------------|
| **Distribution** (what you bill Cursor/clients) | OpenAI | `tiktoken` over the OpenAI-format request & response — exact for text, approx for tool schemas |
| **Supplier** (what `api.aigclaw.ai` bills you) | Anthropic | taken **verbatim** from the upstream `usage`, merged across stream events |

## Why a converter (and not just Cursor → the supplier's OpenAI endpoint)

The upstream *does* expose `/v1/chat/completions`, but two things make it unsuitable for
accurate billing:

1. Its OpenAI `usage` just echoes the Anthropic number (e.g. `prompt_tokens: 434` for a
   6-word message) and zeroes the detail fields — **not** a real OpenAI-tokenizer count.
2. The upstream injects a **~5,900-token cached system prompt**. The true supplier cost
   only appears in the Anthropic stream's `message_delta`
   (`input_tokens` jumps 428 → 1043, plus `cache_creation`/`cache_read`), never in
   `message_start`. Read the wrong field and you under-count by thousands of tokens per call.

This proxy counts OpenAI tokens itself (tiktoken) and reads the *correct* Anthropic
usage field, so both ledgers are right.

## Setup (Windows / PowerShell)

```powershell
cd C:\Users\zc319\Desktop\anthropic_cursor2026
./run.ps1          # creates .venv, installs deps, starts the proxy on :8787
```

Your supplier key is already in `.env`. Edit `.env` to change port, tokenizer, prices, etc.

Manual alternative:

```powershell
python -m pip install -r requirements.txt
python -m proxy.main
```

## Deploy to Zeabur

This repo is deploy-ready for Zeabur (and any Procfile/Python PaaS). `start.py` binds
`0.0.0.0:$PORT` for the container; local dev keeps using `127.0.0.1` via `run.ps1`.

1. In Zeabur: **Create Project → Deploy from GitHub → select `ccmax_cursor`**. Zeabur detects
   Python via `requirements.txt` and starts it with `python start.py` (from `zbpack.json` / `Procfile`).
2. Open the service → **Variables** tab and set:

   | Variable | Value | Required |
   |---|---|---|
   | `UPSTREAM_API_KEY` | your supplier key (`sk-…`) | ✅ yes |
   | `PROXY_API_KEY` | a long random secret you invent | ✅ strongly recommended |
   | `UPSTREAM_BASE_URL` | `https://api.aigclaw.ai` | optional (default) |
   | `TOKENIZER_ENCODING` | `o200k_base` | optional |
   | `PRICE_*_PER_MTOK` | supplier Anthropic prices | optional (enables `supplier_cost_usd`) |

   > ⚠️ **Set `PROXY_API_KEY`.** Without it your deployed proxy is an **open relay** — anyone who
   > finds the URL can burn your supplier credits. With it set, clients must send
   > `Authorization: Bearer <PROXY_API_KEY>`.

3. Zeabur assigns a domain (e.g. `https://ccmax-cursor.zeabur.app`). Use `https://<domain>/v1`
   as the Base URL in Cursor and `PROXY_API_KEY` as the API key.

**Notes**
- `.env` is git-ignored and never deployed — secrets live only in Zeabur's Variables.
- `usage.jsonl` is written to the container's **ephemeral** disk (lost on redeploy); the same
  per-request lines also go to **stderr**, visible in Zeabur logs. For durable accounting, mount
  a volume or ship the logs out.
- `tiktoken` downloads its vocab on first request (needs outbound internet, which Zeabur allows).

## Point Cursor at it

Cursor → **Settings → Models → OpenAI API Key**:

- **Override OpenAI Base URL**: `http://127.0.0.1:8787/v1` (local) or `https://<your-domain>/v1` (Zeabur)
- **API Key**: the value of `PROXY_API_KEY` from `.env` (or any non-empty string if you left it blank)
- Add a custom model whose name the upstream accepts, e.g. `claude-opus-4-8`
- Click **Verify** — Cursor calls `/v1/models` (proxied) and a test completion.

> Cursor allows a custom base URL over `http://localhost`. Keep the proxy running while you use Cursor.

## Token accounting output

Every request appends one line to `usage.jsonl` and prints a summary to the console:

```
[usage] claude-opus-4-8 stream=1 | OpenAI(prompt=14,compl=22,total=36) | Anthropic(in=1043,out=22,cache_w=358,cache_r=5551)
```

`usage.jsonl` line:

```json
{
  "ts": 1780049312, "request_id": "chatcmpl-…", "model": "claude-opus-4-8", "stream": true,
  "openai_protocol":    {"prompt_tokens": 14, "completion_tokens": 22, "total_tokens": 36},
  "anthropic_protocol": {"input_tokens": 1043, "output_tokens": 22,
                          "cache_creation_input_tokens": 358, "cache_read_input_tokens": 5551,
                          "billable_input_total": 6952}
}
```

Non-streaming responses also return the supplier numbers in the `x-upstream-anthropic-usage`
response header.

Set the `PRICE_*_PER_MTOK` values in `.env` to your supplier's Anthropic prices to also get
`supplier_cost_usd` per request.

## Test it

With the proxy running:

```powershell
python test_proxy.py
```

Covers `/v1/models`, non-stream, streaming (+`include_usage`), and tool calling.

## What gets converted

**Request (OpenAI → Anthropic):** system/developer messages → top-level `system`;
`tool` role → `tool_result` blocks (consecutive ones merged into one user turn);
assistant `tool_calls` → `tool_use` blocks; `image_url` (incl. `data:` URLs) → image blocks;
`tools[].function` → `tools[].input_schema`; `tool_choice` (`auto`/`required`/named) mapped;
`stop` → `stop_sequences`; `temperature` clamped to ≤ 1.0; `max_tokens` defaulted if missing.

**Response (Anthropic → OpenAI):** `text` blocks → `message.content`; `tool_use` → `tool_calls`;
`stop_reason` → `finish_reason` (`end_turn`→`stop`, `max_tokens`→`length`, `tool_use`→`tool_calls`);
streaming SSE events → `chat.completion.chunk` deltas with proper tool-call streaming and a
final `usage` chunk when `stream_options.include_usage` is set.

## Configuration (`.env`)

| Key | Meaning |
|-----|---------|
| `UPSTREAM_BASE_URL` / `UPSTREAM_API_KEY` | supplier endpoint + key |
| `PROXY_API_KEY` | if set, clients must send it (keeps the real key out of Cursor) |
| `MODEL_OVERRIDE` | force every request onto one upstream model |
| `TOKENIZER_ENCODING` | `o200k_base` (GPT-4o/5) or `cl100k_base` (GPT-4) |
| `DEFAULT_MAX_TOKENS` | used when the client omits `max_tokens` (Anthropic requires it) |
| `IMAGE_TOKENS_EACH` | flat OpenAI-side token estimate per image part |
| `PRICE_*_PER_MTOK` | optional Anthropic prices → per-request `supplier_cost_usd` |

## Caveats

- **Tool-schema tokens** on the OpenAI side are an approximation (OpenAI doesn't document the
  exact formula). Message/content/completion tokens are exact for the chosen encoding.
- **Image tokens** use a flat estimate (`IMAGE_TOKENS_EACH`), not OpenAI's tile-based formula.
- `tiktoken` downloads its vocab on first run; if offline it falls back to a ~4-chars/token estimate.
- `n > 1`, `logit_bias`, `frequency/presence_penalty`, `seed`, `response_format` have no Anthropic
  equivalent and are dropped.
