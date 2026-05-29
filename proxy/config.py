"""Configuration loaded from environment / .env file.

All knobs that matter for *accurate token accounting* live here:
  - TOKENIZER_ENCODING  : which tiktoken vocab the distribution layer bills with
  - PRICE_*_PER_MTOK    : optional supplier (Anthropic) prices for cost reconciliation
"""
import os


def _load_dotenv(path: str = ".env") -> None:
    """Minimal .env loader (no extra dependency)."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                os.environ.setdefault(key, val)
    except FileNotFoundError:
        pass


_load_dotenv()


def _get(name: str, default: str = "") -> str:
    val = os.environ.get(name)
    return val if val not in (None, "") else default


def _f(name: str, default: float) -> float:
    try:
        return float(_get(name, str(default)))
    except ValueError:
        return default


def _i(name: str, default: int) -> int:
    try:
        return int(_get(name, str(default)))
    except ValueError:
        return default


class Settings:
    # --- upstream (the supplier, Anthropic protocol) ---
    upstream_base_url = _get("UPSTREAM_BASE_URL", "https://api.aigclaw.ai").rstrip("/")
    upstream_api_key = _get("UPSTREAM_API_KEY", "")
    anthropic_version = _get("ANTHROPIC_VERSION", "2023-06-01")
    anthropic_beta = _get("ANTHROPIC_BETA", "")  # optional, e.g. prompt-caching beta header

    # --- this proxy (the distribution layer, OpenAI protocol) ---
    host = _get("HOST", "127.0.0.1")
    port = _i("PORT", 8787)
    proxy_api_key = _get("PROXY_API_KEY", "")  # if set, clients must send it; keeps the real key out of Cursor
    request_timeout = _f("REQUEST_TIMEOUT", 600.0)
    model_override = _get("MODEL_OVERRIDE", "")  # force every request to one upstream model (optional)

    # --- OpenAI-protocol token counting (distribution layer billing) ---
    tokenizer_encoding = _get("TOKENIZER_ENCODING", "o200k_base")  # o200k_base = gpt-4o/4.1/5 family
    default_max_tokens = _i("DEFAULT_MAX_TOKENS", 4096)
    image_tokens_each = _i("IMAGE_TOKENS_EACH", 85)  # flat estimate per image part

    # --- supplier cost reconciliation (Anthropic prices, USD per 1M tokens; 0 = disabled) ---
    price_input = _f("PRICE_INPUT_PER_MTOK", 0.0)
    price_output = _f("PRICE_OUTPUT_PER_MTOK", 0.0)
    price_cache_write = _f("PRICE_CACHE_WRITE_PER_MTOK", 0.0)
    price_cache_read = _f("PRICE_CACHE_READ_PER_MTOK", 0.0)

    # --- logging ---
    usage_log_path = _get("USAGE_LOG_PATH", "usage.jsonl")


settings = Settings()
