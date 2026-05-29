"""Production entrypoint for PaaS (Zeabur / Railway / Render / Docker).

Binds 0.0.0.0 and the platform-provided $PORT. For LOCAL development use
`python -m proxy.main` (binds 127.0.0.1) or `./run.ps1` instead.
"""
import os

import uvicorn

from proxy.config import settings
from proxy.main import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", settings.port))

    if not settings.upstream_api_key:
        print("WARNING: UPSTREAM_API_KEY is empty — set it in your platform's environment variables.")
    if not settings.proxy_api_key:
        print(
            "WARNING: PROXY_API_KEY is empty — this proxy is PUBLICLY OPEN. "
            "Anyone with the URL can spend your upstream credits. "
            "Set PROXY_API_KEY in your platform's environment variables."
        )

    print(f"Starting OpenAI<->Anthropic proxy on 0.0.0.0:{port} -> {settings.upstream_base_url}")
    uvicorn.run(app, host="0.0.0.0", port=port)
