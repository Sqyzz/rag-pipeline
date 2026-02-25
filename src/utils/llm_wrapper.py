import os
import time
from typing import Any

import requests

from .config import cfg
from .telemetry import usage_from_body


def _resolve_api_key() -> str | None:
    configured = os.getenv(cfg.llm.api.api_key_env)
    if configured:
        return configured
    return os.getenv("DASHSCOPE_API_KEY")


def llm_chat(
    messages,
    temperature: float = 0.2,
    max_tokens: int | None = None,
    return_meta: bool = False,
):
    backend = cfg.llm.backend
    if backend == "local":
        base_url = cfg.llm.local.base_url
        model = cfg.llm.local.model
        key = getattr(cfg.llm.local, "api_key", None)
    else:
        base_url = cfg.llm.api.base_url
        model = cfg.llm.api.model
        key = _resolve_api_key()
        if not key:
            raise RuntimeError(
                f"Missing LLM API key. Set {cfg.llm.api.api_key_env} "
                "or DASHSCOPE_API_KEY."
            )

    url = f"{base_url}/chat/completions"
    headers = {}
    if key and key != "EMPTY":
        headers["Authorization"] = f"Bearer {key}"

    payload = {"model": model, "messages": messages, "temperature": temperature}
    if max_tokens is not None:
        payload["max_tokens"] = int(max_tokens)
    t0 = time.perf_counter()
    max_attempts = int(os.getenv("LLM_HTTP_MAX_ATTEMPTS", "3"))
    backoff_sec = float(os.getenv("LLM_HTTP_RETRY_BACKOFF_SEC", "1.5"))
    last_exc: Exception | None = None
    body = None
    for attempt in range(1, max_attempts + 1):
        try:
            r = requests.post(url, json=payload, headers=headers, timeout=120)
            r.raise_for_status()
            body = r.json()
            break
        except requests.exceptions.RequestException as exc:
            last_exc = exc
            retryable = isinstance(exc, (requests.exceptions.Timeout, requests.exceptions.ConnectionError))
            if isinstance(exc, requests.exceptions.HTTPError):
                status = int((exc.response.status_code if exc.response is not None else 0) or 0)
                retryable = status in (429, 500, 502, 503, 504)
            if (not retryable) or attempt >= max_attempts:
                raise
            time.sleep(backoff_sec * attempt)
    if body is None:
        raise RuntimeError(f"LLM request failed after retries: {last_exc}")
    content = body["choices"][0]["message"]["content"]
    if not return_meta:
        return content
    meta: dict[str, Any] = {
        "backend": backend,
        "model": model,
        "latency_ms": int((time.perf_counter() - t0) * 1000),
        "usage": usage_from_body(body),
    }
    return content, meta


class _LLMClient:
    def chat(
        self,
        prompt: str,
        max_tokens: int | None = None,
        return_meta: bool = False,
    ) -> str | tuple[str, dict[str, Any]]:
        messages = [{"role": "user", "content": prompt}]
        return llm_chat(messages, max_tokens=max_tokens, return_meta=return_meta)


llm = _LLMClient()
