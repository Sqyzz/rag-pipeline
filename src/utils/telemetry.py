from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


def now_ms() -> int:
    return int(time.time() * 1000)


def usage_from_body(body: dict[str, Any]) -> dict[str, int]:
    usage = body.get("usage") or {}
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    total_tokens = int(usage.get("total_tokens") or (prompt_tokens + completion_tokens))
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


@dataclass
class Telemetry:
    llm_calls: int = 0
    embedding_calls: int = 0
    llm_latency_ms: int = 0
    embedding_latency_ms: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    def add_llm(self, meta: dict[str, Any] | None) -> None:
        if not meta:
            return
        self.llm_calls += 1
        self.llm_latency_ms += int(meta.get("latency_ms") or 0)
        usage = meta.get("usage") or {}
        self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
        self.completion_tokens += int(usage.get("completion_tokens") or 0)
        self.total_tokens += int(usage.get("total_tokens") or 0)

    def add_embedding(self, meta: dict[str, Any] | None) -> None:
        if not meta:
            return
        self.embedding_calls += 1
        self.embedding_latency_ms += int(meta.get("latency_ms") or 0)
        usage = meta.get("usage") or {}
        self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
        self.total_tokens += int(usage.get("total_tokens") or usage.get("prompt_tokens") or 0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "llm_calls": self.llm_calls,
            "embedding_calls": self.embedding_calls,
            "llm_latency_ms": self.llm_latency_ms,
            "embedding_latency_ms": self.embedding_latency_ms,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "extra": self.extra,
        }
