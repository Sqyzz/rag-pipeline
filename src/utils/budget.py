from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class BudgetManager:
    tokenizer: Any
    cfg: dict[str, Any]
    method: str = ""
    regime: str = ""
    evidence_limit: int = 0
    max_completion: int = 0
    max_calls: int = 0
    max_total: int = 0
    calls_used: int = 0
    total_tokens_used: int = 0
    prompt_tokens_used: int = 0
    completion_tokens_used: int = 0
    events: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.evidence_limit = int(self.cfg.get("evidence_token_limit", 2500))
        self.max_completion = int(self.cfg.get("max_completion_tokens", 800))
        self.max_calls = int(self.cfg.get("max_llm_calls", 2))
        self.max_total = int(self.cfg.get("max_total_tokens", 3800))

    def count(self, text: str) -> int:
        return int(self.tokenizer.count(text))

    def can_add(self, current_text: str, new_text: str) -> bool:
        return self.count((current_text or "") + (new_text or "")) <= self.evidence_limit

    def register_call(self, amount: int = 1) -> None:
        self.calls_used += int(amount)
        if self.calls_used > self.max_calls:
            raise RuntimeError(f"LLM call budget exceeded: {self.calls_used}>{self.max_calls}")

    def register_tokens(self, prompt_tokens: int, completion_tokens: int) -> None:
        p = int(prompt_tokens or 0)
        c = int(completion_tokens or 0)
        self.prompt_tokens_used += p
        self.completion_tokens_used += c
        self.total_tokens_used += p + c
        if self.completion_tokens_used > self.max_completion:
            raise RuntimeError(
                f"Completion token budget exceeded: {self.completion_tokens_used}>{self.max_completion}"
            )
        if self.total_tokens_used > self.max_total:
            raise RuntimeError(f"Total token budget exceeded: {self.total_tokens_used}>{self.max_total}")

    def register_from_telemetry(self, telemetry: dict[str, Any], stage: str) -> None:
        llm_calls = int(telemetry.get("llm_calls", 0))
        prompt = int(telemetry.get("prompt_tokens", 0))
        completion = int(telemetry.get("completion_tokens", 0))
        self.register_call(llm_calls)
        self.register_tokens(prompt, completion)
        self.events.append(
            {
                "stage": stage,
                "method": self.method,
                "regime": self.regime,
                "llm_calls": llm_calls,
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "total_tokens": int(telemetry.get("total_tokens", 0)),
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "regime": self.regime,
            "limits": {
                "evidence_token_limit": self.evidence_limit,
                "max_completion_tokens": self.max_completion,
                "max_llm_calls": self.max_calls,
                "max_total_tokens": self.max_total,
            },
            "used": {
                "llm_calls": self.calls_used,
                "prompt_tokens": self.prompt_tokens_used,
                "completion_tokens": self.completion_tokens_used,
                "total_tokens": self.total_tokens_used,
            },
            "events": self.events,
        }
