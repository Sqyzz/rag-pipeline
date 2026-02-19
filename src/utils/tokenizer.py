from __future__ import annotations

from typing import Any


class TokenizerProvider:
    def __init__(self, backend: str, model_name: str):
        self.backend = backend
        self.model_name = model_name
        self._tokenizer: Any = None

        if backend == "api":
            try:
                import tiktoken

                try:
                    self._tokenizer = tiktoken.encoding_for_model(model_name)
                except Exception:
                    self._tokenizer = tiktoken.get_encoding("cl100k_base")
            except Exception:
                self._tokenizer = None
        elif backend == "local":
            try:
                from transformers import AutoTokenizer

                self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            except Exception:
                self._tokenizer = None

    def encode(self, text: str) -> list[int]:
        text = str(text or "")
        if self._tokenizer is None:
            return [1] * max(len(text.split()), 1)
        if hasattr(self._tokenizer, "encode"):
            return list(self._tokenizer.encode(text))
        return [1] * max(len(text.split()), 1)

    def count(self, text: str) -> int:
        return len(self.encode(text))
