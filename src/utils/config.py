from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml
from dotenv import load_dotenv


# Load environment variables from project root .env if present.
load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")


def _to_ns(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_to_ns(v) for v in value]
    return value


def load_config(path: str = "config.yaml") -> SimpleNamespace:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return _to_ns(data)


cfg = load_config()
