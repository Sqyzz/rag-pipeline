from __future__ import annotations

import re
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

try:
    from utils.llm_wrapper import llm_chat
except ModuleNotFoundError:
    from src.utils.llm_wrapper import llm_chat


_YES_RE = re.compile(r"\byes\b", re.IGNORECASE)
_NO_RE = re.compile(r"\bno\b", re.IGNORECASE)


def _parse_yes_no(text: str) -> int:
    s = str(text or "").strip()
    if _YES_RE.search(s):
        return 1
    if _NO_RE.search(s):
        return 0
    return 0


def semantic_equivalent_yes_no(
    pred: str,
    gold: str,
    model: str = "qwen-flash",
) -> int:
    prompt = (
        "Are the following two answers semantically equivalent?\n"
        "Respond only Yes or No.\n\n"
        f"Answer A: {str(pred or '').strip()}\n"
        f"Answer B: {str(gold or '').strip()}"
    )
    content, _ = llm_chat(
        [{"role": "user", "content": prompt}],
        temperature=0,
        return_meta=True,
        model=model,
    )
    return _parse_yes_no(content)
