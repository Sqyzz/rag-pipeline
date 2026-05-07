from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

try:
    from utils.llm_wrapper import llm_chat
except ModuleNotFoundError:
    from src.utils.llm_wrapper import llm_chat


def main() -> None:
    prompt = 'Return JSON only: {"query": "test?"}'
    try:
        raw, meta = llm_chat(
            [{"role": "user", "content": prompt}],
            return_meta=True,
            max_tokens=32,
        )
    except Exception as exc:
        print(f"llm_chat failed: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return

    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print(raw)


if __name__ == "__main__":
    main()
