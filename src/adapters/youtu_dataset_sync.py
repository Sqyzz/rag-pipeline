from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def sync_chunks_to_youtu_dataset(
    chunks_file: str,
    dataset: str,
    sync_mode: str = "none",
    shared_dir: str | None = None,
) -> dict[str, Any]:
    mode = str(sync_mode or "none")
    if mode == "none":
        return {
            "sync_mode": "none",
            "skipped": True,
            "dataset": dataset,
            "chunks_source": chunks_file,
        }

    if mode != "shared_dir":
        raise ValueError(f"unsupported sync_mode: {mode}")
    if not shared_dir:
        raise ValueError("shared_dir is required when sync_mode=shared_dir")

    src = Path(chunks_file)
    if not src.exists():
        raise FileNotFoundError(f"chunks file not found: {chunks_file}")

    rows: list[dict[str, Any]] = []
    with src.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows.append(
                {
                    "id": str(row.get("chunk_id", "")),
                    "text": str(row.get("text", "")),
                    "metadata": {
                        "doc_id": row.get("doc_id"),
                    },
                }
            )

    out_dir = Path(shared_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{dataset}_corpus.json"
    out_file.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "sync_mode": "shared_dir",
        "skipped": False,
        "dataset": dataset,
        "chunks_source": str(src),
        "written_file": str(out_file),
        "num_chunks": len(rows),
    }
