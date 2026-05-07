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
    src = Path(chunks_file).expanduser().resolve()
    if mode == "none":
        return {
            "sync_mode": "none",
            "skipped": True,
            "dataset": dataset,
            "chunks_source": str(src),
        }

    if mode != "shared_dir":
        raise ValueError(f"unsupported sync_mode: {mode}")
    if not shared_dir:
        raise ValueError("shared_dir is required when sync_mode=shared_dir")

    if not src.exists():
        raise FileNotFoundError(f"chunks file not found: {chunks_file}")

    rows: list[dict[str, Any]] = []
    with src.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            # Support both chunk-level and doc-level jsonl.
            # - chunk jsonl: chunk_id/text/doc_id
            # - doc jsonl: doc_id/text/meta.title
            raw_doc_id = str(row.get("doc_id", "") or "").strip()
            chunk_id = str(row.get("chunk_id", "") or "").strip() or raw_doc_id
            text = str(row.get("text", "") or "")
            doc_id = raw_doc_id or row.get("doc_id")
            # youtu-graphrag expects list[{"title","text"}] (see youtu-graphrag/backend.py::validate_corpus_format).
            # To preserve evidence alignment with enterprise-graphrag evaluation, we ALSO include chunk_id
            # so youtu-graphrag's KTBuilder can reuse it instead of generating nanoid IDs.
            meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
            title = str(meta.get("title", "") or "").strip()
            if not title:
                title = str(doc_id) if doc_id is not None else chunk_id
            if not title:
                title = f"{dataset}_chunk"
            rows.append(
                {
                    "title": title,
                    "text": text,
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                }
            )

    out_dir = Path(shared_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{dataset}_corpus.json"
    out_file.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "sync_mode": "shared_dir",
        "skipped": False,
        "dataset": dataset,
        "chunks_source": str(src),
        "written_file": str(out_file.resolve()),
        "num_chunks": len(rows),
    }
