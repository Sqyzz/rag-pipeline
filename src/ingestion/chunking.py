import argparse
import json
import uuid
from pathlib import Path


def _meta_prefix(doc: dict) -> str:
    source = str(doc.get("source", "") or "").strip()
    meta = doc.get("meta") if isinstance(doc.get("meta"), dict) else {}
    parts = []
    if source:
        parts.append(f"source: {source}")
    for key in ("from", "to", "date", "subject"):
        val = str(meta.get(key, "") or "").strip()
        if val:
            parts.append(f"{key}: {val}")
    if not parts:
        return ""
    return "[META] " + " | ".join(parts)


def chunk_texts(in_file, out_file, chunk_size, overlap):
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    chunks = []
    with open(in_file, encoding="utf-8") as r:
        for line in r:
            d = json.loads(line)
            text = d["text"]
            prefix = _meta_prefix(d)
            pos = 0
            while pos < len(text):
                chunk = text[pos : pos + chunk_size]
                chunk_text = f"{prefix}\n\n{chunk}" if prefix else chunk
                chunks.append(
                    {
                        "chunk_id": str(uuid.uuid4()),
                        "doc_id": d.get("doc_id", "unknown"),
                        "source": d.get("source"),
                        "meta": d.get("meta", {}),
                        "text": chunk_text,
                    }
                )
                pos += chunk_size - overlap

    out_path = Path(out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as w:
        for c in chunks:
            w.write(json.dumps(c, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Chunk docs jsonl into overlapping text chunks.")
    parser.add_argument("--in-file", default="data/processed/enron_docs.jsonl")
    parser.add_argument("--out-file", default="data/processed/chunks.jsonl")
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--overlap", type=int, default=200)
    args = parser.parse_args()
    chunk_texts(args.in_file, args.out_file, args.chunk_size, args.overlap)


if __name__ == "__main__":
    main()
