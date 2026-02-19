import json
import uuid
from pathlib import Path


def chunk_texts(in_file, out_file, chunk_size, overlap):
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    chunks = []
    with open(in_file, encoding="utf-8") as r:
        for line in r:
            d = json.loads(line)
            text = d["text"]
            pos = 0
            while pos < len(text):
                chunk = text[pos : pos + chunk_size]
                chunks.append(
                    {
                        "chunk_id": str(uuid.uuid4()),
                        "doc_id": d.get("doc_id", "unknown"),
                        "text": chunk,
                    }
                )
                pos += chunk_size - overlap

    out_path = Path(out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as w:
        for c in chunks:
            w.write(json.dumps(c, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    chunk_texts("data/processed/enron_docs.jsonl", "data/processed/chunks.jsonl", 1000, 200)
