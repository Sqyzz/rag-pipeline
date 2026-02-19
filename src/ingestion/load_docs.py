# src/ingestion/load_docs.py

import csv
import json
import os
import sys
from pathlib import Path


def split_header_body(message: str):
    """
    Split email header and body.
    Enron emails separate header and body by first blank line.
    """
    parts = message.split("\n\n", 1)
    if len(parts) == 2:
        header, body = parts
    else:
        header = message
        body = ""
    return header, body


def parse_header(header: str):
    meta = {}
    for line in header.split("\n"):
        if ":" in line:
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip()
    return meta


def load_enron_csv(csv_path, output_path):
    csv_path = Path(csv_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Enron CSV rows may contain very large email bodies.
    max_limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(max_limit)
            break
        except OverflowError:
            max_limit = int(max_limit / 10)

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        with open(output_path, "w", encoding="utf-8") as out:
            for row in reader:
                file_id = row["file"]
                message = row["message"]

                header, body = split_header_body(message)
                meta = parse_header(header)

                doc = {
                    "doc_id": file_id,
                    "source": "enron",
                    "text": body.strip(),
                    "meta": {
                        "from": meta.get("From"),
                        "to": meta.get("To"),
                        "date": meta.get("Date"),
                        "subject": meta.get("Subject"),
                    },
                }

                out.write(json.dumps(doc, ensure_ascii=False) + "\n")


def load_enron(raw_dir, out_file):
    raw_dir = Path(raw_dir)
    csv_path = raw_dir / "emails.csv"
    if csv_path.exists():
        load_enron_csv(csv_path, out_file)
        return

    out_path = Path(out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as out:
        for root, _, files in os.walk(raw_dir):
            for name in files:
                if not name.endswith(".txt"):
                    continue
                src = Path(root) / name
                text = src.read_text(encoding="utf-8", errors="ignore")
                doc = {
                    "doc_id": str(src.relative_to(raw_dir)),
                    "source": "enron",
                    "text": text.strip(),
                    "meta": {},
                }
                out.write(json.dumps(doc, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    load_enron("data/raw/enron", "data/processed/enron_docs.jsonl")
