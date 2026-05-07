from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any


def _extract_query_doc_key(row: dict[str, Any]) -> str:
    meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
    for key in ("query_doc_key", "title", "doc_id"):
        value = str(meta.get(key, "") or "").strip()
        if value:
            return value.split("#", 1)[0].strip()
    return ""


def _load_qa_doc_keys(path: str) -> set[str]:
    keys: set[str] = set()
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                continue
            key = _extract_query_doc_key(row)
            if key:
                keys.add(key)
    return keys


def _load_cuad_items(path: str) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload.get("data", []) if isinstance(payload, dict) else []
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict) and str(row.get("title", "") or "").strip()]


def _select_items(
    items: list[dict[str, Any]],
    qa_doc_keys: set[str],
    matched_doc_ratio: float,
    random_seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    matched = [row for row in items if str(row.get("title", "") or "").strip() in qa_doc_keys]
    unmatched = [row for row in items if str(row.get("title", "") or "").strip() not in qa_doc_keys]

    ratio = float(matched_doc_ratio)
    if ratio <= 0 or ratio > 1:
        raise ValueError("matched_doc_ratio must be in (0, 1]")

    rng = random.Random(int(random_seed))
    selected_matched = list(matched)
    if ratio >= 1.0:
        selected_unmatched = []
    else:
        target_total = int(math.ceil(len(selected_matched) / ratio)) if selected_matched else 0
        need_unmatched = max(0, target_total - len(selected_matched))
        if need_unmatched >= len(unmatched):
            selected_unmatched = list(unmatched)
        else:
            selected_unmatched = rng.sample(unmatched, need_unmatched)

    selected = selected_matched + selected_unmatched
    rng.shuffle(selected)
    stats = {
        "num_cuad_docs": len(items),
        "num_qa_doc_keys": len(qa_doc_keys),
        "num_matched_docs_total": len(matched),
        "num_unmatched_docs_total": len(unmatched),
        "num_selected_docs": len(selected),
        "num_selected_matched_docs": len(selected_matched),
        "num_selected_unmatched_docs": len(selected_unmatched),
        "requested_matched_doc_ratio": ratio,
        "actual_matched_doc_ratio": (len(selected_matched) / len(selected)) if selected else 0.0,
    }
    return selected, stats


def _write_docs_jsonl(items: list[dict[str, Any]], out_file: str, split_name: str | None = None) -> dict[str, Any]:
    out_path = Path(out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    num_docs = 0
    num_paragraph_docs = 0
    with out_path.open("w", encoding="utf-8") as out:
        for item in items:
            title = str(item.get("title", "") or "").strip()
            paragraphs = item.get("paragraphs", [])
            if not isinstance(paragraphs, list):
                continue
            num_docs += 1
            for p_idx, para in enumerate(paragraphs):
                if not isinstance(para, dict):
                    continue
                context = str(para.get("context", "") or "").strip()
                if not context:
                    continue
                qas = para.get("qas", [])
                doc = {
                    "doc_id": f"{title}#p{p_idx}",
                    "source": "cuad",
                    "text": context,
                    "meta": {
                        "title": title,
                        "paragraph_index": p_idx,
                        "num_qas": len(qas) if isinstance(qas, list) else 0,
                        "split": split_name,
                    },
                }
                out.write(json.dumps(doc, ensure_ascii=False) + "\n")
                num_paragraph_docs += 1
    return {
        "num_selected_titles_written": num_docs,
        "num_paragraph_docs_written": num_paragraph_docs,
    }


def build_cuad_aligned_docs(
    *,
    cuad_file: str,
    qa_file: str,
    out_docs_file: str,
    matched_doc_ratio: float = 1.0,
    random_seed: int = 42,
    split_name: str | None = "qa_aligned",
) -> dict[str, Any]:
    items = _load_cuad_items(cuad_file)
    qa_doc_keys = _load_qa_doc_keys(qa_file)
    selected, stats = _select_items(
        items=items,
        qa_doc_keys=qa_doc_keys,
        matched_doc_ratio=matched_doc_ratio,
        random_seed=random_seed,
    )
    write_stats = _write_docs_jsonl(selected, out_docs_file, split_name=split_name)
    summary = {
        "cuad_file": cuad_file,
        "qa_file": qa_file,
        "out_docs_file": out_docs_file,
        "random_seed": int(random_seed),
        "split_name": split_name,
        **stats,
        **write_stats,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CUAD docs subset aligned to QA document keys.")
    parser.add_argument("--cuad-file", default="data/raw/cuad/CUADv1.json")
    parser.add_argument("--qa-file", default="data/queries/cuad_capability_queries.jsonl")
    parser.add_argument("--out-docs-file", default="data/processed/cuad_docs_capability_aligned.jsonl")
    parser.add_argument("--matched-doc-ratio", type=float, default=1.0)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--split-name", default="qa_aligned")
    args = parser.parse_args()

    summary = build_cuad_aligned_docs(
        cuad_file=args.cuad_file,
        qa_file=args.qa_file,
        out_docs_file=args.out_docs_file,
        matched_doc_ratio=args.matched_doc_ratio,
        random_seed=args.random_seed,
        split_name=(str(args.split_name).strip() or None),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
