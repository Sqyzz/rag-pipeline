from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _index_by_qid(rows: list[dict[str, Any]], *, source: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        qid = str(row.get("qid", "") or "").strip()
        if not qid:
            raise ValueError(f"Missing qid in {source}")
        if qid in indexed:
            raise ValueError(f"Duplicate qid within {source}: {qid}")
        indexed[qid] = row
    return indexed


def merge_qa_sets(
    *,
    query_files: list[str],
    gold_files: list[str],
    out_queries_file: str,
    out_gold_file: str,
) -> dict[str, Any]:
    if len(query_files) != len(gold_files):
        raise ValueError("query_files and gold_files must have the same length")
    merged_queries: list[dict[str, Any]] = []
    merged_gold: list[dict[str, Any]] = []
    seen_qids: set[str] = set()
    type_counts: dict[str, int] = {}

    for query_file, gold_file in zip(query_files, gold_files, strict=False):
        query_rows = _read_jsonl(query_file)
        gold_rows = _read_jsonl(gold_file)
        query_index = _index_by_qid(query_rows, source=query_file)
        gold_index = _index_by_qid(gold_rows, source=gold_file)
        query_qids = set(query_index)
        gold_qids = set(gold_index)
        if query_qids != gold_qids:
            missing_in_gold = sorted(query_qids - gold_qids)
            missing_in_query = sorted(gold_qids - query_qids)
            raise ValueError(
                "Query/gold qid mismatch for pair "
                f"{query_file} vs {gold_file}; missing_in_gold={missing_in_gold[:5]}, missing_in_query={missing_in_query[:5]}"
            )
        overlap = sorted(seen_qids & query_qids)
        if overlap:
            raise ValueError(f"Duplicate qid across QA sets: {overlap[0]}")
        for qid in sorted(query_qids):
            query_row = query_index[qid]
            gold_row = gold_index[qid]
            merged_queries.append(query_row)
            merged_gold.append(gold_row)
            seen_qids.add(qid)
            qtype = str(query_row.get("type", "unknown") or "unknown")
            type_counts[qtype] = type_counts.get(qtype, 0) + 1

    _write_jsonl(out_queries_file, merged_queries)
    _write_jsonl(out_gold_file, merged_gold)
    return {
        "query_files": list(query_files),
        "gold_files": list(gold_files),
        "out_queries_file": out_queries_file,
        "out_gold_file": out_gold_file,
        "num_queries": len(merged_queries),
        "num_gold": len(merged_gold),
        "type_counts": type_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge staged QA JSONL files into a final evaluation set.")
    parser.add_argument("--query-files", required=True, help="Comma-separated query JSONL files in merge order.")
    parser.add_argument("--gold-files", required=True, help="Comma-separated gold JSONL files in merge order.")
    parser.add_argument("--out-queries-file", default="data/queries/queries.jsonl")
    parser.add_argument("--out-gold-file", default="data/queries/gold.jsonl")
    args = parser.parse_args()

    query_files = [item.strip() for item in str(args.query_files).split(",") if item.strip()]
    gold_files = [item.strip() for item in str(args.gold_files).split(",") if item.strip()]
    stats = merge_qa_sets(
        query_files=query_files,
        gold_files=gold_files,
        out_queries_file=args.out_queries_file,
        out_gold_file=args.out_gold_file,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
