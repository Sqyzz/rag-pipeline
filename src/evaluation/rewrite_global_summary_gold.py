from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.llm_wrapper import llm_chat


def _read_jsonl(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: str, rows: list[dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _doc_prefix(value: str) -> str:
    return str(value or "").split("#", 1)[0].strip()


def _build_chunk_map(chunks_file: str, needed_chunk_ids: set[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with open(chunks_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            cid = str(row.get("chunk_id", "")).strip()
            if (not cid) or (cid not in needed_chunk_ids):
                continue
            out[cid] = row
    return out


def _build_prompt(query: str, doc_key: str, chunk_blocks: list[str]) -> str:
    evidence = "\n\n".join(chunk_blocks)
    return f"""
You are creating a reference answer for evaluating contract global-summary QA.
Use ONLY the evidence excerpts below from one contract.
Write exactly 3 concise bullet points in English:
- key themes/obligations,
- major risks/exposures,
- governance/payment/compliance signals if present.
Do not invent entities or facts not supported by evidence.
No preface, no markdown heading, no legal advice.
If evidence is insufficient, return exactly: NOT_FOUND

Contract:
{doc_key}

Question:
{query}

Evidence excerpts:
{evidence}
"""


def rewrite_global_summary_gold(
    gold_file: str,
    chunks_file: str,
    out_file: str,
    max_chunks: int = 8,
    max_chunk_chars: int = 1200,
    max_completion_tokens: int = 220,
    preserve_non_global: bool = True,
    progress_every: int = 0,
) -> dict[str, Any]:
    gold_rows = _read_jsonl(gold_file)
    global_rows = [r for r in gold_rows if str(r.get("type", "")).strip() == "global_summary"]
    needed_chunk_ids: set[str] = set()
    for row in global_rows:
        for s in row.get("supporting_chunks", []) or []:
            cid = str((s or {}).get("chunk_id", "")).strip()
            if cid:
                needed_chunk_ids.add(cid)

    chunk_map = _build_chunk_map(chunks_file=chunks_file, needed_chunk_ids=needed_chunk_ids)
    rewritten = 0
    fallback_no_chunk = 0
    fallback_error = 0
    progress_step = max(int(progress_every or 0), 0)
    total_global = len(global_rows)
    processed_global = 0

    def maybe_log_progress(stage: str, qid: str) -> None:
        if progress_step <= 0:
            return
        if processed_global <= 0:
            return
        if (processed_global % progress_step) != 0 and processed_global != total_global:
            return
        payload = {
            "stage": stage,
            "qid": qid,
            "processed_global": int(processed_global),
            "total_global": int(total_global),
            "rewritten": int(rewritten),
            "fallback_no_chunk": int(fallback_no_chunk),
            "fallback_error": int(fallback_error),
        }
        print(
            "[rewrite_global_summary_gold] " + json.dumps(payload, ensure_ascii=False),
            file=sys.stderr,
            flush=True,
        )

    out_rows: list[dict] = []
    for row in gold_rows:
        qtype = str(row.get("type", "")).strip()
        if qtype != "global_summary":
            if preserve_non_global:
                out_rows.append(row)
            continue

        new_row = dict(row)
        qid = str(new_row.get("qid", "")).strip()
        meta = dict(new_row.get("meta", {}) or {})
        doc_key = str(meta.get("query_doc_key", "") or meta.get("title", "")).strip()
        supporting = new_row.get("supporting_chunks", []) or []
        chunk_rows: list[dict] = []
        for s in supporting:
            cid = str((s or {}).get("chunk_id", "")).strip()
            if not cid:
                continue
            cr = chunk_map.get(cid)
            if not cr:
                continue
            cr_doc_key = _doc_prefix(str(cr.get("doc_id", "")))
            if doc_key and cr_doc_key and cr_doc_key != doc_key:
                continue
            chunk_rows.append(cr)
            if len(chunk_rows) >= int(max_chunks):
                break

        if not chunk_rows:
            fallback_no_chunk += 1
            meta["global_summary_ref_source"] = "fallback_existing_answer_no_chunk"
            new_row["meta"] = meta
            out_rows.append(new_row)
            processed_global += 1
            maybe_log_progress(stage="fallback_no_chunk", qid=qid)
            continue

        chunk_blocks = []
        for i, c in enumerate(chunk_rows, start=1):
            text = str(c.get("text", "")).strip()[: int(max_chunk_chars)]
            chunk_blocks.append(f"[evidence_{i}] {text}")

        prompt = _build_prompt(
            query=str(new_row.get("query", "")).strip(),
            doc_key=doc_key,
            chunk_blocks=chunk_blocks,
        )
        try:
            answer, _ = llm_chat(
                [{"role": "user", "content": prompt}],
                max_tokens=int(max_completion_tokens),
                return_meta=True,
            )
            answer = str(answer or "").strip()
            if answer:
                new_row["answer"] = answer
                meta["global_summary_ref_source"] = "llm_from_supporting_chunks"
                meta["global_summary_ref_chunks_used"] = len(chunk_rows)
                rewritten += 1
            else:
                fallback_error += 1
                meta["global_summary_ref_source"] = "fallback_existing_answer_empty_llm"
        except Exception as exc:
            fallback_error += 1
            meta["global_summary_ref_source"] = "fallback_existing_answer_llm_error"
            meta["global_summary_ref_error"] = f"{type(exc).__name__}: {exc}"
        new_row["meta"] = meta
        out_rows.append(new_row)
        processed_global += 1
        stage = str(meta.get("global_summary_ref_source", "unknown"))
        maybe_log_progress(stage=stage, qid=qid)

    _write_jsonl(out_file, out_rows)
    return {
        "gold_file": gold_file,
        "chunks_file": chunks_file,
        "out_file": out_file,
        "num_rows_in": len(gold_rows),
        "num_rows_out": len(out_rows),
        "global_summary_rows": len(global_rows),
        "rewritten_global_summary_rows": rewritten,
        "fallback_no_chunk": fallback_no_chunk,
        "fallback_error": fallback_error,
        "max_chunks": int(max_chunks),
        "max_chunk_chars": int(max_chunk_chars),
        "progress_every": int(progress_step),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rewrite global_summary gold answers from supporting chunk evidence."
    )
    parser.add_argument("--gold-file", default="data/queries/cuad_capability_gold.jsonl")
    parser.add_argument("--chunks-file", default="data/processed/cuad_chunks.jsonl")
    parser.add_argument("--out-file", default="data/queries/cuad_capability_gold_docsummary.jsonl")
    parser.add_argument("--max-chunks", type=int, default=8)
    parser.add_argument("--max-chunk-chars", type=int, default=1200)
    parser.add_argument("--max-completion-tokens", type=int, default=220)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=0,
        help="Print progress JSON to stderr every N processed global_summary rows (0 disables progress logs).",
    )
    parser.add_argument("--drop-non-global", action="store_true")
    args = parser.parse_args()

    stats = rewrite_global_summary_gold(
        gold_file=args.gold_file,
        chunks_file=args.chunks_file,
        out_file=args.out_file,
        max_chunks=args.max_chunks,
        max_chunk_chars=args.max_chunk_chars,
        max_completion_tokens=args.max_completion_tokens,
        preserve_non_global=not args.drop_non_global,
        progress_every=args.progress_every,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
