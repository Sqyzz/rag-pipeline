from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def _read_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            obj = json.loads(s)
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _to_set(items: Any) -> set[str]:
    out: set[str] = set()
    if not isinstance(items, list):
        return out
    for x in items:
        s = str(x or "").strip()
        if s:
            out.add(s)
    return out


def _hex40(s: str) -> bool:
    return bool(re.fullmatch(r"[0-9a-f]{40}", str(s or "").strip().lower()))


def _extract_gold(gold_row: dict[str, Any]) -> tuple[set[str], set[str]]:
    chunks: set[str] = set()
    docs: set[str] = set()
    for item in (gold_row.get("supporting_chunks") or []):
        if not isinstance(item, dict):
            continue
        cid = str(item.get("chunk_id", "") or "").strip()
        did = str(item.get("doc_id", "") or "").strip()
        if cid:
            chunks.add(cid)
        if did:
            docs.add(did)
    if not docs and chunks:
        docs = set(chunks)
    return chunks, docs


def _extract_pred(payload: dict[str, Any]) -> tuple[list[str], list[str]]:
    cids: list[str] = []
    dids: list[str] = []
    ev = payload.get("evidence_chunks")
    if not isinstance(ev, list):
        return cids, dids
    for x in ev:
        if not isinstance(x, dict):
            continue
        cid = str(x.get("chunk_id", "") or "").strip()
        did = str(x.get("doc_id", "") or "").strip()
        cids.append(cid)
        dids.append(did)
    return cids, dids


def _safe_div(a: float, b: float) -> float:
    if b <= 0:
        return 0.0
    return float(a / b)


def _detect_methods(pred_rows: list[dict[str, Any]]) -> list[str]:
    found: set[str] = set()
    for row in pred_rows:
        regs = row.get("regimes")
        if not isinstance(regs, dict):
            continue
        for by_method in regs.values():
            if not isinstance(by_method, dict):
                continue
            for m in by_method.keys():
                s = str(m).strip()
                if s:
                    found.add(s)
    return sorted(found)


def evaluate_alignment(
    *,
    pred_file: str,
    gold_file: str,
    methods: list[str] | None = None,
) -> dict[str, Any]:
    pred_rows = _read_jsonl(pred_file)
    gold_rows = _read_jsonl(gold_file)
    gold_map: dict[str, dict[str, Any]] = {str(x.get("qid", "")).strip(): x for x in gold_rows if isinstance(x, dict)}

    if not methods:
        methods = _detect_methods(pred_rows)

    agg: dict[tuple[str, str, str, int], dict[str, float]] = {}

    for row in pred_rows:
        qid = str(row.get("qid", "")).strip()
        if not qid or qid not in gold_map:
            continue
        gold_chunks, gold_docs = _extract_gold(gold_map[qid])

        regs = row.get("regimes")
        if not isinstance(regs, dict):
            continue

        mode = str(row.get("mode", "reject")).strip().lower() or "reject"
        top_k = int(row.get("top_k", 0) or 0)

        for regime, by_method in regs.items():
            if not isinstance(by_method, dict):
                continue
            rg = str(regime)
            for method in methods:
                payload = by_method.get(method)
                if not isinstance(payload, dict):
                    continue

                raw_cids, raw_dids = _extract_pred(payload)
                pred_chunks = {x for x in raw_cids if x}
                pred_docs = {x for x in raw_dids if x}

                overlap_chunks = len(pred_chunks.intersection(gold_chunks))
                overlap_docs = len(pred_docs.intersection(gold_docs))
                chunk_hit = 1.0 if overlap_chunks > 0 else 0.0
                doc_hit = 1.0 if overlap_docs > 0 else 0.0

                key = (rg, method, mode, top_k)
                cur = agg.setdefault(
                    key,
                    {
                        "samples": 0.0,
                        "gold_chunks_total": 0.0,
                        "gold_docs_total": 0.0,
                        "pred_chunks_total": 0.0,
                        "pred_docs_total": 0.0,
                        "pred_chunk_id_non_empty": 0.0,
                        "pred_doc_id_non_empty": 0.0,
                        "pred_chunk_id_hash_like": 0.0,
                        "chunk_overlap_total": 0.0,
                        "doc_overlap_total": 0.0,
                        "chunk_hit_count": 0.0,
                        "doc_hit_count": 0.0,
                    },
                )

                cur["samples"] += 1.0
                cur["gold_chunks_total"] += float(len(gold_chunks))
                cur["gold_docs_total"] += float(len(gold_docs))
                cur["pred_chunks_total"] += float(len(raw_cids))
                cur["pred_docs_total"] += float(len(raw_dids))
                cur["pred_chunk_id_non_empty"] += float(sum(1 for x in raw_cids if x))
                cur["pred_doc_id_non_empty"] += float(sum(1 for x in raw_dids if x))
                cur["pred_chunk_id_hash_like"] += float(sum(1 for x in raw_cids if _hex40(x)))
                cur["chunk_overlap_total"] += float(overlap_chunks)
                cur["doc_overlap_total"] += float(overlap_docs)
                cur["chunk_hit_count"] += chunk_hit
                cur["doc_hit_count"] += doc_hit

    rows: list[dict[str, Any]] = []
    for (regime, method, mode, top_k), m in sorted(agg.items()):
        samples = m["samples"]
        pred_chunks_total = m["pred_chunks_total"]
        pred_docs_total = m["pred_docs_total"]
        row = {
            "regime": regime,
            "method": method,
            "mode": mode,
            "top_k": int(top_k),
            "samples": int(samples),
            "chunk_id_non_empty_rate": round(_safe_div(m["pred_chunk_id_non_empty"], pred_chunks_total), 6),
            "doc_id_non_empty_rate": round(_safe_div(m["pred_doc_id_non_empty"], pred_docs_total), 6),
            "chunk_id_hash_like_rate": round(_safe_div(m["pred_chunk_id_hash_like"], pred_chunks_total), 6),
            "gold_chunk_hit_rate": round(_safe_div(m["chunk_hit_count"], samples), 6),
            "gold_doc_hit_rate": round(_safe_div(m["doc_hit_count"], samples), 6),
            "evidence_recall_chunks": round(_safe_div(m["chunk_overlap_total"], m["gold_chunks_total"]), 6),
            "evidence_recall_docs": round(_safe_div(m["doc_overlap_total"], m["gold_docs_total"]), 6),
            "avg_pred_evidence_chunks": round(_safe_div(pred_chunks_total, samples), 6),
            "avg_pred_evidence_docs": round(_safe_div(pred_docs_total, samples), 6),
        }
        rows.append(row)

    return {
        "pred_file": pred_file,
        "gold_file": gold_file,
        "methods": methods,
        "num_rows": len(rows),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate chunk-id alignment and recall from compare answers jsonl")
    parser.add_argument("--pred-file", required=True)
    parser.add_argument("--gold-file", required=True)
    parser.add_argument("--methods", default="")
    parser.add_argument("--out-json", default="outputs/results/chunk_id_alignment_summary.json")
    parser.add_argument("--out-csv", default="outputs/results/chunk_id_alignment_summary.csv")
    args = parser.parse_args()

    methods = [x.strip() for x in str(args.methods).split(",") if x.strip()]
    result = evaluate_alignment(
        pred_file=args.pred_file,
        gold_file=args.gold_file,
        methods=methods or None,
    )

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "regime",
        "method",
        "mode",
        "top_k",
        "samples",
        "chunk_id_non_empty_rate",
        "doc_id_non_empty_rate",
        "chunk_id_hash_like_rate",
        "gold_chunk_hit_rate",
        "gold_doc_hit_rate",
        "evidence_recall_chunks",
        "evidence_recall_docs",
        "avg_pred_evidence_chunks",
        "avg_pred_evidence_docs",
    ]
    with out_csv.open("w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for row in result["rows"]:
            f.write(",".join(str(row.get(c, "")) for c in cols) + "\n")

    print(json.dumps({"out_json": str(out_json), "out_csv": str(out_csv), "num_rows": result["num_rows"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
