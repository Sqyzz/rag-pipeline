from __future__ import annotations

import argparse
import json
from pathlib import Path


def _doc_prefix(value: str) -> str:
    return str(value or "").split("#", 1)[0].strip()


def _norm_text(value: str) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _pick_answer(answers: list[dict], is_impossible: bool) -> str:
    if is_impossible:
        return ""
    for a in answers:
        if not isinstance(a, dict):
            continue
        text = str(a.get("text", "")).strip()
        if text:
            return text
    return ""


def _load_store_chunks(store_file: str | None) -> dict[str, list[dict]]:
    if not store_file:
        return {}
    p = Path(store_file)
    if not p.exists():
        return {}
    payload = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return {}
    out: dict[str, list[dict]] = {}
    for row in payload:
        if not isinstance(row, dict):
            continue
        doc_id = str(row.get("doc_id", "")).strip()
        if not doc_id:
            continue
        out.setdefault(doc_id, []).append(row)
    return out


def _find_supporting_chunks_for_answers(
    *,
    chunks_by_doc: dict[str, list[dict]],
    doc_id: str,
    answers: list[dict],
) -> list[dict]:
    if not chunks_by_doc:
        return []
    chunks = chunks_by_doc.get(doc_id, [])
    if not chunks:
        return []

    target_texts = []
    for a in answers:
        if not isinstance(a, dict):
            continue
        txt = str(a.get("text", "")).strip()
        if txt:
            target_texts.append(txt)
    if not target_texts:
        return []

    selected: dict[str, dict] = {}
    for chunk in chunks:
        cid = str(chunk.get("chunk_id", "")).strip()
        if not cid:
            continue
        ctext = str(chunk.get("text", "") or "")
        ctext_norm = _norm_text(ctext)
        hit_text = ""
        for t in target_texts:
            t_norm = _norm_text(t)
            if not t_norm:
                continue
            if t in ctext or t_norm in ctext_norm:
                hit_text = t
                break
        if not hit_text:
            continue
        selected[cid] = {
            "chunk_id": cid,
            "doc_id": str(chunk.get("doc_id", "")).strip(),
            "match_text": hit_text,
        }
    return list(selected.values())


def build_cuad_qa(
    raw_file: str,
    queries_out: str,
    gold_out: str,
    split_name: str,
    store_file: str | None = None,
    max_queries: int | None = None,
    answerable_only: bool = False,
    sample_random: bool = False,
    random_seed: int = 42,
) -> dict:
    root = json.loads(Path(raw_file).read_text(encoding="utf-8"))
    data = root.get("data", []) if isinstance(root, dict) else []
    if not isinstance(data, list):
        data = []

    queries: list[dict] = []
    gold: list[dict] = []
    skipped_unanswerable = 0
    chunks_by_doc = _load_store_chunks(store_file)

    for item in data:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title", "")).strip()
        paragraphs = item.get("paragraphs", [])
        if not isinstance(paragraphs, list):
            continue
        for p_idx, para in enumerate(paragraphs):
            if not isinstance(para, dict):
                continue
            qas = para.get("qas", [])
            if not isinstance(qas, list):
                continue
            for qa in qas:
                if not isinstance(qa, dict):
                    continue
                qid = str(qa.get("id", "")).strip()
                query = str(qa.get("question", "")).strip()
                if not qid or not query:
                    continue
                is_impossible = bool(qa.get("is_impossible", False))
                if answerable_only and is_impossible:
                    skipped_unanswerable += 1
                    continue
                answers = qa.get("answers", [])
                if not isinstance(answers, list):
                    answers = []
                answer = _pick_answer(answers=answers, is_impossible=is_impossible)
                qtype = "unanswerable_qa" if is_impossible else "extractive_qa"
                doc_id = f"{title}#p{p_idx}"
                supporting_chunks = (
                    _find_supporting_chunks_for_answers(
                        chunks_by_doc=chunks_by_doc,
                        doc_id=doc_id,
                        answers=answers,
                    )
                    if not is_impossible
                    else []
                )
                query_doc_key = title
                queries.append(
                    {
                        "qid": qid,
                        "type": qtype,
                        "query": query,
                        "meta": {
                            "title": title,
                            "doc_id": doc_id,
                            "split": split_name,
                            "query_doc_key": query_doc_key,
                        },
                    }
                )
                gold.append(
                    {
                        "qid": qid,
                        "type": qtype,
                        "query": query,
                        "answer": answer,
                        "meta": {
                            "title": title,
                            "doc_id": doc_id,
                            "split": split_name,
                            "is_impossible": is_impossible,
                            "query_doc_key": query_doc_key,
                        },
                        **({"supporting_chunks": supporting_chunks} if chunks_by_doc else {}),
                    }
                )
    if max_queries is not None and int(max_queries) >= 0:
        limit = int(max_queries)
        if sample_random and len(queries) > limit:
            import random

            rng = random.Random(int(random_seed))
            picked_idx = sorted(rng.sample(range(len(queries)), limit))
            queries = [queries[i] for i in picked_idx]
            gold = [gold[i] for i in picked_idx]
        else:
            queries = queries[:limit]
            gold = gold[:limit]

    q_path = Path(queries_out)
    q_path.parent.mkdir(parents=True, exist_ok=True)
    with q_path.open("w", encoding="utf-8") as f:
        for row in queries:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    g_path = Path(gold_out)
    g_path.parent.mkdir(parents=True, exist_ok=True)
    with g_path.open("w", encoding="utf-8") as f:
        for row in gold:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    final_answerable_rows = [x for x in gold if x["type"] == "extractive_qa"]
    final_supporting_hit = sum(1 for x in final_answerable_rows if (x.get("supporting_chunks") or []))

    stats = {
        "raw_file": raw_file,
        "queries_out": queries_out,
        "gold_out": gold_out,
        "split_name": split_name,
        "num_queries": len(queries),
        "num_unanswerable": sum(1 for x in gold if x["type"] == "unanswerable_qa"),
        "num_answerable": len(final_answerable_rows),
        "skipped_unanswerable": skipped_unanswerable,
        "store_file": store_file,
        "supporting_chunks_enabled": bool(chunks_by_doc),
        "supporting_chunks_rows_answerable": len(final_answerable_rows),
        "supporting_chunks_rows_with_hit": final_supporting_hit,
        "supporting_chunks_hit_rate": (
            round(final_supporting_hit / len(final_answerable_rows), 6)
            if final_answerable_rows
            else 0.0
        ),
        "answerable_only": bool(answerable_only),
        "max_queries": int(max_queries) if max_queries is not None else None,
        "sample_random": bool(sample_random),
        "random_seed": int(random_seed),
    }
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CUAD queries/gold jsonl files for compare pipeline.")
    parser.add_argument("--raw-file", default="data/raw/cuad/test.json")
    parser.add_argument("--queries-out", default="data/queries/cuad_queries_test.jsonl")
    parser.add_argument("--gold-out", default="data/queries/cuad_gold_test.jsonl")
    parser.add_argument("--split-name", default="test")
    parser.add_argument(
        "--store-file",
        default=None,
        help="Optional chunk store json to backfill supporting_chunks by answer span matching.",
    )
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument(
        "--num-queries",
        type=int,
        default=None,
        help="Alias of --max-queries: keep at most N questions.",
    )
    parser.add_argument("--answerable-only", action="store_true")
    parser.add_argument(
        "--sample-random",
        action="store_true",
        help="When limiting query count, sample randomly instead of taking first N.",
    )
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()
    limit = args.num_queries if args.num_queries is not None else args.max_queries

    stats = build_cuad_qa(
        raw_file=args.raw_file,
        queries_out=args.queries_out,
        gold_out=args.gold_out,
        split_name=args.split_name,
        store_file=args.store_file,
        max_queries=limit,
        answerable_only=args.answerable_only,
        sample_random=args.sample_random,
        random_seed=args.random_seed,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
