from __future__ import annotations

import json
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from baselines.text_mapreduce import mapreduce_answer
from baselines.vector_rag import answer_with_context, retrieve_and_answer
from utils.config import cfg


def _load_queries(path: str) -> list[dict]:
    queries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                queries.append(json.loads(line))
    return queries


def _write_jsonl(path: str, rows: list[dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_queries(
    queries_file: str = "data/queries/queries.jsonl",
    chunks_file: str = "data/processed/chunks.jsonl",
    idx_file: str = "outputs/indexes/faiss.idx",
    store_file: str = "outputs/indexes/chunk_store.json",
    vector_out: str = "outputs/results/vector_answers.jsonl",
    mapreduce_out: str = "outputs/results/mapreduce_answers.jsonl",
    top_k: int | None = None,
    map_count: int = 10,
) -> tuple[int, str, str]:
    queries = _load_queries(queries_file)
    if not queries:
        raise RuntimeError(f"No queries found in {queries_file}")

    retrieval_top_k = top_k if top_k is not None else int(cfg.retrieval.top_k)

    vector_rows = []
    mapreduce_rows = []

    for q in queries:
        qid = q.get("qid")
        qtype = q.get("type")
        query = q["query"]

        contexts = retrieve_and_answer(query, idx_file, store_file, retrieval_top_k)
        vector_answer = answer_with_context(query, contexts)
        vector_rows.append(
            {
                "qid": qid,
                "type": qtype,
                "query": query,
                "contexts": contexts,
                "answer": vector_answer,
            }
        )

        mapreduce_answer_text = mapreduce_answer(query, chunks_file, map_count=map_count)
        mapreduce_rows.append(
            {
                "qid": qid,
                "type": qtype,
                "query": query,
                "answer": mapreduce_answer_text,
            }
        )

    _write_jsonl(vector_out, vector_rows)
    _write_jsonl(mapreduce_out, mapreduce_rows)
    return len(queries), vector_out, mapreduce_out


if __name__ == "__main__":
    total, vector_path, mapreduce_path = run_queries()
    print(f"processed_queries={total}")
    print(f"vector_out={vector_path}")
    print(f"mapreduce_out={mapreduce_path}")
