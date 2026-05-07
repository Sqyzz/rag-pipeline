from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

try:
    from evaluation.build_cuad_capability_qa import (
        _add_example,
        _chunk_meta_from_mentions,
        _chunk_quality_ok,
        _chunk_to_doc_key,
        _community_chunk_map,
        _doc_chunk_consistency_hits,
        _llm_designed_query,
        _load_json,
        _mentions_by_doc,
        _qid_safe,
    )
except ModuleNotFoundError:
    from src.evaluation.build_cuad_capability_qa import (
        _add_example,
        _chunk_meta_from_mentions,
        _chunk_quality_ok,
        _chunk_to_doc_key,
        _community_chunk_map,
        _doc_chunk_consistency_hits,
        _llm_designed_query,
        _load_json,
        _mentions_by_doc,
        _qid_safe,
    )


def _load_chunk_rows(path: str | None) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    chunk_path = Path(path)
    if not chunk_path.exists():
        return {}
    chunk_meta: dict[str, dict[str, Any]] = {}
    with chunk_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                continue
            chunk_id = str(row.get("chunk_id") or row.get("id") or "").strip()
            if not chunk_id:
                continue
            text = str(row.get("text", "") or "").strip()
            doc_id = str(row.get("doc_id", "") or "").strip()
            chunk_meta[chunk_id] = {
                "text": text,
                "doc_id": doc_id,
                "row": row,
            }
    return chunk_meta


def _merge_chunk_meta(*sources: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for source in sources:
        for chunk_id, meta in source.items():
            existing = merged.get(chunk_id, {}).copy()
            existing.update({k: v for k, v in meta.items() if v not in (None, "", [])})
            merged[chunk_id] = existing
    return merged


def _clean_sentence(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def _split_sentences(text: str) -> list[str]:
    cleaned = _clean_sentence(text)
    if not cleaned:
        return []
    parts = re.split(r"(?<=[.!?。！？;；])\s+", cleaned)
    return [part.strip() for part in parts if part.strip()]


def _build_grounded_answer(
    *,
    community_summary: str,
    chunk_texts: list[str],
    answer_mode: str,
    max_chars: int,
    max_sentences: int,
) -> str:
    mode = str(answer_mode or "grounded_chunks").strip().lower()
    unique_sentences: list[str] = []
    seen: set[str] = set()
    for text in chunk_texts:
        for sentence in _split_sentences(text):
            norm = sentence.lower()
            if norm in seen:
                continue
            seen.add(norm)
            unique_sentences.append(sentence)
            if len(unique_sentences) >= int(max_sentences):
                break
        if len(unique_sentences) >= int(max_sentences):
            break

    grounded = " ".join(unique_sentences).strip()
    summary = _clean_sentence(community_summary)
    if mode == "community_summary":
        answer = summary or grounded
    elif mode == "hybrid":
        answer = grounded
        if summary and summary.lower() not in grounded.lower():
            answer = f"{grounded} Theme: {summary}".strip()
    else:
        answer = grounded or summary

    answer = _clean_sentence(answer)
    if int(max_chars) > 0 and len(answer) > int(max_chars):
        answer = answer[: int(max_chars)].rstrip()
    return answer


def _supporting_edge_refs_for_chunks(
    *,
    graph: dict[str, Any],
    community_edge_ids: list[str],
    candidate_chunk_ids: list[str],
) -> list[dict[str, str]]:
    if not community_edge_ids or not candidate_chunk_ids:
        return []
    candidate_chunks = {str(chunk_id).strip() for chunk_id in candidate_chunk_ids if str(chunk_id).strip()}
    if not candidate_chunks:
        return []
    edge_by_id = {str(e.get("edge_id", "")).strip(): e for e in (graph.get("edges", []) or []) if str(e.get("edge_id", "")).strip()}
    refs: list[dict[str, str]] = []
    seen: set[str] = set()
    for edge_id in community_edge_ids:
        eid = str(edge_id).strip()
        if not eid or eid in seen:
            continue
        edge = edge_by_id.get(eid)
        if not edge:
            continue
        mentions = edge.get("mentions", []) or []
        if any(str(m.get("chunk_id", "")).strip() in candidate_chunks for m in mentions):
            seen.add(eid)
            refs.append({"edge_id": eid})
    return refs


def build_cuad_global_from_communities(
    *,
    graph_file: str,
    communities_file: str,
    queries_out: str,
    gold_out: str,
    chunks_file: str | None = None,
    per_type: int = 20,
    random_seed: int = 42,
    global_summary_chunk_cap: int = 30,
    summary_answer_max_chars: int = 420,
    summary_answer_mode: str = "grounded_chunks",
    summary_answer_max_sentences: int = 3,
    question_style: str = "template",
    llm_question_model: str | None = None,
    llm_question_temperature: float = 0.2,
    llm_question_max_tokens: int = 96,
    global_summary_min_high_quality_chunks: int = 2,
    global_summary_target_high_quality_chunks: int = 3,
    global_summary_doc_consistency_min_hits: int = 1,
    min_community_edges: int = 1,
) -> dict[str, Any]:
    graph = _load_json(graph_file)
    communities = _load_json(communities_file)
    community_chunks = _community_chunk_map(graph, communities) if isinstance(communities, dict) else {}
    mentions = _mentions_by_doc(graph)
    chunk_to_doc = _chunk_to_doc_key(mentions)
    mention_chunk_meta = _chunk_meta_from_mentions(mentions)
    file_chunk_meta = _load_chunk_rows(chunks_file)
    chunk_meta = _merge_chunk_meta(mention_chunk_meta, file_chunk_meta)
    rng = random.Random(int(random_seed))

    queries: list[dict[str, Any]] = []
    gold: list[dict[str, Any]] = []
    summary_quality_stats = {
        "candidates_total": 0,
        "candidates_after_quality": 0,
        "dropped_small_community": 0,
        "dropped_low_quality": 0,
        "dropped_doc_consistency": 0,
        "dropped_below_min_chunks": 0,
    }
    llm_query_stats = {
        "attempted": 0,
        "success": 0,
        "fallback": 0,
        "failure_reasons": {},
        "error_samples": [],
    }

    summary_candidates: list[dict[str, Any]] = []
    for community in communities.get("communities", []) or []:
        if not isinstance(community, dict):
            continue
        community_id = str(community.get("community_id", "") or "").strip()
        summary = str(community.get("summary", "") or "").strip()
        edges = [edge for edge in (community.get("edges", []) or []) if str(edge).strip()]
        if not community_id or not summary:
            continue
        if len(edges) < int(min_community_edges):
            summary_quality_stats["dropped_small_community"] += 1
            continue
        doc_keys = sorted({chunk_to_doc.get(chunk_id, "") for chunk_id in community_chunks.get(community_id, set()) if chunk_to_doc.get(chunk_id)})
        for doc_key in doc_keys:
            doc_chunk_ids = sorted([chunk_id for chunk_id in community_chunks.get(community_id, set()) if chunk_to_doc.get(chunk_id) == doc_key])
            if not doc_chunk_ids:
                continue
            summary_quality_stats["candidates_total"] += 1
            qualified_chunk_ids: list[str] = []
            qualified_chunk_texts: list[str] = []
            for chunk_id in doc_chunk_ids:
                meta = chunk_meta.get(chunk_id, {})
                text = str(meta.get("text", "") or "").strip()
                if not _chunk_quality_ok(text, min_chars=100):
                    summary_quality_stats["dropped_low_quality"] += 1
                    continue
                hits = _doc_chunk_consistency_hits(doc_key, text)
                if int(global_summary_doc_consistency_min_hits) > 0 and hits < int(global_summary_doc_consistency_min_hits):
                    summary_quality_stats["dropped_doc_consistency"] += 1
                    continue
                qualified_chunk_ids.append(chunk_id)
                qualified_chunk_texts.append(text)
            if len(qualified_chunk_ids) < int(global_summary_min_high_quality_chunks):
                summary_quality_stats["dropped_below_min_chunks"] += 1
                continue
            summary_quality_stats["candidates_after_quality"] += 1
            if int(global_summary_target_high_quality_chunks) > 0 and len(qualified_chunk_ids) > int(global_summary_target_high_quality_chunks):
                paired = list(zip(qualified_chunk_ids, qualified_chunk_texts, strict=False))
                paired = rng.sample(paired, int(global_summary_target_high_quality_chunks))
                paired.sort(key=lambda item: item[0])
                qualified_chunk_ids = [item[0] for item in paired]
                qualified_chunk_texts = [item[1] for item in paired]
            summary_candidates.append(
                {
                    "doc_key": doc_key,
                    "community_id": community_id,
                    "summary": summary,
                    "community_edge_ids": edges,
                    "chunk_ids": qualified_chunk_ids,
                    "chunk_texts": qualified_chunk_texts,
                }
            )

    rng.shuffle(summary_candidates)
    picked_summary: list[dict[str, Any]] = []
    used_doc_keys: set[str] = set()
    for candidate in summary_candidates:
        doc_key = str(candidate.get("doc_key", "") or "").strip()
        if not doc_key or doc_key in used_doc_keys:
            continue
        used_doc_keys.add(doc_key)
        picked_summary.append(candidate)
        if len(picked_summary) >= int(per_type):
            break

    for index, row in enumerate(picked_summary, start=1):
        doc_key = str(row.get("doc_key", "") or "").strip()
        template_query = (
            f'For contract "{doc_key}", what contract-wide obligations, risks, or themes are directly supported '
            f'by community "{row.get("community_id", "")}"?'
        )
        query = template_query
        if str(question_style).strip().lower() == "llm":
            llm_query_stats["attempted"] += 1
            llm_debug: dict[str, Any] = {}
            query = _llm_designed_query(
                qtype="global_summary",
                template_query=template_query,
                doc_key=doc_key,
                answer=str(row.get("summary", "") or "").strip(),
                evidence_snippets=[str(text).strip() for text in (row.get("chunk_texts") or [])[:3]],
                model=llm_question_model,
                temperature=float(llm_question_temperature),
                max_tokens=int(llm_question_max_tokens),
                debug=llm_debug,
            ) or template_query
            if query == template_query:
                llm_query_stats["fallback"] += 1
                reason = str(llm_debug.get("reason", "") or "unknown").strip() or "unknown"
                llm_query_stats["failure_reasons"][reason] = int(llm_query_stats["failure_reasons"].get(reason, 0)) + 1
                if len(llm_query_stats["error_samples"]) < 5:
                    llm_query_stats["error_samples"].append(
                        {
                            "doc_key": doc_key,
                            "community_id": str(row.get("community_id", "") or "").strip(),
                            "reason": reason,
                            "stage": str(llm_debug.get("stage", "") or "").strip(),
                            "raw": str(llm_debug.get("raw", "") or "").strip()[:500],
                            "candidate": str(llm_debug.get("candidate", "") or "").strip()[:280],
                        }
                    )
            else:
                llm_query_stats["success"] += 1
        candidate_chunk_ids = list(row.get("chunk_ids", []) or [])
        chunk_texts = list(row.get("chunk_texts", []) or [])
        if int(global_summary_chunk_cap) > 0 and len(candidate_chunk_ids) > int(global_summary_chunk_cap):
            paired = list(zip(candidate_chunk_ids, chunk_texts, strict=False))
            paired = rng.sample(paired, int(global_summary_chunk_cap))
            paired.sort(key=lambda item: item[0])
            candidate_chunk_ids = [item[0] for item in paired]
            chunk_texts = [item[1] for item in paired]
        supporting_chunks = []
        for chunk_id in candidate_chunk_ids:
            chunk_ref = {"chunk_id": chunk_id}
            doc_id = str((chunk_meta.get(chunk_id, {}) or {}).get("doc_id", "") or chunk_to_doc.get(chunk_id, "")).strip()
            if doc_id:
                chunk_ref["doc_id"] = doc_id
            supporting_chunks.append(chunk_ref)
        supporting_edges = _supporting_edge_refs_for_chunks(
            graph=graph,
            community_edge_ids=[str(x).strip() for x in (row.get("community_edge_ids", []) or []) if str(x).strip()],
            candidate_chunk_ids=candidate_chunk_ids,
        )
        answer = _build_grounded_answer(
            community_summary=str(row.get("summary", "") or "").strip(),
            chunk_texts=chunk_texts,
            answer_mode=summary_answer_mode,
            max_chars=int(summary_answer_max_chars),
            max_sentences=int(summary_answer_max_sentences),
        )
        _add_example(
            queries=queries,
            gold=gold,
            qid=f'{_qid_safe(doc_key)}__global_summary__{index:04d}',
            qtype="global_summary",
            query=query,
            answer=answer,
            doc_key=doc_key,
            supporting_edges=supporting_edges,
            supporting_communities=[{"community_id": str(row.get("community_id", "") or "").strip()}],
            supporting_chunks=supporting_chunks,
            extra_gold={
                "answer_mode": str(summary_answer_mode),
            },
        )

    Path(queries_out).parent.mkdir(parents=True, exist_ok=True)
    Path(gold_out).parent.mkdir(parents=True, exist_ok=True)
    with Path(queries_out).open("w", encoding="utf-8") as handle:
        for row in queries:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with Path(gold_out).open("w", encoding="utf-8") as handle:
        for row in gold:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {
        "graph_file": graph_file,
        "communities_file": communities_file,
        "chunks_file": chunks_file,
        "queries_out": queries_out,
        "gold_out": gold_out,
        "num_queries": len(queries),
        "num_gold": len(gold),
        "per_type_target": int(per_type),
        "by_type": {"global_summary": len(queries)},
        "summary_answer_mode": str(summary_answer_mode),
        "summary_answer_max_chars": int(summary_answer_max_chars),
        "summary_answer_max_sentences": int(summary_answer_max_sentences),
        "global_summary_min_high_quality_chunks": int(global_summary_min_high_quality_chunks),
        "global_summary_target_high_quality_chunks": int(global_summary_target_high_quality_chunks),
        "global_summary_doc_consistency_min_hits": int(global_summary_doc_consistency_min_hits),
        "min_community_edges": int(min_community_edges),
        "summary_quality_stats": summary_quality_stats,
        "llm_query_stats": llm_query_stats,
        "random_seed": int(random_seed),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CUAD global_summary QA from final graph communities.")
    parser.add_argument("--graph-file", default="outputs/graph/cuad_graph_test.json")
    parser.add_argument("--communities-file", default="outputs/graph/cuad_communities_test.json")
    parser.add_argument("--chunks-file", default=None)
    parser.add_argument("--queries-out", default="data/queries/cuad_global_queries.jsonl")
    parser.add_argument("--gold-out", default="data/queries/cuad_global_gold.jsonl")
    parser.add_argument("--per-type", type=int, default=20)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--global-summary-chunk-cap", type=int, default=30)
    parser.add_argument("--summary-answer-max-chars", type=int, default=420)
    parser.add_argument("--summary-answer-mode", choices=["grounded_chunks", "community_summary", "hybrid"], default="grounded_chunks")
    parser.add_argument("--summary-answer-max-sentences", type=int, default=3)
    parser.add_argument("--question-style", choices=["template", "llm"], default="template")
    parser.add_argument("--llm-question-model", default=None)
    parser.add_argument("--llm-question-temperature", type=float, default=0.2)
    parser.add_argument("--llm-question-max-tokens", type=int, default=96)
    parser.add_argument("--global-summary-min-high-quality-chunks", type=int, default=2)
    parser.add_argument("--global-summary-target-high-quality-chunks", type=int, default=3)
    parser.add_argument("--global-summary-doc-consistency-min-hits", type=int, default=1)
    parser.add_argument("--min-community-edges", type=int, default=1)
    args = parser.parse_args()

    stats = build_cuad_global_from_communities(
        graph_file=args.graph_file,
        communities_file=args.communities_file,
        chunks_file=args.chunks_file,
        queries_out=args.queries_out,
        gold_out=args.gold_out,
        per_type=args.per_type,
        random_seed=args.random_seed,
        global_summary_chunk_cap=args.global_summary_chunk_cap,
        summary_answer_max_chars=args.summary_answer_max_chars,
        summary_answer_mode=args.summary_answer_mode,
        summary_answer_max_sentences=args.summary_answer_max_sentences,
        question_style=args.question_style,
        llm_question_model=args.llm_question_model,
        llm_question_temperature=args.llm_question_temperature,
        llm_question_max_tokens=args.llm_question_max_tokens,
        global_summary_min_high_quality_chunks=args.global_summary_min_high_quality_chunks,
        global_summary_target_high_quality_chunks=args.global_summary_target_high_quality_chunks,
        global_summary_doc_consistency_min_hits=args.global_summary_doc_consistency_min_hits,
        min_community_edges=args.min_community_edges,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
