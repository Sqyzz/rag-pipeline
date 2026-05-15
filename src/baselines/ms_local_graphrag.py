from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.alignment_payloads import build_minimal_trace_bundle
from utils.embedder import embed_texts
from utils.llm_wrapper import llm_chat
from utils.telemetry import Telemetry


_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{1,}")
_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "into",
    "under",
    "shall",
    "agreement",
    "contract",
    "section",
    "article",
    "clause",
    "party",
    "parties",
    "what",
    "which",
    "when",
    "where",
    "whose",
    "role",
    "responsibility",
    "responsibilities",
}


def _load_json(path: str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _doc_prefix(doc_id: Any) -> str:
    return str(doc_id or "").split("#", 1)[0].strip()


def _normalize_answer_mode(answer_mode: str | None) -> str:
    mode = str(answer_mode or "reject").strip().lower()
    return mode if mode in {"reject", "open"} else "reject"


def _tokens(text: Any) -> set[str]:
    return {
        tok.lower()
        for tok in _TOKEN_RE.findall(str(text or ""))
        if len(tok) >= 3 and not tok.isdigit() and tok.lower() not in _STOPWORDS
    }


def _token_score(query_tokens: set[str], text: Any) -> float:
    if not query_tokens:
        return 0.0
    text_tokens = _tokens(text)
    if not text_tokens:
        return 0.0
    overlap = query_tokens & text_tokens
    return len(overlap) / max(1.0, math.sqrt(len(text_tokens)))


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def _load_chunk_store(chunks_file: str | None) -> dict[str, dict]:
    if not chunks_file:
        return {}
    out: dict[str, dict] = {}
    path = Path(chunks_file)
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            chunk_id = str(row.get("chunk_id") or "").strip()
            if chunk_id:
                out[chunk_id] = row
    return out


def _edge_lookup(graph: dict) -> dict[str, dict]:
    return {str(e.get("edge_id")): e for e in graph.get("edges", []) if e.get("edge_id")}


def _all_communities(payload: Any) -> list[dict]:
    if isinstance(payload, list):
        return payload
    return payload.get("communities", []) if isinstance(payload, dict) else []


def _select_level_communities(payload: Any, query_level: int) -> list[dict]:
    communities = _all_communities(payload)
    if not isinstance(payload, dict):
        return communities
    levels = payload.get("levels", [])
    if not levels:
        return communities
    level_ids = {int(x["level"]): set(x["community_ids"]) for x in levels}
    available = sorted(level_ids)
    target = available[-1] if query_level < 0 else (query_level if query_level in level_ids else available[0])
    allowed = level_ids[target]
    return [c for c in communities if c.get("community_id") in allowed]


def _edge_has_doc(edge: dict, doc_prefix_filter: str | None) -> bool:
    target = _doc_prefix(doc_prefix_filter)
    if not target:
        return True
    return any(_doc_prefix(m.get("doc_id")) == target for m in edge.get("mentions", []) or [])


def _filter_edges_by_doc(edges: list[dict], doc_prefix_filter: str | None, strict_doc_scope: bool) -> list[dict]:
    if not doc_prefix_filter:
        return edges
    kept = [e for e in edges if _edge_has_doc(e, doc_prefix_filter)]
    if kept:
        return kept
    return [] if strict_doc_scope else edges


def _filter_communities_by_doc(
    communities: list[dict],
    edge_by_id: dict[str, dict],
    doc_prefix_filter: str | None,
    strict_doc_scope: bool,
) -> list[dict]:
    if not doc_prefix_filter:
        return communities
    kept = []
    for community in communities:
        for edge_id in community.get("edges", []) or []:
            edge = edge_by_id.get(str(edge_id))
            if edge and _edge_has_doc(edge, doc_prefix_filter):
                kept.append(community)
                break
    if kept:
        return kept
    return [] if strict_doc_scope else communities


def _node_label(node: Any) -> str:
    if isinstance(node, dict):
        return str(node.get("label") or node.get("id") or "").strip()
    return str(node or "").strip()


def _rank_entities(graph: dict, query: str, limit: int) -> list[dict]:
    query_l = query.lower()
    query_tokens = _tokens(query)
    ranked = []
    for node in graph.get("nodes", []) or []:
        label = _node_label(node)
        if not label or label == "***":
            continue
        aliases = node.get("aliases", []) if isinstance(node, dict) else []
        type_text = " ".join(node.get("types", []) if isinstance(node, dict) else [])
        text = " ".join([label, type_text, " ".join(str(a) for a in aliases)])
        score = _token_score(query_tokens, text)
        if label.lower() in query_l:
            score += 2.0
        if score <= 0:
            continue
        ranked.append(
            {
                "node_id": str(node.get("id") or label) if isinstance(node, dict) else label,
                "label": label,
                "types": list(node.get("types", []) if isinstance(node, dict) else []),
                "score": round(float(score), 6),
            }
        )
    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked[: max(0, int(limit))]


def _edge_text(edge: dict, mention_limit: int = 4) -> str:
    mentions = edge.get("mentions", []) or []
    mention_text = " ".join(
        str(m.get("text") or m.get("evidence") or "") for m in mentions[: max(0, int(mention_limit))]
    )
    return " ".join(
        [
            str(edge.get("source") or ""),
            str(edge.get("relation") or ""),
            str(edge.get("target") or ""),
            mention_text,
        ]
    )


def _rank_edges(edges: list[dict], query: str, limit: int) -> list[dict]:
    query_l = query.lower()
    query_tokens = _tokens(query)
    ranked = []
    for edge in edges:
        text = _edge_text(edge)
        score = _token_score(query_tokens, text)
        source = str(edge.get("source") or "").lower()
        target = str(edge.get("target") or "").lower()
        relation = str(edge.get("relation") or "").replace("_", " ").lower()
        if source and source in query_l:
            score += 1.5
        if target and target in query_l:
            score += 1.5
        if relation and relation in query_l:
            score += 0.75
        score += min(float(edge.get("weight") or 0.0), 25.0) / 100.0
        if score <= 0:
            continue
        row = dict(edge)
        row["_local_score"] = round(float(score), 6)
        ranked.append(row)
    ranked.sort(key=lambda x: x["_local_score"], reverse=True)
    return ranked[: max(0, int(limit))]


def _rank_communities(
    communities: list[dict],
    query: str,
    limit: int,
    use_community_summaries: bool,
    max_summary_chars: int,
) -> list[dict]:
    query_tokens = _tokens(query)
    ranked = []
    for community in communities:
        nodes = " ".join(str(x) for x in community.get("nodes", [])[:80])
        summary = str(community.get("summary") or "")[:max_summary_chars] if use_community_summaries else ""
        text = f"{summary} {nodes}"
        score = _token_score(query_tokens, text)
        if score <= 0:
            continue
        row = dict(community)
        row["_local_score"] = round(float(score), 6)
        ranked.append(row)
    ranked.sort(key=lambda x: x["_local_score"], reverse=True)
    return ranked[: max(0, int(limit))]


def _incident_edges_for_entities(
    edges: list[dict],
    entity_rows: list[dict],
    limit: int,
) -> list[dict]:
    entity_ids = {str(e.get("node_id") or "").lower() for e in entity_rows}
    entity_labels = {str(e.get("label") or "").lower() for e in entity_rows}
    out = []
    for edge in edges:
        source = str(edge.get("source") or "").lower()
        target = str(edge.get("target") or "").lower()
        if source in entity_ids or source in entity_labels or target in entity_ids or target in entity_labels:
            row = dict(edge)
            row["_local_score"] = max(float(row.get("_local_score") or 0.0), 1.0)
            out.append(row)
        if len(out) >= limit:
            break
    return out


def _community_edges(
    communities: list[dict],
    edge_by_id: dict[str, dict],
    per_community: int,
) -> list[dict]:
    out = []
    for community in communities:
        base_score = float(community.get("_local_score") or 0.0)
        count = 0
        for edge_id in community.get("edges", []) or []:
            edge = edge_by_id.get(str(edge_id))
            if not edge:
                continue
            row = dict(edge)
            row["_local_score"] = max(float(row.get("_local_score") or 0.0), base_score)
            row["_community_id"] = community.get("community_id")
            out.append(row)
            count += 1
            if count >= per_community:
                break
    return out


def _dedupe_edges(edges: list[dict]) -> list[dict]:
    by_id: dict[str, dict] = {}
    for edge in edges:
        edge_id = str(edge.get("edge_id") or "").strip()
        if not edge_id:
            continue
        prev = by_id.get(edge_id)
        if not prev or float(edge.get("_local_score") or 0.0) > float(prev.get("_local_score") or 0.0):
            by_id[edge_id] = edge
    out = list(by_id.values())
    out.sort(key=lambda x: float(x.get("_local_score") or 0.0), reverse=True)
    return out


def _chunk_from_mention(
    mention: dict,
    edge: dict,
    chunk_store: dict[str, dict],
) -> dict | None:
    chunk_id = str(mention.get("chunk_id") or "").strip()
    if not chunk_id:
        return None
    stored = chunk_store.get(chunk_id, {})
    text = str(stored.get("text") or mention.get("text") or mention.get("evidence") or "").strip()
    doc_id = str(stored.get("doc_id") or mention.get("doc_id") or "").strip()
    if not text and not doc_id:
        return None
    return {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "source": stored.get("source"),
        "meta": stored.get("meta", {}),
        "text": text,
        "edge_ids": [edge.get("edge_id")],
        "graph_score": float(edge.get("_local_score") or 0.0),
    }


def _collect_candidate_chunks(
    edges: list[dict],
    chunk_store: dict[str, dict],
    doc_prefix_filter: str | None,
    strict_doc_scope: bool,
    max_candidates: int,
) -> list[dict]:
    wanted = _doc_prefix(doc_prefix_filter)
    by_id: dict[str, dict] = {}
    for edge in edges:
        for mention in edge.get("mentions", []) or []:
            chunk = _chunk_from_mention(mention, edge, chunk_store)
            if not chunk:
                continue
            if wanted and _doc_prefix(chunk.get("doc_id")) != wanted:
                continue
            existing = by_id.get(chunk["chunk_id"])
            if existing:
                existing["graph_score"] += float(chunk.get("graph_score") or 0.0)
                existing["edge_ids"].extend(x for x in chunk.get("edge_ids", []) if x not in existing["edge_ids"])
            else:
                by_id[chunk["chunk_id"]] = chunk
    if not by_id and doc_prefix_filter and not strict_doc_scope:
        return _collect_candidate_chunks(edges, chunk_store, None, False, max_candidates)
    out = list(by_id.values())
    out.sort(key=lambda x: float(x.get("graph_score") or 0.0), reverse=True)
    return out[: max(1, int(max_candidates))]


def _lexical_text_unit_fallback(
    query: str,
    chunk_store: dict[str, dict],
    doc_prefix_filter: str | None,
    limit: int,
) -> list[dict]:
    query_tokens = _tokens(query)
    wanted = _doc_prefix(doc_prefix_filter)
    ranked = []
    for chunk in chunk_store.values():
        if wanted and _doc_prefix(chunk.get("doc_id")) != wanted:
            continue
        score = _token_score(query_tokens, chunk.get("text"))
        if score <= 0:
            continue
        ranked.append(
            {
                "chunk_id": str(chunk.get("chunk_id") or "").strip(),
                "doc_id": str(chunk.get("doc_id") or "").strip(),
                "source": chunk.get("source"),
                "meta": chunk.get("meta", {}),
                "text": str(chunk.get("text") or "").strip(),
                "edge_ids": [],
                "graph_score": 0.0,
                "lexical_fallback_score": round(float(score), 6),
            }
        )
    ranked.sort(key=lambda x: x.get("lexical_fallback_score", 0.0), reverse=True)
    return ranked[: max(0, int(limit))]


def _merge_chunks(primary: list[dict], fallback: list[dict], limit: int) -> list[dict]:
    by_id: dict[str, dict] = {}
    for chunk in primary + fallback:
        chunk_id = str(chunk.get("chunk_id") or "").strip()
        if not chunk_id:
            continue
        if chunk_id not in by_id:
            by_id[chunk_id] = dict(chunk)
            continue
        prev = by_id[chunk_id]
        prev["graph_score"] = float(prev.get("graph_score") or 0.0) + float(chunk.get("graph_score") or 0.0)
        prev["lexical_fallback_score"] = max(
            float(prev.get("lexical_fallback_score") or 0.0),
            float(chunk.get("lexical_fallback_score") or 0.0),
        )
        prev_edges = prev.setdefault("edge_ids", [])
        for edge_id in chunk.get("edge_ids", []) or []:
            if edge_id not in prev_edges:
                prev_edges.append(edge_id)
    out = list(by_id.values())
    out.sort(
        key=lambda x: (
            float(x.get("graph_score") or 0.0),
            float(x.get("lexical_fallback_score") or 0.0),
        ),
        reverse=True,
    )
    return out[: max(1, int(limit))]


def _rerank_chunks(
    query: str,
    chunks: list[dict],
    telemetry: Telemetry,
    use_embedding_rerank: bool,
) -> list[dict]:
    if not chunks:
        return []
    query_tokens = _tokens(query)
    lexical_scores = [_token_score(query_tokens, chunk.get("text")) for chunk in chunks]
    if not use_embedding_rerank:
        rescored = []
        for chunk, lex in zip(chunks, lexical_scores):
            row = dict(chunk)
            row["semantic_score"] = 0.0
            row["lexical_score"] = round(float(lex), 6)
            row["score"] = round(float(lex) + 0.05 * float(row.get("graph_score") or 0.0), 6)
            rescored.append(row)
        rescored.sort(key=lambda x: x["score"], reverse=True)
        return rescored

    q_vec, q_meta = embed_texts([query], return_meta=True)
    c_vec, c_meta = embed_texts([str(c.get("text") or "") for c in chunks], return_meta=True)
    telemetry.add_embedding(q_meta)
    telemetry.add_embedding(c_meta)
    q = _normalize_rows(np.asarray(q_vec, dtype="float32"))[0]
    c = _normalize_rows(np.asarray(c_vec, dtype="float32"))
    sims = c @ q

    graph_scores = [float(chunk.get("graph_score") or 0.0) for chunk in chunks]
    max_graph = max(graph_scores) if graph_scores else 0.0
    rescored = []
    for chunk, sim, lex, graph_score in zip(chunks, sims.tolist(), lexical_scores, graph_scores):
        graph_norm = graph_score / max_graph if max_graph > 0 else 0.0
        row = dict(chunk)
        row["semantic_score"] = round(float(sim), 6)
        row["lexical_score"] = round(float(lex), 6)
        row["score"] = round(float(sim) + 0.15 * float(lex) + 0.10 * graph_norm, 6)
        rescored.append(row)
    rescored.sort(key=lambda x: x["score"], reverse=True)
    return rescored


def _community_report_view(communities: list[dict], max_summary_chars: int) -> list[dict]:
    out = []
    for community in communities:
        summary = str(community.get("summary") or "").strip()
        if not summary:
            summary = " | ".join(str(x) for x in community.get("nodes", [])[:25])
        out.append(
            {
                "community_id": community.get("community_id"),
                "level": community.get("level"),
                "score": community.get("_local_score"),
                "summary": summary[:max_summary_chars],
            }
        )
    return out


def _relationship_view(edges: list[dict], limit: int) -> list[dict]:
    out = []
    for edge in edges[: max(0, int(limit))]:
        out.append(
            {
                "edge_id": edge.get("edge_id"),
                "source": edge.get("source"),
                "relation": edge.get("relation"),
                "target": edge.get("target"),
                "weight": edge.get("weight"),
                "score": edge.get("_local_score"),
            }
        )
    return out


def answer_with_ms_local_graphrag(
    query: str,
    graph_file: str,
    communities_file: str,
    chunks_file: str | None = None,
    top_communities: int = 3,
    max_evidence: int = 12,
    query_level: int = -1,
    use_hierarchy: bool = True,
    use_community_summaries: bool = True,
    max_summary_chars: int = 1200,
    max_completion_tokens: int | None = None,
    doc_prefix_filter: str | None = None,
    strict_doc_scope: bool = False,
    query_type: str | None = None,
    answer_mode: str = "reject",
    max_entities: int = 8,
    max_candidate_edges: int = 80,
    max_candidate_chunks: int = 80,
    use_embedding_rerank: bool = True,
    **_: Any,
) -> dict:
    """A lightweight Microsoft GraphRAG Local Search-style adapter.

    It uses graph entities, relationships, community reports, and source text
    units, but keeps the final answer and RAGAS retrieved_contexts aligned to
    the same reranked chunks.
    """

    answer_mode = _normalize_answer_mode(answer_mode)
    telemetry = Telemetry()
    graph = _load_json(graph_file)
    community_payload = _load_json(communities_file)
    edge_by_id = _edge_lookup(graph)
    chunk_store = _load_chunk_store(chunks_file)

    edges = _filter_edges_by_doc(
        list(graph.get("edges", []) or []),
        doc_prefix_filter=doc_prefix_filter,
        strict_doc_scope=strict_doc_scope,
    )
    communities = (
        _select_level_communities(community_payload, query_level=query_level)
        if use_hierarchy
        else _all_communities(community_payload)
    )
    communities = _filter_communities_by_doc(
        communities,
        edge_by_id=edge_by_id,
        doc_prefix_filter=doc_prefix_filter,
        strict_doc_scope=strict_doc_scope,
    )

    ranked_entities = _rank_entities(graph, query=query, limit=max_entities)
    ranked_edges = _rank_edges(edges, query=query, limit=max_candidate_edges)
    entity_edges = _incident_edges_for_entities(edges, ranked_entities, limit=max_candidate_edges)
    ranked_communities = _rank_communities(
        communities,
        query=query,
        limit=top_communities,
        use_community_summaries=use_community_summaries,
        max_summary_chars=max_summary_chars,
    )
    community_edges = _community_edges(
        ranked_communities,
        edge_by_id=edge_by_id,
        per_community=max(8, int(max_candidate_edges / max(1, top_communities))),
    )
    candidate_edges = _dedupe_edges(ranked_edges + entity_edges + community_edges)[:max_candidate_edges]

    graph_chunks = _collect_candidate_chunks(
        candidate_edges,
        chunk_store=chunk_store,
        doc_prefix_filter=doc_prefix_filter,
        strict_doc_scope=strict_doc_scope,
        max_candidates=max_candidate_chunks,
    )
    fallback_chunks = _lexical_text_unit_fallback(
        query=query,
        chunk_store=chunk_store,
        doc_prefix_filter=doc_prefix_filter,
        limit=max(0, max_candidate_chunks - len(graph_chunks)),
    )
    candidate_chunks = _merge_chunks(
        graph_chunks,
        fallback_chunks,
        limit=max_candidate_chunks,
    )
    ranked_chunks = _rerank_chunks(
        query=query,
        chunks=candidate_chunks,
        telemetry=telemetry,
        use_embedding_rerank=use_embedding_rerank,
    )
    evidence_chunks = ranked_chunks[: max(1, int(max_evidence))]
    for rank, chunk in enumerate(evidence_chunks, start=1):
        chunk["rank"] = rank

    community_reports = _community_report_view(ranked_communities, max_summary_chars=max_summary_chars)
    relationships = _relationship_view(candidate_edges, limit=20)
    context_block = "\n\n".join(
        (
            f"[rank={chunk.get('rank')} score={chunk.get('score')} "
            f"doc_id={chunk.get('doc_id')} chunk_id={chunk.get('chunk_id')}] "
            f"{chunk.get('text')}"
        )
        for chunk in evidence_chunks
    )

    reject_line = "If evidence is insufficient, return exactly: NOT_FOUND"
    open_line = (
        "If evidence is insufficient, you may answer using general knowledge. "
        "Prefix the answer with: OUTSIDE_EVIDENCE:"
    )
    insufficient_policy = reject_line if answer_mode == "reject" else open_line
    if str(query_type or "").strip().lower() == "global_summary":
        prompt = f"""
You are answering with a GraphRAG Local Search context.
Use ONLY the retrieved text units as factual evidence.
Community reports and relationships are navigation context, not standalone proof.
Return 3 concise bullet points (no preface, no legal advice).
{insufficient_policy}

Question:
{query}

Matched entities:
{ranked_entities}

Community reports:
{community_reports}

Relationships:
{relationships}

Retrieved text units:
{context_block}
"""
    else:
        prompt = f"""
Answer the question using ONLY the retrieved text units.
This is an extractive QA task.
Return the shortest exact span copied from the retrieved text units that answers the question.
Do not add explanation, bullets, or legal analysis.
{insufficient_policy}

Question:
{query}

Matched entities:
{ranked_entities}

Community reports:
{community_reports}

Relationships:
{relationships}

Retrieved text units:
{context_block}
"""
    answer, answer_meta = llm_chat(
        [{"role": "user", "content": prompt}],
        max_tokens=max_completion_tokens,
        return_meta=True,
    )
    telemetry.add_llm(answer_meta)

    doc_counts = Counter(_doc_prefix(chunk.get("doc_id")) for chunk in evidence_chunks if _doc_prefix(chunk.get("doc_id")))
    diagnostics = {
        "matched_entities_count": len(ranked_entities),
        "candidate_edges_count": len(candidate_edges),
        "candidate_chunks_count": len(candidate_chunks),
        "selected_chunks_count": len(evidence_chunks),
        "selected_doc_span": len(doc_counts),
        "selected_top_doc_prefix": doc_counts.most_common(1)[0][0] if doc_counts else "",
        "used_text_unit_fallback_count": sum(1 for c in evidence_chunks if c.get("lexical_fallback_score")),
    }

    reasoning_steps = [
        {
            "type": "local_entity_search",
            "matched_entities": ranked_entities,
        },
        {
            "type": "local_relationship_and_community_expansion",
            "selected_community_ids": [c.get("community_id") for c in ranked_communities],
            "candidate_edge_ids": [e.get("edge_id") for e in candidate_edges[:20]],
        },
        {
            "type": "local_text_unit_rerank",
            "selected_chunk_ids": [c.get("chunk_id") for c in evidence_chunks],
            "selected_count": len(evidence_chunks),
        },
    ]

    trace_bundle = build_minimal_trace_bundle(
        method="ms_local_graphrag",
        query=query,
        answer=str(answer or "").strip(),
        evidence_chunks=evidence_chunks,
        doc_prefix_filter=doc_prefix_filter,
        response_for_eval=str(answer or "").strip(),
        response_for_eval_source="ms_local_graphrag_answer",
        orchestration_mode="ms_graphrag_local_search_like",
        aggregation_strategy="entity_relationship_community_text_unit_rerank",
        reasoning_steps=reasoning_steps,
        final_answer_prompt_inputs={
            "query_level": int(query_level),
            "top_communities": int(top_communities),
            "max_evidence": int(max_evidence),
            "max_entities": int(max_entities),
            "use_embedding_rerank": bool(use_embedding_rerank),
            "community_ids": [c.get("community_id") for c in ranked_communities],
            "entity_ids": [e.get("node_id") for e in ranked_entities],
        },
        retrieval_trace_extra={
            "query_level": int(query_level),
            "doc_prefix_filter": doc_prefix_filter or "",
            "strict_doc_scope": bool(strict_doc_scope),
            "matched_entities": ranked_entities,
            "communities": [c.get("community_id") for c in ranked_communities],
            "community_reports": community_reports,
            "relationships": relationships,
            "candidate_edge_ids": [e.get("edge_id") for e in candidate_edges],
            "candidate_chunk_ids": [c.get("chunk_id") for c in candidate_chunks],
            "diagnostics": diagnostics,
        },
        reasoning_trace_extra={
            "matched_entities": ranked_entities,
            "community_reports": community_reports,
            "relationships": relationships,
            "diagnostics": diagnostics,
        },
        answer_trace_extra={
            "final_chunk_selection_strategy": "ms_local_text_unit_rerank",
        },
    )

    return {
        "answer": str(answer or "").strip(),
        "answer_mode": answer_mode,
        "matched_entities": ranked_entities,
        "communities": [c.get("community_id") for c in ranked_communities],
        "community_reports": community_reports,
        "relationships": relationships,
        "evidence_chunks": evidence_chunks,
        "diagnostics": diagnostics,
        "telemetry": telemetry.to_dict(),
        "answer_scope_target_doc_id": trace_bundle["answer_scope_target_doc_id"],
        "answer_composition_mode": trace_bundle["answer_composition_mode"],
        "semantic_alignment": trace_bundle["semantic_alignment"],
        "evaluation_payload": trace_bundle["evaluation_payload"],
        "retrieval_trace": trace_bundle["retrieval_trace"],
        "reasoning_trace": trace_bundle["reasoning_trace"],
        "answer_trace": trace_bundle["answer_trace"],
        "reasoning_steps": trace_bundle["reasoning_steps"],
    }
