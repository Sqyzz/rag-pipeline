from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.embedder import embed_texts
from utils.llm_wrapper import llm_chat
from utils.telemetry import Telemetry

JSON_BLOCK_RE = re.compile(r"```json\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _load_json(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _safe_json_list(content: str) -> list:
    try:
        rows = json.loads(content)
    except json.JSONDecodeError:
        m = JSON_BLOCK_RE.search(content)
        if not m:
            return []
        rows = json.loads(m.group(1))
    return rows if isinstance(rows, list) else []


def _extract_query_entities(query: str) -> tuple[list[str], dict]:
    prompt = f"""
Extract key entities from this query.
Return JSON array only, max 10 items.
Query: {query}
"""
    content, meta = llm_chat([{"role": "user", "content": prompt}], temperature=0, return_meta=True)
    entities = [str(x).strip() for x in _safe_json_list(content) if str(x).strip()]
    return entities, meta


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _tokens(text: str) -> set[str]:
    return {w for w in re.split(r"[^a-zA-Z0-9]+", text.lower()) if len(w) >= 2}


def _normalize_answer_mode(answer_mode: str | None) -> str:
    mode = str(answer_mode or "reject").strip().lower()
    return mode if mode in {"reject", "open"} else "reject"


def _graph_nodes(graph: dict) -> list[str]:
    nodes = []
    for n in graph.get("nodes", []):
        if isinstance(n, dict):
            nodes.append(str(n.get("id", "")).strip())
        else:
            nodes.append(str(n).strip())
    return [n for n in nodes if n]


def _link_entities(
    query: str, entities: list[str], node_names: list[str], max_start_entities: int
) -> list[dict]:
    linked = []
    if not node_names:
        return linked
    node_cache = [(n, _normalize(n), _tokens(n)) for n in node_names]

    for ent in entities:
        ent_norm = _normalize(ent)
        ent_tokens = _tokens(ent)
        best = None
        best_score = 0.0
        method = "none"
        for node, node_norm, node_tokens in node_cache:
            if ent_norm == node_norm:
                score = 100.0
                m = "exact"
            elif ent_norm and (ent_norm in node_norm or node_norm in ent_norm):
                score = 80.0
                m = "substring"
            else:
                inter = ent_tokens.intersection(node_tokens)
                if not inter:
                    continue
                union = ent_tokens.union(node_tokens)
                jaccard = len(inter) / max(len(union), 1)
                score = 40.0 * jaccard + 5.0 * len(inter)
                m = "token_overlap"
            if score > best_score:
                best = node
                best_score = score
                method = m
        if best:
            linked.append(
                {
                    "query_entity": ent,
                    "node_id": best,
                    "score": round(best_score, 4),
                    "method": method,
                }
            )

    if linked:
        linked.sort(key=lambda x: x["score"], reverse=True)
        dedup = []
        seen = set()
        for item in linked:
            nid = item["node_id"]
            if nid in seen:
                continue
            seen.add(nid)
            dedup.append(item)
        return dedup[:max_start_entities]

    # Fallback: infer seeds directly from query terms.
    q_tokens = _tokens(query)
    if not q_tokens:
        return []
    scored = []
    for node, _, node_tokens in node_cache:
        inter = q_tokens.intersection(node_tokens)
        if not inter:
            continue
        scored.append((len(inter), node))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [
        {"query_entity": "(fallback)", "node_id": node, "score": float(score), "method": "fallback"}
        for score, node in scored[:max_start_entities]
    ]


def _build_undirected_adjacency(edges: list[dict]) -> dict[str, list[dict]]:
    adjacency: dict[str, list[dict]] = {}
    for e in edges:
        s = str(e.get("source", "")).strip()
        t = str(e.get("target", "")).strip()
        if not s or not t:
            continue
        adjacency.setdefault(s, []).append(e)
        adjacency.setdefault(t, []).append(e)
    return adjacency


def _multi_hop_traversal(
    start_nodes: list[str],
    edges: list[dict],
    max_hops: int,
    max_edges: int,
    max_nodes: int | None = None,
) -> tuple[set[str], list[dict]]:
    visited_nodes, ranked_edges, _ = _multi_hop_traversal_with_early_stop(
        start_nodes=start_nodes,
        edges=edges,
        max_hops=max_hops,
        max_edges=max_edges,
        max_nodes=max_nodes,
        dynamic_early_stop=False,
        min_hops_before_stop=1,
        relevance_threshold=1.0,
        coverage_threshold=1.0,
        query_tokens=set(),
        coverage_tokens=set(),
    )
    return visited_nodes, ranked_edges


def _edge_relevance(edge: dict, query_tokens: set[str]) -> float:
    if not query_tokens:
        return 0.0
    text = f"{edge.get('source', '')} {edge.get('relation', '')} {edge.get('target', '')}"
    inter = query_tokens.intersection(_tokens(text))
    if not inter:
        return 0.0
    return len(inter) / max(len(query_tokens), 1)


def _node_coverage(visited_nodes: set[str], coverage_tokens: set[str]) -> float:
    if not coverage_tokens:
        return 0.0
    seen_tokens: set[str] = set()
    for n in visited_nodes:
        seen_tokens.update(_tokens(str(n)))
    if not seen_tokens:
        return 0.0
    covered = len(seen_tokens.intersection(coverage_tokens))
    return covered / max(len(coverage_tokens), 1)


def _multi_hop_traversal_with_early_stop(
    start_nodes: list[str],
    edges: list[dict],
    max_hops: int,
    max_edges: int,
    max_nodes: int | None = None,
    dynamic_early_stop: bool = True,
    min_hops_before_stop: int = 1,
    relevance_threshold: float = 0.35,
    coverage_threshold: float = 0.6,
    query_tokens: set[str] | None = None,
    coverage_tokens: set[str] | None = None,
) -> tuple[set[str], list[dict], dict]:
    adjacency = _build_undirected_adjacency(edges)
    frontier = [n for n in start_nodes if str(n).strip()]
    visited_nodes = set(start_nodes)
    collected_edges: dict[str, dict] = {}
    hop_stats: list[dict] = []
    query_tokens = set(query_tokens or set())
    coverage_tokens = set(coverage_tokens or set())
    stop_reason = "max_hops_reached"
    used_hops = 0

    for hop in range(1, int(max_hops) + 1):
        if not frontier:
            stop_reason = "frontier_exhausted"
            break
        if len(collected_edges) >= int(max_edges):
            stop_reason = "max_edges_reached"
            break

        next_frontier: set[str] = set()
        hop_new_edges = 0
        hop_new_nodes = 0
        hop_rel_scores: list[float] = []
        hard_stop = False

        for node in frontier:
            for edge in adjacency.get(node, []):
                edge_id = str(edge.get("edge_id", "")).strip()
                is_new_edge = bool(edge_id) and edge_id not in collected_edges
                if is_new_edge:
                    collected_edges[edge_id] = edge
                    hop_new_edges += 1
                    hop_rel_scores.append(_edge_relevance(edge, query_tokens))
                src = str(edge.get("source", "")).strip()
                tgt = str(edge.get("target", "")).strip()
                nxt = tgt if src == node else src
                if nxt and nxt not in visited_nodes:
                    visited_nodes.add(nxt)
                    next_frontier.add(nxt)
                    hop_new_nodes += 1
                if len(collected_edges) >= int(max_edges):
                    stop_reason = "max_edges_reached"
                    hard_stop = True
                    break
                if max_nodes is not None and len(visited_nodes) >= int(max_nodes):
                    stop_reason = "max_nodes_reached"
                    hard_stop = True
                    break
            if hard_stop:
                break
        used_hops = hop
        hop_relevance = float(sum(hop_rel_scores) / len(hop_rel_scores)) if hop_rel_scores else 0.0
        hop_coverage = _node_coverage(visited_nodes, coverage_tokens)
        hop_stats.append(
            {
                "hop": hop,
                "frontier_size": len(frontier),
                "new_edges": hop_new_edges,
                "new_nodes": hop_new_nodes,
                "traversed_edges_total": len(collected_edges),
                "visited_nodes_total": len(visited_nodes),
                "relevance_score": round(hop_relevance, 4),
                "subgraph_coverage": round(hop_coverage, 4),
            }
        )
        if hard_stop:
            break
        if (
            dynamic_early_stop
            and hop >= int(min_hops_before_stop)
            and hop_relevance >= float(relevance_threshold)
            and hop_coverage >= float(coverage_threshold)
        ):
            stop_reason = "dynamic_early_stop"
            break
        frontier = list(next_frontier)

    ranked_edges = sorted(
        collected_edges.values(),
        key=lambda e: int(e.get("weight", 1)),
        reverse=True,
    )
    traversal_meta = {
        "used_hops": int(used_hops),
        "stop_reason": stop_reason,
        "dynamic_early_stop": bool(dynamic_early_stop),
        "min_hops_before_stop": int(min_hops_before_stop),
        "relevance_threshold": float(relevance_threshold),
        "coverage_threshold": float(coverage_threshold),
        "hop_stats": hop_stats,
    }
    return visited_nodes, ranked_edges, traversal_meta


def _chunk_map_from_store(store_file: str | None) -> dict[str, dict]:
    if not store_file:
        return {}
    p = Path(store_file)
    if not p.exists():
        return {}
    rows = _load_json(store_file)
    result = {}
    for row in rows:
        chunk_id = str(row.get("chunk_id", "")).strip()
        if chunk_id:
            result[chunk_id] = row
    return result


def _collect_traversed_chunks(
    traversed_edges: list[dict],
    chunk_map: dict[str, dict],
    max_chunks: int,
    max_chars: int,
) -> list[dict]:
    score_by_chunk: dict[str, float] = {}
    by_chunk_doc: dict[str, str] = {}
    fallback_text: dict[str, str] = {}

    for edge in traversed_edges:
        edge_weight = float(edge.get("weight", 1))
        for m in edge.get("mentions", []):
            chunk_id = str(m.get("chunk_id", "")).strip()
            if not chunk_id:
                continue
            score_by_chunk[chunk_id] = score_by_chunk.get(chunk_id, 0.0) + edge_weight
            by_chunk_doc[chunk_id] = str(m.get("doc_id", ""))
            if m.get("evidence"):
                fallback_text[chunk_id] = str(m.get("evidence", ""))

    ordered = sorted(score_by_chunk.items(), key=lambda x: x[1], reverse=True)
    selected = []
    total_chars = 0
    for chunk_id, score in ordered:
        source = chunk_map.get(chunk_id, {})
        text = str(source.get("text", "")).strip() or fallback_text.get(chunk_id, "")
        if not text:
            continue
        if len(selected) >= max_chunks:
            break
        if total_chars + len(text) > max_chars:
            break
        selected.append(
            {
                "chunk_id": chunk_id,
                "doc_id": source.get("doc_id") or by_chunk_doc.get(chunk_id),
                "score": round(score, 4),
                "text": text,
            }
        )
        total_chars += len(text)
    return selected


def _doc_prefix(doc_id: str | None) -> str:
    return str(doc_id or "").split("#", 1)[0].strip()


def _filter_chunks_by_doc_prefix(
    chunks: list[dict],
    doc_prefix_filter: str | None,
    strict_doc_scope: bool = True,
) -> list[dict]:
    if not doc_prefix_filter:
        return chunks
    target = _doc_prefix(doc_prefix_filter)
    if not target:
        return chunks
    filtered = [c for c in chunks if _doc_prefix(c.get("doc_id")) == target]
    if filtered:
        return filtered
    return [] if strict_doc_scope else chunks


def _filter_edges_by_doc_prefix(
    edges: list[dict],
    doc_prefix_filter: str | None,
    strict_doc_scope: bool = True,
) -> list[dict]:
    if not doc_prefix_filter:
        return edges
    target = _doc_prefix(doc_prefix_filter)
    if not target:
        return edges
    kept: list[dict] = []
    for e in edges:
        mentions = e.get("mentions", []) or []
        for m in mentions:
            if _doc_prefix(m.get("doc_id")) == target:
                kept.append(e)
                break
    if kept:
        return kept
    return [] if strict_doc_scope else edges


def _compute_kg_diagnostics(
    linked_entities: list[dict],
    traversed_edges: list[dict],
    retrieved_chunks: list[dict],
) -> dict:
    start_nodes = [str(x.get("node_id", "")).strip() for x in linked_entities if str(x.get("node_id", "")).strip()]
    hub_terms = {"agreement", "clause", "party", "contract", "law", "notice", "date", "condition", "obligation"}
    if start_nodes:
        hub_hits = sum(1 for n in start_nodes if n.lower() in hub_terms)
        startnode_hub_ratio = round(hub_hits / len(start_nodes), 4)
    else:
        startnode_hub_ratio = 0.0

    doc_ids = [_doc_prefix(ch.get("doc_id")) for ch in retrieved_chunks]
    doc_ids = [d for d in doc_ids if d]
    doc_span = len(set(doc_ids))
    top_doc = None
    top_doc_share = 0.0
    if doc_ids:
        counts: dict[str, int] = {}
        for d in doc_ids:
            counts[d] = counts.get(d, 0) + 1
        top_doc = max(counts, key=counts.get)
        top_doc_share = round(counts[top_doc] / len(doc_ids), 4)

    return {
        "start_nodes_count": len(start_nodes),
        "start_nodes_unique_count": len(set(start_nodes)),
        "startnode_hub_ratio": startnode_hub_ratio,
        "traversed_edges_count": len(traversed_edges),
        "retrieved_chunks_count": len(retrieved_chunks),
        "retrieved_doc_span": doc_span,
        "retrieved_top_doc_id": top_doc,
        "retrieved_top_doc_share": top_doc_share,
    }


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def _rerank_chunks_by_embedding(query: str, chunks: list[dict], telemetry: Telemetry) -> list[dict]:
    if not chunks:
        return []
    q_vec, q_meta = embed_texts([query], return_meta=True)
    c_vec, c_meta = embed_texts([c.get("text", "") for c in chunks], return_meta=True)
    telemetry.add_embedding(q_meta)
    telemetry.add_embedding(c_meta)

    q = _normalize_rows(np.asarray(q_vec, dtype="float32"))[0]
    c = _normalize_rows(np.asarray(c_vec, dtype="float32"))
    sims = c @ q

    rescored = []
    for sim, ch in zip(sims.tolist(), chunks):
        row = dict(ch)
        row["semantic_score"] = round(float(sim), 6)
        rescored.append(row)
    rescored.sort(key=lambda x: x["semantic_score"], reverse=True)
    return rescored


def answer_with_kg(
    query: str,
    graph_file: str,
    store_file: str | None = None,
    max_hops: int = 2,
    max_start_entities: int = 5,
    top_edges: int = 60,
    max_chunks: int = 12,
    max_context_chars: int = 12000,
    max_nodes: int | None = None,
    use_entity_linking: bool = True,
    use_embedding_rerank: bool = True,
    dynamic_early_stop: bool = True,
    min_hops_before_stop: int = 1,
    relevance_threshold: float = 0.35,
    coverage_threshold: float = 0.6,
    max_completion_tokens: int | None = None,
    doc_prefix_filter: str | None = None,
    strict_doc_scope: bool = True,
    query_type: str | None = None,
    answer_mode: str = "reject",
) -> dict:
    answer_mode = _normalize_answer_mode(answer_mode)
    telemetry = Telemetry()
    graph = _load_json(graph_file)
    entities: list[str] = []
    if use_entity_linking:
        entities, ent_meta = _extract_query_entities(query)
        telemetry.add_llm(ent_meta)
    node_names = _graph_nodes(graph)
    if use_entity_linking:
        linked_entities = _link_entities(
            query=query,
            entities=entities,
            node_names=node_names,
            max_start_entities=max_start_entities,
        )
    else:
        linked_entities = _link_entities(
            query=query,
            entities=[],
            node_names=node_names,
            max_start_entities=max_start_entities,
        )
    start_nodes = [x["node_id"] for x in linked_entities]
    query_tokens = _tokens(query)
    coverage_tokens = set()
    for ent in entities:
        coverage_tokens.update(_tokens(ent))
    if not coverage_tokens:
        coverage_tokens = set(query_tokens)

    traversed_nodes, traversed_edges, traversal_meta = _multi_hop_traversal_with_early_stop(
        start_nodes=start_nodes,
        edges=graph.get("edges", []),
        max_hops=max_hops,
        max_edges=top_edges,
        max_nodes=max_nodes,
        dynamic_early_stop=dynamic_early_stop,
        min_hops_before_stop=min_hops_before_stop,
        relevance_threshold=relevance_threshold,
        coverage_threshold=coverage_threshold,
        query_tokens=query_tokens,
        coverage_tokens=coverage_tokens,
    )
    traversed_edges = _filter_edges_by_doc_prefix(
        traversed_edges,
        doc_prefix_filter,
        strict_doc_scope=strict_doc_scope,
    )
    chunk_map = _chunk_map_from_store(store_file)
    retrieved_chunks = _collect_traversed_chunks(
        traversed_edges=traversed_edges,
        chunk_map=chunk_map,
        max_chunks=max_chunks,
        max_chars=max_context_chars,
    )
    retrieved_chunks = _filter_chunks_by_doc_prefix(
        retrieved_chunks,
        doc_prefix_filter,
        strict_doc_scope=strict_doc_scope,
    )
    if use_embedding_rerank:
        retrieved_chunks = _rerank_chunks_by_embedding(query, retrieved_chunks, telemetry)

    edge_view = [
        {
            "edge_id": e.get("edge_id"),
            "source": e.get("source"),
            "relation": e.get("relation"),
            "target": e.get("target"),
            "weight": e.get("weight"),
        }
        for e in traversed_edges
    ]

    context_lines = []
    for ch in retrieved_chunks:
        context_lines.append(
            f"[doc_id={ch.get('doc_id')} chunk_id={ch.get('chunk_id')} score={ch.get('score')}] {ch.get('text')}"
        )
    context_block = "\n\n".join(context_lines)

    reject_line = "If evidence is insufficient, return exactly: NOT_FOUND"
    open_line = (
        "If evidence is insufficient, you may answer using general knowledge. "
        "Prefix the answer with: OUTSIDE_EVIDENCE:"
    )
    insufficient_policy = reject_line if answer_mode == "reject" else open_line

    if str(query_type or "").strip().lower() == "global_summary":
        prompt = f"""
Summarize the contract-level themes and risks using ONLY the retrieved chunk contexts.
Return 3 concise bullet points (no preface, no legal advice).
{insufficient_policy}

Question:
{query}

Entity linking result:
{linked_entities}

Subgraph edges (from {max_hops}-hop traversal):
{edge_view}

Retrieved chunk contexts:
{context_block}
"""
    else:
        prompt = f"""
Answer the question using ONLY the retrieved chunk contexts from KG traversal.
This is an extractive QA task.
Return the shortest exact span copied from retrieved contexts that answers the question.
Do not add explanation or legal analysis.
{insufficient_policy}

Question:
{query}

Entity linking result:
{linked_entities}

Subgraph edges (from {max_hops}-hop traversal):
{edge_view}

Retrieved chunk contexts:
{context_block}
"""
    answer, answer_meta = llm_chat(
        [{"role": "user", "content": prompt}],
        max_tokens=max_completion_tokens,
        return_meta=True,
    )
    telemetry.add_llm(answer_meta)
    diagnostics = _compute_kg_diagnostics(
        linked_entities=linked_entities,
        traversed_edges=traversed_edges,
        retrieved_chunks=retrieved_chunks,
    )
    return {
        "answer": answer,
        "answer_mode": answer_mode,
        "query_entities": entities,
        "linked_entities": linked_entities,
        "start_nodes": start_nodes,
        "max_hops": max_hops,
        "used_hops": traversal_meta.get("used_hops"),
        "traversal_stop_reason": traversal_meta.get("stop_reason"),
        "traversal_hop_stats": traversal_meta.get("hop_stats"),
        "dynamic_early_stop": dynamic_early_stop,
        "min_hops_before_stop": min_hops_before_stop,
        "relevance_threshold": relevance_threshold,
        "coverage_threshold": coverage_threshold,
        "max_nodes": max_nodes,
        "use_entity_linking": use_entity_linking,
        "use_embedding_rerank": use_embedding_rerank,
        "traversed_nodes": sorted(traversed_nodes),
        "subgraph_edges": edge_view,
        "evidence": retrieved_chunks,
        "diagnostics": diagnostics,
        "traversal_meta": traversal_meta,
        "telemetry": telemetry.to_dict(),
    }
