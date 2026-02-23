from __future__ import annotations

from collections import deque
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
    adjacency = _build_undirected_adjacency(edges)
    queue = deque([(n, 0) for n in start_nodes])
    visited_nodes = set(start_nodes)
    collected_edges: dict[str, dict] = {}

    while queue and len(collected_edges) < max_edges:
        node, hop = queue.popleft()
        if hop >= max_hops:
            continue
        for edge in adjacency.get(node, []):
            edge_id = str(edge.get("edge_id", ""))
            if edge_id and edge_id not in collected_edges:
                collected_edges[edge_id] = edge
            src = str(edge.get("source", ""))
            tgt = str(edge.get("target", ""))
            nxt = tgt if src == node else src
            if nxt and nxt not in visited_nodes:
                visited_nodes.add(nxt)
                queue.append((nxt, hop + 1))
                if max_nodes is not None and len(visited_nodes) >= int(max_nodes):
                    break
            if len(collected_edges) >= max_edges:
                break
        if max_nodes is not None and len(visited_nodes) >= int(max_nodes):
            break

    ranked_edges = sorted(
        collected_edges.values(),
        key=lambda e: int(e.get("weight", 1)),
        reverse=True,
    )
    return visited_nodes, ranked_edges


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
    max_completion_tokens: int | None = None,
) -> dict:
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

    traversed_nodes, traversed_edges = _multi_hop_traversal(
        start_nodes=start_nodes,
        edges=graph.get("edges", []),
        max_hops=max_hops,
        max_edges=top_edges,
        max_nodes=max_nodes,
    )
    chunk_map = _chunk_map_from_store(store_file)
    retrieved_chunks = _collect_traversed_chunks(
        traversed_edges=traversed_edges,
        chunk_map=chunk_map,
        max_chunks=max_chunks,
        max_chars=max_context_chars,
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

    prompt = f"""
Answer the question using ONLY the retrieved chunk contexts from KG traversal.
If evidence is insufficient, explicitly say so.

Question:
{query}

Entity linking result:
{linked_entities}

Subgraph edges (from {max_hops}-hop traversal):
{edge_view}

Retrieved chunk contexts:
{context_block}
Just answer the question briefly. No explanation is needed if not necessary.
"""
    answer, answer_meta = llm_chat(
        [{"role": "user", "content": prompt}],
        max_tokens=max_completion_tokens,
        return_meta=True,
    )
    telemetry.add_llm(answer_meta)
    return {
        "answer": answer,
        "query_entities": entities,
        "linked_entities": linked_entities,
        "start_nodes": start_nodes,
        "max_hops": max_hops,
        "max_nodes": max_nodes,
        "use_entity_linking": use_entity_linking,
        "use_embedding_rerank": use_embedding_rerank,
        "traversed_nodes": sorted(traversed_nodes),
        "subgraph_edges": edge_view,
        "evidence": retrieved_chunks,
        "telemetry": telemetry.to_dict(),
    }
