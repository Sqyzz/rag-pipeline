from __future__ import annotations

import json
import random
import re
import sys
import hashlib
from typing import Any
from pathlib import Path

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.embedder import embed_texts
from utils.llm_wrapper import llm_chat
from utils.telemetry import Telemetry


def _load_json(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _tokenize(text: str) -> set[str]:
    return {w for w in re.split(r"[^a-zA-Z0-9]+", text.lower()) if len(w) >= 3}


def _select_level_communities(payload: Any, query_level: int) -> list[dict]:
    if isinstance(payload, list):
        return payload
    communities = payload.get("communities", [])
    levels = payload.get("levels", [])
    if not levels:
        return communities
    level_ids = {x["level"]: set(x["community_ids"]) for x in levels}
    available = sorted(level_ids.keys())
    if query_level < 0:
        target = available[-1]
    else:
        target = query_level if query_level in level_ids else available[0]
    allow = level_ids[target]
    return [c for c in communities if c.get("community_id") in allow]


def _to_text_for_ranking(community: dict, use_community_summaries: bool, max_summary_chars: int) -> str:
    if use_community_summaries:
        summary = str(community.get("summary", "")).strip()
        if summary:
            return summary[:max_summary_chars]
    nodes = [str(x) for x in community.get("nodes", [])[:50]]
    raw = " | ".join(nodes)
    return raw[:max_summary_chars]


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def _cache_key(
    communities_file: str,
    use_community_summaries: bool,
    cache_text_chars: int,
) -> str:
    p = Path(communities_file)
    stat = p.stat()
    raw = (
        f"{p.resolve()}|{stat.st_size}|{stat.st_mtime_ns}|"
        f"{int(use_community_summaries)}|{int(cache_text_chars)}"
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _all_communities(payload: Any) -> list[dict]:
    if isinstance(payload, list):
        return payload
    return payload.get("communities", [])


def _build_or_load_embedding_cache(
    communities_payload: Any,
    communities_file: str,
    use_community_summaries: bool,
    cache_text_chars: int,
    telemetry: Telemetry,
    embedding_cache_dir: str,
    use_embedding_cache: bool,
) -> tuple[dict[str, np.ndarray], dict]:
    communities = _all_communities(communities_payload)
    texts = [_to_text_for_ranking(c, use_community_summaries, cache_text_chars) for c in communities]
    cids = [str(c.get("community_id", "")) for c in communities]

    cache_meta = {
        "enabled": use_embedding_cache,
        "hit": False,
        "size": len(cids),
        "cache_text_chars": int(cache_text_chars),
    }
    if not use_embedding_cache:
        c_vec, c_meta = embed_texts(texts, return_meta=True)
        telemetry.add_embedding(c_meta)
        arr = _normalize_rows(np.asarray(c_vec, dtype="float32"))
        return {cid: arr[i] for i, cid in enumerate(cids)}, cache_meta

    cache_dir = Path(embedding_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _cache_key(
        communities_file=communities_file,
        use_community_summaries=use_community_summaries,
        cache_text_chars=cache_text_chars,
    )
    vec_file = cache_dir / f"community_emb_{key}.npy"
    id_file = cache_dir / f"community_emb_{key}.json"
    cache_meta["key"] = key

    if vec_file.exists() and id_file.exists():
        ids = json.loads(id_file.read_text(encoding="utf-8"))
        arr = np.load(vec_file).astype("float32")
        if len(ids) == arr.shape[0]:
            cache_meta["hit"] = True
            return {str(ids[i]): arr[i] for i in range(len(ids))}, cache_meta

    c_vec, c_meta = embed_texts(texts, return_meta=True)
    telemetry.add_embedding(c_meta)
    arr = _normalize_rows(np.asarray(c_vec, dtype="float32"))
    np.save(vec_file, arr)
    id_file.write_text(json.dumps(cids, ensure_ascii=False), encoding="utf-8")
    return {cid: arr[i] for i, cid in enumerate(cids)}, cache_meta


def _rank_communities_by_embedding(
    query: str,
    communities: list[dict],
    communities_payload: Any,
    communities_file: str,
    top_k: int,
    telemetry: Telemetry,
    use_community_summaries: bool,
    max_summary_chars: int,
    embedding_cache_dir: str,
    use_embedding_cache: bool,
    cache_text_chars: int,
) -> list[dict]:
    if not communities:
        return []
    emb_map, cache_meta = _build_or_load_embedding_cache(
        communities_payload=communities_payload,
        communities_file=communities_file,
        use_community_summaries=use_community_summaries,
        cache_text_chars=cache_text_chars,
        telemetry=telemetry,
        embedding_cache_dir=embedding_cache_dir,
        use_embedding_cache=use_embedding_cache,
    )
    q_vec, q_meta = embed_texts([query], return_meta=True)
    telemetry.add_embedding(q_meta)

    q = _normalize_rows(np.asarray(q_vec, dtype="float32"))[0]
    c = np.asarray(
        [emb_map.get(str(comm.get("community_id", "")), np.zeros_like(q)) for comm in communities],
        dtype="float32",
    )
    sims = c @ q
    scored = sorted(zip(sims.tolist(), communities), key=lambda x: x[0], reverse=True)
    selected = [x[1] for x in scored[:top_k]]
    for comm in selected:
        comm["_embedding_cache"] = cache_meta
    return selected


def _map_summary_answer(
    query: str,
    community: dict,
    map_keypoints_limit: int = 5,
    max_completion_tokens: int | None = None,
) -> tuple[dict, dict]:
    prompt = f"""
You are in MAP stage of GraphRAG summary reasoning.
Use ONLY this community summary to answer the query partially.
Extract at most {map_keypoints_limit} key bullet points relevant to the query.
Keep each bullet under 20 words. Include uncertainty where needed.

Query:
{query}

Community ID: {community.get("community_id")}
Community Summary:
{community.get("summary", "")}
"""
    answer, meta = llm_chat(
        [{"role": "user", "content": prompt}],
        max_tokens=max_completion_tokens,
        return_meta=True,
    )
    return {
        "community_id": community.get("community_id"),
        "level": community.get("level"),
        "summary": community.get("summary", ""),
        "partial_answer": answer,
    }, meta


def _reduce_answers(
    query: str,
    partials: list[dict],
    max_completion_tokens: int | None = None,
) -> tuple[str, dict]:
    prompt = f"""
You are in REDUCE stage of GraphRAG summary reasoning.
Merge partial answers into one global answer. Resolve conflicts explicitly.
If evidence is insufficient, state insufficient evidence.

Query:
{query}

Partial answers:
{partials}
"""
    return llm_chat(
        [{"role": "user", "content": prompt}],
        max_tokens=max_completion_tokens,
        return_meta=True,
    )


def _single_pass_answer(
    query: str,
    selected: list[dict],
    use_community_summaries: bool,
    max_completion_tokens: int | None = None,
) -> tuple[str, dict]:
    evidence = []
    for c in selected:
        text = (
            str(c.get("summary", ""))
            if use_community_summaries
            else " | ".join(str(x) for x in c.get("nodes", [])[:50])
        )
        evidence.append(
            {
                "community_id": c.get("community_id"),
                "level": c.get("level"),
                "text": text,
            }
        )
    prompt = f"""
Use the following selected graph communities to answer the query.
If evidence is insufficient, state insufficient evidence.

Query:
{query}

Evidence:
{evidence}
"""
    return llm_chat(
        [{"role": "user", "content": prompt}],
        max_tokens=max_completion_tokens,
        return_meta=True,
    )


def _prepare_reasoning_community(
    community: dict,
    use_community_summaries: bool,
    max_summary_chars: int,
) -> dict:
    row = dict(community)
    row["summary"] = _to_text_for_ranking(
        community,
        use_community_summaries=use_community_summaries,
        max_summary_chars=max_summary_chars,
    )
    return row


def _edge_lookup(graph_payload: Any) -> dict[str, dict]:
    if not isinstance(graph_payload, dict):
        return {}
    return {str(e.get("edge_id")): e for e in graph_payload.get("edges", []) if e.get("edge_id")}


def _collect_chunk_evidence_from_communities(
    selected: list[dict],
    edge_by_id: dict[str, dict],
    max_chunks: int,
) -> list[dict]:
    out: list[dict] = []
    seen: set[str] = set()
    if max_chunks <= 0:
        return out

    for c in selected:
        community_id = c.get("community_id")
        for eid in c.get("edges", []) or []:
            edge = edge_by_id.get(str(eid))
            if not edge:
                continue
            for m in edge.get("mentions", []) or []:
                chunk_id = str(m.get("chunk_id", "")).strip()
                if not chunk_id or chunk_id in seen:
                    continue
                seen.add(chunk_id)
                out.append(
                    {
                        "chunk_id": chunk_id,
                        "doc_id": m.get("doc_id"),
                        "community_id": community_id,
                        "edge_id": edge.get("edge_id"),
                    }
                )
                if len(out) >= max_chunks:
                    return out
    return out


def _summarize_community_on_demand(
    community: dict,
    edge_by_id: dict[str, dict],
    max_completion_tokens: int | None = None,
) -> tuple[str, dict]:
    edge_lines = []
    for eid in community.get("edges", [])[:30]:
        e = edge_by_id.get(str(eid))
        if not e:
            continue
        edge_lines.append(
            f"- {e.get('source')} --[{e.get('relation')}]--> {e.get('target')} (weight={e.get('weight',1)})"
        )
    prompt = f"""
You are summarizing a graph community for retrieval.
Return concise text (max 120 words), include main entities, themes and risks.
This is hierarchy level {community.get("level")}.

Nodes:
{community.get("nodes", [])[:80]}

Edges:
{edge_lines}
"""
    return llm_chat(
        [{"role": "user", "content": prompt}],
        max_tokens=max_completion_tokens,
        return_meta=True,
    )


def answer_with_graphrag(
    query: str,
    graph_file: str,
    communities_file: str,
    top_communities: int = 3,
    max_evidence: int = 12,
    query_level: int = 0,
    use_hierarchy: bool = True,
    use_community_summaries: bool = True,
    shuffle_communities: bool = True,
    use_map_reduce: bool = True,
    max_summary_chars: int = 1800,
    map_keypoints_limit: int = 5,
    max_completion_tokens: int | None = None,
    generate_summary_on_demand: bool = True,
    use_embedding_cache: bool = True,
    embedding_cache_dir: str = "outputs/cache",
    embedding_cache_text_chars: int = 4000,
    include_chunk_evidence: bool = True,
) -> dict:
    telemetry = Telemetry()
    graph_payload = _load_json(graph_file) if (generate_summary_on_demand or include_chunk_evidence) else None
    edge_by_id = _edge_lookup(graph_payload)
    communities_payload = _load_json(communities_file)
    if use_hierarchy:
        candidates = _select_level_communities(communities_payload, query_level=query_level)
    else:
        candidates = communities_payload if isinstance(communities_payload, list) else communities_payload.get(
            "communities", []
        )
    selected = _rank_communities_by_embedding(
        query=query,
        communities=candidates,
        communities_payload=communities_payload,
        communities_file=communities_file,
        top_k=top_communities,
        telemetry=telemetry,
        use_community_summaries=use_community_summaries,
        max_summary_chars=max_summary_chars,
        embedding_cache_dir=embedding_cache_dir,
        use_embedding_cache=use_embedding_cache,
        cache_text_chars=embedding_cache_text_chars,
    )
    if shuffle_communities and selected:
        rng = random.Random(hash(query) & 0xFFFFFFFF)
        rng.shuffle(selected)

    partials: list[dict] = []
    reasoning_selected = []
    on_demand_generated = 0
    for c in selected:
        row = _prepare_reasoning_community(
            c,
            use_community_summaries=use_community_summaries,
            max_summary_chars=max_summary_chars,
        )
        if use_community_summaries and not str(row.get("summary", "")).strip() and generate_summary_on_demand:
            summary, meta = _summarize_community_on_demand(
                c,
                edge_by_id=edge_by_id,
                max_completion_tokens=max_completion_tokens,
            )
            telemetry.add_llm(meta)
            row["summary"] = str(summary).strip()
            on_demand_generated += 1
        reasoning_selected.append(row)
    if use_map_reduce:
        for c in reasoning_selected:
            partial, map_meta = _map_summary_answer(
                query,
                c,
                map_keypoints_limit=map_keypoints_limit,
                max_completion_tokens=max_completion_tokens,
            )
            telemetry.add_llm(map_meta)
            partials.append(partial)
        answer, reduce_meta = _reduce_answers(
            query,
            partials,
            max_completion_tokens=max_completion_tokens,
        )
        telemetry.add_llm(reduce_meta)
    else:
        answer, one_meta = _single_pass_answer(
            query=query,
            selected=reasoning_selected,
            use_community_summaries=use_community_summaries,
            max_completion_tokens=max_completion_tokens,
        )
        telemetry.add_llm(one_meta)
        partials = [
            {
                "community_id": c.get("community_id"),
                "level": c.get("level"),
                "summary": _to_text_for_ranking(
                    c, use_community_summaries=use_community_summaries, max_summary_chars=max_summary_chars
                ),
                "partial_answer": "",
            }
            for c in selected
        ]
    summary_evidence = [
        {
            "community_id": p["community_id"],
            "level": p["level"],
            "summary": p["summary"],
        }
        for p in partials[:max_evidence]
    ]
    chunk_evidence = (
        _collect_chunk_evidence_from_communities(
            selected=reasoning_selected,
            edge_by_id=edge_by_id,
            max_chunks=max_evidence,
        )
        if include_chunk_evidence
        else []
    )

    return {
        "answer": answer,
        "communities": [c["community_id"] for c in selected],
        "community_summaries": [c["summary"] for c in reasoning_selected],
        "query_level": query_level,
        "use_hierarchy": use_hierarchy,
        "use_community_summaries": use_community_summaries,
        "shuffle_communities": shuffle_communities,
        "use_map_reduce": use_map_reduce,
        "map_keypoints_limit": map_keypoints_limit,
        "generate_summary_on_demand": generate_summary_on_demand,
        "embedding_cache": (selected[0].get("_embedding_cache") if selected else {"enabled": use_embedding_cache}),
        "on_demand_summaries_generated": on_demand_generated,
        "map_partial_answers": partials,
        "subgraph_edges": [],
        "evidence": summary_evidence,
        "evidence_chunks": chunk_evidence,
        "telemetry": telemetry.to_dict(),
    }
