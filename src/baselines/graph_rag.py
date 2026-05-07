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
from utils.alignment_payloads import build_minimal_trace_bundle
from utils.telemetry import Telemetry


def _load_json(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _tokenize(text: str) -> set[str]:
    return {w for w in re.split(r"[^a-zA-Z0-9]+", text.lower()) if len(w) >= 3}


def _normalize_answer_mode(answer_mode: str | None) -> str:
    mode = str(answer_mode or "reject").strip().lower()
    return mode if mode in {"reject", "open"} else "reject"


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
    query_type: str | None = None,
    answer_mode: str = "reject",
) -> tuple[str, dict]:
    answer_mode = _normalize_answer_mode(answer_mode)
    reject_line = "If evidence is insufficient, return exactly: NOT_FOUND"
    open_line = (
        "If evidence is insufficient, you may answer using general knowledge. "
        "Prefix the answer with: OUTSIDE_EVIDENCE:"
    )
    insufficient_policy = reject_line if answer_mode == "reject" else open_line
    if str(query_type or "").strip().lower() == "global_summary":
        prompt = f"""
You are in REDUCE stage of GraphRAG summary reasoning.
Merge partial answers into a concise global summary.
Return exactly 3 bullet points covering key themes and risks.
Do not add preface or legal advice.
{insufficient_policy}

Query:
{query}

Partial answers:
{partials}
"""
    else:
        prompt = f"""
You are in REDUCE stage of GraphRAG summary reasoning.
Merge partial answers into one final extractive answer.
Return the shortest exact span supported by the partial answers.
Do not add explanation.
{insufficient_policy}

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
    query_type: str | None = None,
    answer_mode: str = "reject",
) -> tuple[str, dict]:
    answer_mode = _normalize_answer_mode(answer_mode)
    reject_line = "If evidence is insufficient, return exactly: NOT_FOUND"
    open_line = (
        "If evidence is insufficient, you may answer using general knowledge. "
        "Prefix the answer with: OUTSIDE_EVIDENCE:"
    )
    insufficient_policy = reject_line if answer_mode == "reject" else open_line
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
    if str(query_type or "").strip().lower() == "global_summary":
        prompt = f"""
Use the following selected graph communities to produce a concise global summary.
Return exactly 3 bullet points on key themes and risks.
Do not add preface or legal advice.
{insufficient_policy}

Query:
{query}

Evidence:
{evidence}
"""
    else:
        prompt = f"""
Use the following selected graph communities to answer the query.
This is an extractive QA task.
Return the shortest exact span supported by the evidence.
Do not add explanation.
{insufficient_policy}

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
    raw_summary = str(community.get("summary", "")).strip()
    has_precomputed_summary = bool(raw_summary)
    row["_has_precomputed_summary"] = has_precomputed_summary
    if use_community_summaries:
        # Keep summary empty when precomputed summary is missing, so
        # query-time on-demand summarization can be triggered.
        row["summary"] = raw_summary[:max_summary_chars]
    else:
        row["summary"] = _to_text_for_ranking(
            community,
            use_community_summaries=False,
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
                text = str(m.get("text", "") or m.get("evidence", "") or "").strip()
                out.append(
                    {
                        "chunk_id": chunk_id,
                        "doc_id": m.get("doc_id"),
                        "text": text,
                        "community_id": community_id,
                        "edge_id": edge.get("edge_id"),
                    }
                )
                if len(out) >= max_chunks:
                    return out
    return out


def _doc_prefix(doc_id: str | None) -> str:
    return str(doc_id or "").split("#", 1)[0].strip()


def _filter_communities_by_doc_prefix(
    communities: list[dict],
    edge_by_id: dict[str, dict],
    doc_prefix_filter: str | None,
    strict_doc_scope: bool = True,
) -> list[dict]:
    if not doc_prefix_filter:
        return communities
    target = _doc_prefix(doc_prefix_filter)
    if not target:
        return communities
    out: list[dict] = []
    for c in communities:
        hit = False
        for eid in c.get("edges", []) or []:
            edge = edge_by_id.get(str(eid))
            if not edge:
                continue
            for m in edge.get("mentions", []) or []:
                if _doc_prefix(m.get("doc_id")) == target:
                    hit = True
                    break
            if hit:
                break
        if hit:
            out.append(c)
    if out:
        return out
    return [] if strict_doc_scope else communities


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
    doc_prefix_filter: str | None = None,
    strict_doc_scope: bool = True,
    query_type: str | None = None,
    answer_mode: str = "reject",
) -> dict:
    answer_mode = _normalize_answer_mode(answer_mode)
    telemetry = Telemetry()
    graph_payload = _load_json(graph_file) if (generate_summary_on_demand or include_chunk_evidence or doc_prefix_filter) else None
    edge_by_id = _edge_lookup(graph_payload)
    communities_payload = _load_json(communities_file)
    if use_hierarchy:
        candidates = _select_level_communities(communities_payload, query_level=query_level)
    else:
        candidates = communities_payload if isinstance(communities_payload, list) else communities_payload.get(
            "communities", []
        )
    candidates = _filter_communities_by_doc_prefix(
        candidates,
        edge_by_id=edge_by_id,
        doc_prefix_filter=doc_prefix_filter,
        strict_doc_scope=strict_doc_scope,
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
    selected_missing_precomputed_summaries = 0
    for c in selected:
        row = _prepare_reasoning_community(
            c,
            use_community_summaries=use_community_summaries,
            max_summary_chars=max_summary_chars,
        )
        if use_community_summaries and not bool(row.get("_has_precomputed_summary")):
            selected_missing_precomputed_summaries += 1
            if generate_summary_on_demand:
                summary, meta = _summarize_community_on_demand(
                    c,
                    edge_by_id=edge_by_id,
                    max_completion_tokens=max_completion_tokens,
                )
                telemetry.add_llm(meta)
                row["summary"] = str(summary).strip()
                on_demand_generated += 1
        if use_community_summaries and not str(row.get("summary", "")).strip():
            # Final fallback to avoid empty MAP/REDUCE inputs when on-demand
            # summary is disabled or generation fails.
            row["summary"] = _to_text_for_ranking(
                c,
                use_community_summaries=False,
                max_summary_chars=max_summary_chars,
            )
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
            query_type=query_type,
            answer_mode=answer_mode,
        )
        telemetry.add_llm(reduce_meta)
    else:
        answer, one_meta = _single_pass_answer(
            query=query,
            selected=reasoning_selected,
            use_community_summaries=use_community_summaries,
            max_completion_tokens=max_completion_tokens,
            query_type=query_type,
            answer_mode=answer_mode,
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
    reasoning_steps = [
        {
            "type": "community_map",
            "community_id": partial.get("community_id"),
            "level": partial.get("level"),
            "summary": str(partial.get("summary", "") or "")[:400],
            "partial_answer": str(partial.get("partial_answer", "") or "").strip(),
        }
        for partial in partials
    ]
    reasoning_steps.append(
        {
            "type": "community_reduce" if use_map_reduce else "community_single_pass",
            "selected_community_ids": [c.get("community_id") for c in selected],
            "selected_count": len(selected),
            "evidence_chunk_count": len(chunk_evidence),
            "answer_preview": str(answer or "").strip()[:300],
        }
    )
    trace_bundle = build_minimal_trace_bundle(
        method="graph_rag",
        query=query,
        answer=str(answer or "").strip(),
        evidence_chunks=chunk_evidence,
        doc_prefix_filter=doc_prefix_filter,
        response_for_eval=str(answer or "").strip(),
        response_for_eval_source="graph_rag_answer",
        orchestration_mode="graph_local_community_retrieval",
        aggregation_strategy=(
            "graph_map_reduce_community_summaries_with_chunk_backing"
            if use_map_reduce
            else "graph_single_pass_community_summaries_with_chunk_backing"
        ),
        reasoning_steps=reasoning_steps,
        final_answer_prompt_inputs={
            "query_level": int(query_level),
            "use_hierarchy": bool(use_hierarchy),
            "use_map_reduce": bool(use_map_reduce),
            "use_community_summaries": bool(use_community_summaries),
            "map_keypoints_limit": int(map_keypoints_limit),
            "community_ids": [c.get("community_id") for c in selected],
            "community_summary_count": len(reasoning_selected),
        },
        retrieval_trace_extra={
            "query_level": int(query_level),
            "use_hierarchy": bool(use_hierarchy),
            "use_community_summaries": bool(use_community_summaries),
            "shuffle_communities": bool(shuffle_communities),
            "use_map_reduce": bool(use_map_reduce),
            "communities": [c.get("community_id") for c in selected],
            "community_summaries": [c.get("summary", "") for c in reasoning_selected],
            "map_partial_answers": partials,
        },
        reasoning_trace_extra={
            "query_level": int(query_level),
            "selected_communities": [c.get("community_id") for c in selected],
            "community_summaries": [c.get("summary", "") for c in reasoning_selected],
            "map_partial_answers": partials,
            "selected_precomputed_summary_coverage": (
                round(
                    (len(selected) - selected_missing_precomputed_summaries) / len(selected),
                    4,
                )
                if selected
                else 0.0
            ),
        },
        answer_trace_extra={
            "final_chunk_selection_strategy": "graph_selected_community_mentions",
        },
    )

    return {
        "answer": answer,
        "answer_mode": answer_mode,
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
        "selected_missing_precomputed_summaries": selected_missing_precomputed_summaries,
        "selected_precomputed_summary_coverage": (
            round(
                (len(selected) - selected_missing_precomputed_summaries) / len(selected),
                4,
            )
            if selected
            else 0.0
        ),
        "map_partial_answers": partials,
        "subgraph_edges": [],
        "evidence": summary_evidence,
        "evidence_chunks": chunk_evidence,
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
