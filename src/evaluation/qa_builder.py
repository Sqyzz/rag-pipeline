from __future__ import annotations

import argparse
import json
import random
import re
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

GENERIC_ENTITY_TERMS = {
    "person",
    "people",
    "organization",
    "org",
    "org_unit",
    "department",
    "team",
    "project",
    "document",
    "system",
    "meeting",
    "location",
    "entity",
    "company",
    "group",
    "user",
    "manager",
    "employee",
}


def _load_json(path: str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_chunk_store(path: str) -> list[dict]:
    p = Path(path)
    if p.suffix == ".json":
        rows = _load_json(path)
        return rows if isinstance(rows, list) else []
    rows = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _read_jsonl(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    rows: list[dict] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: str, rows: list[dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _safe_text(text: str, max_len: int = 180) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _looks_like_noise_entity(name: str) -> bool:
    s = str(name or "").strip()
    if not s:
        return True
    if len(s) > 72 and len(set(s)) <= 10:
        return True
    alnum = sum(ch.isalnum() for ch in s)
    if alnum == 0:
        return True
    upper = sum(ch.isupper() for ch in s)
    lower = sum(ch.islower() for ch in s)
    digits = sum(ch.isdigit() for ch in s)
    if len(s) >= 24 and upper >= 10 and lower <= 2 and digits <= 4:
        return True
    if len(s) >= 24 and len(set(s)) <= 8:
        return True
    return False


def _is_generic_entity(name: str) -> bool:
    s = re.sub(r"\s+", " ", str(name or "")).strip().lower()
    if not s:
        return True
    if s in GENERIC_ENTITY_TERMS:
        return True
    if "_" in s and s.replace("_", " ") in GENERIC_ENTITY_TERMS:
        return True
    return False


def _allow_edge_for_question(source: str, target: str) -> bool:
    a = str(source or "").strip()
    b = str(target or "").strip()
    if not a or not b:
        return False
    if _looks_like_noise_entity(a) or _looks_like_noise_entity(b):
        return False
    if _is_generic_entity(a) and _is_generic_entity(b):
        return False
    return True


def _community_payload(payload: Any) -> tuple[list[dict], list[dict]]:
    if isinstance(payload, list):
        return payload, []
    return payload.get("communities", []), payload.get("levels", [])


def _resolve_level(communities: list[dict], levels: list[dict], qa_level: int) -> int:
    if levels:
        available = sorted(int(x.get("level")) for x in levels)
    else:
        available = sorted({int(c.get("level", 0)) for c in communities})
    if not available:
        return 0
    if qa_level < 0:
        return available[-1]
    if qa_level in available:
        return qa_level
    return available[0]


def _communities_at_level(communities: list[dict], levels: list[dict], level: int) -> list[dict]:
    if levels:
        allow = set()
        for lv in levels:
            if int(lv.get("level", -1)) == int(level):
                allow = set(lv.get("community_ids", []))
                break
        if allow:
            return [c for c in communities if c.get("community_id") in allow]
    return [c for c in communities if int(c.get("level", 0)) == int(level)]


def _node_to_community_map(level_communities: list[dict]) -> dict[str, str]:
    node2community: dict[str, tuple[str, int]] = {}
    for c in level_communities:
        cid = str(c.get("community_id"))
        size = int(c.get("size", len(c.get("nodes", []))))
        for n in c.get("nodes", []):
            node = str(n).strip()
            if not node:
                continue
            prev = node2community.get(node)
            if prev is None or size > prev[1]:
                node2community[node] = (cid, size)
    return {k: v[0] for k, v in node2community.items()}


def _edge_obj(edge: dict) -> dict:
    return {
        "edge_id": edge.get("edge_id"),
        "source": edge.get("source"),
        "relation": edge.get("relation"),
        "target": edge.get("target"),
    }


def _community_ref(cid: str, community_by_id: dict[str, dict], default_level: int) -> dict:
    c = community_by_id.get(cid, {})
    return {"community_id": cid, "level": int(c.get("level", default_level))}


def _extract_key_points(summary: str, k: int = 5) -> list[str]:
    parts = re.split(r"[.;!?]\s+", str(summary or ""))
    points = []
    for p in parts:
        p = _safe_text(p, max_len=90)
        if p:
            points.append(p)
        if len(points) >= k:
            break
    return points


def _community_focus_terms(community: dict, k: int = 3) -> list[str]:
    terms: list[str] = []
    for node in community.get("nodes", []):
        t = _safe_text(node, max_len=28)
        if t:
            terms.append(t)
        if len(terms) >= k:
            break
    return terms


def _global_summary_query(template: str, community: dict) -> str:
    cid = str(community.get("community_id", "")).strip()
    terms = _community_focus_terms(community, k=3)
    if terms:
        return f"{template} Focus on community {cid} ({', '.join(terms)})."
    return f"{template} Focus on community {cid}."


def _fallback_summary_from_edges(community: dict, edge_by_id: dict[str, dict], max_edges: int = 3) -> str:
    facts: list[str] = []
    for eid in community.get("edges", []):
        edge = edge_by_id.get(str(eid))
        if not edge:
            continue
        s = _safe_text(edge.get("source", ""), max_len=40)
        r = _safe_text(edge.get("relation", ""), max_len=30)
        t = _safe_text(edge.get("target", ""), max_len=40)
        if s and r and t:
            facts.append(f"{s} {r} {t}")
        if len(facts) >= max_edges:
            break
    if not facts:
        return ""
    return "Key relations in this community include: " + "; ".join(facts) + "."


def _pick_edge_chunks(edge: dict, chunk_map: dict[str, dict], k: int = 2) -> list[dict]:
    out = []
    seen = set()
    for m in edge.get("mentions", []):
        cid = str(m.get("chunk_id", "")).strip()
        if not cid or cid in seen or cid not in chunk_map:
            continue
        seen.add(cid)
        out.append({"chunk_id": cid, "doc_id": chunk_map[cid].get("doc_id")})
        if len(out) >= k:
            break
    return out


def _build_undirected_adjacency(edges: list[dict]) -> dict[str, list[tuple[dict, str]]]:
    adj: dict[str, list[tuple[dict, str]]] = {}
    for e in edges:
        s = str(e.get("source", "")).strip()
        t = str(e.get("target", "")).strip()
        if not s or not t or s == t:
            continue
        adj.setdefault(s, []).append((e, t))
        adj.setdefault(t, []).append((e, s))
    return adj


def _validate_gold(rows: list[dict], chunk_map: dict[str, dict], edge_by_id: dict[str, dict], community_by_id: dict[str, dict]) -> None:
    for r in rows:
        for ch in r.get("supporting_chunks", []):
            cid = ch.get("chunk_id")
            if cid not in chunk_map:
                raise ValueError(f"missing chunk_id in chunk_store: {cid}")
        for e in r.get("supporting_edges", []):
            eid = e.get("edge_id")
            if eid not in edge_by_id:
                raise ValueError(f"missing edge_id in graph: {eid}")
        for c in r.get("supporting_communities", []):
            cid = c.get("community_id")
            if cid not in community_by_id:
                raise ValueError(f"missing community_id in communities: {cid}")


def build_qa(
    graph_file: str,
    communities_file: str,
    chunk_store_file: str,
    out_gold: str,
    out_queries: str,
    out_gold_answer: str,
    n_local: int,
    n_cross: int,
    n_global: int,
    n_trace: int,
    qa_community_level: int,
    seed: int,
    append: bool = False,
    trace_min_chunks: int = 3,
    trace_allow_backoff: bool = True,
) -> dict:
    rng = random.Random(seed)
    graph = _load_json(graph_file)
    communities_payload = _load_json(communities_file)
    chunk_rows = _read_chunk_store(chunk_store_file)
    chunk_map = {str(x.get("chunk_id", "")).strip(): x for x in chunk_rows if str(x.get("chunk_id", "")).strip()}
    edges = graph.get("edges", [])
    edge_by_id = {str(e.get("edge_id")): e for e in edges if e.get("edge_id")}
    total_mentions = 0
    matched_mentions = 0
    for e in edges:
        for m in e.get("mentions", []):
            total_mentions += 1
            cid = str(m.get("chunk_id", "")).strip()
            if cid in chunk_map:
                matched_mentions += 1

    communities, levels = _community_payload(communities_payload)
    level = _resolve_level(communities, levels, qa_community_level)
    level_communities = _communities_at_level(communities, levels, level=level)
    community_by_id = {str(c.get("community_id")): c for c in communities if c.get("community_id")}
    node2community = _node_to_community_map(level_communities)

    edges_ranked = sorted(
        [e for e in edges if str(e.get("source", "")).strip() and str(e.get("target", "")).strip()],
        key=lambda x: int(x.get("weight", 1)),
        reverse=True,
    )

    existing_rows: list[dict] = _read_jsonl(out_gold) if append else []
    rows: list[dict] = list(existing_rows)
    new_rows: list[dict] = []
    used_queries = {str(r.get("query", "")).strip().lower() for r in rows if str(r.get("query", "")).strip()}
    used_qids = {str(r.get("qid", "")).strip() for r in rows if str(r.get("qid", "")).strip()}
    qnums = []
    for qid in used_qids:
        m = re.match(r"^q(\d+)$", qid, re.IGNORECASE)
        if m:
            qnums.append(int(m.group(1)))
    qnum = (max(qnums) + 1) if qnums else 1
    produced_new = {
        "local_factual": 0,
        "cross_doc_reasoning": 0,
        "global_summary": 0,
        "evidence_tracing": 0,
    }
    global_fallback_used = 0
    trace_added_by_min_chunks: dict[int, int] = {}

    def next_qid() -> str:
        nonlocal qnum
        while True:
            qid = f"q{qnum:03d}"
            qnum += 1
            if qid not in used_qids:
                used_qids.add(qid)
                return qid

    def add_row(row: dict) -> bool:
        q = row["query"].strip().lower()
        if not q or q in used_queries:
            return False
        used_queries.add(q)
        rows.append(row)
        new_rows.append(row)
        t = str(row.get("type", ""))
        if t in produced_new:
            produced_new[t] += 1
        return True

    # local_factual
    local_templates = [
        "What does {a} {r}?",
        "Which entity is {r} by {a}?",
        "What is the object of relation '{r}' for {a}?",
    ]
    for e in edges_ranked:
        if produced_new["local_factual"] >= n_local:
            break
        a = str(e.get("source", "")).strip()
        r = str(e.get("relation", "")).strip()
        b = str(e.get("target", "")).strip()
        if not a or not r or not b:
            continue
        if not _allow_edge_for_question(a, b):
            continue
        chunks = _pick_edge_chunks(e, chunk_map, k=2)
        if not chunks:
            continue
        tpl = local_templates[len(rows) % len(local_templates)]
        query = tpl.format(a=a, r=r)
        communities_ref = []
        for node in (a, b):
            cid = node2community.get(node)
            if cid and cid not in {x["community_id"] for x in communities_ref}:
                communities_ref.append(_community_ref(cid, community_by_id, level))
        add_row(
            {
                "qid": next_qid(),
                "type": "local_factual",
                "query": query,
                "answer": b,
                "key_points": [f"{a} --[{r}]--> {b}"],
                "supporting_chunks": chunks,
                "supporting_edges": [_edge_obj(e)],
                "supporting_communities": communities_ref,
            }
        )

    # cross_doc_reasoning
    adj = _build_undirected_adjacency(edges_ranked)
    path_candidates: list[tuple[str, dict, dict, str, str]] = []
    for mid, links in adj.items():
        if len(links) < 2:
            continue
        for (e1, a), (e2, c) in combinations(links, 2):
            if a == c:
                continue
            path_candidates.append((mid, e1, e2, a, c))
    rng.shuffle(path_candidates)

    cross_templates = [
        "What intermediate entity links {a} and {c}?",
        "How is {a} connected to {c}?",
        "Through which entity does {a} relate to {c}?",
    ]
    cross_count = 0
    for mid, e1, e2, a, c in path_candidates:
        if cross_count >= n_cross or produced_new["cross_doc_reasoning"] >= n_cross:
            break
        if _looks_like_noise_entity(mid) or _looks_like_noise_entity(a) or _looks_like_noise_entity(c):
            continue
        if _is_generic_entity(a) and _is_generic_entity(c):
            continue
        chunks = _pick_edge_chunks(e1, chunk_map, k=2) + _pick_edge_chunks(e2, chunk_map, k=2)
        # unique chunk refs
        uniq = []
        seen = set()
        for ch in chunks:
            cid = ch["chunk_id"]
            if cid in seen:
                continue
            seen.add(cid)
            uniq.append(ch)
        if len(uniq) < 2:
            continue
        tpl = cross_templates[cross_count % len(cross_templates)]
        query = tpl.format(a=a, c=c)
        communities_ref = []
        for node in (a, mid, c):
            cid = node2community.get(node)
            if cid and cid not in {x["community_id"] for x in communities_ref}:
                communities_ref.append(_community_ref(cid, community_by_id, level))
        add_row(
            {
                "qid": next_qid(),
                "type": "cross_doc_reasoning",
                "query": query,
                "answer": f"{a} is connected to {c} through {mid}.",
                "key_points": [_safe_text(f"{a} -> {mid} -> {c}", 90)],
                "supporting_chunks": uniq[:4],
                "supporting_edges": [_edge_obj(e1), _edge_obj(e2)],
                "supporting_communities": communities_ref,
            }
        )
        if produced_new["cross_doc_reasoning"] > cross_count:
            cross_count += 1

    # global_summary
    global_templates = [
        "What are the main issues and risks discussed in this context?",
        "Summarize the major themes and risks in this knowledge base.",
        "Provide a high-level summary of the dominant topics in this context.",
    ]
    summary_candidates = [c for c in level_communities if str(c.get("summary", "")).strip()]
    summary_candidates.sort(key=lambda x: int(x.get("size", 0)), reverse=True)
    fallback_candidates = [c for c in level_communities if not str(c.get("summary", "")).strip()]
    fallback_candidates.sort(key=lambda x: int(x.get("size", 0)), reverse=True)
    global_candidates = summary_candidates + fallback_candidates
    for i, c in enumerate(global_candidates):
        if produced_new["global_summary"] >= n_global:
            break
        cid = str(c.get("community_id"))
        summary = str(c.get("summary", "")).strip()
        used_fallback_summary = False
        if not summary:
            summary = _fallback_summary_from_edges(c, edge_by_id=edge_by_id, max_edges=3)
            used_fallback_summary = bool(summary)
        if not summary:
            continue
        edge_refs = []
        chunk_refs = []
        seen_chunks = set()
        for eid in c.get("edges", [])[:3]:
            e = edge_by_id.get(str(eid))
            if not e:
                continue
            edge_refs.append(_edge_obj(e))
            for ch in _pick_edge_chunks(e, chunk_map, k=1):
                if ch["chunk_id"] in seen_chunks:
                    continue
                seen_chunks.add(ch["chunk_id"])
                chunk_refs.append(ch)
        query = _global_summary_query(global_templates[i % len(global_templates)], c)
        added = add_row(
            {
                "qid": next_qid(),
                "type": "global_summary",
                "query": query,
                "answer": summary,
                "key_points": _extract_key_points(summary, k=5)
                + (["summary_source=fallback_edges"] if used_fallback_summary else ["summary_source=community_summary"]),
                "supporting_chunks": chunk_refs,
                "supporting_edges": edge_refs,
                "supporting_communities": [_community_ref(cid, community_by_id, level)],
            }
        )
        if added and used_fallback_summary:
            global_fallback_used += 1

    # evidence_tracing
    min_chunks = max(1, int(trace_min_chunks))
    trace_thresholds = [min_chunks]
    if trace_allow_backoff:
        for x in range(min_chunks - 1, 0, -1):
            trace_thresholds.append(x)
    edge_chunk_cache: dict[str, list[dict]] = {}
    used_trace_edges: set[str] = set()
    for required_chunks in trace_thresholds:
        trace_added_by_min_chunks.setdefault(required_chunks, 0)
        if produced_new["evidence_tracing"] >= n_trace:
            break
        for e in edges_ranked:
            if produced_new["evidence_tracing"] >= n_trace:
                break
            edge_key = str(e.get("edge_id", "")).strip() or f"{e.get('source')}|{e.get('relation')}|{e.get('target')}"
            if edge_key in used_trace_edges:
                continue
            base_chunks = edge_chunk_cache.get(edge_key)
            if base_chunks is None:
                base_chunks = _pick_edge_chunks(e, chunk_map, k=6)
                edge_chunk_cache[edge_key] = base_chunks
            if len(base_chunks) < required_chunks:
                continue
            a = str(e.get("source", "")).strip()
            r = str(e.get("relation", "")).strip()
            b = str(e.get("target", "")).strip()
            if not _allow_edge_for_question(a, b):
                continue
            query = (
                f"Provide evidence for the claim that {a} {r} {b}, "
                "including document IDs and quoted snippets."
            )
            doc_ids = [str(chunk_map[ch["chunk_id"]].get("doc_id")) for ch in base_chunks[: max(1, required_chunks)]]
            answer = (
                f"Evidence indicates {a} {r} {b}. Supporting documents include "
                + ", ".join(doc_ids)
                + "."
            )
            communities_ref = []
            for node in (a, b):
                cid = node2community.get(node)
                if cid and cid not in {x["community_id"] for x in communities_ref}:
                    communities_ref.append(_community_ref(cid, community_by_id, level))
            added = add_row(
                {
                    "qid": next_qid(),
                    "type": "evidence_tracing",
                    "query": query,
                    "answer": answer,
                    "key_points": [
                        f"Claim: {a} {r} {b}",
                        f"Docs: {', '.join(doc_ids)}",
                        f"evidence_chunks={len(base_chunks)} (min_required={required_chunks})",
                    ],
                    "supporting_chunks": base_chunks[:4],
                    "supporting_edges": [_edge_obj(e)],
                    "supporting_communities": communities_ref,
                }
            )
            if added:
                used_trace_edges.add(edge_key)
                trace_added_by_min_chunks[required_chunks] += 1

    _validate_gold(new_rows, chunk_map, edge_by_id, community_by_id)

    queries_rows = [{"qid": r["qid"], "type": r["type"], "query": r["query"]} for r in rows]
    gold_answer_rows = [{"qid": r["qid"], "answer": r["answer"]} for r in rows]

    _write_jsonl(out_gold, rows)
    _write_jsonl(out_queries, queries_rows)
    _write_jsonl(out_gold_answer, gold_answer_rows)

    by_type = {}
    for r in rows:
        by_type[r["type"]] = by_type.get(r["type"], 0) + 1

    requested = {
        "local_factual": int(n_local),
        "cross_doc_reasoning": int(n_cross),
        "global_summary": int(n_global),
        "evidence_tracing": int(n_trace),
    }
    warnings = []
    for t, n in requested.items():
        produced = int(produced_new.get(t, 0))
        if produced < n:
            warnings.append(f"type={t}: requested={n}, produced={produced}")
    coverage = round(matched_mentions / max(total_mentions, 1), 4)
    if coverage < 0.2:
        warnings.append(
            "low graph↔chunk alignment coverage; check that chunk_store_file matches graph/triples assets"
        )

    return {
        "out_gold": out_gold,
        "out_queries": out_queries,
        "out_gold_answer": out_gold_answer,
        "num_rows": len(rows),
        "num_existing_rows": len(existing_rows),
        "num_added_rows": len(new_rows),
        "by_type": by_type,
        "added_by_type": produced_new,
        "requested_by_type": requested,
        "graph_chunk_mention_coverage": {
            "matched_mentions": matched_mentions,
            "total_mentions": total_mentions,
            "ratio": coverage,
        },
        "warnings": warnings,
        "qa_community_level": level,
        "seed": seed,
        "append": append,
        "global_summary_fallback_count": global_fallback_used,
        "evidence_trace_policy": {
            "trace_min_chunks": min_chunks,
            "trace_allow_backoff": bool(trace_allow_backoff),
            "added_by_min_required_chunks": trace_added_by_min_chunks,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build gold QA aligned with current pipeline assets")
    parser.add_argument("--graph-file", default="outputs/graph/graph.json")
    parser.add_argument("--communities-file", default="outputs/graph/communities.json")
    parser.add_argument("--chunk-store-file", default="outputs/indexes/chunk_store_sampled.json")
    parser.add_argument("--out-gold", default="data/queries/gold_qa.jsonl")
    parser.add_argument("--out-queries", default="data/queries/queries.jsonl")
    parser.add_argument("--out-gold-answer", default="data/queries/gold.jsonl")
    parser.add_argument("--n_local", type=int, default=20)
    parser.add_argument("--n_cross", type=int, default=20)
    parser.add_argument("--n_global", type=int, default=20)
    parser.add_argument("--n_trace", type=int, default=20)
    parser.add_argument("--qa-community-level", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trace-min-chunks", type=int, default=3, help="Preferred minimum chunks for evidence_tracing")
    parser.add_argument(
        "--trace-allow-backoff",
        action="store_true",
        help="If enabled, back off chunk requirement (e.g., 3->2->1) to fill requested evidence_tracing rows",
    )
    parser.add_argument(
        "--no-trace-allow-backoff",
        dest="trace_allow_backoff",
        action="store_false",
        help="Disable backoff and keep strict minimum chunks for evidence_tracing",
    )
    parser.set_defaults(trace_allow_backoff=True)
    parser.add_argument("--append", action="store_true", help="Append new QA rows to existing outputs")
    parser.add_argument(
        "--append-type",
        choices=["local_factual", "cross_doc_reasoning", "global_summary", "evidence_tracing"],
        default=None,
        help="Generate only this QA type (use with --append-count)",
    )
    parser.add_argument("--append-count", type=int, default=None, help="How many rows to generate for --append-type")
    args = parser.parse_args()

    if args.append_count is not None and args.append_count < 0:
        parser.error("--append-count must be >= 0")
    if (args.append_type is None) != (args.append_count is None):
        parser.error("--append-type and --append-count must be used together")

    n_local = args.n_local
    n_cross = args.n_cross
    n_global = args.n_global
    n_trace = args.n_trace
    if args.append_type is not None:
        n_local, n_cross, n_global, n_trace = 0, 0, 0, 0
        if args.append_type == "local_factual":
            n_local = int(args.append_count)
        elif args.append_type == "cross_doc_reasoning":
            n_cross = int(args.append_count)
        elif args.append_type == "global_summary":
            n_global = int(args.append_count)
        elif args.append_type == "evidence_tracing":
            n_trace = int(args.append_count)

    summary = build_qa(
        graph_file=args.graph_file,
        communities_file=args.communities_file,
        chunk_store_file=args.chunk_store_file,
        out_gold=args.out_gold,
        out_queries=args.out_queries,
        out_gold_answer=args.out_gold_answer,
        n_local=n_local,
        n_cross=n_cross,
        n_global=n_global,
        n_trace=n_trace,
        qa_community_level=args.qa_community_level,
        seed=args.seed,
        append=args.append,
        trace_min_chunks=args.trace_min_chunks,
        trace_allow_backoff=args.trace_allow_backoff,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
