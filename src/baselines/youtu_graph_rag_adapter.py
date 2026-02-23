from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

try:
    from adapters.youtu_client import YoutuClient
    from utils.config import cfg
except ModuleNotFoundError:
    from src.adapters.youtu_client import YoutuClient
    from src.utils.config import cfg


def _g(obj: Any, key: str, default: Any) -> Any:
    if obj is None:
        return default
    return getattr(obj, key, default)


def _to_int(v: Any) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def _collect_latency(items: Any) -> int:
    if not isinstance(items, list):
        return 0
    return int(sum(_to_int((x or {}).get("latency_ms")) for x in items if isinstance(x, dict)))


def _resolve_youtu_settings() -> dict[str, Any]:
    y = getattr(cfg, "youtu", None)
    return {
        "base_url": str(_g(y, "base_url", "http://127.0.0.1:8080")),
        "dataset": str(_g(y, "dataset", "enterprise")),
        "timeout_sec": int(_g(y, "timeout_sec", 120)),
        "max_retries": int(_g(y, "max_retries", 2)),
        "retry_sleep_sec": float(_g(y, "retry_sleep_sec", 1.0)),
    }


def _to_chunk_evidence(data: dict[str, Any]) -> list[dict[str, Any]]:
    raw = data.get("retrieved_chunks") or data.get("evidence_chunks") or data.get("chunks") or []
    out: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, dict):
            out.append(
                {
                    "chunk_id": str(item.get("chunk_id", "") or ""),
                    "text": str(item.get("text", "") or ""),
                    "doc_id": item.get("doc_id"),
                    "score": item.get("score"),
                    "community_id": str(item.get("community_id", "") or ""),
                    "edge_id": str(item.get("edge_id", "") or ""),
                }
            )
        else:
            out.append(
                {
                    "chunk_id": "",
                    "text": str(item or ""),
                    "doc_id": None,
                    "score": None,
                    "community_id": "",
                    "edge_id": "",
                }
            )
    return out


def _to_subgraph_edges(data: dict[str, Any]) -> list[dict[str, Any]]:
    raw = data.get("retrieved_triples") or data.get("retrieved_edges") or data.get("subgraph_edges") or []
    out: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, dict):
            out.append(
                {
                    "edge_id": str(item.get("edge_id", "") or ""),
                    "source": item.get("source"),
                    "relation": item.get("relation"),
                    "target": item.get("target"),
                    "text": item.get("text"),
                    "weight": item.get("weight"),
                }
            )
        else:
            out.append(
                {
                    "edge_id": "",
                    "source": None,
                    "relation": None,
                    "target": None,
                    "text": str(item or ""),
                    "weight": None,
                }
            )
    return out


def _to_communities(data: dict[str, Any]) -> tuple[list[str], list[str]]:
    raw = data.get("communities") or []
    ids: list[str] = []
    summaries: list[str] = []
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                cid = str(item.get("community_id", "") or "").strip()
                if cid:
                    ids.append(cid)
                s = str(item.get("summary", "") or "").strip()
                if s:
                    summaries.append(s)
            else:
                cid = str(item or "").strip()
                if cid:
                    ids.append(cid)

    if not summaries:
        cs = data.get("community_summaries") or []
        if isinstance(cs, list):
            summaries = [str(x or "") for x in cs]
    return ids, summaries


def _to_evidence(
    data: dict[str, Any],
    evidence_chunks: list[dict[str, Any]],
    max_evidence: int,
) -> list[dict[str, Any]]:
    raw = data.get("evidence") or data.get("community_evidence") or []
    out: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        out.append(
            {
                "community_id": str(item.get("community_id", "") or ""),
                "chunk_id": str(item.get("chunk_id", "") or ""),
                "edge_id": str(item.get("edge_id", "") or ""),
                "text": str(item.get("text", "") or item.get("summary", "") or ""),
                "summary": str(item.get("summary", "") or ""),
                "level": item.get("level"),
            }
        )

    if out:
        return out[:max_evidence]

    derived: list[dict[str, Any]] = []
    for item in evidence_chunks[:max_evidence]:
        derived.append(
            {
                "community_id": str(item.get("community_id", "") or ""),
                "chunk_id": str(item.get("chunk_id", "") or ""),
                "edge_id": str(item.get("edge_id", "") or ""),
                "text": str(item.get("text", "") or ""),
                "summary": str(item.get("text", "") or ""),
                "level": None,
            }
        )
    return derived


def _to_map_partials(data: dict[str, Any]) -> list[dict[str, Any]]:
    raw = data.get("map_partial_answers") or data.get("partials") or []
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, dict):
            out.append(item)
        else:
            out.append({"partial_answer": str(item or "")})
    return out


def _build_telemetry(meta: dict[str, Any]) -> dict[str, Any]:
    usage = meta.get("usage") if isinstance(meta.get("usage"), dict) else {}
    prompt_tokens = _to_int(usage.get("prompt_tokens"))
    completion_tokens = _to_int(usage.get("completion_tokens"))

    total_tokens_raw = usage.get("total_tokens")
    total_tokens = _to_int(total_tokens_raw) if total_tokens_raw is not None else (prompt_tokens + completion_tokens)

    llm_calls_raw = meta.get("llm_calls") if isinstance(meta.get("llm_calls"), list) else []
    embedding_calls_raw = meta.get("embedding_calls") if isinstance(meta.get("embedding_calls"), list) else []
    llm_calls = _to_int(meta.get("llm_call_count")) or len(llm_calls_raw)
    embedding_calls = _to_int(meta.get("embedding_call_count")) or len(embedding_calls_raw)

    missing_fields: list[str] = []
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        if usage.get(key) is None:
            missing_fields.append(f"meta.usage.{key}")
    if not meta.get("llm_calls") and meta.get("llm_call_count") is None:
        missing_fields.append("meta.llm_calls")
    if not meta.get("embedding_calls") and meta.get("embedding_call_count") is None:
        missing_fields.append("meta.embedding_calls")

    usage_complete = not any(x.startswith("meta.usage") for x in missing_fields)
    return {
        "llm_calls": llm_calls,
        "embedding_calls": embedding_calls,
        "llm_latency_ms": _collect_latency(llm_calls_raw),
        "embedding_latency_ms": _collect_latency(embedding_calls_raw),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "extra": {
            "missing_fields": missing_fields,
            "usage_complete": usage_complete,
        },
    }


def map_youtu_to_graph_payload(
    response: dict[str, Any],
    *,
    query_level: int,
    use_hierarchy: bool,
    use_community_summaries: bool,
    shuffle_communities: bool,
    use_map_reduce: bool,
    map_keypoints_limit: int,
    max_evidence: int,
) -> dict[str, Any]:
    data = response.get("data") if isinstance(response.get("data"), dict) else {}
    meta = response.get("meta") if isinstance(response.get("meta"), dict) else {}

    evidence_chunks = _to_chunk_evidence(data)
    subgraph_edges = _to_subgraph_edges(data)
    communities, community_summaries = _to_communities(data)
    payload = {
        "answer": str(data.get("answer", "") or ""),
        "communities": communities,
        "community_summaries": community_summaries,
        "query_level": int(query_level),
        "use_hierarchy": bool(use_hierarchy),
        "use_community_summaries": bool(use_community_summaries),
        "shuffle_communities": bool(shuffle_communities),
        "use_map_reduce": bool(use_map_reduce),
        "map_keypoints_limit": int(map_keypoints_limit),
        "map_partial_answers": _to_map_partials(data),
        "subgraph_edges": subgraph_edges,
        "evidence_chunks": evidence_chunks,
        "evidence": _to_evidence(data, evidence_chunks=evidence_chunks, max_evidence=max_evidence),
        "telemetry": _build_telemetry(meta),
        "generate_summary_on_demand": False,
        "embedding_cache": {"enabled": False, "source": "youtu"},
        "on_demand_summaries_generated": 0,
    }
    return payload


def answer_with_youtu_graphrag(
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
    youtu_base_url: str | None = None,
    youtu_dataset: str | None = None,
    timeout_sec: int | None = None,
) -> dict[str, Any]:
    del graph_file, communities_file

    settings = _resolve_youtu_settings()
    base_url = youtu_base_url or settings["base_url"]
    dataset = youtu_dataset or settings["dataset"]
    client = YoutuClient(base_url=base_url, timeout_sec=timeout_sec or settings["timeout_sec"])

    payload = {
        "query": query,
        "question": query,
        "top_communities": int(top_communities),
        "max_evidence": int(max_evidence),
        "query_level": int(query_level),
        "use_hierarchy": bool(use_hierarchy),
        "use_community_summaries": bool(use_community_summaries),
        "shuffle_communities": bool(shuffle_communities),
        "use_map_reduce": bool(use_map_reduce),
        "max_summary_chars": int(max_summary_chars),
        "map_keypoints_limit": int(map_keypoints_limit),
    }
    if max_completion_tokens is not None:
        payload["max_completion_tokens"] = int(max_completion_tokens)

    retries = max(1, int(settings["max_retries"]))
    sleep_sec = float(settings["retry_sleep_sec"])
    last_error: str | None = None
    for attempt in range(1, retries + 1):
        try:
            response = client.search(dataset_name=dataset, payload=payload)
            return map_youtu_to_graph_payload(
                response,
                query_level=query_level,
                use_hierarchy=use_hierarchy,
                use_community_summaries=use_community_summaries,
                shuffle_communities=shuffle_communities,
                use_map_reduce=use_map_reduce,
                map_keypoints_limit=map_keypoints_limit,
                max_evidence=max_evidence,
            )
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            if attempt < retries:
                time.sleep(sleep_sec)

    return {
        "answer": "",
        "communities": [],
        "community_summaries": [],
        "query_level": int(query_level),
        "use_hierarchy": bool(use_hierarchy),
        "use_community_summaries": bool(use_community_summaries),
        "shuffle_communities": bool(shuffle_communities),
        "use_map_reduce": bool(use_map_reduce),
        "map_keypoints_limit": int(map_keypoints_limit),
        "map_partial_answers": [],
        "subgraph_edges": [],
        "evidence": [],
        "evidence_chunks": [],
        "telemetry": {
            "llm_calls": 0,
            "embedding_calls": 0,
            "llm_latency_ms": 0,
            "embedding_latency_ms": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "extra": {
                "missing_fields": ["response"],
                "usage_complete": False,
                "error": last_error,
            },
        },
        "generate_summary_on_demand": False,
        "embedding_cache": {"enabled": False, "source": "youtu"},
        "on_demand_summaries_generated": 0,
    }
