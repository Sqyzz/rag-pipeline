from __future__ import annotations

import hashlib
import json
import re
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
        "route_type": str(_g(y, "route_type", "")).strip().lower(),
        "client_id": str(_g(y, "client_id", "")).strip(),
        "timeout_sec": int(_g(y, "timeout_sec", 120)),
        "max_retries": int(_g(y, "max_retries", 2)),
        "retry_sleep_sec": float(_g(y, "retry_sleep_sec", 1.0)),
    }


def _norm_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def _doc_prefix(value: Any) -> str:
    return str(value or "").split("#", 1)[0].strip()


def _text_hash(text: Any) -> str:
    return hashlib.sha1(_norm_text(text).encode("utf-8")).hexdigest()


def _sha256_file(path: str) -> str:
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


_CHUNK_ID_INLINE_RE = re.compile(r"\[chunk_id=([^\]]+)\]")
_TRIPLE_SCORE_RE = re.compile(r"\)\s*\[score:\s*(-?\d+(?:\.\d+)?)\]\s*$", flags=re.IGNORECASE)


def _extract_inline_chunk_id(text: Any) -> str:
    m = _CHUNK_ID_INLINE_RE.search(str(text or ""))
    if not m:
        return ""
    return str(m.group(1) or "").strip()


def _clean_triple_node_text(value: Any) -> str:
    s = str(value or "").strip()
    if not s:
        return ""
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1].strip()
    s = re.sub(r"^(unknown\s+node\s*:)\s*", "", s, flags=re.IGNORECASE)
    return s.strip()


def _split_top_level_csv(text: str, expected_parts: int = 3) -> list[str]:
    parts: list[str] = []
    cur: list[str] = []
    square = 0
    round_ = 0
    curly = 0
    for ch in text:
        if ch == "[":
            square += 1
        elif ch == "]":
            square = max(0, square - 1)
        elif ch == "(":
            round_ += 1
        elif ch == ")":
            round_ = max(0, round_ - 1)
        elif ch == "{":
            curly += 1
        elif ch == "}":
            curly = max(0, curly - 1)

        if ch == "," and square == 0 and round_ == 0 and curly == 0 and len(parts) < expected_parts - 1:
            parts.append("".join(cur).strip())
            cur = []
            continue
        cur.append(ch)
    if cur:
        parts.append("".join(cur).strip())
    return parts


def _parse_triple_text(text: Any) -> tuple[str, str, str, float | None] | None:
    raw = str(text or "").strip()
    if not raw:
        return None

    score: float | None = None
    score_match = _TRIPLE_SCORE_RE.search(raw)
    if score_match:
        try:
            score = float(score_match.group(1))
        except (TypeError, ValueError):
            score = None
        raw = raw[: score_match.start() + 1].strip()

    if not (raw.startswith("(") and raw.endswith(")")):
        return None

    inner = raw[1:-1].strip()
    parts = _split_top_level_csv(inner, expected_parts=3)
    if len(parts) != 3:
        return None

    source = _clean_triple_node_text(parts[0])
    relation = str(parts[1] or "").strip()
    target = _clean_triple_node_text(parts[2])
    if not source or not relation or not target:
        return None
    return source, relation, target, score


def _load_or_build_chunk_id_map(store_file: str | None) -> dict[str, str]:
    if not store_file:
        return {}
    p = Path(store_file)
    if not p.exists():
        return {}

    suffix = "sampled" if "sampled" in p.name else "full"
    cache_file = Path("outputs/cache") / f"youtu_chunk_id_map_{suffix}.json"
    source_fp = _sha256_file(str(p))

    if cache_file.exists():
        try:
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            if (
                isinstance(cached, dict)
                and str(cached.get("source_store_file", "")) == str(p)
                and str(cached.get("source_store_fingerprint", "")) == source_fp
                and isinstance(cached.get("by_text_hash"), dict)
            ):
                return {str(k): str(v) for k, v in cached["by_text_hash"].items()}
        except Exception:  # noqa: BLE001
            pass

    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}
    rows = raw if isinstance(raw, list) else (raw.get("chunks") if isinstance(raw, dict) else None)
    if not isinstance(rows, list):
        return {}

    mapping: dict[str, str] = {}
    for item in rows:
        if not isinstance(item, dict):
            continue
        cid = str(item.get("chunk_id", "") or "").strip()
        txt = str(item.get("text", "") or "")
        if not cid or not txt:
            continue
        mapping[_text_hash(txt)] = cid

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(
        json.dumps(
            {
                "version": 1,
                "source_store_file": str(p),
                "source_store_fingerprint": source_fp,
                "by_text_hash": mapping,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return mapping


def _load_or_build_chunk_doc_map(store_file: str | None) -> dict[str, str]:
    if not store_file:
        return {}
    p = Path(store_file)
    if not p.exists():
        return {}

    suffix = "sampled" if "sampled" in p.name else "full"
    cache_file = Path("outputs/cache") / f"youtu_chunk_id_map_{suffix}.json"
    source_fp = _sha256_file(str(p))

    if cache_file.exists():
        try:
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            if (
                isinstance(cached, dict)
                and str(cached.get("source_store_file", "")) == str(p)
                and str(cached.get("source_store_fingerprint", "")) == source_fp
                and isinstance(cached.get("by_chunk_id_doc"), dict)
            ):
                return {str(k): str(v) for k, v in cached["by_chunk_id_doc"].items()}
        except Exception:  # noqa: BLE001
            pass

    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}
    rows = raw if isinstance(raw, list) else (raw.get("chunks") if isinstance(raw, dict) else None)
    if not isinstance(rows, list):
        return {}

    text_map: dict[str, str] = {}
    doc_map: dict[str, str] = {}
    for item in rows:
        if not isinstance(item, dict):
            continue
        cid = str(item.get("chunk_id", "") or "").strip()
        doc_id = str(item.get("doc_id", "") or "").strip()
        txt = str(item.get("text", "") or "")
        if cid and doc_id:
            doc_map[cid] = doc_id
        if cid and txt:
            text_map[_text_hash(txt)] = cid

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(
        json.dumps(
            {
                "version": 2,
                "source_store_file": str(p),
                "source_store_fingerprint": source_fp,
                "by_text_hash": text_map,
                "by_chunk_id_doc": doc_map,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return doc_map


def _to_chunk_evidence(
    data: dict[str, Any],
    chunk_id_map: dict[str, str] | None = None,
    chunk_doc_map: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    def _is_virtual_chunk_id(value: Any) -> bool:
        return str(value or "").strip() in {"community_context"}

    # Prefer structured chunk payload when backend provides alignment fields.
    raw = data.get("retrieved_chunk_items") or data.get("retrieved_chunks") or data.get("evidence_chunks") or data.get("chunks") or []
    # youtu-graphrag backend returns:
    # - retrieved_chunk_ids: List[str]
    # - retrieved_chunks: List[str]
    # We must zip them to keep chunk_id evidence aligned with gold QA.
    if (
        isinstance(raw, list)
        and raw
        and all(isinstance(x, str) for x in raw)
        and isinstance(data.get("retrieved_chunk_ids"), list)
    ):
        ids = [str(x or "").strip() for x in (data.get("retrieved_chunk_ids") or [])]
        if ids:
            zipped = []
            length_match = len(ids) == len(raw)
            for i, text in enumerate(raw):
                chunk_id = ids[i] if (length_match and i < len(ids)) else ""
                zipped.append(
                    {
                        "chunk_id": chunk_id,
                        "text": str(text or ""),
                        "alignment_source": "paired_chunk_ids" if chunk_id else "paired_chunk_ids_length_mismatch",
                    }
                )
            raw = zipped
    out: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, dict):
            item_text = str(item.get("text", "") or item.get("content", "") or item.get("chunk_text", "") or "")
            resolved_chunk_id = str(item.get("chunk_id", "") or "").strip()
            alignment_source = str(item.get("alignment_source", "") or "").strip() or ("backend_structured" if resolved_chunk_id else "unknown")
            if not resolved_chunk_id and item_text:
                resolved_chunk_id = _extract_inline_chunk_id(item_text)
                if resolved_chunk_id:
                    alignment_source = "inline_chunk_id"
            resolved_doc_id = (
                str(item.get("doc_id", "") or item.get("document_id", "") or item.get("file_path", "") or "").strip() or None
            )
            if chunk_id_map and item_text and not resolved_chunk_id:
                mapped_chunk_id = chunk_id_map.get(_text_hash(item_text), "")
                mapped_doc_id = str(chunk_doc_map.get(mapped_chunk_id) or "").strip() if (chunk_doc_map and mapped_chunk_id) else ""
                if mapped_chunk_id and resolved_doc_id and mapped_doc_id and resolved_doc_id != mapped_doc_id:
                    alignment_source = "text_hash_doc_conflict"
                elif mapped_chunk_id:
                    resolved_chunk_id = mapped_chunk_id
                    alignment_source = "text_hash_fallback"
            if not resolved_doc_id and resolved_chunk_id and chunk_doc_map:
                resolved_doc_id = chunk_doc_map.get(resolved_chunk_id)
                if resolved_doc_id and alignment_source == "unknown":
                    alignment_source = "chunk_doc_map"
            if not resolved_doc_id and resolved_chunk_id and "#p" in resolved_chunk_id:
                # Some youtu responses inline doc-style ids into chunk_id.
                resolved_doc_id = resolved_chunk_id
                if alignment_source == "unknown":
                    alignment_source = "chunk_id_doc_style"
            if _is_virtual_chunk_id(resolved_chunk_id):
                continue
            out.append(
                {
                    "chunk_id": resolved_chunk_id,
                    "text": item_text,
                    "doc_id": resolved_doc_id,
                    "score": item.get("score"),
                    "community_id": str(item.get("community_id", "") or ""),
                    "edge_id": str(item.get("edge_id", "") or ""),
                    "alignment_source": alignment_source,
                }
            )
        else:
            item_text = str(item or "")
            resolved_chunk_id = _extract_inline_chunk_id(item_text)
            resolved_doc_id = None
            alignment_source = "inline_chunk_id" if resolved_chunk_id else "unknown"
            if resolved_chunk_id and chunk_doc_map:
                resolved_doc_id = chunk_doc_map.get(resolved_chunk_id)
                if resolved_doc_id:
                    alignment_source = "chunk_doc_map"
            if not resolved_doc_id and resolved_chunk_id and "#p" in resolved_chunk_id:
                resolved_doc_id = resolved_chunk_id
                alignment_source = "chunk_id_doc_style"
            if _is_virtual_chunk_id(resolved_chunk_id):
                continue
            out.append(
                {
                    "chunk_id": resolved_chunk_id,
                    "text": item_text,
                    "doc_id": resolved_doc_id,
                    "score": None,
                    "community_id": "",
                    "edge_id": "",
                    "alignment_source": alignment_source,
                }
            )
    return out


def _to_subgraph_edges(data: dict[str, Any]) -> list[dict[str, Any]]:
    raw = data.get("retrieved_triples") or data.get("retrieved_edges") or data.get("subgraph_edges") or []
    # Prefer structured triples if provided by youtu-graphrag backend.
    if isinstance(data.get("retrieved_triples_struct"), list) and data.get("retrieved_triples_struct"):
        raw = data.get("retrieved_triples_struct") or raw
    out: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, dict):
            # youtu-graphrag uses subject/relation/object (+score) naming.
            source = item.get("source") if item.get("source") is not None else item.get("subject")
            relation = item.get("relation")
            target = item.get("target") if item.get("target") is not None else item.get("object")
            # Some backend variants only provide textual triples.
            if not (source and relation and target):
                parsed = _parse_triple_text(item.get("text"))
                if parsed:
                    source, relation, target, parsed_score = parsed
                    if item.get("score") is None and item.get("weight") is None and parsed_score is not None:
                        item = {**item, "score": parsed_score}
            edge_id = str(item.get("edge_id", "") or "").strip()
            if not edge_id and source and relation and target:
                edge_id = f"{source}|{relation}|{target}"
            out.append(
                {
                    "edge_id": edge_id,
                    "source": source,
                    "relation": relation,
                    "target": target,
                    "text": item.get("text"),
                    "weight": item.get("weight") if item.get("weight") is not None else item.get("score"),
                }
            )
        else:
            item_text = str(item or "")
            parsed = _parse_triple_text(item_text)
            if parsed:
                source, relation, target, parsed_score = parsed
                out.append(
                    {
                        "edge_id": f"{source}|{relation}|{target}",
                        "source": source,
                        "relation": relation,
                        "target": target,
                        "text": item_text,
                        "weight": parsed_score,
                    }
                )
                continue
            out.append(
                {
                    "edge_id": "",
                    "source": None,
                    "relation": None,
                    "target": None,
                    "text": item_text,
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
                "doc_id": item.get("doc_id"),
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
                "doc_id": item.get("doc_id"),
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


def _to_float_or_none(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _to_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    s = str(v or "").strip().lower()
    return s in {"1", "true", "yes", "y", "on"}


def _normalize_answer_mode(answer_mode: str | None) -> str:
    mode = str(answer_mode or "reject").strip().lower()
    return mode if mode in {"reject", "open"} else "reject"


def _normalize_route_type(route_type: str | None) -> str:
    mode = str(route_type or "").strip().lower()
    if mode in {"", "none", "auto", "default"}:
        return ""
    # Keep backend aliases (for example "soft") unchanged.
    return mode


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
    store_file: str | None = None,
    doc_prefix_filter: str | None = None,
    strict_doc_scope: bool = True,
    answer_mode_requested: str = "reject",
    answer_mode_effective: str = "reject",
    answer_mode_backend_supported: bool = False,
) -> dict[str, Any]:
    data = response.get("data") if isinstance(response.get("data"), dict) else {}
    meta = response.get("meta") if isinstance(response.get("meta"), dict) else {}

    chunk_id_map = _load_or_build_chunk_id_map(store_file)
    chunk_doc_map = _load_or_build_chunk_doc_map(store_file)
    evidence_chunks = _to_chunk_evidence(
        data,
        chunk_id_map=chunk_id_map,
        chunk_doc_map=chunk_doc_map,
    )
    subgraph_edges = _to_subgraph_edges(data)
    communities, community_summaries = _to_communities(data)
    telemetry = _build_telemetry(meta)
    route_type = str(meta.get("route_type", "") or "").strip()
    route_confidence = _to_float_or_none(meta.get("route_confidence"))
    route_reason = str(meta.get("route_reason", "") or "").strip()
    route_fallback = _to_bool(meta.get("route_fallback"))
    decomposition_debug = meta.get("decomposition_debug") if isinstance(meta.get("decomposition_debug"), dict) else {}
    wanted_prefix = _doc_prefix(doc_prefix_filter) if doc_prefix_filter else ""
    if wanted_prefix:
        scoped_chunks = [
            x for x in evidence_chunks
            if _doc_prefix((x or {}).get("doc_id")) == wanted_prefix
        ]
        if scoped_chunks:
            evidence_chunks = scoped_chunks
        elif strict_doc_scope:
            evidence_chunks = []
    if evidence_chunks:
        empty_chunk_ids = sum(1 for x in evidence_chunks if not str((x or {}).get("chunk_id", "")).strip())
        telemetry.setdefault("extra", {})
        telemetry["extra"]["alignment_chunk_id_empty_rate"] = round(float(empty_chunk_ids / len(evidence_chunks)), 6)
    if subgraph_edges:
        empty_edge_ids = sum(1 for x in subgraph_edges if not str((x or {}).get("edge_id", "")).strip())
        telemetry.setdefault("extra", {})
        telemetry["extra"]["alignment_edge_id_empty_rate"] = round(float(empty_edge_ids / len(subgraph_edges)), 6)
    if (telemetry.get("extra") or {}).get("alignment_chunk_id_empty_rate", 0.0) >= 0.95:
        telemetry.setdefault("extra", {})
        telemetry["extra"]["alignment_warning"] = "chunk_id mostly empty; evidence id alignment likely broken"
    payload = {
        "answer": str(data.get("answer", "") or ""),
        "sub_questions": data.get("sub_questions") if isinstance(data.get("sub_questions"), list) else [],
        "answer_mode": _normalize_answer_mode(answer_mode_effective),
        "answer_mode_requested": _normalize_answer_mode(answer_mode_requested),
        "answer_mode_backend_supported": bool(answer_mode_backend_supported),
        "route_type": route_type,
        "route_confidence": route_confidence,
        "route_reason": route_reason,
        "route_fallback": route_fallback,
        "decomposition_debug": decomposition_debug,
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
        "telemetry": telemetry,
        "retrieval_trace": data.get("retrieval_trace") if isinstance(data.get("retrieval_trace"), dict) else {},
        "reasoning_trace": data.get("reasoning_trace") if isinstance(data.get("reasoning_trace"), dict) else {},
        "answer_trace": data.get("answer_trace") if isinstance(data.get("answer_trace"), dict) else {},
        "reasoning_steps": data.get("reasoning_steps") if isinstance(data.get("reasoning_steps"), list) else [],
        "evaluation_payload": data.get("evaluation_payload") if isinstance(data.get("evaluation_payload"), dict) else {},
        "generate_summary_on_demand": False,
        "embedding_cache": {"enabled": False, "source": "youtu"},
        "on_demand_summaries_generated": 0,
    }
    if wanted_prefix:
        scoped_evidence = [
            x for x in (payload.get("evidence") or [])
            if _doc_prefix((x or {}).get("doc_id")) == wanted_prefix
        ]
        if scoped_evidence:
            payload["evidence"] = scoped_evidence
        elif strict_doc_scope:
            payload["evidence"] = []
            payload["answer"] = "NOT_FOUND"
            telemetry.setdefault("extra", {})
            telemetry["extra"]["doc_scope_filtered_empty"] = True
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
    store_file: str | None = None,
    doc_prefix_filter: str | None = None,
    strict_doc_scope: bool = True,
    answer_mode: str = "reject",
    route_type: str | None = None,
    client_id: str | None = None,
) -> dict[str, Any]:
    del graph_file, communities_file
    answer_mode = _normalize_answer_mode(answer_mode)

    settings = _resolve_youtu_settings()
    base_url = youtu_base_url or settings["base_url"]
    dataset = youtu_dataset or settings["dataset"]
    effective_route_type = _normalize_route_type(str(route_type or settings.get("route_type", "")))
    effective_client_id = str(client_id or settings.get("client_id", "")).strip()
    client = YoutuClient(base_url=base_url, timeout_sec=timeout_sec or settings["timeout_sec"])

    payload = {
        "query": query,
        "question": query,
        "answer_mode": answer_mode,
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
    if effective_route_type:
        payload["route_type"] = effective_route_type
    if effective_client_id:
        payload["client_id"] = effective_client_id

    retries = max(1, int(settings["max_retries"]))
    sleep_sec = float(settings["retry_sleep_sec"])
    last_error: str | None = None
    for attempt in range(1, retries + 1):
        try:
            response = client.search(dataset_name=dataset, payload=payload)
            data = response.get("data") if isinstance(response.get("data"), dict) else {}
            meta = response.get("meta") if isinstance(response.get("meta"), dict) else {}
            backend_supported = (
                ("answer_mode" in data)
                or ("answer_mode" in meta)
                or ("supported_answer_modes" in data)
                or ("supported_answer_modes" in meta)
            )
            effective_mode_raw = (
                str(data.get("answer_mode", "") or "")
                or str(meta.get("answer_mode", "") or "")
                or answer_mode
            )
            effective_mode = _normalize_answer_mode(effective_mode_raw)
            return map_youtu_to_graph_payload(
                response,
                query_level=query_level,
                use_hierarchy=use_hierarchy,
                use_community_summaries=use_community_summaries,
                shuffle_communities=shuffle_communities,
                use_map_reduce=use_map_reduce,
                map_keypoints_limit=map_keypoints_limit,
                max_evidence=max_evidence,
                store_file=store_file,
                doc_prefix_filter=doc_prefix_filter,
                strict_doc_scope=strict_doc_scope,
                answer_mode_requested=answer_mode,
                answer_mode_effective=effective_mode,
                answer_mode_backend_supported=backend_supported,
            )
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            if attempt < retries:
                time.sleep(sleep_sec)

    return {
        "answer": "",
        "answer_mode": answer_mode,
        "answer_mode_requested": answer_mode,
        "answer_mode_backend_supported": False,
        "route_type": "",
        "route_confidence": None,
        "route_reason": last_error or "",
        "route_fallback": True,
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
