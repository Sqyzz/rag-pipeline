from __future__ import annotations

import asyncio
import hashlib
import json
import re
import shutil
import sys
import threading
import time
import concurrent.futures
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.config import cfg
from utils.alignment_payloads import build_minimal_trace_bundle

_INDEX_LOCK = threading.Lock()
_CHUNK_CACHE: dict[str, dict[str, Any]] = {}
_QUERY_LOOP_LOCK = threading.Lock()
_QUERY_LOOP: asyncio.AbstractEventLoop | None = None
_QUERY_LOOP_THREAD: threading.Thread | None = None


def _normalize_answer_mode(answer_mode: str | None) -> str:
    mode = str(answer_mode or "reject").strip().lower()
    return mode if mode in {"reject", "open"} else "reject"


def _resolve_api_key(env_name: str, fallback: str | None = None) -> str:
    import os

    value = str(os.getenv(env_name, "")).strip()
    if value:
        return value
    if fallback:
        alt = str(os.getenv(fallback, "")).strip()
        if alt:
            return alt
    raise RuntimeError(f"Missing API key env: {env_name}")


def _ensure_query_loop() -> asyncio.AbstractEventLoop:
    global _QUERY_LOOP, _QUERY_LOOP_THREAD
    with _QUERY_LOOP_LOCK:
        if _QUERY_LOOP is not None and _QUERY_LOOP_THREAD is not None and _QUERY_LOOP_THREAD.is_alive():
            return _QUERY_LOOP

        ready = threading.Event()
        loop_holder: dict[str, asyncio.AbstractEventLoop] = {}

        def _runner() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop_holder["loop"] = loop
            ready.set()
            loop.run_forever()

        thread = threading.Thread(
            target=_runner,
            name="lightrag-query-loop",
            daemon=True,
        )
        thread.start()
        ready.wait()
        _QUERY_LOOP = loop_holder["loop"]
        _QUERY_LOOP_THREAD = thread
        return _QUERY_LOOP


def _run_query_coro(coro: Any) -> Any:
    loop = _ensure_query_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    try:
        return future.result()
    except concurrent.futures.CancelledError as exc:
        raise RuntimeError("LightRAG query coroutine was cancelled") from exc


def _file_fingerprint(path: str) -> str:
    p = Path(path)
    st = p.stat()
    raw = f"{p.resolve()}|{st.st_size}|{st.st_mtime_ns}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _hash_text(text: Any) -> str:
    norm = re.sub(r"\s+", " ", str(text or "")).strip().lower()
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()


def _to_positive_int(value: Any, default: int = 1) -> int:
    try:
        v = int(value)
    except (TypeError, ValueError):
        return int(default)
    return v if v > 0 else int(default)


def _resolve_merge_chunks(merge_chunks: int | None = None) -> int:
    if merge_chunks is not None:
        return _to_positive_int(merge_chunks, default=1)
    lightrag_cfg = getattr(cfg, "lightrag", None)
    for key in ("merge_chunks", "chunk_merge_size", "merge_group_size"):
        raw = getattr(lightrag_cfg, key, None) if lightrag_cfg is not None else None
        if raw is not None:
            return _to_positive_int(raw, default=1)
    return 1


def _load_chunks_assets(chunks_file: str) -> dict[str, Any]:
    fp = _file_fingerprint(chunks_file)
    cached = _CHUNK_CACHE.get(chunks_file)
    if cached and cached.get("fingerprint") == fp:
        return cached

    rows: list[dict[str, str]] = []
    by_hash: dict[str, dict[str, str]] = {}
    with open(chunks_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = str(obj.get("text", "") or "").strip()
            if not text:
                continue
            chunk_id = str(obj.get("chunk_id", "") or "").strip()
            doc_id = str(obj.get("doc_id", "") or "").strip()
            if not chunk_id:
                continue
            row = {
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "text": text,
            }
            rows.append(row)
            by_hash.setdefault(_hash_text(text), row)

    payload = {
        "fingerprint": fp,
        "rows": rows,
        "by_hash": by_hash,
    }
    _CHUNK_CACHE[chunks_file] = payload
    return payload


def _prepare_rows_for_index(assets: dict[str, Any], merge_chunks: int) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    size = _to_positive_int(merge_chunks, default=1)
    if size <= 1:
        return list(assets["rows"]), dict(assets["by_hash"])

    cache = assets.setdefault("prepared_cache", {})
    cache_key = f"merge_{size}"
    cached = cache.get(cache_key)
    if isinstance(cached, dict) and isinstance(cached.get("rows"), list) and isinstance(cached.get("by_hash"), dict):
        return cached["rows"], cached["by_hash"]

    rows: list[dict[str, Any]] = []
    by_hash: dict[str, dict[str, Any]] = {}
    source_rows = assets["rows"]
    i = 0
    while i < len(source_rows):
        head = source_rows[i]
        doc_id = str(head.get("doc_id", "") or "").strip()
        group = [head]
        j = i + 1
        while j < len(source_rows) and len(group) < size:
            nxt = source_rows[j]
            if str(nxt.get("doc_id", "") or "").strip() != doc_id:
                break
            group.append(nxt)
            j += 1
        merged_text = "\n".join(str(x.get("text", "") or "") for x in group).strip()
        if merged_text:
            merged_chunk_ids = [str(x.get("chunk_id", "") or "").strip() for x in group if str(x.get("chunk_id", "") or "").strip()]
            rep_chunk_id = merged_chunk_ids[0] if merged_chunk_ids else str(head.get("chunk_id", "") or "").strip()
            row = {
                "chunk_id": rep_chunk_id,
                "doc_id": doc_id,
                "text": merged_text,
                "chunk_ids": merged_chunk_ids,
            }
            rows.append(row)
            by_hash.setdefault(_hash_text(merged_text), row)
        i = j

    prepared = {"rows": rows, "by_hash": by_hash}
    cache[cache_key] = prepared
    return rows, by_hash


def _resolve_working_dir(chunks_file: str, working_dir: str | None = None) -> Path:
    if working_dir:
        return Path(working_dir)
    stem = Path(chunks_file).stem
    return Path("outputs/lightrag") / stem


def _state_file(working_dir: Path) -> Path:
    return working_dir / "_pipeline_state.json"


def _load_state(working_dir: Path) -> dict[str, Any]:
    sf = _state_file(working_dir)
    if not sf.exists():
        return {}
    try:
        return json.loads(sf.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_state(working_dir: Path, state: dict[str, Any]) -> None:
    working_dir.mkdir(parents=True, exist_ok=True)
    state_file = _state_file(working_dir)
    tmp_file = state_file.with_name(state_file.name + ".tmp")
    tmp_file.write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp_file.replace(state_file)


def _has_reusable_index_artifacts(working_dir: Path) -> bool:
    required = (
        "graph_chunk_entity_relation.graphml",
        "kv_store_doc_status.json",
        "kv_store_full_docs.json",
        "kv_store_text_chunks.json",
        "vdb_chunks.json",
        "vdb_entities.json",
        "vdb_relationships.json",
    )
    return working_dir.exists() and all((working_dir / name).exists() for name in required)


def _read_lightrag_graph_stats(working_dir: Path) -> dict[str, int]:
    graphml_file = working_dir / "graph_chunk_entity_relation.graphml"
    if not graphml_file.exists():
        return {"num_nodes": 0, "num_edges": 0}

    try:
        root = ET.parse(graphml_file).getroot()
    except Exception:
        return {"num_nodes": 0, "num_edges": 0}

    num_nodes = 0
    num_edges = 0
    for elem in root.iter():
        tag = str(elem.tag or "")
        if tag.endswith("node"):
            num_nodes += 1
        elif tag.endswith("edge"):
            num_edges += 1
    return {"num_nodes": int(num_nodes), "num_edges": int(num_edges)}


def _import_lightrag():
    root = Path(__file__).resolve().parents[2] / "LightRAG"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from lightrag import LightRAG, QueryParam
    from lightrag.llm.openai import openai_complete_if_cache, openai_embed
    from lightrag.stage_trace import get_current_stage
    from lightrag.utils import TokenTracker, wrap_embedding_func_with_attrs

    return (
        LightRAG,
        QueryParam,
        openai_complete_if_cache,
        openai_embed,
        get_current_stage,
        TokenTracker,
        wrap_embedding_func_with_attrs,
    )


def _usage_snapshot(tracker: Any) -> dict[str, int]:
    raw = tracker.get_usage() if hasattr(tracker, "get_usage") else {}
    return {
        "call_count": int((raw or {}).get("call_count", 0) or 0),
        "prompt_tokens": int((raw or {}).get("prompt_tokens", 0) or 0),
        "completion_tokens": int((raw or {}).get("completion_tokens", 0) or 0),
        "total_tokens": int((raw or {}).get("total_tokens", 0) or 0),
    }


def _usage_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    out: dict[str, int] = {}
    for k in ("call_count", "prompt_tokens", "completion_tokens", "total_tokens"):
        out[k] = int(after.get(k, 0) or 0) - int(before.get(k, 0) or 0)
    return out


async def _create_rag(working_dir: Path):
    (
        LightRAG,
        _QueryParam,
        openai_complete_if_cache,
        openai_embed,
        get_current_stage,
        TokenTracker,
        wrap_embedding_func_with_attrs,
    ) = _import_lightrag()

    llm_key = _resolve_api_key(str(cfg.llm.api.api_key_env), fallback="DASHSCOPE_API_KEY")
    emb_key = _resolve_api_key(str(cfg.embedding.api.api_key_env), fallback="DASHSCOPE_API_KEY")
    llm_tracker = TokenTracker()
    emb_tracker = TokenTracker()

    llm_model = str(cfg.llm.api.model)
    llm_base = str(cfg.llm.api.base_url)
    emb_model = str(cfg.embedding.api.model)
    emb_base = str(cfg.embedding.api.base_url)
    emb_dim = 1536
    phase_stats: dict[str, dict[str, int]] = {}

    def _phase(name: str) -> dict[str, int]:
        key = str(name or "unspecified")
        bucket = phase_stats.get(key)
        if bucket is None:
            bucket = {
                "llm_calls": 0,
                "embedding_calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "latency_ms_total": 0,
            }
            phase_stats[key] = bucket
        return bucket

    # `openai_complete_if_cache(model, prompt, ...)` expects model as the first positional arg.
    # If model is bound by keyword, later prompt positional args can collide and raise:
    # "got multiple values for argument 'model'".
    async def _llm_func(prompt: str, **kwargs: Any):
        stage = str(get_current_stage() or "unspecified")
        before = _usage_snapshot(llm_tracker)
        t0 = time.perf_counter()
        result = await openai_complete_if_cache(
            llm_model,
            prompt,
            base_url=llm_base,
            api_key=llm_key,
            token_tracker=llm_tracker,
            **kwargs,
        )
        delta = _usage_delta(before, _usage_snapshot(llm_tracker))
        bucket = _phase(stage)
        bucket["llm_calls"] += int(max(0, delta.get("call_count", 0)))
        bucket["prompt_tokens"] += int(max(0, delta.get("prompt_tokens", 0)))
        bucket["completion_tokens"] += int(max(0, delta.get("completion_tokens", 0)))
        bucket["total_tokens"] += int(max(0, delta.get("total_tokens", 0)))
        bucket["latency_ms_total"] += int((time.perf_counter() - t0) * 1000)
        return result

    async def _embed_func(texts: list[str], **kwargs: Any):
        stage = str(get_current_stage() or "unspecified")
        before = _usage_snapshot(emb_tracker)
        t0 = time.perf_counter()
        ret = await openai_embed.func(
            texts=texts,
            model=emb_model,
            base_url=emb_base,
            api_key=emb_key,
            token_tracker=emb_tracker,
            embedding_dim=emb_dim,
        )
        delta = _usage_delta(before, _usage_snapshot(emb_tracker))
        bucket = _phase(stage)
        bucket["embedding_calls"] += int(max(0, delta.get("call_count", 0)))
        bucket["prompt_tokens"] += int(max(0, delta.get("prompt_tokens", 0)))
        bucket["total_tokens"] += int(max(0, delta.get("total_tokens", 0)))
        bucket["latency_ms_total"] += int((time.perf_counter() - t0) * 1000)
        return ret

    embed_func = wrap_embedding_func_with_attrs(
        embedding_dim=emb_dim,
        max_token_size=8192,
        model_name=emb_model,
    )(_embed_func)

    rag = LightRAG(
        working_dir=str(working_dir),
        llm_model_func=_llm_func,
        embedding_func=embed_func,
    )
    await rag.initialize_storages()
    return rag, llm_tracker, emb_tracker, phase_stats


def _target_state(chunks_file: str, merge_chunks: int) -> dict[str, Any]:
    return {
        "chunks_file": str(Path(chunks_file).resolve()),
        "chunks_fingerprint": _file_fingerprint(chunks_file),
        "llm_model": str(cfg.llm.api.model),
        "embedding_model": str(cfg.embedding.api.model),
        "embedding_dim": 1536,
        "merge_chunks": int(_to_positive_int(merge_chunks, default=1)),
    }


def _format_state_mismatch(current: dict[str, Any], target: dict[str, Any]) -> str:
    mismatches: list[str] = []
    # `llm_model` affects answer generation, but does not invalidate an existing index.
    for key in ("chunks_file", "chunks_fingerprint", "embedding_model", "embedding_dim", "merge_chunks"):
        if current.get(key) != target.get(key):
            mismatches.append(f"{key}: current={current.get(key)!r}, requested={target.get(key)!r}")
    return "; ".join(mismatches) if mismatches else "unknown state mismatch"


def _ensure_indexed(
    chunks_file: str,
    working_dir: Path,
    force_rebuild: bool = False,
    merge_chunks: int | None = None,
) -> dict[str, Any]:
    merge_size = _resolve_merge_chunks(merge_chunks)
    target = _target_state(chunks_file, merge_chunks=merge_size)
    with _INDEX_LOCK:
        current = _load_state(working_dir)
        comparable_current = dict(current or {})
        comparable_target = dict(target)
        comparable_current.pop("llm_model", None)
        comparable_target.pop("llm_model", None)
        if (not force_rebuild) and comparable_current == comparable_target and working_dir.exists():
            return {
                "rebuilt": False,
                "reused_cache": True,
                "working_dir": str(working_dir),
                "chunks_file": str(Path(chunks_file).resolve()),
                "llm_calls": 0,
                "embedding_calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "latency_ms_total": 0,
                "usage_source": "lightrag_token_tracker",
                "merge_chunks": int(merge_size),
                "phase_breakdown": {},
                **_read_lightrag_graph_stats(working_dir),
            }
        has_reusable_artifacts = _has_reusable_index_artifacts(working_dir)
        if (not force_rebuild) and working_dir.exists() and current and comparable_current != comparable_target:
            raise RuntimeError(
                "LightRAG working_dir already contains an index built for a different configuration. "
                f"working_dir={working_dir}; {_format_state_mismatch(current, target)}. "
                "Use a different --lightrag-working-dir or pass the matching --chunks-file/--merge-chunks. "
                "If you intentionally want to overwrite it, rerun with force_rebuild=true."
            )
        if (not force_rebuild) and (not current) and has_reusable_artifacts:
            raise RuntimeError(
                "LightRAG working_dir already contains reusable index artifacts but is missing "
                f"{_state_file(working_dir).name}, so provenance cannot be verified safely. "
                f"working_dir={working_dir}. Use a different --lightrag-working-dir or rebuild explicitly "
                "with force_rebuild=true."
            )

        if working_dir.exists():
            shutil.rmtree(working_dir)
        working_dir.mkdir(parents=True, exist_ok=True)

        assets = _load_chunks_assets(chunks_file)
        rows, _ = _prepare_rows_for_index(assets, merge_chunks=merge_size)

        async def _index() -> dict[str, Any]:
            rag, llm_tracker, emb_tracker, phase_stats = await _create_rag(working_dir)
            t0 = time.perf_counter()
            try:
                if rows:
                    texts = [r["text"] for r in rows]
                    ids = [r["chunk_id"] for r in rows]
                    file_paths = [r["doc_id"] for r in rows]
                    await rag.ainsert(texts, ids=ids, file_paths=file_paths)
            finally:
                await rag.finalize_storages()
            llm_usage = llm_tracker.get_usage()
            emb_usage = emb_tracker.get_usage()
            result = {
                "llm_calls": int(llm_usage.get("call_count", 0)),
                "embedding_calls": int(emb_usage.get("call_count", 0)),
                "prompt_tokens": int(llm_usage.get("prompt_tokens", 0)) + int(emb_usage.get("prompt_tokens", 0)),
                "completion_tokens": int(llm_usage.get("completion_tokens", 0)),
                "total_tokens": int(llm_usage.get("total_tokens", 0)) + int(emb_usage.get("total_tokens", 0)),
                "latency_ms_total": int((time.perf_counter() - t0) * 1000),
            }
            result["phase_breakdown"] = phase_stats
            return result

        usage_stats = asyncio.run(_index())
        _save_state(working_dir, target)
        if not _state_file(working_dir).exists():
            raise RuntimeError(f"Failed to persist LightRAG pipeline state: {working_dir}")
        return {
            "rebuilt": True,
            "reused_cache": False,
            "working_dir": str(working_dir),
            "chunks_file": str(Path(chunks_file).resolve()),
            "usage_source": "lightrag_token_tracker",
            "merge_chunks": int(merge_size),
            **_read_lightrag_graph_stats(working_dir),
            **usage_stats,
        }


def ensure_lightrag_assets(
    chunks_file: str,
    working_dir: str | None = None,
    force_rebuild: bool = False,
    merge_chunks: int | None = None,
) -> dict[str, Any]:
    work_dir = _resolve_working_dir(chunks_file=chunks_file, working_dir=working_dir)
    return _ensure_indexed(
        chunks_file=chunks_file,
        working_dir=work_dir,
        force_rebuild=force_rebuild,
        merge_chunks=merge_chunks,
    )


def _build_user_prompt(answer_mode: str, query_type: str | None = None) -> str:
    mode = _normalize_answer_mode(answer_mode)
    if mode == "reject":
        policy = "If evidence is insufficient, return exactly: NOT_FOUND"
    else:
        policy = (
            "If evidence is insufficient, you may answer using general knowledge. "
            "Prefix your answer with: OUTSIDE_EVIDENCE:"
        )
    if str(query_type or "").strip().lower() == "global_summary":
        return (
            "Return exactly 3 concise bullet points covering key themes and risks. "
            "Do not add preface or legal advice. "
            + policy
        )
    return "Return a concise direct answer without extra explanation. " + policy


def answer_with_lightrag(
    query: str,
    chunks_file: str,
    top_k: int = 10,
    max_completion_tokens: int | None = None,
    answer_mode: str = "reject",
    query_type: str | None = None,
    working_dir: str | None = None,
    lightrag_mode: str = "hybrid",
    force_rebuild: bool = False,
    merge_chunks: int | None = None,
    doc_prefix_filter: str | None = None,
) -> dict:
    mode = _normalize_answer_mode(answer_mode)
    merge_size = _resolve_merge_chunks(merge_chunks)
    work_dir = _resolve_working_dir(chunks_file, working_dir=working_dir)
    build_info = _ensure_indexed(
        chunks_file=chunks_file,
        working_dir=work_dir,
        force_rebuild=force_rebuild,
        merge_chunks=merge_size,
    )
    assets = _load_chunks_assets(chunks_file)
    _, by_hash = _prepare_rows_for_index(assets, merge_chunks=merge_size)

    async def _query_once() -> dict:
        (
            _LightRAG,
            QueryParam,
            _openai_complete_if_cache,
            _openai_embed,
            _get_current_stage,
            _TokenTracker,
            _wrap_embedding_func_with_attrs,
        ) = _import_lightrag()
        rag, llm_tracker, emb_tracker, _phase_stats = await _create_rag(work_dir)
        t0 = time.perf_counter()
        try:
            param = QueryParam(
                mode=lightrag_mode,
                top_k=int(max(1, top_k)),
                chunk_top_k=int(max(1, top_k)),
                stream=False,
                include_references=True,
                user_prompt=_build_user_prompt(mode, query_type=query_type),
            )
            if max_completion_tokens is not None:
                # LightRAG does not expose per-call max_tokens directly in QueryParam;
                # keep answer length constrained via user prompt policy above.
                _ = int(max_completion_tokens)

            payload = await rag.aquery_llm(query, param=param)
            latency_ms = int((time.perf_counter() - t0) * 1000)
            llm_usage = llm_tracker.get_usage()
            emb_usage = emb_tracker.get_usage()

            llm_response = (payload.get("llm_response") or {}) if isinstance(payload, dict) else {}
            answer = str(llm_response.get("content") or "").strip()
            chunks = (((payload.get("data") or {}).get("chunks") or []) if isinstance(payload, dict) else [])
            evidence_chunks: list[dict[str, Any]] = []
            seen: set[str] = set()
            for item in chunks:
                if not isinstance(item, dict):
                    continue
                text = str(item.get("content", "") or "")
                h = _hash_text(text)
                mapped = by_hash.get(h, {})
                chunk_id = str(mapped.get("chunk_id") or item.get("chunk_id") or "").strip()
                doc_id = str(mapped.get("doc_id") or item.get("file_path") or "").strip()
                if not chunk_id and text:
                    chunk_id = h
                if not chunk_id or chunk_id in seen:
                    continue
                seen.add(chunk_id)
                evidence_chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "doc_id": doc_id,
                        "text": text,
                        "chunk_ids": list(mapped.get("chunk_ids") or []),
                    }
                )

            telemetry = {
                "llm_calls": int(llm_usage.get("call_count", 0)),
                "embedding_calls": int(emb_usage.get("call_count", 0)),
                "llm_latency_ms": int(latency_ms),
                "embedding_latency_ms": 0,
                "prompt_tokens": int(llm_usage.get("prompt_tokens", 0)) + int(emb_usage.get("prompt_tokens", 0)),
                "completion_tokens": int(llm_usage.get("completion_tokens", 0)),
                "total_tokens": int(llm_usage.get("total_tokens", 0)) + int(emb_usage.get("total_tokens", 0)),
                "extra": {
                    "usage_complete": True,
                    "lightrag_mode": lightrag_mode,
                    "working_dir": str(work_dir),
                    "lightrag_build": build_info,
                    "merge_chunks": int(merge_size),
                },
            }
            reasoning_steps = [
                {
                    "type": "lightrag_query",
                    "query_mode": lightrag_mode,
                    "top_k": int(max(1, top_k)),
                    "retrieved_chunk_count": len(evidence_chunks),
                    "answer_preview": answer[:300],
                }
            ]
            trace_bundle = build_minimal_trace_bundle(
                method="lightrag",
                query=query,
                answer=answer,
                evidence_chunks=evidence_chunks,
                doc_prefix_filter=doc_prefix_filter,
                response_for_eval=answer,
                response_for_eval_source="lightrag_answer",
                orchestration_mode=f"lightrag_{lightrag_mode}",
                aggregation_strategy="lightrag_direct_generation_with_retrieved_chunk_backing",
                reasoning_steps=reasoning_steps,
                final_answer_prompt_inputs={
                    "query_mode": str(lightrag_mode or "").strip(),
                    "top_k": int(max(1, top_k)),
                    "merge_chunks": int(merge_size),
                    "working_dir": str(work_dir),
                },
                retrieval_trace_extra={
                    "query_mode": str(lightrag_mode or "").strip(),
                    "top_k": int(max(1, top_k)),
                    "merge_chunks": int(merge_size),
                    "working_dir": str(work_dir),
                },
                reasoning_trace_extra={
                    "query_mode": str(lightrag_mode or "").strip(),
                    "top_k": int(max(1, top_k)),
                    "merge_chunks": int(merge_size),
                },
                answer_trace_extra={
                    "final_chunk_selection_strategy": "lightrag_referenced_chunks",
                },
            )
            return {
                "answer": answer,
                "answer_mode": mode,
                "query_mode": lightrag_mode,
                "evidence": [],
                "subgraph_edges": [],
                "evidence_chunks": evidence_chunks,
                "telemetry": telemetry,
                "answer_scope_target_doc_id": trace_bundle["answer_scope_target_doc_id"],
                "answer_composition_mode": trace_bundle["answer_composition_mode"],
                "semantic_alignment": trace_bundle["semantic_alignment"],
                "evaluation_payload": trace_bundle["evaluation_payload"],
                "retrieval_trace": trace_bundle["retrieval_trace"],
                "reasoning_trace": trace_bundle["reasoning_trace"],
                "answer_trace": trace_bundle["answer_trace"],
                "reasoning_steps": trace_bundle["reasoning_steps"],
            }
        finally:
            await rag.finalize_storages()

    return _run_query_coro(_query_once())
