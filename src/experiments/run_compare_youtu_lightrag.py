from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from baselines.graph_rag import answer_with_graphrag
from baselines.lightrag_adapter import answer_with_lightrag, ensure_lightrag_assets
from evaluation.run_eval import run_eval
from experiments.run_compare import (  # Reuse canonical compare/eval protocol helpers.
    _YOUTU_AVAILABLE,
    _check_budget,
    _ensure_youtu_graph_assets,
    ensure_graph_assets,
    _extract_query_doc_key,
    _infer_dataset_tag,
    _load_queries,
    _load_yaml,
    _method_payload_ready,
    _normalize_youtu_route_type,
    _parse_bool,
    _query_consistency_snapshot,
    _rebuild_aggregate_views,
    _regime_settings,
    _row_is_complete_for_compare,
    _resolve_answer_modes,
    _resolve_regimes,
    _scope_query_for_cuad,
    _write_json,
    _write_jsonl,
)

try:
    from adapters.youtu_schema_adapter import load_and_adapt_schema
    from baselines.youtu_graph_rag_adapter import answer_with_youtu_graphrag
except ImportError:
    from src.adapters.youtu_schema_adapter import load_and_adapt_schema
    from src.baselines.youtu_graph_rag_adapter import answer_with_youtu_graphrag


def _progress(message: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[run_compare_youtu_lightrag {ts}] {message}", flush=True)


def _parse_int_list(raw: str, default: list[int]) -> list[int]:
    vals: list[int] = []
    for x in str(raw or "").split(","):
        x = x.strip()
        if not x:
            continue
        vals.append(int(x))
    return vals or list(default)


def _parse_mode_list(raw: str, default: list[str]) -> list[str]:
    vals: list[str] = []
    for x in str(raw or "").split(","):
        mode = x.strip().lower()
        if mode in {"reject", "open"} and mode not in vals:
            vals.append(mode)
    return vals or list(default)


def _parse_rag_list(raw: str, default: list[str]) -> list[str]:
    alias = {
        "lightrag": "lightrag",
        "youtu": "youtu",
        "youtu_graph_rag": "youtu",
        "youtu-graphrag": "youtu",
        "graph_rag": "graph_rag",
        "graphrag": "graph_rag",
        "graph-rag": "graph_rag",
    }
    vals: list[str] = []
    for x in str(raw or "").split(","):
        key = str(x or "").strip().lower()
        if not key:
            continue
        mapped = alias.get(key)
        if mapped and mapped not in vals:
            vals.append(mapped)
    return vals or list(default)


def _write_md_table(path: Path, df: pd.DataFrame) -> None:
    lines = [
        "| Method | Dataset | Regime | Mode | Top20 | Top10 | Tokens | Time |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in df.iterrows():
        lines.append(
            "| "
            + f"{row.get('method', '')} | {row.get('dataset', '')} | {row.get('regime', '')} | {row.get('mode', '')} | "
            + f"{row.get('top20_accuracy', '')} | {row.get('top10_accuracy', '')} | "
            + f"{row.get('tokens', '')} | {row.get('time', '')} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _to_int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _pick_answer_accuracy(item: dict[str, Any]) -> float | None:
    yesno = item.get("answer_semantic_yesno")
    if yesno is not None:
        try:
            return float(yesno)
        except (TypeError, ValueError):
            pass
    exact_relaxed = item.get("answer_exact_relaxed")
    if exact_relaxed is not None:
        try:
            return float(exact_relaxed)
        except (TypeError, ValueError):
            pass
    return None


def _extract_usage_triplet(payload: dict[str, Any]) -> tuple[int | None, int | None, int | None]:
    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
    return (
        _to_int_or_none(usage.get("prompt_tokens")),
        _to_int_or_none(usage.get("completion_tokens")),
        _to_int_or_none(usage.get("total_tokens")),
    )


def _selected_method_names(selected_rags: set[str], include_youtu: bool) -> set[str]:
    methods: set[str] = set()
    if "graph_rag" in selected_rags:
        methods.add("graph_rag")
    if "lightrag" in selected_rags:
        methods.add("lightrag")
    if "youtu" in selected_rags and include_youtu:
        methods.add("youtu_graph_rag")
    return methods


def _select_queries_balanced_by_type(queries: list[dict[str, Any]], max_queries: int | None) -> list[dict[str, Any]]:
    if max_queries is None or int(max_queries) <= 0 or len(queries) <= int(max_queries):
        return queries

    limit = int(max_queries)
    buckets: dict[str, list[dict[str, Any]]] = {}
    type_order: list[str] = []
    for q in queries:
        qtype = str(q.get("type", "unknown")).strip() or "unknown"
        if qtype not in buckets:
            buckets[qtype] = []
            type_order.append(qtype)
        buckets[qtype].append(q)

    active_types = [t for t in type_order if buckets.get(t)]
    if not active_types:
        return queries[:limit]

    counts = {t: 0 for t in active_types}
    remaining = limit

    # First pass: give each type a near-equal base allocation, capped by availability.
    for idx, qtype in enumerate(active_types):
        slots_left = len(active_types) - idx
        base = remaining // slots_left if slots_left > 0 else 0
        take = min(len(buckets[qtype]), base)
        counts[qtype] = take
        remaining -= take

    # Second pass: distribute leftover capacity round-robin to keep per-type counts within 1 when possible.
    while remaining > 0:
        progressed = False
        for qtype in active_types:
            if remaining <= 0:
                break
            if counts[qtype] >= len(buckets[qtype]):
                continue
            counts[qtype] += 1
            remaining -= 1
            progressed = True
        if not progressed:
            break

    selected_by_type = {t: buckets[t][: counts[t]] for t in active_types}
    selected: list[dict[str, Any]] = []
    while len(selected) < limit:
        progressed = False
        for qtype in active_types:
            bucket = selected_by_type[qtype]
            if not bucket:
                continue
            selected.append(bucket.pop(0))
            progressed = True
            if len(selected) >= limit:
                break
        if not progressed:
            break
    return selected[:limit]


def _merge_retrieve_rows_with_existing(
    current_rows: list[dict[str, Any]],
    existing_rows: list[dict[str, Any]],
    selected_methods: set[str],
) -> tuple[list[dict[str, Any]], bool]:
    if not existing_rows or not selected_methods:
        return current_rows, False

    existing_by_qid: dict[str, dict[str, Any]] = {}
    for row in existing_rows:
        if not isinstance(row, dict):
            continue
        qid = str(row.get("qid", "")).strip()
        if qid and qid not in existing_by_qid:
            existing_by_qid[qid] = row

    merged_rows: list[dict[str, Any]] = []
    merged_any = False
    for row in current_rows:
        qid = str(row.get("qid", "")).strip()
        existing = existing_by_qid.get(qid)
        if not existing:
            merged_rows.append(row)
            continue
        same_query_shape = (
            str(existing.get("query", "")).strip() == str(row.get("query", "")).strip()
            and str(existing.get("type", "")).strip() == str(row.get("type", "")).strip()
            and str(existing.get("mode", "")).strip() == str(row.get("mode", "")).strip()
            and int(existing.get("top_k", 0) or 0) == int(row.get("top_k", 0) or 0)
        )
        if not same_query_shape:
            merged_rows.append(row)
            continue

        row_regimes = row.get("regimes") if isinstance(row.get("regimes"), dict) else {}
        existing_regimes = existing.get("regimes") if isinstance(existing.get("regimes"), dict) else {}
        merged_row = dict(row)
        merged_regimes: dict[str, Any] = {}
        for regime_name, regime_payload in row_regimes.items():
            merged_payload = dict(regime_payload) if isinstance(regime_payload, dict) else {}
            existing_payload = existing_regimes.get(regime_name) if isinstance(existing_regimes.get(regime_name), dict) else {}
            for method_name, method_result in existing_payload.items():
                if method_name in selected_methods:
                    continue
                if method_name not in merged_payload:
                    merged_payload[method_name] = method_result
                    merged_any = True
            merged_regimes[regime_name] = merged_payload
        merged_row["regimes"] = merged_regimes
        merged_rows.append(merged_row)

    return merged_rows, merged_any


def _row_is_complete_for_selected_methods(
    row: dict[str, Any],
    regime_names: list[str],
    selected_methods: set[str],
) -> bool:
    return _row_is_complete_for_compare(row, regime_names=regime_names, methods=sorted(selected_methods))


def _materialize_retrieve_rows(
    rows: list[dict[str, Any]],
    existing_rows: list[dict[str, Any]],
    selected_methods: set[str],
) -> tuple[list[dict[str, Any]], bool]:
    if not existing_rows:
        return list(rows), False
    return _merge_retrieve_rows_with_existing(
        current_rows=rows,
        existing_rows=existing_rows,
        selected_methods=selected_methods,
    )


def _build_compare_summary(
    *,
    rows: list[dict[str, Any]],
    queries_file: str,
    dataset_tag: str,
    query_snapshot: dict[str, Any],
    regime_names: list[str],
    methods: list[str],
    top_k: int,
    answer_mode: str,
    budget: dict[str, Any],
    budget_config_file: str,
    cuad_doc_scope: bool,
    strict_doc_scope: bool,
    lightrag_mode: str,
    lightrag_working_dir: str | None,
    graph_file: str,
    communities_file: str,
    retrieve_reuse_cache: bool,
    selected_rags: set[str],
    include_youtu: bool,
    youtu_base_url: str,
    youtu_dataset: str,
    youtu_route_type: str | None,
    youtu_client_id: str | None,
    incremental_update: bool,
    incremental_only: bool,
    num_existing_answers: int,
    youtu_assets_metrics: dict[str, Any],
    results_file: str,
) -> dict[str, Any]:
    aggregate_view, by_type_view = _rebuild_aggregate_views(
        rows=rows,
        regime_names=regime_names,
        methods=methods,
    )
    return {
        "pipeline": "youtu_vs_lightrag",
        "stage": "retrieve",
        "queries_file": queries_file,
        "dataset_tag": dataset_tag,
        "query_consistency": {
            "effective": query_snapshot,
            "max_queries_effective": query_snapshot["total_queries"],
        },
        "num_queries": len(rows),
        "regimes": regime_names,
        "methods": methods,
        "top_k": int(top_k),
        "answer_mode": answer_mode,
        "budget": budget,
        "budget_config_file": budget_config_file,
        "cuad_doc_scope": bool(cuad_doc_scope),
        "strict_doc_scope": bool(strict_doc_scope),
        "lightrag_mode": lightrag_mode,
        "lightrag_working_dir": str(lightrag_working_dir or ""),
        "graph_file": str(graph_file or ""),
        "communities_file": str(communities_file or ""),
        "retrieve_reuse_cache": bool(retrieve_reuse_cache),
        "include_rags": sorted(selected_rags),
        "include_youtu": bool(include_youtu),
        "youtu_base_url": youtu_base_url,
        "youtu_dataset": youtu_dataset,
        "youtu_route_type": str(youtu_route_type or ""),
        "youtu_client_id": str(youtu_client_id or ""),
        "incremental_update": bool(incremental_update),
        "incremental_only": bool(incremental_only),
        "num_existing_answers": int(num_existing_answers),
        "youtu_graph_assets_metrics": youtu_assets_metrics,
        "aggregate_metrics": aggregate_view,
        "aggregate_metrics_by_type": by_type_view,
        "results_file": results_file,
    }


def _write_retrieve_summary_outputs(
    *,
    retrieve_out_dir: Path,
    retrieve_summary_csv: str | None,
    retrieve_summary_md: str | None,
    retrieve_summary_json: str | None,
    queries_file: str,
    gold_file: str,
    regime: str,
    topk_list: list[int],
    answer_modes: list[str],
    judge_mode: str,
    judge_model: str,
    selected_rags: set[str],
    include_youtu_effective: bool,
    lightrag_mode: str,
    merge_chunks: int,
    retrieve_reuse_cache: bool,
    build_force_rebuild: bool,
    runs: list[dict[str, Any]],
    wide: dict[tuple[str, str, str, str], dict[str, Any]],
) -> dict[str, Any]:
    df = pd.DataFrame(list(wide.values()))
    if not df.empty:
        for col in (
            "top10_accuracy",
            "top20_accuracy",
            "tokens",
            "time",
            "tokens_top10",
            "tokens_top20",
            "time_top10",
            "time_top20",
        ):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.sort_values(["dataset", "mode", "method"]).reset_index(drop=True)

    csv_file = (
        Path(retrieve_summary_csv)
        if str(retrieve_summary_csv or "").strip()
        else (retrieve_out_dir / "youtu_lightrag_topk_mode_summary.csv")
    )
    md_file = (
        Path(retrieve_summary_md)
        if str(retrieve_summary_md or "").strip()
        else (retrieve_out_dir / "youtu_lightrag_topk_mode_summary.md")
    )
    csv_file.parent.mkdir(parents=True, exist_ok=True)
    md_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_file, index=False)
    _write_md_table(md_file, df)

    retrieve_summary = {
        "queries_file": queries_file,
        "gold_file": gold_file,
        "regime": regime,
        "topk_list": [int(x) for x in topk_list],
        "answer_modes": answer_modes,
        "judge_mode": judge_mode,
        "judge_model": judge_model,
        "include_rags": sorted(selected_rags),
        "include_youtu": bool(include_youtu_effective),
        "lightrag_mode": str(lightrag_mode),
        "merge_chunks": int(merge_chunks),
        "retrieve_reuse_cache": bool(retrieve_reuse_cache),
        "build_force_rebuild": bool(build_force_rebuild),
        "runs": runs,
        "summary_csv": str(csv_file),
        "summary_md": str(md_file),
        "num_rows": int(len(df)),
        "accuracy_definition": "mean(answer_score_primary), secondary=mean(answer_semantic_yesno)",
        "retrieval_eval_scope": "answer_only_no_community_expand",
    }
    summary_file = (
        Path(retrieve_summary_json)
        if str(retrieve_summary_json or "").strip()
        else (retrieve_out_dir / "youtu_lightrag_topk_mode_summary.json")
    )
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    summary_file.write_text(json.dumps(retrieve_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    retrieve_summary["summary_json"] = str(summary_file)
    return retrieve_summary


def _build_paper_metrics_from_eval(eval_payload: dict[str, Any], top_k: int) -> dict[str, Any]:
    by_type = eval_payload.get("summary", {}).get("by_type", [])
    overall = eval_payload.get("summary", {}).get("overall", [])

    type_alias = {
        "local_factual": "local_f1",
        "cross_clause": "structural_f1",
        "global_summary": "global_f1",
    }
    result: dict[str, Any] = {
        "top_k": int(top_k),
        "f1_metrics": {name: {} for name in type_alias.values()},
        "retrieval_metrics": {
            f"chunk_recall_at_{int(top_k)}": {},
            f"chunk_hit_at_{int(top_k)}": {},
            f"doc_recall_at_{int(top_k)}": {},
            f"doc_hit_at_{int(top_k)}": {},
        },
    }

    for item in by_type:
        qtype = str(item.get("type", "")).strip()
        metric_name = type_alias.get(qtype)
        method = str(item.get("method", "")).strip()
        if not metric_name or not method:
            continue
        result["f1_metrics"][metric_name][method] = item.get("answer_token_f1")

    for item in overall:
        method = str(item.get("method", "")).strip()
        if not method:
            continue
        result["retrieval_metrics"][f"chunk_recall_at_{int(top_k)}"][method] = item.get("chunk_recall_at_k")
        result["retrieval_metrics"][f"chunk_hit_at_{int(top_k)}"][method] = item.get("chunk_hit_rate_at_k")
        result["retrieval_metrics"][f"doc_recall_at_{int(top_k)}"][method] = item.get("doc_recall_at_k")
        result["retrieval_metrics"][f"doc_hit_at_{int(top_k)}"][method] = item.get("doc_hit_rate_at_k")

    return result


def _write_paper_metrics_outputs(base_path: Path, payload: dict[str, Any]) -> dict[str, str]:
    json_path = base_path.with_suffix(".json")
    csv_path = base_path.with_suffix(".csv")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(str(json_path), payload)

    metric_rows: list[dict[str, Any]] = []
    for metric_name, method_map in payload.get("f1_metrics", {}).items():
        row = {"metric": metric_name}
        row.update(method_map if isinstance(method_map, dict) else {})
        metric_rows.append(row)
    for metric_name, method_map in payload.get("retrieval_metrics", {}).items():
        row = {"metric": metric_name}
        row.update(method_map if isinstance(method_map, dict) else {})
        metric_rows.append(row)
    pd.DataFrame(metric_rows).to_csv(csv_path, index=False)
    return {"json": str(json_path), "csv": str(csv_path)}


def _extract_graph_build_stats(local_metrics: dict[str, Any]) -> dict[str, Any]:
    triple = local_metrics.get("triple_extract") if isinstance(local_metrics.get("triple_extract"), dict) else {}
    graph_build = local_metrics.get("graph_build") if isinstance(local_metrics.get("graph_build"), dict) else {}
    community = local_metrics.get("community_build") if isinstance(local_metrics.get("community_build"), dict) else {}
    triple_t = triple.get("telemetry") if isinstance(triple.get("telemetry"), dict) else {}
    comm_t = community.get("telemetry") if isinstance(community.get("telemetry"), dict) else {}
    llm_calls = int(triple.get("llm_calls", 0) or 0) + int(comm_t.get("llm_calls", 0) or 0)
    embedding_calls = int(triple_t.get("embedding_calls", 0) or 0) + int(comm_t.get("embedding_calls", 0) or 0)
    prompt_tokens = int(triple_t.get("prompt_tokens", 0) or 0) + int(comm_t.get("prompt_tokens", 0) or 0)
    completion_tokens = int(triple_t.get("completion_tokens", 0) or 0) + int(comm_t.get("completion_tokens", 0) or 0)
    total_tokens = int(triple_t.get("total_tokens", 0) or 0) + int(comm_t.get("total_tokens", 0) or 0)
    latency_ms_total = (
        int(triple_t.get("llm_latency_ms", 0) or 0)
        + int(triple_t.get("embedding_latency_ms", 0) or 0)
        + int(comm_t.get("llm_latency_ms", 0) or 0)
        + int(comm_t.get("embedding_latency_ms", 0) or 0)
    )
    wall_time_ms_total = (
        int(triple.get("wall_time_ms", 0) or 0)
        + int(graph_build.get("wall_time_ms", 0) or 0)
        + int(community.get("wall_time_ms", 0) or 0)
    )
    phase_breakdown = {
        "triple_extract": {
            "llm_calls": int(triple.get("llm_calls", 0) or 0),
            "embedding_calls": int(triple_t.get("embedding_calls", 0) or 0),
            "prompt_tokens": int(triple_t.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(triple_t.get("completion_tokens", 0) or 0),
            "total_tokens": int(triple_t.get("total_tokens", 0) or 0),
            "latency_ms_total": int(triple_t.get("llm_latency_ms", 0) or 0) + int(triple_t.get("embedding_latency_ms", 0) or 0),
            "wall_time_ms": int(triple.get("wall_time_ms", 0) or 0),
            "skipped": bool(triple.get("skipped", False)),
        },
        "graph_build": {
            "llm_calls": 0,
            "embedding_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "latency_ms_total": 0,
            "wall_time_ms": int(graph_build.get("wall_time_ms", 0) or 0),
            "skipped": bool(graph_build.get("skipped", False)),
        },
        "community_build": {
            "llm_calls": int(comm_t.get("llm_calls", 0) or 0),
            "embedding_calls": int(comm_t.get("embedding_calls", 0) or 0),
            "prompt_tokens": int(comm_t.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(comm_t.get("completion_tokens", 0) or 0),
            "total_tokens": int(comm_t.get("total_tokens", 0) or 0),
            "latency_ms_total": int(comm_t.get("llm_latency_ms", 0) or 0) + int(comm_t.get("embedding_latency_ms", 0) or 0),
            "wall_time_ms": int(community.get("wall_time_ms", 0) or 0),
            "skipped": bool(community.get("skipped", False)),
        },
    }
    phase_breakdown["construct_total"] = {
        "llm_calls": llm_calls,
        "embedding_calls": embedding_calls,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "latency_ms_total": latency_ms_total,
        "wall_time_ms": wall_time_ms_total,
        "usage_source": "local_graph_pipeline_telemetry",
        "skipped": bool(
            phase_breakdown["triple_extract"]["skipped"]
            and phase_breakdown["graph_build"]["skipped"]
            and phase_breakdown["community_build"]["skipped"]
        ),
    }
    return {
        "llm_calls": llm_calls,
        "embedding_calls": embedding_calls,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "latency_ms_total": latency_ms_total,
        "wall_time_ms_total": wall_time_ms_total,
        "usage_source": "local_graph_pipeline_telemetry",
        "num_nodes": int(graph_build.get("num_nodes", 0) or 0),
        "num_edges": int(graph_build.get("num_edges", 0) or 0),
        "phase_breakdown": phase_breakdown,
    }


def _prepare_chunks_file_for_merge(chunks_file: str, merge_chunks: int, out_dir: Path) -> str:
    size = int(max(1, int(merge_chunks)))
    if size <= 1:
        return chunks_file
    rows: list[dict[str, Any]] = []
    with open(chunks_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
    merged: list[dict[str, Any]] = []
    i = 0
    while i < len(rows):
        head = rows[i]
        doc_id = str(head.get("doc_id", "") or "").strip()
        group = [head]
        j = i + 1
        while j < len(rows) and len(group) < size:
            nxt = rows[j]
            if str(nxt.get("doc_id", "") or "").strip() != doc_id:
                break
            group.append(nxt)
            j += 1
        parts = [str(x.get("text", "") or "").strip() for x in group if str(x.get("text", "") or "").strip()]
        if parts:
            merged_row = dict(head)
            merged_row["text"] = "\n".join(parts)
            merged_row["merged_chunk_ids"] = [str(x.get("chunk_id", "") or "").strip() for x in group]
            merged_row["chunk_id"] = str(head.get("chunk_id", "") or "").strip() or f"merged_{i}"
            merged.append(merged_row)
        i = j

    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{Path(chunks_file).stem}_merge{size}.jsonl"
    with out_file.open("w", encoding="utf-8") as f:
        for row in merged:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return str(out_file)


def _build_stage_metrics(
    *,
    chunks_file: str,
    triples_file: str,
    graph_file: str,
    communities_file: str,
    youtu_base_url: str,
    youtu_dataset: str,
    youtu_graph_state_file: str,
    youtu_sync_mode: str,
    youtu_shared_corpus_dir: str,
    youtu_corpus_source_file: str | None,
    youtu_construct_poll_sec: int,
    youtu_construct_timeout_sec: int,
    youtu_require_fingerprint_match: bool,
    youtu_schema_file: str | None,
    youtu_reuse_graph: bool,
    build_force_rebuild: bool,
    lightrag_working_dir: str | None,
    merge_chunks: int,
    out_file: str,
    selected_rags: set[str],
    build_parallel: bool,
) -> dict[str, Any]:
    include_lightrag = "lightrag" in selected_rags
    include_graph_rag = "graph_rag" in selected_rags
    include_youtu = ("youtu" in selected_rags) and bool(_YOUTU_AVAILABLE)
    out_path = Path(out_file)

    def _compose_summary(
        lightrag_stats_value: dict[str, Any],
        graph_rag_metrics_value: dict[str, Any],
        youtu_metrics_value: dict[str, Any],
    ) -> dict[str, Any]:
        youtu_build = (((youtu_metrics_value.get("youtu_construct") or {}).get("build_stats") or {}) if isinstance(youtu_metrics_value, dict) else {})
        youtu_prompt, youtu_completion, youtu_total = _extract_usage_triplet(youtu_build)
        youtu_elapsed_sec = (youtu_metrics_value.get("youtu_construct") or {}).get("elapsed_sec") if isinstance(youtu_metrics_value, dict) else None
        youtu_latency_ms = _to_int_or_none(youtu_build.get("latency_ms_total"))
        if youtu_latency_ms is None and youtu_elapsed_sec is not None:
            youtu_latency_ms = int(float(youtu_elapsed_sec) * 1000)
        youtu_phase_breakdown = (youtu_build.get("phase_breakdown") if isinstance(youtu_build, dict) else {}) or {}
        youtu_phase_breakdown = dict(youtu_phase_breakdown)
        youtu_phase_breakdown["construct_total"] = {
            "llm_calls": int((youtu_build.get("llm_calls") or 0)) if isinstance(youtu_build, dict) else 0,
            "prompt_tokens": youtu_prompt,
            "completion_tokens": youtu_completion,
            "total_tokens": youtu_total,
            "latency_ms_total": youtu_latency_ms,
            "wall_time_ms": int(float(youtu_elapsed_sec or 0.0) * 1000),
            "usage_source": str(youtu_build.get("usage_source", "unavailable")) if isinstance(youtu_build, dict) else "unavailable",
            "skipped": bool((youtu_metrics_value.get("youtu_construct") or {}).get("skipped", False)) if isinstance(youtu_metrics_value, dict) else False,
        }
        youtu_final_status = (youtu_metrics_value.get("youtu_construct") or {}).get("final_status") if isinstance(youtu_metrics_value, dict) else None
        youtu_graph_stats = {}
        if isinstance(youtu_final_status, dict):
            youtu_graph_data = (youtu_final_status.get("data") or {}).get("graph_data") if isinstance(youtu_final_status.get("data"), dict) else {}
            if isinstance(youtu_graph_data, dict):
                youtu_graph_stats = youtu_graph_data.get("stats", {}) if isinstance(youtu_graph_data.get("stats"), dict) else {}
        youtu_num_nodes = _to_int_or_none(youtu_graph_stats.get("total_nodes"))
        youtu_num_edges = _to_int_or_none(youtu_graph_stats.get("total_edges"))
        graph_build_stats = _extract_graph_build_stats(graph_rag_metrics_value if isinstance(graph_rag_metrics_value, dict) else {})

        summary_local = {
            "stage": "build",
            "chunks_file": chunks_file,
            "methods": (
                (["graph_rag"] if include_graph_rag else [])
                + (["lightrag"] if include_lightrag else [])
                + (["youtu_graph_rag"] if include_youtu else [])
            ),
            "selected_rags": sorted(selected_rags),
            "build_force_rebuild": bool(build_force_rebuild),
            "build_parallel": bool(build_parallel and len([x for x in (include_lightrag, include_graph_rag, include_youtu) if x]) > 1),
            "build_benchmark": {
                "graph_rag": {
                    "method": "graph_rag",
                    "rebuilt": bool(
                        isinstance(graph_rag_metrics_value, dict)
                        and any(k in graph_rag_metrics_value for k in ("triple_extract", "graph_build", "community_build"))
                    ),
                    "reused_cache": bool(
                        include_graph_rag
                        and isinstance(graph_rag_metrics_value, dict)
                        and not any(k in graph_rag_metrics_value for k in ("triple_extract", "graph_build", "community_build"))
                    ),
                    "llm_calls": int(graph_build_stats.get("llm_calls", 0) or 0),
                    "embedding_calls": int(graph_build_stats.get("embedding_calls", 0) or 0),
                    "prompt_tokens": int(graph_build_stats.get("prompt_tokens", 0) or 0),
                    "completion_tokens": int(graph_build_stats.get("completion_tokens", 0) or 0),
                    "total_tokens": int(graph_build_stats.get("total_tokens", 0) or 0),
                    "latency_ms_total": int(graph_build_stats.get("latency_ms_total", 0) or 0),
                    "wall_time_ms_total": int(graph_build_stats.get("wall_time_ms_total", 0) or 0),
                    "usage_source": str(graph_build_stats.get("usage_source", "local_graph_pipeline_telemetry")),
                    "num_nodes": int(graph_build_stats.get("num_nodes", 0) or 0),
                    "num_edges": int(graph_build_stats.get("num_edges", 0) or 0),
                    "phase_breakdown": graph_build_stats.get("phase_breakdown", {}),
                },
                "lightrag": {
                    "method": "lightrag",
                    "rebuilt": bool(lightrag_stats_value.get("rebuilt", False)),
                    "reused_cache": bool(lightrag_stats_value.get("reused_cache", False)),
                    "llm_calls": int(lightrag_stats_value.get("llm_calls", 0) or 0),
                    "embedding_calls": int(lightrag_stats_value.get("embedding_calls", 0) or 0),
                    "prompt_tokens": int(lightrag_stats_value.get("prompt_tokens", 0) or 0),
                    "completion_tokens": int(lightrag_stats_value.get("completion_tokens", 0) or 0),
                    "total_tokens": int(lightrag_stats_value.get("total_tokens", 0) or 0),
                    "latency_ms_total": int(lightrag_stats_value.get("latency_ms_total", 0) or 0),
                    "usage_source": str(lightrag_stats_value.get("usage_source", "lightrag_token_tracker")),
                    "num_nodes": int(lightrag_stats_value.get("num_nodes", 0) or 0),
                    "num_edges": int(lightrag_stats_value.get("num_edges", 0) or 0),
                    "phase_breakdown": (
                        lightrag_stats_value.get("phase_breakdown")
                        if isinstance(lightrag_stats_value.get("phase_breakdown"), dict)
                        else {
                            "index": {
                                "llm_calls": int(lightrag_stats_value.get("llm_calls", 0) or 0),
                                "embedding_calls": int(lightrag_stats_value.get("embedding_calls", 0) or 0),
                                "prompt_tokens": int(lightrag_stats_value.get("prompt_tokens", 0) or 0),
                                "completion_tokens": int(lightrag_stats_value.get("completion_tokens", 0) or 0),
                                "total_tokens": int(lightrag_stats_value.get("total_tokens", 0) or 0),
                                "latency_ms_total": int(lightrag_stats_value.get("latency_ms_total", 0) or 0),
                                "skipped": bool(lightrag_stats_value.get("skipped", False)),
                            }
                        }
                    ),
                },
                "youtu_graph_rag": {
                    "method": "youtu_graph_rag",
                    "rebuilt": not bool((youtu_metrics_value.get("youtu_construct") or {}).get("skipped", False)) if isinstance(youtu_metrics_value, dict) else False,
                    "reused_cache": bool((youtu_metrics_value.get("graph_reuse") or {}).get("used_cached_graph", False)) if isinstance(youtu_metrics_value, dict) else False,
                    "llm_calls": int((youtu_build.get("llm_calls") or 0)) if isinstance(youtu_build, dict) else 0,
                    "embedding_calls": 0,
                    "prompt_tokens": youtu_prompt,
                    "completion_tokens": youtu_completion,
                    "total_tokens": youtu_total,
                    "latency_ms_total": youtu_latency_ms,
                    "usage_source": str(youtu_build.get("usage_source", "unavailable")) if isinstance(youtu_build, dict) else "unavailable",
                    "num_nodes": youtu_num_nodes if youtu_num_nodes is not None else 0,
                    "num_edges": youtu_num_edges if youtu_num_edges is not None else 0,
                    "phase_breakdown": youtu_phase_breakdown,
                },
            },
            "raw": {
                "graph_rag": graph_rag_metrics_value,
                "lightrag": lightrag_stats_value,
                "youtu": youtu_metrics_value,
            },
        }

        if out_path.exists():
            try:
                existing = json.loads(out_path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                existing = {}
            if isinstance(existing, dict):
                existing_bb = existing.get("build_benchmark") if isinstance(existing.get("build_benchmark"), dict) else {}
                existing_raw = existing.get("raw") if isinstance(existing.get("raw"), dict) else {}
                if "lightrag" not in selected_rags:
                    if "lightrag" in existing_bb and "lightrag" in summary_local["build_benchmark"]:
                        summary_local["build_benchmark"]["lightrag"] = existing_bb["lightrag"]
                    if "lightrag" in existing_raw and "lightrag" in summary_local["raw"]:
                        summary_local["raw"]["lightrag"] = existing_raw["lightrag"]
                if "graph_rag" not in selected_rags:
                    if "graph_rag" in existing_bb and "graph_rag" in summary_local["build_benchmark"]:
                        summary_local["build_benchmark"]["graph_rag"] = existing_bb["graph_rag"]
                    if "graph_rag" in existing_raw and "graph_rag" in summary_local["raw"]:
                        summary_local["raw"]["graph_rag"] = existing_raw["graph_rag"]
                if "youtu" not in selected_rags:
                    if "youtu_graph_rag" in existing_bb and "youtu_graph_rag" in summary_local["build_benchmark"]:
                        summary_local["build_benchmark"]["youtu_graph_rag"] = existing_bb["youtu_graph_rag"]
                    if "youtu" in existing_raw and "youtu" in summary_local["raw"]:
                        summary_local["raw"]["youtu"] = existing_raw["youtu"]
                merged_methods = []
                for m in list(existing.get("methods", []) or []) + list(summary_local.get("methods", []) or []):
                    if isinstance(m, str) and m and m not in merged_methods:
                        merged_methods.append(m)
                summary_local["methods"] = merged_methods
                summary_local["incremental_update"] = True
        return summary_local

    def _persist_build_progress(
        lightrag_stats_value: dict[str, Any],
        graph_rag_metrics_value: dict[str, Any],
        youtu_metrics_value: dict[str, Any],
    ) -> dict[str, Any]:
        summary_local = _compose_summary(
            lightrag_stats_value=lightrag_stats_value,
            graph_rag_metrics_value=graph_rag_metrics_value,
            youtu_metrics_value=youtu_metrics_value,
        )
        _write_json(out_file, summary_local)
        return summary_local

    def _run_lightrag() -> dict[str, Any]:
        _progress("stage=build: collecting LightRAG build metrics")
        return ensure_lightrag_assets(
            chunks_file=chunks_file,
            working_dir=lightrag_working_dir,
            force_rebuild=bool(build_force_rebuild),
            merge_chunks=int(merge_chunks),
        )

    def _run_youtu() -> dict[str, Any]:
        _progress("stage=build: collecting youtu-GraphRAG build metrics")
        youtu_schema, youtu_schema_meta = load_and_adapt_schema(youtu_schema_file)
        return _ensure_youtu_graph_assets(
            chunks_file=chunks_file,
            youtu_corpus_source_file=youtu_corpus_source_file,
            triples_file=triples_file,
            graph_file=graph_file,
            communities_file=communities_file,
            youtu_base_url=youtu_base_url,
            youtu_dataset=youtu_dataset,
            youtu_schema=youtu_schema,
            youtu_schema_meta=youtu_schema_meta,
            graph_state_file=youtu_graph_state_file,
            reuse_graph=youtu_reuse_graph,
            force_rebuild=bool(build_force_rebuild),
            sync_mode=youtu_sync_mode,
            shared_corpus_dir=youtu_shared_corpus_dir,
            construct_poll_sec=youtu_construct_poll_sec,
            construct_timeout_sec=youtu_construct_timeout_sec,
            require_fingerprint_match=bool(youtu_require_fingerprint_match),
        )

    def _run_graph_rag_build() -> dict[str, Any]:
        if not str(triples_file or "").strip():
            raise ValueError("graph_rag build requires --triples-file")
        if not str(graph_file or "").strip():
            raise ValueError("graph_rag build requires --graph-file")
        if not str(communities_file or "").strip():
            raise ValueError("graph_rag build requires --communities-file")
        if bool(build_force_rebuild):
            for p in (triples_file, graph_file, communities_file):
                pp = Path(p)
                if pp.exists():
                    pp.unlink()
        _progress("stage=build: collecting graph_rag build metrics")
        chunks_for_graph = _prepare_chunks_file_for_merge(
            chunks_file=chunks_file,
            merge_chunks=int(merge_chunks),
            out_dir=Path("outputs/cache/rags"),
        )
        return ensure_graph_assets(
            chunks_file=chunks_for_graph,
            triples_file=triples_file,
            graph_file=graph_file,
            communities_file=communities_file,
        )

    lightrag_stats: dict[str, Any] = {"skipped": True, "reason": "not_selected"}
    graph_rag_metrics: dict[str, Any] = {"skipped": True, "reason": "not_selected"}
    youtu_metrics: dict[str, Any] = {"skipped": True, "reason": "not_selected"}
    if bool(build_parallel) and len([x for x in (include_lightrag, include_graph_rag, include_youtu) if x]) > 1:
        _progress("stage=build: running selected builds in parallel")
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
            fut_map: dict[concurrent.futures.Future, str] = {}
            if include_lightrag:
                fut_map[ex.submit(_run_lightrag)] = "lightrag"
            if include_graph_rag:
                fut_map[ex.submit(_run_graph_rag_build)] = "graph_rag"
            if include_youtu:
                fut_map[ex.submit(_run_youtu)] = "youtu"
            for fut, name in fut_map.items():
                result = fut.result()
                if name == "lightrag":
                    lightrag_stats = result
                elif name == "graph_rag":
                    graph_rag_metrics = result
                elif name == "youtu":
                    youtu_metrics = result
                _persist_build_progress(
                    lightrag_stats_value=lightrag_stats,
                    graph_rag_metrics_value=graph_rag_metrics,
                    youtu_metrics_value=youtu_metrics,
                )
    else:
        if include_lightrag:
            lightrag_stats = _run_lightrag()
            _persist_build_progress(
                lightrag_stats_value=lightrag_stats,
                graph_rag_metrics_value=graph_rag_metrics,
                youtu_metrics_value=youtu_metrics,
            )
        if include_graph_rag:
            graph_rag_metrics = _run_graph_rag_build()
            _persist_build_progress(
                lightrag_stats_value=lightrag_stats,
                graph_rag_metrics_value=graph_rag_metrics,
                youtu_metrics_value=youtu_metrics,
            )
        if include_youtu:
            youtu_metrics = _run_youtu()
            _persist_build_progress(
                lightrag_stats_value=lightrag_stats,
                graph_rag_metrics_value=graph_rag_metrics,
                youtu_metrics_value=youtu_metrics,
            )
        elif "youtu" in selected_rags and not _YOUTU_AVAILABLE:
            youtu_metrics = {"skipped": True, "reason": "youtu_not_enabled"}
    return _persist_build_progress(
        lightrag_stats_value=lightrag_stats,
        graph_rag_metrics_value=graph_rag_metrics,
        youtu_metrics_value=youtu_metrics,
    )


def _run_pair_compare_once(
    *,
    queries_file: str,
    chunks_file: str,
    graph_file: str,
    communities_file: str,
    top_k: int,
    out_file: str,
    metrics_file: str,
    regimes: str,
    budget_config_file: str,
    max_queries: int | None,
    answer_mode: str,
    selected_rags: set[str],
    youtu_base_url: str,
    youtu_dataset: str,
    youtu_route_type: str | None,
    youtu_client_id: str | None,
    youtu_graph_state_file: str,
    youtu_reuse_graph: bool,
    youtu_force_rebuild: bool,
    youtu_sync_mode: str,
    youtu_shared_corpus_dir: str,
    youtu_corpus_source_file: str | None,
    youtu_construct_poll_sec: int,
    youtu_construct_timeout_sec: int,
    youtu_require_fingerprint_match: bool,
    youtu_schema_file: str | None,
    lightrag_mode: str,
    lightrag_working_dir: str | None,
    merge_chunks: int,
    retrieve_reuse_cache: bool,
    cuad_doc_scope: bool,
    strict_doc_scope: bool,
    graph_query_level: int | None,
    incremental_only: bool = False,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    include_lightrag = "lightrag" in selected_rags
    include_graph_rag = "graph_rag" in selected_rags
    include_youtu = ("youtu" in selected_rags) and bool(_YOUTU_AVAILABLE)
    selected_methods = _selected_method_names(selected_rags, include_youtu=bool(include_youtu))
    if include_graph_rag:
        if not str(graph_file or "").strip():
            raise ValueError("graph_rag retrieve requires --graph-file")
        if not str(communities_file or "").strip():
            raise ValueError("graph_rag retrieve requires --communities-file")
    if not include_lightrag and not include_graph_rag and not include_youtu:
        raise RuntimeError("No runnable RAG selected. Supported now: graph_rag, lightrag, youtu")
    queries = _select_queries_balanced_by_type(_load_queries(queries_file), None)

    dataset_tag = _infer_dataset_tag(queries_file=queries_file, queries=queries)
    query_snapshot = _query_consistency_snapshot(queries)
    use_cuad_doc_scope = (dataset_tag in {"cuad", "cuad_like"}) and bool(cuad_doc_scope)
    regime_names = _resolve_regimes(regimes)
    budget_yaml = _load_yaml(budget_config_file)
    budget = {
        "max_llm_calls": 4,
        "evidence_token_limit": 2500,
        "max_completion_tokens": 800,
        "max_total_tokens": 3800,
    }
    yaml_budget = (budget_yaml or {}).get("budget", {})
    if yaml_budget:
        budget["evidence_token_limit"] = int(yaml_budget.get("evidence_token_limit", budget["evidence_token_limit"]))
        budget["max_completion_tokens"] = int(yaml_budget.get("max_completion_tokens", budget["max_completion_tokens"]))
        budget["max_total_tokens"] = int(yaml_budget.get("max_total_tokens", budget["max_total_tokens"]))

    settings = _regime_settings(
        top_k=top_k,
        budget_cfg_yaml=budget_yaml,
        graph_query_level_override=graph_query_level,
    )

    youtu_assets_metrics: dict[str, Any] = {}
    if include_youtu and _YOUTU_AVAILABLE:
        youtu_schema, youtu_schema_meta = load_and_adapt_schema(youtu_schema_file)
        youtu_assets_metrics = _ensure_youtu_graph_assets(
            chunks_file=chunks_file,
            youtu_corpus_source_file=youtu_corpus_source_file,
            triples_file="",
            graph_file=graph_file,
            communities_file=communities_file,
            youtu_base_url=youtu_base_url,
            youtu_dataset=youtu_dataset,
            youtu_schema=youtu_schema,
            youtu_schema_meta=youtu_schema_meta,
            graph_state_file=youtu_graph_state_file,
            reuse_graph=youtu_reuse_graph,
            force_rebuild=bool(youtu_force_rebuild),
            sync_mode=youtu_sync_mode,
            shared_corpus_dir=youtu_shared_corpus_dir,
            construct_poll_sec=youtu_construct_poll_sec,
            construct_timeout_sec=youtu_construct_timeout_sec,
            require_fingerprint_match=bool(youtu_require_fingerprint_match),
        )

    out_path = Path(out_file)
    existing_rows: list[dict[str, Any]] = []
    if out_path.exists():
        try:
            existing_rows = [
                json.loads(line)
                for line in out_path.read_text(encoding="utf-8").splitlines()
                if str(line).strip()
            ]
        except Exception:  # noqa: BLE001
            existing_rows = []

    rows: list[dict[str, Any]] = []
    methods = (
        (["graph_rag"] if include_graph_rag else [])
        + (["lightrag"] if include_lightrag else [])
        + (["youtu_graph_rag"] if include_youtu else [])
    )
    num_existing_answers = len(existing_rows)
    if incremental_only:
        filtered_queries: list[dict[str, Any]] = []
        kept_rows: list[dict[str, Any]] = []
        skipped_completed = 0
        existing_by_qid: dict[str, dict[str, Any]] = {}
        for row in existing_rows:
            qid = str(row.get("qid", "")).strip()
            if qid and qid not in existing_by_qid:
                existing_by_qid[qid] = row
            if _row_is_complete_for_selected_methods(row, regime_names=regime_names, selected_methods=selected_methods):
                kept_rows.append(row)

        for q in queries:
            qid = str(q.get("qid", "")).strip()
            existing_row = existing_by_qid.get(qid)
            if existing_row is not None:
                same_query_shape = (
                    str(existing_row.get("query", "")).strip() == str(q.get("query", "")).strip()
                    and str(existing_row.get("type", "")).strip() == str(q.get("type", "")).strip()
                    and str(existing_row.get("mode", "")).strip() == str(answer_mode).strip()
                    and int(existing_row.get("top_k", 0) or 0) == int(top_k)
                )
                if same_query_shape and _row_is_complete_for_selected_methods(
                    existing_row,
                    regime_names=regime_names,
                    selected_methods=selected_methods,
                ):
                    skipped_completed += 1
                    continue
            filtered_queries.append(q)

        queries = filtered_queries
        rows = kept_rows
        _progress(
            "incremental retrieve scope "
            + json.dumps(
                {
                    "existing_rows": len(existing_rows),
                    "kept_rows": len(rows),
                    "pending_queries": len(queries),
                    "skipped_completed": skipped_completed,
                },
                ensure_ascii=False,
            )
        )

    if max_queries is not None and int(max_queries) > 0:
        queries = queries[: int(max_queries)]
        _progress(f"max_queries applied: processing={len(queries)}")

    for idx, q in enumerate(queries, start=1):
        qid = str(q.get("qid", "")).strip()
        qtext = str(q.get("query", "")).strip()
        qtype = str(q.get("type", "unknown")).strip() or "unknown"
        query_doc_key = _extract_query_doc_key(q)
        scoped_query = _scope_query_for_cuad(
            raw_query=qtext,
            query_doc_key=query_doc_key,
            dataset_tag=dataset_tag,
            enabled=bool(use_cuad_doc_scope),
        )
        doc_prefix_filter = query_doc_key if (use_cuad_doc_scope and strict_doc_scope and query_doc_key) else None

        row: dict[str, Any] = {
            "qid": qid,
            "query": qtext,
            "type": qtype,
            "mode": answer_mode,
            "top_k": int(top_k),
            "regimes": {},
        }

        for rg in regime_names:
            is_budget = rg == "budget_matched"
            graph_kwargs = settings[rg]["graph_kwargs"]

            regime_payload: dict[str, Any] = {}
            if include_graph_rag:
                graph_out = answer_with_graphrag(
                    query=scoped_query,
                    graph_file=graph_file,
                    communities_file=communities_file,
                    top_communities=graph_kwargs.get("top_communities", 3),
                    max_evidence=graph_kwargs.get("max_evidence", 12),
                    query_level=graph_kwargs.get("query_level", 0),
                    use_hierarchy=graph_kwargs.get("use_hierarchy", True),
                    use_community_summaries=graph_kwargs.get("use_community_summaries", True),
                    shuffle_communities=graph_kwargs.get("shuffle_communities", True),
                    use_map_reduce=graph_kwargs.get("use_map_reduce", True),
                    max_summary_chars=graph_kwargs.get("max_summary_chars", 1800),
                    map_keypoints_limit=graph_kwargs.get("map_keypoints_limit", 5),
                    max_completion_tokens=budget["max_completion_tokens"] if is_budget else None,
                    doc_prefix_filter=doc_prefix_filter,
                    strict_doc_scope=bool(strict_doc_scope),
                    query_type=qtype,
                    answer_mode=answer_mode,
                )
                graph_telemetry = graph_out.get("telemetry") if isinstance(graph_out, dict) else {}
                graph_out["budget_check"] = _check_budget(graph_telemetry, budget) if is_budget else {"within_budget": None}
                regime_payload["graph_rag"] = graph_out
            if include_lightrag:
                lightrag_out = answer_with_lightrag(
                    query=scoped_query,
                    chunks_file=chunks_file,
                    top_k=int(top_k),
                    max_completion_tokens=budget["max_completion_tokens"] if is_budget else None,
                    answer_mode=answer_mode,
                    query_type=qtype,
                    working_dir=lightrag_working_dir,
                    lightrag_mode=lightrag_mode,
                    force_rebuild=not bool(retrieve_reuse_cache),
                    merge_chunks=int(merge_chunks),
                )
                lightrag_telemetry = lightrag_out.get("telemetry") if isinstance(lightrag_out, dict) else {}
                lightrag_out["budget_check"] = _check_budget(lightrag_telemetry, budget) if is_budget else {"within_budget": None}
                regime_payload["lightrag"] = lightrag_out

            if include_youtu:
                youtu_out = answer_with_youtu_graphrag(
                    query=scoped_query,
                    graph_file=graph_file,
                    communities_file=communities_file,
                    top_communities=graph_kwargs.get("top_communities", 3),
                    max_evidence=graph_kwargs.get("max_evidence", 12),
                    query_level=graph_kwargs.get("query_level", 0),
                    use_hierarchy=graph_kwargs.get("use_hierarchy", True),
                    use_community_summaries=graph_kwargs.get("use_community_summaries", True),
                    shuffle_communities=graph_kwargs.get("shuffle_communities", True),
                    use_map_reduce=graph_kwargs.get("use_map_reduce", True),
                    max_summary_chars=graph_kwargs.get("max_summary_chars", 1800),
                    map_keypoints_limit=graph_kwargs.get("map_keypoints_limit", 5),
                    max_completion_tokens=budget["max_completion_tokens"] if is_budget else None,
                    youtu_base_url=youtu_base_url,
                    youtu_dataset=youtu_dataset,
                    store_file=None,
                    doc_prefix_filter=doc_prefix_filter,
                    strict_doc_scope=bool(strict_doc_scope),
                    answer_mode=answer_mode,
                    route_type=youtu_route_type,
                    client_id=youtu_client_id,
                )
                youtu_telemetry = youtu_out.get("telemetry") if isinstance(youtu_out, dict) else {}
                youtu_out["budget_check"] = _check_budget(youtu_telemetry, budget) if is_budget else {"within_budget": None}
                regime_payload["youtu_graph_rag"] = youtu_out

            row["regimes"][rg] = regime_payload
        rows.append(row)
        materialized_rows, _ = _materialize_retrieve_rows(
            rows=rows,
            existing_rows=existing_rows,
            selected_methods=selected_methods,
        )
        _write_jsonl(out_file, materialized_rows)
        partial_summary = _build_compare_summary(
            rows=materialized_rows,
            queries_file=queries_file,
            dataset_tag=dataset_tag,
            query_snapshot=_query_consistency_snapshot(queries),
            regime_names=regime_names,
            methods=methods,
            top_k=top_k,
            answer_mode=answer_mode,
            budget=budget,
            budget_config_file=budget_config_file,
            cuad_doc_scope=cuad_doc_scope,
            strict_doc_scope=strict_doc_scope,
            lightrag_mode=lightrag_mode,
            lightrag_working_dir=lightrag_working_dir,
            graph_file=graph_file,
            communities_file=communities_file,
            retrieve_reuse_cache=retrieve_reuse_cache,
            selected_rags=selected_rags,
            include_youtu=bool(include_youtu),
            youtu_base_url=youtu_base_url,
            youtu_dataset=youtu_dataset,
            youtu_route_type=youtu_route_type,
            youtu_client_id=youtu_client_id,
            incremental_update=bool(existing_rows),
            incremental_only=incremental_only,
            num_existing_answers=num_existing_answers,
            youtu_assets_metrics=youtu_assets_metrics,
            results_file=out_file,
        )
        _write_json(metrics_file, partial_summary)
        if progress_callback is not None:
            progress_callback(partial_summary)
        _progress(f"query {idx}/{len(queries)} done: qid={qid} mode={answer_mode}")

    rows, incremental_update = _materialize_retrieve_rows(
        rows=rows,
        existing_rows=existing_rows,
        selected_methods=selected_methods,
    )

    _write_jsonl(out_file, rows)
    summary = _build_compare_summary(
        rows=rows,
        queries_file=queries_file,
        dataset_tag=dataset_tag,
        query_snapshot=_query_consistency_snapshot(queries),
        regime_names=regime_names,
        methods=methods,
        top_k=top_k,
        answer_mode=answer_mode,
        budget=budget,
        budget_config_file=budget_config_file,
        cuad_doc_scope=cuad_doc_scope,
        strict_doc_scope=strict_doc_scope,
        lightrag_mode=lightrag_mode,
        lightrag_working_dir=lightrag_working_dir,
        graph_file=graph_file,
        communities_file=communities_file,
        retrieve_reuse_cache=retrieve_reuse_cache,
        selected_rags=selected_rags,
        include_youtu=bool(include_youtu),
        youtu_base_url=youtu_base_url,
        youtu_dataset=youtu_dataset,
        youtu_route_type=youtu_route_type,
        youtu_client_id=youtu_client_id,
        incremental_update=incremental_update,
        incremental_only=incremental_only,
        num_existing_answers=num_existing_answers,
        youtu_assets_metrics=youtu_assets_metrics,
        results_file=out_file,
    )
    _write_json(metrics_file, summary)
    return summary


def run_compare_youtu_lightrag(
    *,
    stage: str,
    queries_file: str,
    gold_file: str,
    chunks_file: str,
    graph_file: str,
    communities_file: str,
    triples_file: str,
    topk_list: list[int],
    answer_modes: list[str],
    regime: str,
    budget_config_file: str,
    max_queries: int | None,
    judge_mode: str,
    judge_model: str,
    results_dir: str,
    retrieve_results_dir: str | None,
    include_rags: list[str],
    youtu_base_url: str,
    youtu_dataset: str,
    youtu_route_type: str | None,
    youtu_client_id: str | None,
    youtu_schema_file: str | None,
    youtu_graph_state_file: str,
    youtu_reuse_graph: bool,
    youtu_force_rebuild: bool,
    youtu_sync_mode: str,
    youtu_shared_corpus_dir: str,
    youtu_corpus_source_file: str | None,
    youtu_construct_poll_sec: int,
    youtu_construct_timeout_sec: int,
    youtu_require_fingerprint_match: bool,
    lightrag_mode: str,
    lightrag_working_dir: str | None,
    merge_chunks: int,
    build_force_rebuild: bool,
    retrieve_reuse_cache: bool,
    incremental_only: bool,
    cuad_doc_scope: bool,
    strict_doc_scope: bool,
    graph_query_level: int | None,
    build_parallel: bool,
    build_metrics_file: str | None,
    retrieve_summary_csv: str | None,
    retrieve_summary_md: str | None,
    retrieve_summary_json: str | None,
    pipeline_summary_file: str | None,
) -> dict[str, Any]:
    stage_norm = str(stage or "all").strip().lower()
    if stage_norm not in {"build", "retrieve", "all"}:
        raise ValueError("stage must be one of: build, retrieve, all")

    build_out_dir = Path(results_dir)
    build_out_dir.mkdir(parents=True, exist_ok=True)
    retrieve_out_dir = Path(retrieve_results_dir) if str(retrieve_results_dir or "").strip() else build_out_dir
    retrieve_out_dir.mkdir(parents=True, exist_ok=True)
    selected_rags = {x for x in include_rags if x in {"graph_rag", "lightrag", "youtu"}}
    if not selected_rags:
        raise ValueError("include_rags is empty; supported values now: graph_rag, lightrag, youtu")
    include_youtu_effective = bool(("youtu" in selected_rags) and _YOUTU_AVAILABLE)

    output: dict[str, Any] = {
        "pipeline": "youtu_vs_lightrag",
        "stage": stage_norm,
        "results_dir": str(build_out_dir),
        "retrieve_results_dir": str(retrieve_out_dir),
    }

    if stage_norm in {"build", "all"}:
        build_file = Path(build_metrics_file) if str(build_metrics_file or "").strip() else (build_out_dir / "youtu_lightrag_build_metrics.json")
        output["build"] = _build_stage_metrics(
            chunks_file=chunks_file,
            triples_file=triples_file,
            graph_file=graph_file,
            communities_file=communities_file,
            youtu_base_url=youtu_base_url,
            youtu_dataset=youtu_dataset,
            youtu_graph_state_file=youtu_graph_state_file,
            youtu_sync_mode=youtu_sync_mode,
            youtu_shared_corpus_dir=youtu_shared_corpus_dir,
            youtu_corpus_source_file=youtu_corpus_source_file,
            youtu_construct_poll_sec=youtu_construct_poll_sec,
            youtu_construct_timeout_sec=youtu_construct_timeout_sec,
            youtu_require_fingerprint_match=youtu_require_fingerprint_match,
            youtu_schema_file=youtu_schema_file,
            youtu_reuse_graph=youtu_reuse_graph,
            build_force_rebuild=build_force_rebuild,
            lightrag_working_dir=lightrag_working_dir,
            merge_chunks=int(merge_chunks),
            out_file=str(build_file),
            selected_rags=selected_rags,
            build_parallel=bool(build_parallel),
        )
        output["build_metrics_file"] = str(build_file)

    if stage_norm in {"retrieve", "all"}:
        wide: dict[tuple[str, str, str], dict[str, Any]] = {}
        runs: list[dict[str, Any]] = []
        run_registry: dict[tuple[int, str], dict[str, Any]] = {}

        final_file = (
            Path(pipeline_summary_file)
            if str(pipeline_summary_file or "").strip()
            else (build_out_dir / "youtu_lightrag_pipeline_summary.json")
        )
        final_file.parent.mkdir(parents=True, exist_ok=True)

        def _update_wide_for_run(compare_summary: dict[str, Any], eval_payload: dict[str, Any], mode: str, top_k: int) -> None:
            dataset = str(compare_summary.get("dataset_tag", "unknown"))
            selected_regimes = set(_resolve_regimes(regime))
            selected_methods_eval: set[str] = set()
            if "graph_rag" in selected_rags:
                selected_methods_eval.add("graph_rag")
            if "lightrag" in selected_rags:
                selected_methods_eval.add("lightrag")
            if include_youtu_effective:
                selected_methods_eval.add("youtu_graph_rag")
            for item in eval_payload.get("summary", {}).get("overall", []):
                item_regime = str(item.get("regime", ""))
                if item_regime not in selected_regimes:
                    continue
                method = str(item.get("method", "")).strip()
                if method not in selected_methods_eval:
                    continue
                key = (method, dataset, item_regime, mode)
                row = wide.setdefault(
                    key,
                    {
                        "method": method,
                        "dataset": dataset,
                        "regime": item_regime,
                        "mode": mode,
                        "top10_accuracy": None,
                        "top20_accuracy": None,
                        "tokens": None,
                        "time": None,
                        "tokens_top10": None,
                        "tokens_top20": None,
                        "time_top10": None,
                        "time_top20": None,
                    },
                )
                row[f"top{int(top_k)}_accuracy"] = _pick_answer_accuracy(item)
                row[f"tokens_top{int(top_k)}"] = item.get("total_tokens")
                row[f"time_top{int(top_k)}"] = item.get("latency_ms_total")
                if int(top_k) == 20:
                    row["tokens"] = item.get("total_tokens")
                    row["time"] = item.get("latency_ms_total")
                elif row.get("tokens") is None:
                    row["tokens"] = item.get("total_tokens")
                    row["time"] = item.get("latency_ms_total")

        def _refresh_global_retrieve_outputs() -> None:
            retrieve_summary = _write_retrieve_summary_outputs(
                retrieve_out_dir=retrieve_out_dir,
                retrieve_summary_csv=retrieve_summary_csv,
                retrieve_summary_md=retrieve_summary_md,
                retrieve_summary_json=retrieve_summary_json,
                queries_file=queries_file,
                gold_file=gold_file,
                regime=regime,
                topk_list=topk_list,
                answer_modes=answer_modes,
                judge_mode=judge_mode,
                judge_model=judge_model,
                selected_rags=selected_rags,
                include_youtu_effective=include_youtu_effective,
                lightrag_mode=lightrag_mode,
                merge_chunks=int(merge_chunks),
                retrieve_reuse_cache=retrieve_reuse_cache,
                build_force_rebuild=build_force_rebuild,
                runs=runs,
                wide=wide,
            )
            output["retrieve"] = retrieve_summary
            final_file.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

        for top_k in topk_list:
            for mode in answer_modes:
                suffix = f"top{int(top_k)}_{mode}"
                compare_out = retrieve_out_dir / f"youtu_lightrag_compare_answers_{suffix}.jsonl"
                compare_metrics = retrieve_out_dir / f"youtu_lightrag_compare_metrics_{suffix}.json"
                eval_out = retrieve_out_dir / f"youtu_lightrag_eval_{suffix}.csv"
                eval_summary = retrieve_out_dir / f"youtu_lightrag_eval_{suffix}_summary.json"
                paper_metrics_base = retrieve_out_dir / f"youtu_lightrag_paper_metrics_{suffix}"
                run_key = (int(top_k), str(mode))
                run_registry[run_key] = {
                    "mode": mode,
                    "top_k": int(top_k),
                    "compare_answers": str(compare_out),
                    "compare_metrics": str(compare_metrics),
                    "eval_csv": str(eval_out),
                    "eval_summary": str(eval_summary),
                    "paper_metrics_json": str(paper_metrics_base.with_suffix(".json")),
                    "paper_metrics_csv": str(paper_metrics_base.with_suffix(".csv")),
                }
                runs = list(run_registry.values())

                def _progress_refresh(compare_summary: dict[str, Any], *, _mode: str = mode, _top_k: int = int(top_k)) -> None:
                    eval_payload = run_eval(
                        pred_file=str(compare_out),
                        gold_file=gold_file,
                        out_csv=str(eval_out),
                        out_summary=str(eval_summary),
                        graph_file=None,
                        communities_file=None,
                        expand_community_chunks=False,
                        make_plots=False,
                        method_mode="all",
                        judge_mode=judge_mode,
                        judge_model=judge_model,
                    )
                    _update_wide_for_run(compare_summary, eval_payload, _mode, _top_k)
                    _write_paper_metrics_outputs(
                        paper_metrics_base,
                        _build_paper_metrics_from_eval(eval_payload, top_k=_top_k),
                    )
                    _refresh_global_retrieve_outputs()

                compare_summary = _run_pair_compare_once(
                    queries_file=queries_file,
                    chunks_file=chunks_file,
                    graph_file=graph_file,
                    communities_file=communities_file,
                    top_k=int(top_k),
                    out_file=str(compare_out),
                    metrics_file=str(compare_metrics),
                    regimes=regime,
                    budget_config_file=budget_config_file,
                    max_queries=max_queries,
                    answer_mode=mode,
                    selected_rags=selected_rags,
                    youtu_base_url=youtu_base_url,
                    youtu_dataset=youtu_dataset,
                    youtu_route_type=youtu_route_type,
                    youtu_client_id=youtu_client_id,
                    youtu_graph_state_file=youtu_graph_state_file,
                    youtu_reuse_graph=youtu_reuse_graph,
                    youtu_force_rebuild=youtu_force_rebuild,
                    youtu_sync_mode=youtu_sync_mode,
                    youtu_shared_corpus_dir=youtu_shared_corpus_dir,
                    youtu_corpus_source_file=youtu_corpus_source_file,
                    youtu_construct_poll_sec=youtu_construct_poll_sec,
                    youtu_construct_timeout_sec=youtu_construct_timeout_sec,
                    youtu_require_fingerprint_match=youtu_require_fingerprint_match,
                    youtu_schema_file=youtu_schema_file,
                    lightrag_mode=lightrag_mode,
                    lightrag_working_dir=lightrag_working_dir,
                    merge_chunks=int(merge_chunks),
                    retrieve_reuse_cache=retrieve_reuse_cache,
                    incremental_only=incremental_only,
                    cuad_doc_scope=cuad_doc_scope,
                    strict_doc_scope=strict_doc_scope,
                    graph_query_level=graph_query_level,
                    progress_callback=_progress_refresh,
                )

                eval_payload = run_eval(
                    pred_file=str(compare_out),
                    gold_file=gold_file,
                    out_csv=str(eval_out),
                    out_summary=str(eval_summary),
                    graph_file=None,
                    communities_file=None,
                    expand_community_chunks=False,
                    make_plots=False,
                    method_mode="all",
                    judge_mode=judge_mode,
                    judge_model=judge_model,
                )
                _update_wide_for_run(compare_summary, eval_payload, mode, int(top_k))
                _write_paper_metrics_outputs(
                    paper_metrics_base,
                    _build_paper_metrics_from_eval(eval_payload, top_k=int(top_k)),
                )
                runs = list(run_registry.values())
                _refresh_global_retrieve_outputs()

    final_file = (
        Path(pipeline_summary_file)
        if str(pipeline_summary_file or "").strip()
        else (build_out_dir / "youtu_lightrag_pipeline_summary.json")
    )
    final_file.parent.mkdir(parents=True, exist_ok=True)
    final_file.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    output["pipeline_summary_file"] = str(final_file)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Run independent youtu-GraphRAG vs LightRAG compare pipeline")
    parser.add_argument("--stage", choices=["build", "retrieve", "all"], default="all")

    parser.add_argument("--queries-file", default="data/queries/queries.jsonl")
    parser.add_argument("--gold-file", default="data/queries/gold_qa.jsonl")
    parser.add_argument("--chunks-file", default="data/processed/chunks_sampled.jsonl")
    parser.add_argument("--triples-file", default="")
    parser.add_argument("--graph-file", default="")
    parser.add_argument("--communities-file", default="")

    parser.add_argument("--topk-list", default="10,20")
    parser.add_argument("--answer-modes", default="reject,open")
    parser.add_argument("--regime", choices=["best_effort", "budget_matched", "both"], default="best_effort")
    parser.add_argument("--budget-config-file", default="config_budget.yaml")
    parser.add_argument("--max-queries", type=int, default=None)

    parser.add_argument("--judge-mode", choices=["off", "llm_yesno"], default="off")
    parser.add_argument("--judge-model", default="qwen-flash")

    parser.add_argument("--results-dir", default="outputs/results/rags")
    parser.add_argument("--retrieve-results-dir", default="outputs/results/test")
    parser.add_argument("--build-metrics-file", default="")
    parser.add_argument("--retrieve-summary-csv", default="")
    parser.add_argument("--retrieve-summary-md", default="")
    parser.add_argument("--retrieve-summary-json", default="")
    parser.add_argument("--pipeline-summary-file", default="")

    parser.add_argument("--build-force-rebuild", default="true")
    parser.add_argument("--build-parallel", default="false")
    parser.add_argument("--retrieve-reuse-cache", default="true")
    parser.add_argument("--incremental-only", default="false")

    parser.add_argument("--include-rag", default="lightrag,youtu")
    parser.add_argument("--youtu-base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--youtu-dataset", default="enterprise")
    parser.add_argument("--youtu-route-type", default="")
    parser.add_argument("--youtu-client-id", default="")
    parser.add_argument("--youtu-schema-file", default="")
    parser.add_argument("--youtu-graph-state-file", default="outputs/graph/youtu_graph_state.json")
    parser.add_argument("--youtu-reuse-graph", default="true")
    parser.add_argument("--youtu-force-rebuild", default="false")
    parser.add_argument("--youtu-sync-mode", default="none")
    parser.add_argument("--youtu-shared-corpus-dir", default="outputs/youtu_sync")
    parser.add_argument("--youtu-corpus-source-file", default="")
    parser.add_argument("--youtu-construct-poll-sec", type=int, default=2)
    parser.add_argument("--youtu-construct-timeout-sec", type=int, default=1800)
    parser.add_argument("--youtu-require-fingerprint-match", default="true")

    parser.add_argument("--lightrag-mode", default="hybrid")
    parser.add_argument("--lightrag-working-dir", default="")
    parser.add_argument("--merge-chunks", type=int, default=1)

    parser.add_argument("--cuad-doc-scope", default="false")
    parser.add_argument("--strict-doc-scope", default="true")
    parser.add_argument("--graph-query-level", type=int, default=None)

    args = parser.parse_args()

    topk_list = _parse_int_list(args.topk_list, [10, 20])
    answer_modes = _parse_mode_list(args.answer_modes, ["reject", "open"])
    answer_modes = _resolve_answer_modes(answer_mode=answer_modes[0] if answer_modes else "reject", answer_modes=",".join(answer_modes))
    youtu_route_type = _normalize_youtu_route_type(args.youtu_route_type)

    include_rags = _parse_rag_list(args.include_rag, ["lightrag", "youtu"])

    summary = run_compare_youtu_lightrag(
        stage=args.stage,
        queries_file=args.queries_file,
        gold_file=args.gold_file,
        chunks_file=args.chunks_file,
        graph_file=args.graph_file,
        communities_file=args.communities_file,
        triples_file=args.triples_file,
        topk_list=topk_list,
        answer_modes=answer_modes,
        regime=args.regime,
        budget_config_file=args.budget_config_file,
        max_queries=args.max_queries,
        judge_mode=args.judge_mode,
        judge_model=args.judge_model,
        results_dir=args.results_dir,
        retrieve_results_dir=(str(args.retrieve_results_dir).strip() or None),
        include_rags=include_rags,
        youtu_base_url=args.youtu_base_url,
        youtu_dataset=args.youtu_dataset,
        youtu_route_type=youtu_route_type,
        youtu_client_id=(str(args.youtu_client_id).strip() or None),
        youtu_schema_file=(str(args.youtu_schema_file).strip() or None),
        youtu_graph_state_file=args.youtu_graph_state_file,
        youtu_reuse_graph=_parse_bool(args.youtu_reuse_graph, True),
        youtu_force_rebuild=_parse_bool(args.youtu_force_rebuild, False),
        youtu_sync_mode=args.youtu_sync_mode,
        youtu_shared_corpus_dir=args.youtu_shared_corpus_dir,
        youtu_corpus_source_file=(str(args.youtu_corpus_source_file).strip() or None),
        youtu_construct_poll_sec=args.youtu_construct_poll_sec,
        youtu_construct_timeout_sec=args.youtu_construct_timeout_sec,
        youtu_require_fingerprint_match=_parse_bool(args.youtu_require_fingerprint_match, True),
        lightrag_mode=args.lightrag_mode,
        lightrag_working_dir=(str(args.lightrag_working_dir).strip() or None),
        merge_chunks=int(args.merge_chunks),
        build_force_rebuild=_parse_bool(args.build_force_rebuild, True),
        retrieve_reuse_cache=_parse_bool(args.retrieve_reuse_cache, True),
        incremental_only=_parse_bool(args.incremental_only, False),
        cuad_doc_scope=_parse_bool(args.cuad_doc_scope, False),
        strict_doc_scope=_parse_bool(args.strict_doc_scope, True),
        graph_query_level=args.graph_query_level,
        build_parallel=_parse_bool(args.build_parallel, False),
        build_metrics_file=(str(args.build_metrics_file).strip() or None),
        retrieve_summary_csv=(str(args.retrieve_summary_csv).strip() or None),
        retrieve_summary_md=(str(args.retrieve_summary_md).strip() or None),
        retrieve_summary_json=(str(args.retrieve_summary_json).strip() or None),
        pipeline_summary_file=(str(args.pipeline_summary_file).strip() or None),
    )
    _progress("run_compare_youtu_lightrag completed")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
