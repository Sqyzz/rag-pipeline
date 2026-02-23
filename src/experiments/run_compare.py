from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import sys
import time
from pathlib import Path

import yaml

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from baselines.graph_rag import answer_with_graphrag
from baselines.kg_rag import answer_with_kg
from baselines.vector_rag import answer_with_context, retrieve_with_evidence
from evaluation.graph_structure_metrics import compute_graph_structure_metrics
from graph_build.build_communities import build_communities
from graph_build.build_graph import build_graph
from graph_build.extract_triples import extract_triples
from utils.budget import BudgetManager
from utils.config import cfg
from utils.telemetry import Telemetry
from utils.tokenizer import TokenizerProvider


def _progress(message: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[run_compare {ts}] {message}", flush=True)


def _g(obj, key: str, default):
    if obj is None:
        return default
    return getattr(obj, key, default)


def _load_queries(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _load_jsonl(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    rows = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _maybe_read_json(path: str) -> dict | None:
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _load_yaml(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _write_json(path: str, payload: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: str, rows: list[dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    idx = q * (len(ordered) - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return float(ordered[lo])
    w = idx - lo
    return float(ordered[lo] * (1 - w) + ordered[hi] * w)


def _latency_stats(samples_ms: list[int]) -> dict:
    vals = [float(x) for x in samples_ms]
    return {
        "sum": int(sum(samples_ms)),
        "p50": round(_percentile(vals, 0.5), 2),
        "p95": round(_percentile(vals, 0.95), 2),
    }


def _merge_telemetry(dst: Telemetry, src: dict | None) -> None:
    if not src:
        return
    dst.llm_calls += int(src.get("llm_calls", 0))
    dst.embedding_calls += int(src.get("embedding_calls", 0))
    dst.llm_latency_ms += int(src.get("llm_latency_ms", 0))
    dst.embedding_latency_ms += int(src.get("embedding_latency_ms", 0))
    dst.prompt_tokens += int(src.get("prompt_tokens", 0))
    dst.completion_tokens += int(src.get("completion_tokens", 0))
    dst.total_tokens += int(src.get("total_tokens", 0))


def _check_budget(telemetry: dict, budget: dict) -> dict:
    llm_calls = int(telemetry.get("llm_calls", 0))
    prompt_tokens = int(telemetry.get("prompt_tokens", 0))
    completion_tokens = int(telemetry.get("completion_tokens", 0))
    total_tokens = int(telemetry.get("total_tokens", 0))
    limits = {
        "max_llm_calls": int(budget["max_llm_calls"]),
        "max_completion_tokens": int(budget["max_completion_tokens"]),
        "max_total_tokens": int(budget["max_total_tokens"]),
        "evidence_token_limit": int(budget.get("evidence_token_limit", 0)),
    }
    within = (
        llm_calls <= limits["max_llm_calls"]
        and completion_tokens <= limits["max_completion_tokens"]
        and total_tokens <= limits["max_total_tokens"]
    )
    return {
        "within_budget": within,
        "observed": {
            "llm_calls": llm_calls,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        },
        "limits": limits,
    }


def _pack_contexts_with_budget(
    contexts: list[str],
    budget_manager: BudgetManager | None,
    max_chunks: int | None = None,
    max_context_chars: int | None = None,
) -> list[str]:
    picked: list[str] = []
    current = ""
    for c in contexts:
        if max_chunks is not None and len(picked) >= int(max_chunks):
            break
        text = str(c)
        if not text:
            continue
        if max_context_chars is not None and len(text) > int(max_context_chars):
            text = text[: int(max_context_chars)]
        if budget_manager is not None:
            if not budget_manager.can_add(current, text + "\n\n"):
                continue
        picked.append(text)
        current += text + "\n\n"
    return picked


def _run_vector(
    query: str,
    idx_file: str,
    store_file: str,
    top_k: int,
    max_context_chars: int,
    max_chunks: int | None = None,
    max_completion_tokens: int | None = None,
    budget_manager: BudgetManager | None = None,
) -> tuple[dict, dict]:
    evidence, retrieve_meta = retrieve_with_evidence(
        query=query,
        idx_file=idx_file,
        store_file=store_file,
        top_k=top_k,
        return_meta=True,
    )
    contexts = _pack_contexts_with_budget(
        [x["text"] for x in evidence],
        budget_manager=budget_manager,
        max_chunks=max_chunks,
        max_context_chars=max_context_chars,
    )
    answer, answer_meta = answer_with_context(
        query=query,
        contexts=contexts,
        max_completion_tokens=max_completion_tokens,
        return_meta=True,
    )
    t = Telemetry()
    t.add_embedding((retrieve_meta or {}).get("embedding"))
    t.add_llm(answer_meta)
    payload = {
        "answer": answer,
        "evidence": evidence,
        "telemetry": {
            "embedding": (retrieve_meta or {}).get("embedding"),
            "generation": answer_meta,
            "aggregate": t.to_dict(),
        },
    }
    return payload, t.to_dict()


def ensure_graph_assets(chunks_file: str, triples_file: str, graph_file: str, communities_file: str) -> dict:
    metrics = {}
    if not Path(triples_file).exists():
        metrics["triple_extract"] = extract_triples(chunks_file, triples_file)
    if not Path(graph_file).exists():
        metrics["graph_build"] = build_graph(triples_file, graph_file)
    if not Path(communities_file).exists():
        metrics["community_build"] = build_communities(graph_file, communities_file)
    return metrics


def _resolve_regimes(regimes: str) -> list[str]:
    if regimes == "both":
        return ["best_effort", "budget_matched"]
    return [regimes]


def _load_indexing_metrics(graph_build_metrics: dict) -> dict:
    vector = _maybe_read_json("outputs/results/index_build_metrics.json") or _maybe_read_json(
        "outputs/results/index_build_metrics_batch.json"
    )
    triple_extract = graph_build_metrics.get("triple_extract") or _maybe_read_json(
        "outputs/results/triple_extract_metrics.json"
    )
    graph_build = graph_build_metrics.get("graph_build") or _maybe_read_json(
        "outputs/results/graph_build_metrics.json"
    )
    community_build = graph_build_metrics.get("community_build") or _maybe_read_json(
        "outputs/results/community_build_metrics.json"
    )
    return {
        "vector_index": vector,
        "triple_extract": triple_extract,
        "graph_build": graph_build,
        "community_build": community_build,
    }


def _regime_settings(top_k: int, budget_cfg_yaml: dict | None = None) -> dict[str, dict]:
    comp = getattr(cfg, "comparison", None)
    best = getattr(comp, "best_effort", None)
    budget = getattr(comp, "budget_matched", None)
    best_kg = getattr(best, "kg", None)
    best_graph = getattr(best, "graph", None)
    budget_kg = getattr(budget, "kg", None)
    budget_graph = getattr(budget, "graph", None)
    out = {
        "best_effort": {
            "vector_top_k": int(_g(best, "vector_top_k", top_k)),
            "vector_max_context_chars": int(_g(best, "vector_max_context_chars", 12000)),
            "kg_kwargs": {
                "max_hops": int(_g(best_kg, "max_hops", 2)),
                "max_start_entities": int(_g(best_kg, "max_start_entities", 5)),
                "top_edges": int(_g(best_kg, "top_edges", 60)),
                "max_chunks": int(_g(best_kg, "max_chunks", 12)),
                "max_context_chars": int(_g(best_kg, "max_context_chars", 12000)),
                "max_nodes": int(_g(best_kg, "max_nodes", 0)) or None,
                "use_entity_linking": True,
                "use_embedding_rerank": True,
            },
            "graph_kwargs": {
                "top_communities": int(_g(best_graph, "top_communities", 3)),
                "max_evidence": int(_g(best_graph, "max_evidence", 12)),
                "query_level": int(_g(best_graph, "query_level", -1)),
                "use_hierarchy": bool(_g(best_graph, "use_hierarchy", True)),
                "use_community_summaries": bool(_g(best_graph, "use_community_summaries", True)),
                "shuffle_communities": bool(_g(best_graph, "shuffle_communities", True)),
                "use_map_reduce": bool(_g(best_graph, "use_map_reduce", True)),
                "max_summary_chars": int(_g(best_graph, "max_summary_chars", 1800)),
                "map_keypoints_limit": int(_g(best_graph, "map_keypoints_limit", 5)),
            },
        },
        "budget_matched": {
            "vector_top_k": int(_g(budget, "vector_top_k", min(top_k, 3))),
            "vector_max_context_chars": int(_g(budget, "vector_max_context_chars", 3600)),
            "kg_kwargs": {
                "max_hops": int(_g(budget_kg, "max_hops", 1)),
                "max_start_entities": int(_g(budget_kg, "max_start_entities", 3)),
                "top_edges": int(_g(budget_kg, "top_edges", 24)),
                "max_chunks": int(_g(budget_kg, "max_chunks", 6)),
                "max_context_chars": int(_g(budget_kg, "max_context_chars", 3600)),
                "max_nodes": int(_g(budget_kg, "max_nodes", 30)),
                "use_entity_linking": True,
                "use_embedding_rerank": True,
            },
            "graph_kwargs": {
                "top_communities": int(_g(budget_graph, "top_communities", 1)),
                "max_evidence": int(_g(budget_graph, "max_evidence", 6)),
                "query_level": int(_g(budget_graph, "query_level", 0)),
                "use_hierarchy": bool(_g(budget_graph, "use_hierarchy", True)),
                "use_community_summaries": bool(_g(budget_graph, "use_community_summaries", True)),
                "shuffle_communities": bool(_g(budget_graph, "shuffle_communities", False)),
                "use_map_reduce": bool(_g(budget_graph, "use_map_reduce", True)),
                "max_summary_chars": int(_g(budget_graph, "max_summary_chars", 1200)),
                "map_keypoints_limit": int(_g(budget_graph, "map_keypoints_limit", 5)),
            },
        },
    }
    y_budget = ((budget_cfg_yaml or {}).get("budget") or {})
    if y_budget:
        out["budget_matched"]["vector_top_k"] = int(
            ((y_budget.get("vector") or {}).get("max_chunks"))
            or out["budget_matched"]["vector_top_k"]
        )
        out["budget_matched"]["kg_kwargs"]["max_hops"] = int(
            ((y_budget.get("kg") or {}).get("max_hops"))
            or out["budget_matched"]["kg_kwargs"]["max_hops"]
        )
        out["budget_matched"]["kg_kwargs"]["max_nodes"] = int(
            ((y_budget.get("kg") or {}).get("max_nodes"))
            or out["budget_matched"]["kg_kwargs"].get("max_nodes")
            or 30
        )
        out["budget_matched"]["graph_kwargs"]["top_communities"] = int(
            ((y_budget.get("graph") or {}).get("max_communities"))
            or out["budget_matched"]["graph_kwargs"]["top_communities"]
        )
        out["budget_matched"]["graph_kwargs"]["query_level"] = int(
            ((y_budget.get("graph") or {}).get("summary_level"))
            or out["budget_matched"]["graph_kwargs"]["query_level"]
        )
        out["budget_matched"]["graph_kwargs"]["map_keypoints_limit"] = int(
            ((y_budget.get("graph") or {}).get("map_keypoints_limit"))
            or out["budget_matched"]["graph_kwargs"].get("map_keypoints_limit")
            or 5
        )
    return out


def run_compare(
    queries_file: str,
    chunks_file: str,
    idx_file: str,
    store_file: str,
    triples_file: str,
    graph_file: str,
    communities_file: str,
    top_k: int,
    out_file: str,
    metrics_file: str,
    regimes: str,
    budget_config_file: str,
    incremental_only: bool = False,
) -> dict:
    _progress(f"loading queries: {queries_file}")
    queries = _load_queries(queries_file)
    _progress(f"loaded queries: {len(queries)}")
    _progress("ensuring graph assets")
    graph_metrics = ensure_graph_assets(chunks_file, triples_file, graph_file, communities_file)
    _progress("graph assets ready")
    regime_names = _resolve_regimes(regimes)
    _progress(f"regimes: {regime_names}")
    _progress(f"loading budget config: {budget_config_file}")
    budget_yaml = _load_yaml(budget_config_file)

    budget = {
        "evidence_token_limit": 2500,
        "max_completion_tokens": 800,
        "max_total_tokens": 3800,
        "max_llm_calls": 2,
    }
    yaml_budget = (budget_yaml.get("budget") or {})
    if yaml_budget:
        budget["evidence_token_limit"] = int(yaml_budget.get("evidence_token_limit", budget["evidence_token_limit"]))
        budget["max_completion_tokens"] = int(yaml_budget.get("max_completion_tokens", budget["max_completion_tokens"]))
        budget["max_total_tokens"] = int(yaml_budget.get("max_total_tokens", budget["max_total_tokens"]))
        budget["max_llm_calls"] = int(yaml_budget.get("max_llm_calls", budget["max_llm_calls"]))
    settings = _regime_settings(top_k=top_k, budget_cfg_yaml=budget_yaml)
    tokenizer = TokenizerProvider(
        backend=cfg.llm.backend,
        model_name=(cfg.llm.api.model if cfg.llm.backend == "api" else cfg.llm.local.model),
    )

    existing_rows: list[dict] = _load_jsonl(out_file) if incremental_only else []
    existing_qids = {
        str(r.get("qid", "")).strip()
        for r in existing_rows
        if str(r.get("qid", "")).strip()
    }
    existing_queries = {
        str(r.get("query", "")).strip().lower()
        for r in existing_rows
        if str(r.get("query", "")).strip()
    }
    if incremental_only:
        filtered_queries: list[dict] = []
        for q in queries:
            qid = str(q.get("qid", "")).strip()
            qtext = str(q.get("query", "")).strip().lower()
            if (qid and qid in existing_qids) or (qtext and qtext in existing_queries):
                continue
            filtered_queries.append(q)
        _progress(
            f"incremental mode: existing={len(existing_rows)}, pending={len(filtered_queries)}, skipped={len(queries) - len(filtered_queries)}"
        )
        queries = filtered_queries

    rows: list[dict] = list(existing_rows)
    new_rows: list[dict] = []
    aggregate: dict[str, dict[str, Telemetry]] = {
        rg: {"vector_rag": Telemetry(), "kg_rag": Telemetry(), "graph_rag": Telemetry()}
        for rg in regime_names
    }
    by_type: dict[str, dict[str, dict[str, Telemetry]]] = {rg: {} for rg in regime_names}
    latency_samples: dict[str, dict[str, list[int]]] = {
        rg: {"vector_rag": [], "kg_rag": [], "graph_rag": []} for rg in regime_names
    }

    for idx, q in enumerate(queries, start=1):
        query = q["query"]
        qid = q.get("qid")
        qtype = q.get("type", "unknown")
        _progress(f"query {idx}/{len(queries)} start: qid={qid} type={qtype}")
        row = {
            "qid": qid,
            "type": qtype,
            "query": query,
            "regimes": {},
        }
        for rg in regime_names:
            _progress(f"query {idx}/{len(queries)} regime={rg}: start")
            is_budget = rg == "budget_matched"
            regime_cfg = settings[rg]
            vector_top_k = regime_cfg["vector_top_k"]
            vector_context_chars = regime_cfg["vector_max_context_chars"]
            kg_kwargs = regime_cfg["kg_kwargs"]
            graph_kwargs = regime_cfg["graph_kwargs"]
            if is_budget:
                vector_manager = BudgetManager(tokenizer=tokenizer, cfg=budget, method="vector_rag", regime=rg)
                kg_manager = BudgetManager(tokenizer=tokenizer, cfg=budget, method="kg_rag", regime=rg)
                graph_manager = BudgetManager(tokenizer=tokenizer, cfg=budget, method="graph_rag", regime=rg)
            else:
                vector_manager = None
                kg_manager = None
                graph_manager = None

            _progress(f"query {idx}/{len(queries)} regime={rg}: running 3 branches in parallel")
            branch_results: dict[str, dict | tuple[dict, dict]] = {}

            def _run_vector_branch() -> tuple[dict, dict]:
                return _run_vector(
                    query=query,
                    idx_file=idx_file,
                    store_file=store_file,
                    top_k=vector_top_k,
                    max_context_chars=vector_context_chars,
                    max_chunks=vector_top_k if is_budget else None,
                    max_completion_tokens=budget["max_completion_tokens"] if is_budget else None,
                    budget_manager=vector_manager,
                )

            def _run_kg_branch() -> dict:
                return answer_with_kg(
                    query=query,
                    graph_file=graph_file,
                    store_file=store_file,
                    max_completion_tokens=budget["max_completion_tokens"] if is_budget else None,
                    **kg_kwargs,
                )

            def _run_graph_branch() -> dict:
                return answer_with_graphrag(
                    query=query,
                    graph_file=graph_file,
                    communities_file=communities_file,
                    max_completion_tokens=budget["max_completion_tokens"] if is_budget else None,
                    **graph_kwargs,
                )

            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = {
                    executor.submit(_run_vector_branch): "vector_rag",
                    executor.submit(_run_kg_branch): "kg_rag",
                    executor.submit(_run_graph_branch): "graph_rag",
                }
                for future in as_completed(futures):
                    name = futures[future]
                    branch_results[name] = future.result()
                    _progress(f"query {idx}/{len(queries)} regime={rg}: {name} done")

            vector_out, vector_t = branch_results["vector_rag"]  # type: ignore[assignment]
            kg_out = branch_results["kg_rag"]  # type: ignore[assignment]
            graph_out = branch_results["graph_rag"]  # type: ignore[assignment]

            vector_agg_t = vector_t
            kg_agg_t = kg_out.get("telemetry", {})
            graph_agg_t = graph_out.get("telemetry", {})

            # Adaptive enforcement for GraphRAG under strict budget.
            if is_budget and not _check_budget(graph_agg_t, budget).get("within_budget", False):
                _progress(
                    f"query {idx}/{len(queries)} regime={rg}: graph_rag over budget, applying adaptive shrink"
                )
                tighter_graph_kwargs = dict(graph_kwargs)
                tighter_graph_kwargs["top_communities"] = 1
                tighter_graph_kwargs["use_map_reduce"] = False
                tighter_graph_kwargs["shuffle_communities"] = False
                tighter_graph_kwargs["max_summary_chars"] = min(
                    int(graph_kwargs.get("max_summary_chars", 1200)), 500
                )
                tighter_graph_kwargs["map_keypoints_limit"] = min(
                    int(graph_kwargs.get("map_keypoints_limit", 5)), 3
                )
                graph_out = answer_with_graphrag(
                    query=query,
                    graph_file=graph_file,
                    communities_file=communities_file,
                    max_completion_tokens=budget["max_completion_tokens"],
                    **tighter_graph_kwargs,
                )
                graph_out["budget_adaptation"] = {
                    "enabled": True,
                    "original_graph_kwargs": graph_kwargs,
                    "adapted_graph_kwargs": tighter_graph_kwargs,
                }
                graph_agg_t = graph_out.get("telemetry", {})

            _merge_telemetry(aggregate[rg]["vector_rag"], vector_agg_t)
            _merge_telemetry(aggregate[rg]["kg_rag"], kg_agg_t)
            _merge_telemetry(aggregate[rg]["graph_rag"], graph_agg_t)

            by_type[rg].setdefault(
                qtype,
                {"vector_rag": Telemetry(), "kg_rag": Telemetry(), "graph_rag": Telemetry()},
            )
            _merge_telemetry(by_type[rg][qtype]["vector_rag"], vector_agg_t)
            _merge_telemetry(by_type[rg][qtype]["kg_rag"], kg_agg_t)
            _merge_telemetry(by_type[rg][qtype]["graph_rag"], graph_agg_t)

            latency_samples[rg]["vector_rag"].append(
                int(vector_agg_t.get("llm_latency_ms", 0)) + int(vector_agg_t.get("embedding_latency_ms", 0))
            )
            latency_samples[rg]["kg_rag"].append(
                int(kg_agg_t.get("llm_latency_ms", 0)) + int(kg_agg_t.get("embedding_latency_ms", 0))
            )
            latency_samples[rg]["graph_rag"].append(
                int(graph_agg_t.get("llm_latency_ms", 0)) + int(graph_agg_t.get("embedding_latency_ms", 0))
            )

            if is_budget:
                v_err = None
                k_err = None
                g_err = None
                try:
                    vector_manager.register_from_telemetry(vector_agg_t, stage="vector_answer")
                except RuntimeError as e:
                    v_err = str(e)
                try:
                    kg_manager.register_from_telemetry(kg_agg_t, stage="kg_answer")
                except RuntimeError as e:
                    k_err = str(e)
                try:
                    graph_manager.register_from_telemetry(graph_agg_t, stage="graph_answer")
                except RuntimeError as e:
                    g_err = str(e)

                vector_out["budget_check"] = {
                    **_check_budget(vector_agg_t, budget),
                    "manager": vector_manager.to_dict(),
                    "error": v_err,
                }
                kg_out["budget_check"] = {
                    **_check_budget(kg_agg_t, budget),
                    "manager": kg_manager.to_dict(),
                    "error": k_err,
                }
                graph_out["budget_check"] = {
                    **_check_budget(graph_agg_t, budget),
                    "manager": graph_manager.to_dict(),
                    "error": g_err,
                }
                _progress(
                    "budget usage "
                    + json.dumps(
                        {
                            "method": "vector_rag",
                            "regime": rg,
                            "llm_calls": vector_agg_t.get("llm_calls", 0),
                            "prompt_tokens": vector_agg_t.get("prompt_tokens", 0),
                            "completion_tokens": vector_agg_t.get("completion_tokens", 0),
                        },
                        ensure_ascii=False,
                    )
                )
                _progress(
                    "budget usage "
                    + json.dumps(
                        {
                            "method": "kg_rag",
                            "regime": rg,
                            "llm_calls": kg_agg_t.get("llm_calls", 0),
                            "prompt_tokens": kg_agg_t.get("prompt_tokens", 0),
                            "completion_tokens": kg_agg_t.get("completion_tokens", 0),
                        },
                        ensure_ascii=False,
                    )
                )
                _progress(
                    "budget usage "
                    + json.dumps(
                        {
                            "method": "graph_rag",
                            "regime": rg,
                            "llm_calls": graph_agg_t.get("llm_calls", 0),
                            "prompt_tokens": graph_agg_t.get("prompt_tokens", 0),
                            "completion_tokens": graph_agg_t.get("completion_tokens", 0),
                        },
                        ensure_ascii=False,
                    )
                )

            row["regimes"][rg] = {
                "vector_rag": vector_out,
                "kg_rag": kg_out,
                "graph_rag": graph_out,
            }
            _progress(f"query {idx}/{len(queries)} regime={rg}: done")
        rows.append(row)
        new_rows.append(row)
        _progress(f"query {idx}/{len(queries)} done: qid={qid}")

    _progress(f"writing answers: {out_file}")
    _write_jsonl(out_file, rows)

    aggregate_view = {}
    for rg in regime_names:
        aggregate_view[rg] = {}
        for method in ("vector_rag", "kg_rag", "graph_rag"):
            telemetry_dict = aggregate[rg][method].to_dict()
            telemetry_dict["latency_ms"] = _latency_stats(latency_samples[rg][method])
            aggregate_view[rg][method] = telemetry_dict

    by_type_view = {}
    for rg in regime_names:
        by_type_view[rg] = {}
        for qtype, bundle in by_type[rg].items():
            by_type_view[rg][qtype] = {}
            for method in ("vector_rag", "kg_rag", "graph_rag"):
                by_type_view[rg][qtype][method] = bundle[method].to_dict()

    summary = {
        "queries_file": queries_file,
        "num_queries": len(rows),
        "num_existing_answers": len(existing_rows),
        "num_processed_this_run": len(new_rows),
        "regimes": regime_names,
        "top_k": top_k,
        "budget": budget,
        "budget_config_file": budget_config_file,
        "incremental_only": incremental_only,
        "graph_assets_metrics": graph_metrics,
        "indexing_metrics": _load_indexing_metrics(graph_metrics),
        "graph_structure_metrics": {},
        "aggregate_metrics": aggregate_view,
        "aggregate_metrics_by_type": by_type_view,
        "results_file": out_file,
    }
    try:
        summary["graph_structure_metrics"] = compute_graph_structure_metrics(
            graph_file=graph_file,
            communities_file=communities_file,
            out_json="outputs/results/graph_structure_metrics.json",
            out_dir="outputs/results/graph_plots",
        )
    except Exception as exc:
        summary["graph_structure_metrics"] = {"error": str(exc)}
    _progress(f"writing metrics: {metrics_file}")
    _write_json(metrics_file, summary)
    _progress("run_compare completed")
    return summary


def main() -> None:
    comp = getattr(cfg, "comparison", None)
    best = getattr(comp, "best_effort", None)
    default_top_k = int(_g(best, "vector_top_k", int(cfg.retrieval.top_k)))
    parser = argparse.ArgumentParser(description="Run VectorRAG, KG-RAG, GraphRAG comparison")
    parser.add_argument("--queries-file", default="data/queries/queries.jsonl")
    parser.add_argument("--chunks-file", default="data/processed/chunks_sampled.jsonl")
    parser.add_argument("--idx-file", default="outputs/indexes/faiss_sampled.idx")
    parser.add_argument("--store-file", default="outputs/indexes/chunk_store_sampled.json")
    parser.add_argument("--triples-file", default="outputs/graph/triples.jsonl")
    parser.add_argument("--graph-file", default="outputs/graph/graph.json")
    parser.add_argument("--communities-file", default="outputs/graph/communities.json")
    parser.add_argument("--top-k", type=int, default=default_top_k)
    parser.add_argument("--out-file", default="outputs/results/compare_answers.jsonl")
    parser.add_argument("--metrics-file", default="outputs/results/compare_metrics.json")
    parser.add_argument(
        "--regimes",
        choices=["best_effort", "budget_matched", "both"],
        default="both",
    )
    parser.add_argument("--budget-config-file", default="config_budget.yaml")
    parser.add_argument(
        "--incremental-only",
        action="store_true",
        help="Only process unanswered queries and keep existing answers in out-file",
    )
    args = parser.parse_args()

    summary = run_compare(
        queries_file=args.queries_file,
        chunks_file=args.chunks_file,
        idx_file=args.idx_file,
        store_file=args.store_file,
        triples_file=args.triples_file,
        graph_file=args.graph_file,
        communities_file=args.communities_file,
        top_k=args.top_k,
        out_file=args.out_file,
        metrics_file=args.metrics_file,
        regimes=args.regimes,
        budget_config_file=args.budget_config_file,
        incremental_only=args.incremental_only,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
