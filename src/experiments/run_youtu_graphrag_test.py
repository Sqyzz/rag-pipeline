from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import yaml

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

try:
    from adapters.youtu_client import YoutuClient
    from adapters.youtu_dataset_sync import sync_chunks_to_youtu_dataset
    from adapters.youtu_graph_state import build_state_payload, decide_graph_reuse, save_graph_state
    from baselines.youtu_graph_rag_adapter import answer_with_youtu_graphrag
    from evaluation.graph_structure_metrics import compute_graph_structure_metrics
    from experiments.run_compare import ensure_graph_assets
    from utils.budget import BudgetManager
    from utils.config import cfg
    from utils.telemetry import Telemetry
    from utils.tokenizer import TokenizerProvider
except ModuleNotFoundError:
    from src.adapters.youtu_client import YoutuClient
    from src.adapters.youtu_dataset_sync import sync_chunks_to_youtu_dataset
    from src.adapters.youtu_graph_state import build_state_payload, decide_graph_reuse, save_graph_state
    from src.baselines.youtu_graph_rag_adapter import answer_with_youtu_graphrag
    from src.evaluation.graph_structure_metrics import compute_graph_structure_metrics
    from src.experiments.run_compare import ensure_graph_assets
    from src.utils.budget import BudgetManager
    from src.utils.config import cfg
    from src.utils.telemetry import Telemetry
    from src.utils.tokenizer import TokenizerProvider


def _progress(message: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[run_youtu_graphrag_test {ts}] {message}", flush=True)


def _g(obj, key: str, default):
    if obj is None:
        return default
    return getattr(obj, key, default)


def _parse_bool(raw: str | bool) -> bool:
    if isinstance(raw, bool):
        return raw
    val = str(raw).strip().lower()
    return val in {"1", "true", "yes", "y", "on"}


def _load_queries(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
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
    graph_build = graph_build_metrics.get("graph_build") or _maybe_read_json("outputs/results/graph_build_metrics.json")
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
    best_graph = getattr(best, "graph", None)
    budget_graph = getattr(budget, "graph", None)
    out = {
        "best_effort": {
            "vector_top_k": int(_g(best, "vector_top_k", top_k)),
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
            ((y_budget.get("vector") or {}).get("max_chunks")) or out["budget_matched"]["vector_top_k"]
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
            or out["budget_matched"]["graph_kwargs"]["map_keypoints_limit"]
        )
    return out


def _empty_method_payload() -> dict[str, Any]:
    return {
        "answer": "",
        "communities": [],
        "community_summaries": [],
        "map_partial_answers": [],
        "subgraph_edges": [],
        "evidence": [],
        "evidence_chunks": [],
        "telemetry": Telemetry().to_dict(),
    }


def ensure_youtu_graph_assets(
    *,
    chunks_file: str,
    triples_file: str,
    graph_file: str,
    communities_file: str,
    youtu_base_url: str,
    youtu_dataset: str,
    graph_state_file: str,
    reuse_graph: bool,
    force_rebuild: bool,
    export_youtu_artifacts: bool,
    construct_poll_sec: int,
    construct_timeout_sec: int,
    sync_mode: str,
    shared_corpus_dir: str,
) -> dict[str, Any]:
    Path(graph_state_file).parent.mkdir(parents=True, exist_ok=True)

    build_params = {
        "dataset": youtu_dataset,
        "chunks_file": chunks_file,
        "sync_mode": sync_mode,
        "graph_file": graph_file,
        "communities_file": communities_file,
    }

    require_local_assets = bool(export_youtu_artifacts)
    decision = decide_graph_reuse(
        graph_state_file=graph_state_file,
        chunks_file=chunks_file,
        dataset=youtu_dataset,
        build_params=build_params,
        reuse_graph=reuse_graph,
        force_rebuild=force_rebuild,
        require_local_assets=require_local_assets,
        graph_file=graph_file,
        communities_file=communities_file,
    )

    result: dict[str, Any] = {
        "graph_reuse": {
            "used_cached_graph": bool(decision["used_cached_graph"]),
            "reason": decision["reason"],
            "fingerprint": decision["fingerprint"],
            "fingerprint_method": decision.get("fingerprint_method"),
        }
    }

    # Bootstrap local reuse when state file is missing but local artifacts already exist.
    if (
        not decision["used_cached_graph"]
        and reuse_graph
        and not force_rebuild
        and not decision.get("state")
        and Path(graph_file).exists()
        and Path(communities_file).exists()
    ):
        state_payload = build_state_payload(
            dataset=youtu_dataset,
            fingerprint=decision["fingerprint"],
            build_params={
                "dataset": youtu_dataset,
                "chunks_source": chunks_file,
                "chunks_fingerprint": decision["fingerprint"],
                "sync_mode": sync_mode,
                "bootstrap_from_local_assets": True,
            },
            graph_task_id=None,
        )
        save_graph_state(graph_state_file, state_payload)
        result["graph_reuse"] = {
            **result["graph_reuse"],
            "used_cached_graph": True,
            "reason": "fingerprint_match",
            "bootstrapped_state": True,
        }
        result["youtu_construct"] = {"skipped": True}
        result["youtu_sync"] = {"sync_mode": "none", "skipped": True}
        result["youtu_artifacts"] = {"skipped": True, "reused_local": True}
        return result

    if decision["used_cached_graph"]:
        result["youtu_construct"] = {"skipped": True}
        result["youtu_sync"] = {"sync_mode": "none", "skipped": True}
        result["youtu_artifacts"] = {"skipped": True, "reused_local": True}
        return result

    youtu_client = YoutuClient(base_url=youtu_base_url, timeout_sec=120)
    sync_meta = sync_chunks_to_youtu_dataset(
        chunks_file=chunks_file,
        dataset=youtu_dataset,
        sync_mode=sync_mode,
        shared_dir=shared_corpus_dir,
    )
    result["youtu_sync"] = sync_meta

    construct_started = time.time()
    task_id = youtu_client.construct_graph(
        dataset_name=youtu_dataset,
        chunks_source=sync_meta.get("written_file") or sync_meta.get("chunks_source"),
        chunks_fingerprint=decision["fingerprint"],
    )
    final_status = youtu_client.poll_construct(
        task_id=task_id,
        timeout_sec=construct_timeout_sec,
        poll_sec=construct_poll_sec,
    )
    result["youtu_construct"] = {
        "task_id": task_id,
        "elapsed_sec": round(time.time() - construct_started, 3),
        "final_status": final_status,
    }

    artifact_meta: dict[str, Any]
    if export_youtu_artifacts:
        try:
            artifact_meta = youtu_client.export_graph_artifacts(
                dataset_name=youtu_dataset,
                graph_file=graph_file,
                communities_file=communities_file,
                task_id=task_id,
            )
        except Exception as exc:  # noqa: BLE001
            artifact_meta = {
                "downloaded": False,
                "error": str(exc),
                "path": "B_local_fallback",
            }
            try:
                local_metrics = ensure_graph_assets(chunks_file, triples_file, graph_file, communities_file)
            except Exception as local_exc:  # noqa: BLE001
                local_metrics = {"error": str(local_exc)}
            result["local_graph_assets"] = local_metrics
    else:
        artifact_meta = {
            "downloaded": False,
            "skipped": True,
            "path": "B_no_artifact_export",
        }
    result["youtu_artifacts"] = artifact_meta

    state_payload = build_state_payload(
        dataset=youtu_dataset,
        fingerprint=decision["fingerprint"],
        build_params={
            "dataset": youtu_dataset,
            "chunks_source": sync_meta.get("written_file") or sync_meta.get("chunks_source"),
            "chunks_fingerprint": decision["fingerprint"],
            "sync_mode": sync_meta.get("sync_mode", sync_mode),
        },
        graph_task_id=task_id,
    )
    save_graph_state(graph_state_file, state_payload)
    return result


def run_youtu_graphrag_test(
    queries_file: str,
    chunks_file: str,
    triples_file: str,
    graph_file: str,
    communities_file: str,
    top_k: int,
    out_file: str,
    metrics_file: str,
    regimes: str,
    budget_config_file: str,
    graph_state_file: str,
    reuse_graph: bool,
    force_rebuild: bool,
    youtu_base_url: str,
    youtu_dataset: str,
    export_youtu_artifacts: bool,
    construct_poll_sec: int,
    construct_timeout_sec: int,
    sync_mode: str,
    shared_corpus_dir: str,
    max_queries: int | None,
) -> dict[str, Any]:
    _progress(f"loading queries: {queries_file}")
    queries = _load_queries(queries_file)
    if max_queries is not None and int(max_queries) > 0:
        queries = queries[: int(max_queries)]
    _progress(f"loaded queries: {len(queries)}")

    _progress("ensuring youtu graph assets")
    graph_metrics = ensure_youtu_graph_assets(
        chunks_file=chunks_file,
        triples_file=triples_file,
        graph_file=graph_file,
        communities_file=communities_file,
        youtu_base_url=youtu_base_url,
        youtu_dataset=youtu_dataset,
        graph_state_file=graph_state_file,
        reuse_graph=reuse_graph,
        force_rebuild=force_rebuild,
        export_youtu_artifacts=export_youtu_artifacts,
        construct_poll_sec=construct_poll_sec,
        construct_timeout_sec=construct_timeout_sec,
        sync_mode=sync_mode,
        shared_corpus_dir=shared_corpus_dir,
    )
    _progress("graph assets ready")

    regime_names = _resolve_regimes(regimes)
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

    rows: list[dict] = []
    graph_attempts = 0
    graph_error_count = 0
    graph_success_count = 0
    graph_error_samples: list[str] = []
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
        row = {"qid": qid, "type": qtype, "query": query, "regimes": {}}

        for rg in regime_names:
            is_budget = rg == "budget_matched"
            graph_kwargs = settings[rg]["graph_kwargs"]
            graph_manager = BudgetManager(tokenizer=tokenizer, cfg=budget, method="graph_rag", regime=rg) if is_budget else None

            graph_out = answer_with_youtu_graphrag(
                query=query,
                graph_file=graph_file,
                communities_file=communities_file,
                max_completion_tokens=budget["max_completion_tokens"] if is_budget else None,
                youtu_base_url=youtu_base_url,
                youtu_dataset=youtu_dataset,
                **graph_kwargs,
            )
            graph_agg_t = graph_out.get("telemetry", {})
            graph_attempts += 1
            graph_error = str((((graph_agg_t.get("extra") or {}).get("error")) or "")).strip()
            if graph_error:
                graph_error_count += 1
                if len(graph_error_samples) < 3:
                    graph_error_samples.append(graph_error)
            elif str(graph_out.get("answer", "")).strip():
                graph_success_count += 1
            budget_view = _check_budget(graph_agg_t, budget)
            usage_complete = bool((((graph_agg_t.get("extra") or {}).get("usage_complete", True))))

            if is_budget and (not budget_view.get("within_budget", False) or not usage_complete):
                tighter_graph_kwargs = dict(graph_kwargs)
                tighter_graph_kwargs["top_communities"] = 1
                tighter_graph_kwargs["use_map_reduce"] = False
                tighter_graph_kwargs["shuffle_communities"] = False
                tighter_graph_kwargs["max_summary_chars"] = min(int(graph_kwargs.get("max_summary_chars", 1200)), 500)
                tighter_graph_kwargs["map_keypoints_limit"] = min(int(graph_kwargs.get("map_keypoints_limit", 5)), 3)

                graph_out = answer_with_youtu_graphrag(
                    query=query,
                    graph_file=graph_file,
                    communities_file=communities_file,
                    max_completion_tokens=budget["max_completion_tokens"],
                    youtu_base_url=youtu_base_url,
                    youtu_dataset=youtu_dataset,
                    **tighter_graph_kwargs,
                )
                graph_out["budget_adaptation"] = {
                    "enabled": True,
                    "original_graph_kwargs": graph_kwargs,
                    "adapted_graph_kwargs": tighter_graph_kwargs,
                    "reason": "over_budget_or_incomplete_usage",
                }
                graph_agg_t = graph_out.get("telemetry", {})
                graph_error = str((((graph_agg_t.get("extra") or {}).get("error")) or "")).strip()
                if graph_error:
                    graph_error_count += 1
                    if len(graph_error_samples) < 3:
                        graph_error_samples.append(graph_error)
                elif str(graph_out.get("answer", "")).strip():
                    graph_success_count += 1
                budget_view = _check_budget(graph_agg_t, budget)

            _merge_telemetry(aggregate[rg]["graph_rag"], graph_agg_t)
            by_type[rg].setdefault(
                qtype,
                {"vector_rag": Telemetry(), "kg_rag": Telemetry(), "graph_rag": Telemetry()},
            )
            _merge_telemetry(by_type[rg][qtype]["graph_rag"], graph_agg_t)

            latency_samples[rg]["vector_rag"].append(0)
            latency_samples[rg]["kg_rag"].append(0)
            latency_samples[rg]["graph_rag"].append(
                int(graph_agg_t.get("llm_latency_ms", 0)) + int(graph_agg_t.get("embedding_latency_ms", 0))
            )

            vector_out = _empty_method_payload()
            kg_out = _empty_method_payload()
            if is_budget:
                err = None
                if not usage_complete:
                    err = "telemetry usage incomplete"
                try:
                    graph_manager.register_from_telemetry(graph_agg_t, stage="graph_answer")
                except RuntimeError as exc:
                    err = str(exc)

                graph_out["budget_check"] = {
                    **budget_view,
                    "manager": graph_manager.to_dict(),
                    "error": err,
                }
                vector_out["budget_check"] = {
                    **_check_budget(vector_out["telemetry"], budget),
                    "manager": {"method": "vector_rag", "regime": rg, "used": {"llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}},
                    "error": None,
                }
                kg_out["budget_check"] = {
                    **_check_budget(kg_out["telemetry"], budget),
                    "manager": {"method": "kg_rag", "regime": rg, "used": {"llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}},
                    "error": None,
                }

            row["regimes"][rg] = {
                "vector_rag": vector_out,
                "kg_rag": kg_out,
                "graph_rag": graph_out,
            }

        rows.append(row)
        _progress(f"query {idx}/{len(queries)} done: qid={qid}")

    _progress(f"writing answers: {out_file}")
    _write_jsonl(out_file, rows)

    if graph_attempts > 0 and graph_error_count >= graph_attempts and graph_success_count == 0:
        raise RuntimeError(
            "All youtu graph queries failed; no valid graph_rag answers were produced. "
            + "samples="
            + json.dumps(graph_error_samples, ensure_ascii=False)
        )

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
        "regimes": regime_names,
        "top_k": top_k,
        "budget": budget,
        "budget_config_file": budget_config_file,
        "graph_assets_metrics": graph_metrics,
        "indexing_metrics": _load_indexing_metrics(graph_metrics),
        "graph_structure_metrics": {},
        "aggregate_metrics": aggregate_view,
        "aggregate_metrics_by_type": by_type_view,
        "results_file": out_file,
    }

    if Path(graph_file).exists() and Path(communities_file).exists():
        try:
            summary["graph_structure_metrics"] = compute_graph_structure_metrics(
                graph_file=graph_file,
                communities_file=communities_file,
                out_json="outputs/results/graph_structure_metrics.json",
                out_dir="outputs/results/graph_plots",
            )
        except Exception as exc:  # noqa: BLE001
            summary["graph_structure_metrics"] = {"error": str(exc)}
    else:
        summary["graph_structure_metrics"] = {"error": "graph/communities artifacts not available"}

    _progress(f"writing metrics: {metrics_file}")
    _write_json(metrics_file, summary)
    _progress("run_youtu_graphrag_test completed")
    return summary


def main() -> None:
    comp = getattr(cfg, "comparison", None)
    best = getattr(comp, "best_effort", None)
    youtu = getattr(cfg, "youtu", None)

    parser = argparse.ArgumentParser(description="Run independent youtu-GraphRAG test pipeline")
    parser.add_argument("--queries-file", default="data/queries/queries.jsonl")
    parser.add_argument("--chunks-file", default="data/processed/chunks_sampled.jsonl")
    parser.add_argument("--triples-file", default="outputs/graph/triples.jsonl")
    parser.add_argument("--graph-file", default="outputs/graph/graph.json")
    parser.add_argument("--communities-file", default="outputs/graph/communities.json")
    parser.add_argument("--top-k", type=int, default=int(_g(best, "vector_top_k", int(cfg.retrieval.top_k))))
    parser.add_argument("--out-file", default="outputs/results/youtu_compare_answers.jsonl")
    parser.add_argument("--metrics-file", default="outputs/results/youtu_compare_metrics.json")
    parser.add_argument("--regimes", choices=["best_effort", "budget_matched", "both"], default="both")
    parser.add_argument("--budget-config-file", default="config_budget.yaml")

    parser.add_argument("--reuse-graph", default="true")
    parser.add_argument("--force-rebuild", default="false")
    parser.add_argument("--graph-state-file", default="outputs/graph/youtu_graph_state.json")

    parser.add_argument("--youtu-base-url", default=str(_g(youtu, "base_url", "http://127.0.0.1:8000")))
    parser.add_argument("--youtu-dataset", default=str(_g(youtu, "dataset", "enterprise")))
    parser.add_argument("--export-youtu-artifacts", default="true")
    parser.add_argument("--construct-poll-sec", type=int, default=int(_g(youtu, "construct_poll_sec", 2)))
    parser.add_argument("--construct-timeout-sec", type=int, default=int(_g(youtu, "construct_timeout_sec", 1800)))

    parser.add_argument("--sync-mode", choices=["none", "shared_dir"], default="none")
    parser.add_argument("--shared-corpus-dir", default="outputs/youtu_sync")
    parser.add_argument("--max-queries", type=int, default=None)

    args = parser.parse_args()
    summary = run_youtu_graphrag_test(
        queries_file=args.queries_file,
        chunks_file=args.chunks_file,
        triples_file=args.triples_file,
        graph_file=args.graph_file,
        communities_file=args.communities_file,
        top_k=args.top_k,
        out_file=args.out_file,
        metrics_file=args.metrics_file,
        regimes=args.regimes,
        budget_config_file=args.budget_config_file,
        graph_state_file=args.graph_state_file,
        reuse_graph=_parse_bool(args.reuse_graph),
        force_rebuild=_parse_bool(args.force_rebuild),
        youtu_base_url=args.youtu_base_url,
        youtu_dataset=args.youtu_dataset,
        export_youtu_artifacts=_parse_bool(args.export_youtu_artifacts),
        construct_poll_sec=args.construct_poll_sec,
        construct_timeout_sec=args.construct_timeout_sec,
        sync_mode=args.sync_mode,
        shared_corpus_dir=args.shared_corpus_dir,
        max_queries=args.max_queries,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
