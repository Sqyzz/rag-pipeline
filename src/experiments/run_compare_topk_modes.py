from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from evaluation.run_eval import run_eval
from experiments.run_compare import run_compare


def _parse_int_list(raw: str, default: list[int]) -> list[int]:
    vals = []
    for x in str(raw or "").split(","):
        x = x.strip()
        if not x:
            continue
        vals.append(int(x))
    return vals or list(default)


def _parse_mode_list(raw: str, default: list[str]) -> list[str]:
    vals = []
    for x in str(raw or "").split(","):
        x = x.strip().lower()
        if x in {"reject", "open"} and x not in vals:
            vals.append(x)
    return vals or list(default)


def _write_md_table(path: Path, df: pd.DataFrame) -> None:
    lines = [
        "| Method | Dataset | Mode | Top20 | Top10 | Tokens | Time |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in df.iterrows():
        lines.append(
            "| "
            + f"{row.get('method', '')} | {row.get('dataset', '')} | {row.get('mode', '')} | "
            + f"{row.get('top20_accuracy', '')} | {row.get('top10_accuracy', '')} | "
            + f"{row.get('tokens', '')} | {row.get('time', '')} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_compare_topk_modes(
    *,
    queries_file: str,
    gold_file: str,
    chunks_file: str,
    idx_file: str,
    store_file: str,
    triples_file: str,
    graph_file: str,
    communities_file: str,
    budget_config_file: str = "config_budget.yaml",
    topk_list: list[int] | None = None,
    answer_modes: list[str] | None = None,
    regime: str = "best_effort",
    include_youtu: bool = False,
    include_lightrag: bool = False,
    lightrag_mode: str = "hybrid",
    lightrag_working_dir: str | None = None,
    lightrag_force_rebuild: bool = False,
    method_mode: str = "all",
    judge_mode: str = "off",
    judge_model: str = "qwen-flash",
    max_queries: int | None = None,
    incremental_only: bool = False,
    results_dir: str = "outputs/results",
    youtu_base_url: str = "http://127.0.0.1:8000",
    youtu_dataset: str = "enterprise",
    youtu_route_type: str | None = None,
    youtu_client_id: str | None = None,
    youtu_schema_file: str | None = None,
    youtu_graph_state_file: str = "outputs/graph/youtu_graph_state.json",
    youtu_reuse_graph: bool = True,
    youtu_force_rebuild: bool = False,
    youtu_sync_mode: str = "none",
    youtu_shared_corpus_dir: str = "outputs/youtu_sync",
    youtu_corpus_source_file: str | None = None,
    youtu_construct_poll_sec: int = 2,
    youtu_construct_timeout_sec: int = 1800,
    youtu_require_fingerprint_match: bool = True,
    cuad_doc_scope: bool = False,
    strict_doc_scope: bool = True,
    graph_edge_merge_mode: str = "global",
    graph_node_merge_mode: str = "normalized",
    graph_node_scope_with_type: bool = True,
    graph_query_level: int | None = None,
    triple_schema_file: str | None = "config_triple_schema.json",
    schema_apply_mode: str = "strict",
) -> dict[str, Any]:
    topk_list = list(topk_list or [10, 20])
    answer_modes = list(answer_modes or ["reject", "open"])
    if method_mode == "only_youtu" and not include_youtu:
        raise ValueError("method_mode=only_youtu requires --include-youtu")
    effective_include_youtu = bool(include_youtu and method_mode != "exclude_youtu")
    execution_mode = "only_youtu" if method_mode == "only_youtu" else "all"
    sync_mode_effective = youtu_sync_mode
    if execution_mode == "only_youtu" and str(sync_mode_effective).strip().lower() == "none":
        # youtu-only runs commonly pass JSONL corpus sources; shared_dir converts to backend corpus.json format safely.
        sync_mode_effective = "shared_dir"
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wide: dict[tuple[str, str, str], dict[str, Any]] = {}
    runs: list[dict[str, Any]] = []
    for top_k in topk_list:
        for mode in answer_modes:
            suffix = f"top{int(top_k)}_{mode}"
            compare_out = out_dir / f"compare_answers_{suffix}.jsonl"
            compare_metrics = out_dir / f"compare_metrics_{suffix}.json"
            eval_out = out_dir / f"eval_{suffix}.csv"
            eval_summary = out_dir / f"eval_{suffix}_summary.json"

            compare_summary = run_compare(
                queries_file=queries_file,
                chunks_file=chunks_file,
                idx_file=idx_file,
                store_file=store_file,
                triples_file=triples_file,
                graph_file=graph_file,
                communities_file=communities_file,
                top_k=int(top_k),
                out_file=str(compare_out),
                metrics_file=str(compare_metrics),
                regimes=regime,
                budget_config_file=budget_config_file,
                incremental_only=incremental_only,
                warmup_graphrag=False,
                max_queries=max_queries,
                include_lightrag=include_lightrag,
                lightrag_mode=lightrag_mode,
                lightrag_working_dir=lightrag_working_dir,
                lightrag_force_rebuild=lightrag_force_rebuild,
                include_youtu=effective_include_youtu,
                youtu_base_url=youtu_base_url,
                youtu_dataset=youtu_dataset,
                youtu_client_id=(str(youtu_client_id).strip() or None),
                youtu_route_type=(str(youtu_route_type).strip().lower() or None),
                youtu_schema_file=youtu_schema_file,
                youtu_graph_state_file=youtu_graph_state_file,
                youtu_reuse_graph=youtu_reuse_graph,
                youtu_force_rebuild=youtu_force_rebuild,
                youtu_sync_mode=sync_mode_effective,
                youtu_shared_corpus_dir=youtu_shared_corpus_dir,
                youtu_corpus_source_file=youtu_corpus_source_file,
                youtu_construct_poll_sec=youtu_construct_poll_sec,
                youtu_construct_timeout_sec=youtu_construct_timeout_sec,
                youtu_require_fingerprint_match=youtu_require_fingerprint_match,
                cuad_doc_scope=cuad_doc_scope,
                strict_doc_scope=strict_doc_scope,
                graph_edge_merge_mode=graph_edge_merge_mode,
                graph_node_merge_mode=graph_node_merge_mode,
                graph_node_scope_with_type=graph_node_scope_with_type,
                graph_query_level=graph_query_level,
                triple_schema_file=triple_schema_file,
                schema_apply_mode=schema_apply_mode,
                answer_mode=mode,
                answer_modes=None,
                execution_mode=execution_mode,
            )

            eval_payload = run_eval(
                pred_file=str(compare_out),
                gold_file=gold_file,
                out_csv=str(eval_out),
                out_summary=str(eval_summary),
                graph_file=graph_file,
                communities_file=communities_file,
                expand_community_chunks=True,
                make_plots=False,
                method_mode=method_mode,
                judge_mode=judge_mode,
                judge_model=judge_model,
            )
            dataset = str(compare_summary.get("dataset_tag", "unknown"))
            for item in eval_payload.get("summary", {}).get("overall", []):
                if str(item.get("regime", "")) != str(regime):
                    continue
                method = str(item.get("method", "")).strip()
                if not method:
                    continue
                key = (method, dataset, mode)
                row = wide.setdefault(
                    key,
                    {
                        "method": method,
                        "dataset": dataset,
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
                row[f"top{int(top_k)}_accuracy"] = item.get("topk_accuracy")
                row[f"tokens_top{int(top_k)}"] = item.get("total_tokens")
                row[f"time_top{int(top_k)}"] = item.get("latency_ms_total")
                if int(top_k) == 20:
                    row["tokens"] = item.get("total_tokens")
                    row["time"] = item.get("latency_ms_total")
                elif row.get("tokens") is None:
                    row["tokens"] = item.get("total_tokens")
                    row["time"] = item.get("latency_ms_total")
            runs.append(
                {
                    "mode": mode,
                    "top_k": int(top_k),
                    "compare_answers": str(compare_out),
                    "compare_metrics": str(compare_metrics),
                    "eval_csv": str(eval_out),
                    "eval_summary": str(eval_summary),
                }
            )

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

    csv_file = out_dir / "topk_mode_summary.csv"
    md_file = out_dir / "topk_mode_summary.md"
    df.to_csv(csv_file, index=False)
    _write_md_table(md_file, df)

    summary = {
        "queries_file": queries_file,
        "gold_file": gold_file,
        "regime": regime,
        "topk_list": [int(x) for x in topk_list],
        "answer_modes": answer_modes,
        "incremental_only": bool(incremental_only),
        "judge_mode": judge_mode,
        "judge_model": judge_model,
        "method_mode": method_mode,
        "cuad_doc_scope": bool(cuad_doc_scope),
        "strict_doc_scope": bool(strict_doc_scope),
        "include_youtu_requested": bool(include_youtu),
        "include_youtu_effective": bool(effective_include_youtu),
        "include_lightrag": bool(include_lightrag),
        "lightrag_mode": str(lightrag_mode),
        "lightrag_working_dir": str(lightrag_working_dir or ""),
        "lightrag_force_rebuild": bool(lightrag_force_rebuild),
        "execution_mode": execution_mode,
        "youtu_sync_mode_requested": youtu_sync_mode,
        "youtu_sync_mode_effective": sync_mode_effective,
        "youtu_route_type": (str(youtu_route_type).strip().lower() or ""),
        "youtu_client_id": (str(youtu_client_id).strip() or ""),
        "runs": runs,
        "summary_csv": str(csv_file),
        "summary_md": str(md_file),
        "num_rows": int(len(df)),
    }
    summary_file = out_dir / "topk_mode_summary.json"
    summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["summary_json"] = str(summary_file)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run compare+eval for TopK x answer_mode combinations.")
    parser.add_argument("--queries-file", default="data/queries/cuad_capability_queries.jsonl")
    parser.add_argument("--gold-file", default="data/queries/cuad_capability_gold.jsonl")
    parser.add_argument("--chunks-file", default="data/processed/chunks_sampled.jsonl")
    parser.add_argument("--idx-file", default="outputs/indexes/faiss_sampled.idx")
    parser.add_argument("--store-file", default="outputs/indexes/chunk_store_sampled.json")
    parser.add_argument("--triples-file", default="outputs/graph/triples.jsonl")
    parser.add_argument("--graph-file", default="outputs/graph/graph.json")
    parser.add_argument("--communities-file", default="outputs/graph/communities.json")
    parser.add_argument("--budget-config-file", default="config_budget.yaml")
    parser.add_argument("--topk-list", default="10,20")
    parser.add_argument("--answer-modes", default="reject,open")
    parser.add_argument("--regime", choices=["best_effort", "budget_matched"], default="best_effort")
    parser.add_argument("--method-mode", choices=["all", "exclude_youtu", "only_youtu"], default="all")
    parser.add_argument("--judge-mode", choices=["off", "llm_yesno"], default="off")
    parser.add_argument("--judge-model", default="qwen-flash")
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument(
        "--incremental-only",
        action="store_true",
        help="Only process unanswered queries and keep existing answers in compare_outputs",
    )
    parser.add_argument("--results-dir", default="outputs/results")
    parser.add_argument("--include-youtu", action="store_true")
    parser.add_argument("--include-lightrag", action="store_true")
    parser.add_argument("--lightrag-mode", choices=["local", "global", "hybrid", "naive", "mix", "bypass"], default="hybrid")
    parser.add_argument("--lightrag-working-dir", default="")
    parser.add_argument("--lightrag-force-rebuild", default="false")
    parser.add_argument("--youtu-base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--youtu-dataset", default="enterprise")
    parser.add_argument("--youtu-client-id", default="")
    parser.add_argument("--youtu-route-type", default="")
    parser.add_argument("--youtu-schema-file", default=None)
    parser.add_argument("--youtu-graph-state-file", default="outputs/graph/youtu_graph_state.json")
    parser.add_argument("--youtu-reuse-graph", default="true")
    parser.add_argument("--youtu-force-rebuild", default="false")
    parser.add_argument("--youtu-sync-mode", choices=["none", "shared_dir"], default="none")
    parser.add_argument("--youtu-shared-corpus-dir", default="outputs/youtu_sync")
    parser.add_argument("--youtu-corpus-source-file", default="")
    parser.add_argument("--youtu-construct-poll-sec", type=int, default=2)
    parser.add_argument("--youtu-construct-timeout-sec", type=int, default=1800)
    parser.add_argument("--youtu-require-fingerprint-match", default="true")
    parser.add_argument("--cuad-doc-scope", default="false")
    parser.add_argument("--strict-doc-scope", default="true")
    parser.add_argument("--graph-edge-merge-mode", choices=["global", "doc_scoped"], default="global")
    parser.add_argument("--graph-node-merge-mode", choices=["exact", "casefold", "normalized"], default="normalized")
    parser.add_argument("--graph-node-scope-with-type", default="true")
    parser.add_argument("--graph-query-level", type=int, default=None)
    parser.add_argument("--triple-schema-file", default="config_triple_schema.json")
    parser.add_argument("--schema-apply-mode", choices=["strict", "prompt_only", "disabled"], default="strict")
    args = parser.parse_args()

    summary = run_compare_topk_modes(
        queries_file=args.queries_file,
        gold_file=args.gold_file,
        chunks_file=args.chunks_file,
        idx_file=args.idx_file,
        store_file=args.store_file,
        triples_file=args.triples_file,
        graph_file=args.graph_file,
        communities_file=args.communities_file,
        budget_config_file=args.budget_config_file,
        topk_list=_parse_int_list(args.topk_list, [10, 20]),
        answer_modes=_parse_mode_list(args.answer_modes, ["reject", "open"]),
        regime=args.regime,
        include_youtu=args.include_youtu,
        include_lightrag=args.include_lightrag,
        lightrag_mode=args.lightrag_mode,
        lightrag_working_dir=(str(args.lightrag_working_dir).strip() or None),
        lightrag_force_rebuild=(
            str(args.lightrag_force_rebuild).strip().lower() in {"1", "true", "yes", "y", "on"}
        ),
        method_mode=args.method_mode,
        judge_mode=args.judge_mode,
        judge_model=args.judge_model,
        max_queries=args.max_queries,
        incremental_only=args.incremental_only,
        results_dir=args.results_dir,
        youtu_base_url=args.youtu_base_url,
        youtu_dataset=args.youtu_dataset,
        youtu_client_id=(str(args.youtu_client_id).strip() or None),
        youtu_route_type=(str(args.youtu_route_type).strip().lower() or None),
        youtu_schema_file=(str(args.youtu_schema_file).strip() or None),
        youtu_graph_state_file=args.youtu_graph_state_file,
        youtu_reuse_graph=str(args.youtu_reuse_graph).strip().lower() in {"1", "true", "yes", "y", "on"},
        youtu_force_rebuild=str(args.youtu_force_rebuild).strip().lower() in {"1", "true", "yes", "y", "on"},
        youtu_sync_mode=args.youtu_sync_mode,
        youtu_shared_corpus_dir=args.youtu_shared_corpus_dir,
        youtu_corpus_source_file=(str(args.youtu_corpus_source_file).strip() or None),
        youtu_construct_poll_sec=args.youtu_construct_poll_sec,
        youtu_construct_timeout_sec=args.youtu_construct_timeout_sec,
        youtu_require_fingerprint_match=(
            str(args.youtu_require_fingerprint_match).strip().lower() in {"1", "true", "yes", "y", "on"}
        ),
        cuad_doc_scope=str(args.cuad_doc_scope).strip().lower() in {"1", "true", "yes", "y", "on"},
        strict_doc_scope=str(args.strict_doc_scope).strip().lower() in {"1", "true", "yes", "y", "on"},
        graph_edge_merge_mode=args.graph_edge_merge_mode,
        graph_node_merge_mode=args.graph_node_merge_mode,
        graph_node_scope_with_type=(
            str(args.graph_node_scope_with_type).strip().lower() in {"1", "true", "yes", "y", "on"}
        ),
        graph_query_level=args.graph_query_level,
        triple_schema_file=args.triple_schema_file,
        schema_apply_mode=args.schema_apply_mode,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
