from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from evaluation.metrics import similarity

METHODS = ("vector_rag", "kg_rag", "graph_rag")
PLOT_STYLE = "whitegrid"


def _read_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_json(path: str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _norm_text(text: Any) -> str:
    text = "" if text is None else str(text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _binary(value: bool) -> int:
    return 1 if value else 0


def _match_scores(pred: str, gold: str) -> dict:
    p = _norm_text(pred)
    g = _norm_text(gold)
    exact = p == g and bool(g)
    contains = bool(g) and (g in p or p in g)
    return {
        "answer_similarity": round(float(similarity(str(pred or ""), str(gold or ""))), 6),
        "answer_exact": _binary(exact),
        "answer_contains": _binary(contains),
    }


def _precision_recall_f1(pred_set: set[str], gold_set: set[str]) -> tuple[float, float, float]:
    if not gold_set:
        return 0.0, 0.0, 0.0
    if not pred_set:
        return 0.0, 0.0, 0.0
    hit = len(pred_set & gold_set)
    if hit == 0:
        return 0.0, 0.0, 0.0
    p = hit / len(pred_set)
    r = hit / len(gold_set)
    f1 = 2 * p * r / (p + r)
    return p, r, f1


def _collect_pred_ids(method_payload: dict, community_chunk_map: dict[str, set[str]] | None = None) -> dict[str, set[str]]:
    chunk_ids = set()
    edge_ids = set()
    community_ids = set()

    for e in method_payload.get("evidence", []) or []:
        if isinstance(e, dict):
            cid = str(e.get("chunk_id", "")).strip()
            if cid:
                chunk_ids.add(cid)
            comm = str(e.get("community_id", "")).strip()
            if comm:
                community_ids.add(comm)

    for e in method_payload.get("subgraph_edges", []) or []:
        if isinstance(e, dict):
            eid = str(e.get("edge_id", "")).strip()
            if eid:
                edge_ids.add(eid)

    for c in method_payload.get("communities", []) or []:
        cid = str(c).strip()
        if cid:
            community_ids.add(cid)

    for e in method_payload.get("evidence_chunks", []) or []:
        if isinstance(e, dict):
            cid = str(e.get("chunk_id", "")).strip()
            if cid:
                chunk_ids.add(cid)

    if community_chunk_map:
        for cid in community_ids:
            chunk_ids.update(community_chunk_map.get(cid, set()))

    return {
        "chunks": chunk_ids,
        "edges": edge_ids,
        "communities": community_ids,
    }


def _build_community_chunk_map(graph_file: str | None, communities_file: str | None) -> dict[str, set[str]]:
    if not graph_file or not communities_file:
        return {}
    gp = Path(graph_file)
    cp = Path(communities_file)
    if not gp.exists() or not cp.exists():
        return {}

    graph = _load_json(str(gp))
    communities_payload = _load_json(str(cp))
    communities = communities_payload if isinstance(communities_payload, list) else communities_payload.get("communities", [])
    edge_by_id = {str(e.get("edge_id")): e for e in graph.get("edges", []) if e.get("edge_id")}

    out: dict[str, set[str]] = {}
    for c in communities:
        cid = str(c.get("community_id", "")).strip()
        if not cid:
            continue
        chunk_ids: set[str] = set()
        for eid in c.get("edges", []) or []:
            edge = edge_by_id.get(str(eid))
            if not edge:
                continue
            for m in edge.get("mentions", []) or []:
                chunk_id = str(m.get("chunk_id", "")).strip()
                if chunk_id:
                    chunk_ids.add(chunk_id)
        if chunk_ids:
            out[cid] = chunk_ids
    return out


def _collect_gold_ids(gold_row: dict) -> dict[str, set[str]]:
    chunk_ids = {
        str(x.get("chunk_id", "")).strip()
        for x in gold_row.get("supporting_chunks", []) or []
        if str(x.get("chunk_id", "")).strip()
    }
    edge_ids = {
        str(x.get("edge_id", "")).strip()
        for x in gold_row.get("supporting_edges", []) or []
        if str(x.get("edge_id", "")).strip()
    }
    community_ids = {
        str(x.get("community_id", "")).strip()
        for x in gold_row.get("supporting_communities", []) or []
        if str(x.get("community_id", "")).strip()
    }
    return {
        "chunks": chunk_ids,
        "edges": edge_ids,
        "communities": community_ids,
    }


def _has_compare_layout(row: dict) -> bool:
    regimes = row.get("regimes")
    return isinstance(regimes, dict) and bool(regimes)


def _detect_compare_methods(pred_rows: list[dict]) -> tuple[str, ...]:
    found: set[str] = set()
    for row in pred_rows:
        if not isinstance(row, dict):
            continue
        regimes = row.get("regimes")
        if not isinstance(regimes, dict):
            continue
        for by_method in regimes.values():
            if not isinstance(by_method, dict):
                continue
            for method in by_method.keys():
                m = str(method).strip()
                if m:
                    found.add(m)
    if found:
        return tuple(sorted(found))
    return METHODS


def _eval_compare(
    pred_rows: list[dict],
    gold_map: dict[str, dict],
    include_evidence: bool,
    methods: tuple[str, ...],
    community_chunk_map: dict[str, set[str]] | None = None,
) -> list[dict]:
    res = []
    for row in pred_rows:
        qid = str(row.get("qid"))
        gold = gold_map.get(qid)
        if not gold:
            continue

        regimes = row.get("regimes", {}) or {}
        for regime, by_method in regimes.items():
            for method in methods:
                payload = (by_method or {}).get(method, {}) or {}
                pred_answer = str(payload.get("answer", "") or "")
                gold_answer = str(gold.get("answer", "") or "")
                base = {
                    "qid": qid,
                    "type": str(row.get("type", gold.get("type", "unknown"))),
                    "query": str(row.get("query", gold.get("query", ""))),
                    "regime": str(regime),
                    "method": method,
                    "gold_answer": gold_answer,
                    "pred_answer": pred_answer,
                }
                base.update(_match_scores(pred_answer, gold_answer))

                budget = payload.get("budget_check") or {}
                base["budget_within"] = budget.get("within_budget")
                base["budget_error"] = budget.get("error")

                if include_evidence:
                    pred_ids = _collect_pred_ids(payload, community_chunk_map=community_chunk_map)
                    gold_ids = _collect_gold_ids(gold)
                    p_c, r_c, f_c = _precision_recall_f1(pred_ids["chunks"], gold_ids["chunks"])
                    p_e, r_e, f_e = _precision_recall_f1(pred_ids["edges"], gold_ids["edges"])
                    p_m, r_m, f_m = _precision_recall_f1(pred_ids["communities"], gold_ids["communities"])
                    p_a, r_a, f_a = _precision_recall_f1(
                        pred_ids["chunks"] | pred_ids["edges"] | pred_ids["communities"],
                        gold_ids["chunks"] | gold_ids["edges"] | gold_ids["communities"],
                    )
                    base.update(
                        {
                            "evidence_precision_chunks": round(p_c, 6),
                            "evidence_recall_chunks": round(r_c, 6),
                            "evidence_f1_chunks": round(f_c, 6),
                            "evidence_precision_edges": round(p_e, 6),
                            "evidence_recall_edges": round(r_e, 6),
                            "evidence_f1_edges": round(f_e, 6),
                            "evidence_precision_communities": round(p_m, 6),
                            "evidence_recall_communities": round(r_m, 6),
                            "evidence_f1_communities": round(f_m, 6),
                            "evidence_precision_all": round(p_a, 6),
                            "evidence_recall_all": round(r_a, 6),
                            "evidence_f1_all": round(f_a, 6),
                            "pred_chunks": len(pred_ids["chunks"]),
                            "pred_edges": len(pred_ids["edges"]),
                            "pred_communities": len(pred_ids["communities"]),
                            "gold_chunks": len(gold_ids["chunks"]),
                            "gold_edges": len(gold_ids["edges"]),
                            "gold_communities": len(gold_ids["communities"]),
                        }
                    )
                res.append(base)
    return res


def _eval_legacy(pred_rows: list[dict], gold_map: dict[str, dict]) -> list[dict]:
    res = []
    for row in pred_rows:
        qid = str(row.get("qid"))
        gold = gold_map.get(qid)
        if not gold:
            continue
        pred_answer = str(row.get("answer", "") or "")
        gold_answer = str(gold.get("answer", "") or "")
        item = {
            "qid": qid,
            "type": str(gold.get("type", "unknown")),
            "query": str(gold.get("query", "")),
            "regime": "legacy",
            "method": "legacy",
            "gold_answer": gold_answer,
            "pred_answer": pred_answer,
        }
        item.update(_match_scores(pred_answer, gold_answer))
        res.append(item)
    return res


def _build_summary(rows: list[dict]) -> dict:
    df = pd.DataFrame(rows)
    if df.empty:
        return {"num_rows": 0, "overall": [], "by_type": []}

    metric_cols = [
        c
        for c in (
            "answer_similarity",
            "answer_exact",
            "answer_contains",
            "evidence_precision_all",
            "evidence_recall_all",
            "evidence_f1_all",
            "evidence_recall_chunks",
            "evidence_recall_edges",
            "evidence_recall_communities",
        )
        if c in df.columns
    ]

    agg = {m: "mean" for m in metric_cols}
    agg["qid"] = "count"
    overall = (
        df.groupby(["regime", "method"], dropna=False)
        .agg(agg)
        .rename(columns={"qid": "num_queries"})
        .reset_index()
    )

    by_type = (
        df.groupby(["regime", "method", "type"], dropna=False)
        .agg(agg)
        .rename(columns={"qid": "num_queries"})
        .reset_index()
    )

    if "budget_within" in df.columns:
        budget_numeric = pd.to_numeric(df["budget_within"], errors="coerce")
        bw = (
            pd.DataFrame({"regime": df["regime"], "method": df["method"], "budget_within": budget_numeric})
            .groupby(["regime", "method"], dropna=False)["budget_within"]
            .mean()
            .reset_index()
            .rename(columns={"budget_within": "budget_within_rate"})
        )
        overall = overall.merge(bw, on=["regime", "method"], how="left")

    for d in (overall, by_type):
        for c in d.columns:
            if c not in ("regime", "method", "type", "num_queries"):
                d[c] = d[c].astype(float).round(6)

    return {
        "num_rows": int(len(df)),
        "overall": overall.to_dict(orient="records"),
        "by_type": by_type.to_dict(orient="records"),
    }


def _dedup_compare_eval_rows(rows: list[dict]) -> list[dict]:
    """Keep the latest row for each qid+regime+method to avoid incremental duplicates."""
    dedup: dict[tuple[str, str, str], dict] = {}
    ordered_keys: list[tuple[str, str, str]] = []
    for row in rows:
        key = (
            str(row.get("qid", "")),
            str(row.get("regime", "")),
            str(row.get("method", "")),
        )
        if key not in dedup:
            ordered_keys.append(key)
        dedup[key] = row
    return [dedup[k] for k in ordered_keys]


def _plot_metric_bar(
    df: pd.DataFrame,
    metric: str,
    title: str,
    out_file: Path,
    hue: str = "regime",
) -> None:
    if metric not in df.columns:
        return
    plt.figure(figsize=(8, 4.8))
    ax = sns.barplot(data=df, x="method", y=metric, hue=hue)
    ax.set_title(title)
    ax.set_xlabel("Method")
    ax.set_ylabel(metric)
    ax.tick_params(axis="x", rotation=0)
    plt.tight_layout()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=160)
    plt.close()


def _generate_plots(summary: dict, plots_dir: str) -> list[str]:
    out_paths: list[str] = []
    payload = summary.get("summary", {})
    overall = pd.DataFrame(payload.get("overall", []))
    by_type = pd.DataFrame(payload.get("by_type", []))
    out_dir = Path(plots_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not overall.empty:
        metrics = [
            ("answer_similarity", "Answer Similarity by Method/Regime", "answer_similarity.png"),
            ("answer_contains", "Answer Contains Rate by Method/Regime", "answer_contains.png"),
            ("answer_exact", "Answer Exact Match Rate by Method/Regime", "answer_exact.png"),
            ("evidence_recall_chunks", "Evidence Chunk Recall by Method/Regime", "evidence_recall_chunks.png"),
            ("evidence_recall_edges", "Evidence Edge Recall by Method/Regime", "evidence_recall_edges.png"),
            ("evidence_recall_all", "Evidence Recall (All IDs) by Method/Regime", "evidence_recall_all.png"),
            ("budget_within_rate", "Budget Within Rate by Method/Regime", "budget_within_rate.png"),
        ]
        for metric, title, filename in metrics:
            out_file = out_dir / filename
            _plot_metric_bar(overall, metric=metric, title=title, out_file=out_file)
            if out_file.exists():
                out_paths.append(str(out_file))

    if not by_type.empty and "answer_similarity" in by_type.columns:
        g = sns.catplot(
            data=by_type,
            x="method",
            y="answer_similarity",
            hue="regime",
            col="type",
            kind="bar",
            height=4,
            aspect=1,
            col_wrap=2,
            sharey=True,
        )
        g.fig.subplots_adjust(top=0.9)
        g.fig.suptitle("Answer Similarity by Type / Method / Regime")
        out_file = out_dir / "answer_similarity_by_type.png"
        g.savefig(out_file, dpi=160)
        plt.close(g.fig)
        if out_file.exists():
            out_paths.append(str(out_file))

    if not by_type.empty and "evidence_recall_chunks" in by_type.columns:
        g = sns.catplot(
            data=by_type,
            x="method",
            y="evidence_recall_chunks",
            hue="regime",
            col="type",
            kind="bar",
            height=4,
            aspect=1,
            col_wrap=2,
            sharey=True,
        )
        g.fig.subplots_adjust(top=0.9)
        g.fig.suptitle("Evidence Chunk Recall by Type / Method / Regime")
        out_file = out_dir / "evidence_recall_chunks_by_type.png"
        g.savefig(out_file, dpi=160)
        plt.close(g.fig)
        if out_file.exists():
            out_paths.append(str(out_file))

    return out_paths


def run_eval(
    pred_file: str,
    gold_file: str,
    out_csv: str,
    out_summary: str,
    graph_file: str | None = None,
    communities_file: str | None = None,
    expand_community_chunks: bool = True,
    make_plots: bool = True,
    plots_dir: str = "outputs/results/eval_plots",
) -> dict:
    sns.set_theme(style=PLOT_STYLE)
    pred_rows = _read_jsonl(pred_file)
    gold_rows = _read_jsonl(gold_file)
    gold_map = {str(x.get("qid")): x for x in gold_rows if x.get("qid")}
    include_evidence = any("supporting_chunks" in x or "supporting_edges" in x for x in gold_rows)
    community_chunk_map = (
        _build_community_chunk_map(graph_file=graph_file, communities_file=communities_file)
        if expand_community_chunks
        else {}
    )

    has_compare_layout = any(isinstance(x, dict) and _has_compare_layout(x) for x in pred_rows)
    if has_compare_layout:
        methods = _detect_compare_methods(pred_rows)
        rows = _eval_compare(
            pred_rows,
            gold_map,
            include_evidence=include_evidence,
            methods=methods,
            community_chunk_map=community_chunk_map,
        )
        rows = _dedup_compare_eval_rows(rows)
    else:
        methods = ("legacy",)
        rows = _eval_legacy(pred_rows, gold_map)

    out_csv_path = Path(out_csv)
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    summary = {
        "pred_file": pred_file,
        "gold_file": gold_file,
        "out_csv": out_csv,
        "num_pred_rows": len(pred_rows),
        "num_gold_rows": len(gold_rows),
        "mode": "compare" if has_compare_layout else "legacy",
        "methods": list(methods),
        "expand_community_chunks": bool(expand_community_chunks),
        "community_chunk_map_size": len(community_chunk_map),
        "summary": _build_summary(rows),
    }
    if make_plots:
        summary["plots"] = _generate_plots(summary, plots_dir=plots_dir)
        summary["plots_dir"] = plots_dir
    out_summary_path = Path(out_summary)
    out_summary_path.parent.mkdir(parents=True, exist_ok=True)
    out_summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate comparison answers against gold references.")
    parser.add_argument("--pred-file", default="outputs/results/compare_answers.jsonl")
    parser.add_argument("--gold-file", default="data/queries/gold_qa.jsonl")
    parser.add_argument("--out-csv", default="outputs/results/eval_compare.csv")
    parser.add_argument("--out-summary", default="outputs/results/eval_compare_summary.json")
    parser.add_argument("--graph-file", default="outputs/graph/graph.json")
    parser.add_argument("--communities-file", default="outputs/graph/communities.json")
    parser.add_argument("--disable-community-chunk-expand", action="store_true")
    parser.add_argument("--disable-plots", action="store_true")
    parser.add_argument("--plots-dir", default="outputs/results/eval_plots")
    args = parser.parse_args()

    summary = run_eval(
        pred_file=args.pred_file,
        gold_file=args.gold_file,
        out_csv=args.out_csv,
        out_summary=args.out_summary,
        graph_file=args.graph_file,
        communities_file=args.communities_file,
        expand_community_chunks=not args.disable_community_chunk_expand,
        make_plots=not args.disable_plots,
        plots_dir=args.plots_dir,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
