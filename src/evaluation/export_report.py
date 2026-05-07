from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _compute_pareto_frontier(df: pd.DataFrame, token_col: str, acc_col: str) -> pd.DataFrame:
    work = df[[token_col, acc_col]].copy()
    work[token_col] = pd.to_numeric(work[token_col], errors="coerce")
    work[acc_col] = pd.to_numeric(work[acc_col], errors="coerce")
    work = work.dropna().sort_values(token_col, ascending=True)
    if work.empty:
        return work
    keep_idx = []
    best_acc = float("-inf")
    for idx, row in work.iterrows():
        acc = float(row[acc_col])
        if acc > best_acc:
            keep_idx.append(idx)
            best_acc = acc
    return df.loc[keep_idx].sort_values(token_col, ascending=True)


def _to_number_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(dtype="float64")
    return pd.to_numeric(df[col], errors="coerce")


def _safe_mean(df: pd.DataFrame, value_col: str, topk_col: str, topk: int) -> float | None:
    if value_col not in df.columns:
        return None
    if topk_col in df.columns:
        vals = pd.to_numeric(df.loc[df[topk_col] == topk, value_col], errors="coerce").dropna()
    else:
        vals = pd.to_numeric(df[value_col], errors="coerce").dropna()
    if vals.empty:
        return None
    return float(vals.mean())


def _safe_sum(df: pd.DataFrame, value_col: str, topk_col: str, topk: int) -> float | None:
    if value_col not in df.columns:
        return None
    if topk_col in df.columns:
        vals = pd.to_numeric(df.loc[df[topk_col] == topk, value_col], errors="coerce").dropna()
    else:
        vals = pd.to_numeric(df[value_col], errors="coerce").dropna()
    if vals.empty:
        return None
    return float(vals.sum())


def _from_eval_rows(df: pd.DataFrame) -> pd.DataFrame:
    required = {"method", "mode", "topk_accuracy", "total_tokens", "latency_ms_total"}
    missing = sorted([c for c in required if c not in df.columns])
    if missing:
        raise ValueError(f"Eval-style CSV missing required columns: {missing}")

    work = df.copy()
    if "dataset" not in work.columns:
        work["dataset"] = None
    if "regime" not in work.columns:
        work["regime"] = None
    if "top_k" not in work.columns:
        work["top_k"] = None

    group_cols = ["method", "dataset", "mode", "regime"]
    regimes = [str(x).strip() for x in work["regime"].dropna().unique() if str(x).strip()]
    include_regime_in_name = len(regimes) > 1

    rows = []
    for keys, g in work.groupby(group_cols, dropna=False):
        key_map = dict(zip(group_cols, keys))
        top10 = _safe_mean(g, "topk_accuracy", "top_k", 10)
        top20 = _safe_mean(g, "topk_accuracy", "top_k", 20)
        if top10 is None and top20 is None:
            fallback_acc = pd.to_numeric(g["topk_accuracy"], errors="coerce").dropna()
            top10 = float(fallback_acc.mean()) if not fallback_acc.empty else None

        tokens_top10 = _safe_sum(g, "total_tokens", "top_k", 10)
        tokens_top20 = _safe_sum(g, "total_tokens", "top_k", 20)
        time_top10 = _safe_sum(g, "latency_ms_total", "top_k", 10)
        time_top20 = _safe_sum(g, "latency_ms_total", "top_k", 20)

        tokens = tokens_top20 if tokens_top20 is not None else tokens_top10
        time = time_top20 if time_top20 is not None else time_top10
        if tokens is None:
            tok_any = pd.to_numeric(g["total_tokens"], errors="coerce").dropna()
            tokens = float(tok_any.sum()) if not tok_any.empty else None
        if time is None:
            t_any = pd.to_numeric(g["latency_ms_total"], errors="coerce").dropna()
            time = float(t_any.sum()) if not t_any.empty else None

        method = str(key_map["method"])
        regime = str(key_map["regime"] or "").strip()
        if include_regime_in_name and regime:
            method = f"{method}[{regime}]"

        rows.append(
            {
                "method": method,
                "dataset": key_map["dataset"],
                "mode": key_map["mode"],
                "top20_accuracy": top20,
                "top10_accuracy": top10,
                "tokens": tokens,
                "time": time,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    for c in ("top20_accuracy", "top10_accuracy", "tokens", "time"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.sort_values(["dataset", "mode", "method"], na_position="last").reset_index(drop=True)


def _to_report_df(df: pd.DataFrame) -> pd.DataFrame:
    if {"Method", "Dataset", "Mode", "Top20", "Top10", "Tokens", "Time"}.issubset(df.columns):
        return df[["Method", "Dataset", "Mode", "Top20", "Top10", "Tokens", "Time"]].copy()

    is_eval_style = {"method", "mode", "topk_accuracy", "total_tokens", "latency_ms_total"}.issubset(df.columns)
    if is_eval_style:
        df = _from_eval_rows(df)

    cols = {
        "method": "Method",
        "dataset": "Dataset",
        "mode": "Mode",
        "top20_accuracy": "Top20",
        "top10_accuracy": "Top10",
        "tokens": "Tokens",
        "time": "Time",
    }
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = None
    out = out[list(cols.keys())].rename(columns=cols)
    return out


def _write_md(path: Path, report_df: pd.DataFrame) -> None:
    lines = [
        "| Method | Dataset | Mode | Top20 | Top10 | Tokens | Time |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for _, row in report_df.iterrows():
        lines.append(
            f"| {row['Method']} | {row['Dataset']} | {row['Mode']} | "
            f"{row['Top20']} | {row['Top10']} | {row['Tokens']} | {row['Time']} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_report(
    in_csv: str = "outputs/results/topk_mode_summary.csv",
    out_csv: str = "outputs/results/final_report.csv",
    out_md: str = "outputs/results/final_report.md",
    pareto_out: str = "outputs/results/pareto_frontier.png",
    accuracy_col: str = "top20_accuracy",
) -> dict:
    src = Path(in_csv)
    if not src.exists():
        raise FileNotFoundError(f"Input CSV not found: {in_csv}")
    df = pd.read_csv(src)
    report_df = _to_report_df(df)

    out_csv_p = Path(out_csv)
    out_md_p = Path(out_md)
    pareto_p = Path(pareto_out)
    out_csv_p.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(out_csv_p, index=False)
    _write_md(out_md_p, report_df)

    token_col = "Tokens"
    requested_acc_col = "Top20" if accuracy_col == "top20_accuracy" else "Top10"
    acc_col = requested_acc_col
    acc_vals = _to_number_series(report_df, acc_col).dropna()
    if acc_vals.empty:
        fallback = "Top10" if requested_acc_col == "Top20" else "Top20"
        fallback_vals = _to_number_series(report_df, fallback).dropna()
        if not fallback_vals.empty:
            acc_col = fallback
    pareto_df = _compute_pareto_frontier(report_df, token_col=token_col, acc_col=acc_col)

    plt.figure(figsize=(8.5, 5.5))
    for _, row in report_df.iterrows():
        x = pd.to_numeric(row.get(token_col), errors="coerce")
        y = pd.to_numeric(row.get(acc_col), errors="coerce")
        if pd.isna(x) or pd.isna(y):
            continue
        label = f"{row.get('Method', '')}-{row.get('Mode', '')}"
        plt.scatter(float(x), float(y), s=36, alpha=0.8)
        plt.annotate(label, (float(x), float(y)), fontsize=8, xytext=(4, 4), textcoords="offset points")
    if not pareto_df.empty:
        px = pd.to_numeric(pareto_df[token_col], errors="coerce")
        py = pd.to_numeric(pareto_df[acc_col], errors="coerce")
        m = (~px.isna()) & (~py.isna())
        plt.plot(px[m], py[m], linewidth=2.0)
    plt.xlabel("Tokens")
    plt.ylabel("Accuracy")
    plt.title(f"Pareto Frontier ({acc_col})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    pareto_p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(pareto_p, dpi=180)
    plt.close()

    return {
        "input_csv": str(src),
        "final_report_csv": str(out_csv_p),
        "final_report_md": str(out_md_p),
        "pareto_frontier_png": str(pareto_p),
        "accuracy_col": accuracy_col,
        "accuracy_col_effective": ("top20_accuracy" if acc_col == "Top20" else "top10_accuracy"),
        "num_rows": int(len(report_df)),
        "pareto_points": int(len(pareto_df)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export final report table and Pareto frontier plot.")
    parser.add_argument("--in-csv", default="outputs/results/topk_mode_summary.csv")
    parser.add_argument("--out-csv", default="outputs/results/final_report.csv")
    parser.add_argument("--out-md", default="outputs/results/final_report.md")
    parser.add_argument("--pareto-out", default="outputs/results/pareto_frontier.png")
    parser.add_argument("--accuracy-col", choices=["top20_accuracy", "top10_accuracy"], default="top20_accuracy")
    args = parser.parse_args()
    result = export_report(
        in_csv=args.in_csv,
        out_csv=args.out_csv,
        out_md=args.out_md,
        pareto_out=args.pareto_out,
        accuracy_col=args.accuracy_col,
    )
    print(result)


if __name__ == "__main__":
    main()
