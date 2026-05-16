#!/usr/bin/env python3
"""Plot grouped RAGAS summary metrics from ragas_eval_summary.csv."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DEFAULT_CSV = Path(
    "outputs/results/ragas/merged_vector_prued_all_test/ragas_eval_summary.csv"
)

METRIC_GROUPS = {
    "retrieval": ["context_recall", "context_precision", "context_mrr"],
    "answer_quality": ["faithfulness", "answer_correctness"],
    "semantic": ["answer_relevance", "semantic_similarity"],
}

METHOD_LABELS = {
    "vector_rag": "VectorRAG",
    "graph_rag": "GraphRAG",
    "lightrag": "LightRAG",
    "E-GraphRAG": "E-GraphRAG",
}

SYNTHESIZER_LABELS = {
    "multi_hop_abstract_query_synthesizer": "Global",
    "multi_hop_specific_query_synthesizer": "Structure",
    "single_hop_specific_query_synthesizer": "Local",
}

FACET_ORDER = ["Local", "Structure", "Global", "Overall"]

PALETTE = {
    "VectorRAG": "#8DB7E8",
    "GraphRAG": "#7CCBC4",
    "LightRAG": "#A9D18E",
    "E-GraphRAG": "#F6EE7A",
}

FONT_SIZES = {
    "subplot_title": 17,
    "axis_label": 16,
    "tick_label": 14,
    "legend": 14,
    "figure_title": 19,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize RAGAS evaluation summary metrics as grouped bar charts."
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Input CSV path.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to a 'figures' folder next to the CSV.",
    )
    parser.add_argument(
        "--level",
        choices=["synthesizer", "method"],
        default="synthesizer",
        help=(
            "Plot per synthesizer as multiple subplots, or only overall method rows. "
            "The reference image corresponds to 'synthesizer'."
        ),
    )
    parser.add_argument(
        "--format",
        default="png",
        choices=["png", "pdf", "svg"],
        help="Output image format.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Output DPI for raster images.")
    return parser.parse_args()


def prepare_data(df: pd.DataFrame, level: str) -> pd.DataFrame:
    group_type = "method_synthesizer" if level == "synthesizer" else "method"
    plot_df = df.loc[df["group_type"].eq(group_type)].copy()
    if plot_df.empty:
        raise ValueError(f"No rows found for group_type={group_type!r}.")

    plot_df["method_label"] = plot_df["method"].map(METHOD_LABELS).fillna(plot_df["method"])
    if level == "synthesizer":
        plot_df["facet"] = (
            plot_df["synthesizer_name"]
            .map(SYNTHESIZER_LABELS)
            .fillna(plot_df["synthesizer_name"])
        )
    else:
        plot_df["facet"] = "Overall"
    return plot_df


def plot_metric_group(
    data: pd.DataFrame,
    metrics: list[str],
    title: str,
    out_path: Path,
    dpi: int,
) -> None:
    missing = [metric for metric in metrics if metric not in data.columns]
    if missing:
        raise ValueError(f"Missing metric column(s): {', '.join(missing)}")

    available_facets = list(dict.fromkeys(data["facet"].dropna()))
    facets = [facet for facet in FACET_ORDER if facet in available_facets]
    facets.extend(facet for facet in available_facets if facet not in facets)
    fig, axes = plt.subplots(
        1,
        len(facets),
        figsize=(4.8 * len(facets), 3.6),
        sharey=True,
        constrained_layout=True,
    )
    if len(facets) == 1:
        axes = [axes]

    method_order = [label for label in PALETTE if label in set(data["method_label"])]

    for ax, facet in zip(axes, facets):
        facet_df = data.loc[data["facet"].eq(facet)]
        long_df = facet_df.melt(
            id_vars=["method_label"],
            value_vars=metrics,
            var_name="metric",
            value_name="score",
        )
        sns.barplot(
            data=long_df,
            x="metric",
            y="score",
            hue="method_label",
            hue_order=method_order,
            palette=PALETTE,
            ax=ax,
            edgecolor="white",
            linewidth=0.8,
        )
        ax.set_title(str(facet), fontsize=FONT_SIZES["subplot_title"])
        ax.set_xlabel("")
        ax.set_ylabel(
            "Score" if ax is axes[0] else "",
            fontsize=FONT_SIZES["axis_label"],
        )
        ax.set_ylim(0, 1.05)
        ax.tick_params(
            axis="x",
            labelrotation=18,
            labelsize=FONT_SIZES["tick_label"],
        )
        ax.tick_params(axis="y", labelsize=FONT_SIZES["tick_label"])
        ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
        ax.set_axisbelow(True)
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(labels),
        frameon=False,
        bbox_to_anchor=(0.5, 1.08),
        fontsize=FONT_SIZES["legend"],
    )
    fig.suptitle(title, y=1.16, fontsize=FONT_SIZES["figure_title"])
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    csv_path = args.csv.expanduser().resolve()
    out_dir = (args.out_dir or csv_path.parent / "figures").expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    plot_df = prepare_data(df, args.level)

    for group_name, metrics in METRIC_GROUPS.items():
        out_path = out_dir / f"ragas_{group_name}_{args.level}.{args.format}"
        title = group_name.replace("_", " ").title()
        plot_metric_group(plot_df, metrics, title, out_path, args.dpi)
        print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
