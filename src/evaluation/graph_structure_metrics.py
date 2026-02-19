from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

try:
    import igraph as ig
except Exception as exc:  # pragma: no cover
    ig = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


def _load_json(path: str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


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


def _stats(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "min": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0, "mean": 0.0}
    vals = [float(x) for x in values]
    return {
        "count": int(len(vals)),
        "min": round(min(vals), 6),
        "p50": round(_percentile(vals, 0.5), 6),
        "p95": round(_percentile(vals, 0.95), 6),
        "max": round(max(vals), 6),
        "mean": round(sum(vals) / len(vals), 6),
    }


def _build_igraph(graph_payload: dict) -> Any:
    if ig is None:
        raise RuntimeError("python-igraph is required to compute graph structure metrics.") from _IMPORT_ERROR
    node_names = []
    for n in graph_payload.get("nodes", []):
        node_names.append(n["id"] if isinstance(n, dict) else str(n))
    node_names = sorted(set(node_names))
    node_to_idx = {n: i for i, n in enumerate(node_names)}

    g = ig.Graph(directed=False)
    g.add_vertices(len(node_names))
    g.vs["name"] = node_names

    edge_weights: dict[tuple[int, int], float] = {}
    for e in graph_payload.get("edges", []):
        s = str(e.get("source", "")).strip()
        t = str(e.get("target", "")).strip()
        if not s or not t or s == t:
            continue
        if s not in node_to_idx or t not in node_to_idx:
            continue
        a, b = sorted((node_to_idx[s], node_to_idx[t]))
        edge_weights[(a, b)] = edge_weights.get((a, b), 0.0) + float(e.get("weight", 1.0))

    edge_list = list(edge_weights.keys())
    g.add_edges(edge_list)
    g.es["weight"] = [edge_weights[k] for k in edge_list]
    return g


def _extract_community_sizes(communities_payload: Any) -> list[int]:
    communities = communities_payload if isinstance(communities_payload, list) else communities_payload.get("communities", [])
    sizes = []
    for c in communities:
        size = c.get("size")
        if size is None:
            size = len(c.get("nodes", []))
        try:
            iv = int(size)
        except Exception:
            continue
        if iv >= 0:
            sizes.append(iv)
    return sizes


def _plot_hist(values: list[float], title: str, xlabel: str, out_file: Path, bins: int = 30) -> str | None:
    if not values:
        return None
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4.8))
    plt.hist(values, bins=bins, edgecolor="black", alpha=0.85)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_file, dpi=160)
    plt.close()
    return str(out_file)


def compute_graph_structure_metrics(
    graph_file: str,
    communities_file: str,
    out_json: str | None = None,
    out_dir: str = "outputs/results/graph_plots",
) -> dict:
    graph_payload = _load_json(graph_file)
    communities_payload = _load_json(communities_file)
    g = _build_igraph(graph_payload)

    n = int(g.vcount())
    m = int(g.ecount())
    degrees = [int(x) for x in g.degree()] if n > 0 else []
    weights = [float(x) for x in (g.es["weight"] if m > 0 else [])]
    community_sizes = _extract_community_sizes(communities_payload)

    if n > 0:
        avg_degree = float((2 * m) / n)
        components = g.connected_components()
        largest_cc = max((len(c) for c in components), default=0)
        largest_cc_ratio = float(largest_cc / n)
        degree_one_ratio = float(sum(1 for d in degrees if d == 1) / n)
        clustering_global = float(g.transitivity_undirected(mode="zero"))
        clustering_avg_local = float(g.transitivity_avglocal_undirected(mode="zero"))
    else:
        avg_degree = 0.0
        largest_cc_ratio = 0.0
        degree_one_ratio = 0.0
        clustering_global = 0.0
        clustering_avg_local = 0.0

    out_dir_path = Path(out_dir)
    out_paths = []
    p = _plot_hist(
        values=community_sizes,
        title="Community Size Distribution",
        xlabel="Community Size",
        out_file=out_dir_path / "community_size_distribution.png",
        bins=30,
    )
    if p:
        out_paths.append(p)
    p = _plot_hist(
        values=weights,
        title="Edge Weight Distribution",
        xlabel="Edge Weight",
        out_file=out_dir_path / "edge_weight_distribution.png",
        bins=30,
    )
    if p:
        out_paths.append(p)

    metrics = {
        "graph_file": graph_file,
        "communities_file": communities_file,
        "num_nodes": n,
        "num_edges": m,
        "average_degree": round(avg_degree, 6),
        "largest_connected_component_ratio": round(largest_cc_ratio, 6),
        "clustering_coefficient_global": round(clustering_global, 6),
        "clustering_coefficient_avg_local": round(clustering_avg_local, 6),
        "degree_one_ratio": round(degree_one_ratio, 6),
        "degree_stats": _stats([float(x) for x in degrees]),
        "weight_stats": _stats(weights),
        "community_size_stats": _stats([float(x) for x in community_sizes]),
        "plots": out_paths,
        "plots_dir": out_dir,
    }

    if out_json:
        out_path = Path(out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute graph structure metrics and plot distributions.")
    parser.add_argument("--graph-file", default="outputs/graph/graph.json")
    parser.add_argument("--communities-file", default="outputs/graph/communities.json")
    parser.add_argument("--out-json", default="outputs/results/graph_structure_metrics.json")
    parser.add_argument("--out-dir", default="outputs/results/graph_plots")
    args = parser.parse_args()

    metrics = compute_graph_structure_metrics(
        graph_file=args.graph_file,
        communities_file=args.communities_file,
        out_json=args.out_json,
        out_dir=args.out_dir,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
