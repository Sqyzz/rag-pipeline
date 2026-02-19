from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.llm_wrapper import llm_chat
from utils.telemetry import Telemetry

try:
    import igraph as ig
    import leidenalg
except Exception as exc:  # pragma: no cover
    ig = None
    leidenalg = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


def _progress(message: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[build_communities {ts}] {message}", flush=True)


def _assert_leiden_available() -> None:
    if ig is None or leidenalg is None:
        raise RuntimeError(
            "Leiden dependencies missing. Please install python-igraph and leidenalg."
        ) from _IMPORT_ERROR


def _build_igraph(graph: dict[str, Any]) -> tuple[Any, list[str], dict[str, int]]:
    node_names = []
    for n in graph.get("nodes", []):
        node_names.append(n["id"] if isinstance(n, dict) else str(n))
    node_names = sorted(set(node_names))
    node_to_idx = {n: i for i, n in enumerate(node_names)}
    g = ig.Graph(directed=False)
    g.add_vertices(len(node_names))
    g.vs["name"] = node_names

    edge_weights: dict[tuple[int, int], float] = {}
    for e in graph.get("edges", []):
        s = str(e["source"])
        t = str(e["target"])
        if s not in node_to_idx or t not in node_to_idx or s == t:
            continue
        a, b = sorted((node_to_idx[s], node_to_idx[t]))
        edge_weights[(a, b)] = edge_weights.get((a, b), 0.0) + float(e.get("weight", 1))

    edges = list(edge_weights.keys())
    g.add_edges(edges)
    g.es["weight"] = [edge_weights[e] for e in edges]
    return g, node_names, node_to_idx


def _run_leiden_levels(
    g: Any, node_names: list[str], resolutions: list[float]
) -> list[dict[str, Any]]:
    levels: list[dict[str, Any]] = []
    for level_idx, resolution in enumerate(resolutions):
        _progress(f"leiden level={level_idx} resolution={resolution}")
        partition = leidenalg.find_partition(
            g,
            leidenalg.RBConfigurationVertexPartition,
            weights=g.es["weight"] if g.ecount() > 0 else None,
            resolution_parameter=resolution,
        )
        groups: dict[int, list[str]] = {}
        for vid, cluster_id in enumerate(partition.membership):
            groups.setdefault(cluster_id, []).append(node_names[vid])
        communities = []
        for order, (_, members) in enumerate(
            sorted(groups.items(), key=lambda x: len(x[1]), reverse=True),
            start=1,
        ):
            communities.append(
                {
                    "community_id": f"l{level_idx}_c{order:04d}",
                    "level": level_idx,
                    "resolution": resolution,
                    "size": len(members),
                    "nodes": sorted(members),
                }
            )
        levels.append(
            {
                "level": level_idx,
                "resolution": resolution,
                "communities": communities,
            }
        )
        _progress(f"leiden level={level_idx} communities={len(communities)}")
    return levels


def _attach_hierarchy(levels: list[dict[str, Any]]) -> None:
    for i in range(1, len(levels)):
        prev_communities = levels[i - 1]["communities"]
        cur_communities = levels[i]["communities"]
        prev_sets = {c["community_id"]: set(c["nodes"]) for c in prev_communities}
        for c in cur_communities:
            current_nodes = set(c["nodes"])
            parent_id = None
            max_overlap = -1
            for pid, pnodes in prev_sets.items():
                overlap = len(current_nodes.intersection(pnodes))
                if overlap > max_overlap:
                    max_overlap = overlap
                    parent_id = pid
            c["parent_id"] = parent_id
        children: dict[str, list[str]] = {}
        for c in cur_communities:
            children.setdefault(c["parent_id"], []).append(c["community_id"])
        for p in prev_communities:
            p["children_ids"] = sorted(children.get(p["community_id"], []))
    if levels:
        for c in levels[0]["communities"]:
            c.setdefault("parent_id", None)
        for c in levels[-1]["communities"]:
            c.setdefault("children_ids", [])


def _summarize_community(nodes: list[str], edges: list[dict], level: int) -> tuple[str, dict]:
    edge_lines = [
        f"- {e['source']} --[{e['relation']}]--> {e['target']} (weight={e['weight']})"
        for e in edges[:30]
    ]
    prompt = f"""
You are summarizing a graph community for retrieval.
Return concise text (max 120 words), include main entities, themes and risks.
This is hierarchy level {level}.

Nodes:
{nodes[:80]}

Edges:
{edge_lines}
"""
    return llm_chat([{"role": "user", "content": prompt}], temperature=0.1, return_meta=True)


def build_communities(
    graph_file: str,
    out_file: str,
    resolutions: list[float] | None = None,
    summary_level_max: int = 0,
    summary_min_size: int = 8,
    summary_top_per_level: int = 0,
) -> dict:
    _assert_leiden_available()
    if resolutions is None:
        resolutions = [0.6, 1.0, 1.6]

    _progress(f"start graph_file={graph_file}")
    graph = json.loads(Path(graph_file).read_text(encoding="utf-8"))
    edges = graph.get("edges", [])
    _progress(f"graph loaded nodes={len(graph.get('nodes', []))}, edges={len(edges)}")
    g, _, _ = _build_igraph(graph)
    _progress(f"igraph built vertices={g.vcount()} edges={g.ecount()}")
    levels = _run_leiden_levels(g, g.vs["name"], resolutions=resolutions)
    _attach_hierarchy(levels)
    _progress(f"hierarchy attached levels={len(levels)}")

    telemetry = Telemetry()
    all_communities: list[dict[str, Any]] = []
    total_communities = sum(len(lv["communities"]) for lv in levels)
    plan_to_summarize = 0
    for lv in levels:
        level_rows = lv["communities"]
        top_limit = int(summary_top_per_level) if int(summary_top_per_level) > 0 else len(level_rows)
        for idx, c in enumerate(level_rows, start=1):
            should_summarize = (
                int(c["level"]) <= int(summary_level_max)
                and int(c.get("size", 0)) >= int(summary_min_size)
                and idx <= top_limit
            )
            if should_summarize:
                plan_to_summarize += 1
    _progress(
        f"summary policy: level_max={summary_level_max}, min_size={summary_min_size}, top_per_level={summary_top_per_level}; planned={plan_to_summarize}/{total_communities}"
    )

    processed = 0
    summarized = 0
    for level in levels:
        _progress(f"summarizing level={level['level']} communities={len(level['communities'])}")
        level_rows = level["communities"]
        top_limit = int(summary_top_per_level) if int(summary_top_per_level) > 0 else len(level_rows)
        for idx, community in enumerate(level_rows, start=1):
            comp_set = set(community["nodes"])
            comp_edges = [
                e for e in edges if e["source"] in comp_set and e["target"] in comp_set
            ]
            should_summarize = (
                int(community["level"]) <= int(summary_level_max)
                and int(community.get("size", 0)) >= int(summary_min_size)
                and idx <= top_limit
            )
            community["edges"] = [e["edge_id"] for e in comp_edges]
            if should_summarize:
                summary, meta = _summarize_community(
                    community["nodes"], comp_edges, level=community["level"]
                )
                telemetry.add_llm(meta)
                community["summary"] = summary
                community["summary_generated"] = True
            else:
                community["summary"] = ""
                community["summary_generated"] = False
                community["summary_skipped_reason"] = {
                    "level": int(community["level"]),
                    "size": int(community.get("size", 0)),
                    "rank_in_level": idx,
                    "thresholds": {
                        "summary_level_max": int(summary_level_max),
                        "summary_min_size": int(summary_min_size),
                        "summary_top_per_level": int(summary_top_per_level),
                    },
                }
            all_communities.append(community)
            processed += 1
            if should_summarize:
                summarized += 1
            if processed % 20 == 0 or processed == total_communities:
                _progress(f"processed={processed}/{total_communities}, summarized={summarized}/{plan_to_summarize}")

    payload = {
        "algorithm": "leiden",
        "is_hierarchical": True,
        "resolutions": resolutions,
        "summary_policy": {
            "summary_level_max": int(summary_level_max),
            "summary_min_size": int(summary_min_size),
            "summary_top_per_level": int(summary_top_per_level),
        },
        "levels": [
            {
                "level": lv["level"],
                "resolution": lv["resolution"],
                "community_ids": [c["community_id"] for c in lv["communities"]],
            }
            for lv in levels
        ],
        "communities": all_communities,
    }

    out_path = Path(out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _progress(f"done num_levels={len(levels)} num_communities={len(all_communities)} -> {out_file}")
    summarized_count = sum(1 for c in all_communities if c.get("summary_generated"))
    return {
        "graph_file": graph_file,
        "community_file": out_file,
        "algorithm": "leiden",
        "num_levels": len(levels),
        "num_communities": len(all_communities),
        "summarized_communities": summarized_count,
        "telemetry": telemetry.to_dict(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build graph communities and summaries")
    parser.add_argument("--graph-file", default="outputs/graph/graph.json")
    parser.add_argument("--out-file", default="outputs/graph/communities.json")
    parser.add_argument("--metrics-file", default="outputs/results/community_build_metrics.json")
    parser.add_argument(
        "--resolutions",
        default="0.6,1.0,1.6",
        help="comma-separated leiden resolution parameters, low->high",
    )
    parser.add_argument(
        "--summary-level-max",
        type=int,
        default=0,
        help="only pre-generate summaries for communities with level <= this value",
    )
    parser.add_argument(
        "--summary-min-size",
        type=int,
        default=8,
        help="only pre-generate summaries for communities with size >= this value",
    )
    parser.add_argument(
        "--summary-top-per-level",
        type=int,
        default=0,
        help="only pre-generate summaries for top-N largest communities per level; 0 means no top-N limit",
    )
    args = parser.parse_args()
    resolutions = [float(x.strip()) for x in args.resolutions.split(",") if x.strip()]
    metrics = build_communities(
        args.graph_file,
        args.out_file,
        resolutions=resolutions,
        summary_level_max=args.summary_level_max,
        summary_min_size=args.summary_min_size,
        summary_top_per_level=args.summary_top_per_level,
    )
    metrics_path = Path(args.metrics_file)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
