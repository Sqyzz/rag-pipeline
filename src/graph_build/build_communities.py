from __future__ import annotations

import argparse
import json
import sys
import time
import re
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


_GENERIC_HUB_LABELS = {
    "agreement",
    "this agreement",
    "party",
    "parties",
    "each party",
    "other party",
    "the other party",
    "law",
    "applicable law",
    "date",
    "event",
    "clause",
    "document",
    "notice",
    "payment",
    "jurisdiction",
    "confidential_info",
    "confidential information",
    "receiving party",
    "disclosing party",
    "indemnifying party",
    "indemnified party",
    "indemnitee",
}


def _normalize_label(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).casefold()


def _looks_like_numeric_hub(label: str) -> bool:
    text = _normalize_label(label)
    if not text:
        return False
    if re.fullmatch(r"[\d\W_]+", text):
        return True
    if re.fullmatch(r"\d+(?:\.\d+)+", text):
        return True
    if re.fullmatch(r"\d+\s*(day|days|month|months|year|years|business day|business days)", text):
        return True
    return False


def _prune_graph_for_communities(
    graph: dict[str, Any],
    hub_doc_threshold: int = 0,
    hub_degree_threshold: int = 0,
    drop_generic_hubs: bool = False,
    drop_numeric_hubs: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    edges = graph.get("edges", []) or []
    nodes = graph.get("nodes", []) or []

    node_doc_sets: dict[str, set[str]] = {}
    node_degree: dict[str, int] = {}
    for edge in edges:
        source = str(edge.get("source", "")).strip()
        target = str(edge.get("target", "")).strip()
        if not source or not target:
            continue
        node_degree[source] = node_degree.get(source, 0) + 1
        node_degree[target] = node_degree.get(target, 0) + 1
        docs = {
            str(m.get("doc_id", "")).split("#", 1)[0].strip()
            for m in (edge.get("mentions") or [])
            if str(m.get("doc_id", "")).strip()
        }
        node_doc_sets.setdefault(source, set()).update(docs)
        node_doc_sets.setdefault(target, set()).update(docs)

    pruned_nodes: set[str] = set()
    prune_reasons = {
        "generic_hub": 0,
        "numeric_hub": 0,
        "doc_frequency": 0,
        "degree": 0,
    }
    for node in nodes:
        node_id = str(node.get("id", "")).strip()
        if not node_id:
            continue
        norm = _normalize_label(node_id)
        docs = node_doc_sets.get(node_id, set())
        degree = int(node_degree.get(node_id, 0) or 0)
        reason = None
        if drop_generic_hubs and norm in _GENERIC_HUB_LABELS:
            reason = "generic_hub"
        elif drop_numeric_hubs and _looks_like_numeric_hub(node_id):
            reason = "numeric_hub"
        elif int(hub_doc_threshold) > 0 and len(docs) >= int(hub_doc_threshold):
            reason = "doc_frequency"
        elif int(hub_degree_threshold) > 0 and degree >= int(hub_degree_threshold):
            reason = "degree"
        if reason:
            pruned_nodes.add(node_id)
            prune_reasons[reason] += 1

    filtered_edges = [
        edge for edge in edges
        if str(edge.get("source", "")).strip() not in pruned_nodes
        and str(edge.get("target", "")).strip() not in pruned_nodes
    ]
    kept_nodes = {
        str(edge.get("source", "")).strip() for edge in filtered_edges
    }.union({
        str(edge.get("target", "")).strip() for edge in filtered_edges
    })
    filtered_nodes = [node for node in nodes if str(node.get("id", "")).strip() in kept_nodes]
    filtered_adjacency = {
        str(node_id): [
            item for item in (graph.get("adjacency", {}) or {}).get(str(node_id), [])
            if str(item.get("target", "")).strip() in kept_nodes
        ]
        for node_id in kept_nodes
    }
    diagnostics = {
        "hub_doc_threshold": int(hub_doc_threshold),
        "hub_degree_threshold": int(hub_degree_threshold),
        "drop_generic_hubs": bool(drop_generic_hubs),
        "drop_numeric_hubs": bool(drop_numeric_hubs),
        "pruned_nodes": len(pruned_nodes),
        "kept_nodes": len(filtered_nodes),
        "pruned_edges": max(0, len(edges) - len(filtered_edges)),
        "kept_edges": len(filtered_edges),
        "prune_reasons": prune_reasons,
    }
    return {
        "nodes": filtered_nodes,
        "edges": filtered_edges,
        "adjacency": filtered_adjacency,
    }, diagnostics


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


def _attach_hierarchy(levels: list[dict[str, Any]], min_parent_overlap: float = 0.8) -> dict[str, Any]:
    linked = 0
    unlinked = 0
    for i in range(1, len(levels)):
        prev_communities = levels[i - 1]["communities"]
        cur_communities = levels[i]["communities"]
        prev_sets = {c["community_id"]: set(c["nodes"]) for c in prev_communities}
        for c in cur_communities:
            current_nodes = set(c["nodes"])
            parent_id = None
            max_overlap = -1
            best_ratio = 0.0
            for pid, pnodes in prev_sets.items():
                overlap = len(current_nodes.intersection(pnodes))
                if overlap > max_overlap:
                    max_overlap = overlap
                    parent_id = pid
                    best_ratio = (
                        (overlap / max(len(current_nodes), 1))
                        if current_nodes
                        else 0.0
                    )
            if best_ratio < float(min_parent_overlap):
                parent_id = None
                unlinked += 1
            else:
                linked += 1
            c["parent_id"] = parent_id
            c["parent_overlap_ratio"] = round(best_ratio, 4)
        children: dict[str, list[str]] = {}
        for c in cur_communities:
            if c["parent_id"]:
                children.setdefault(c["parent_id"], []).append(c["community_id"])
        for p in prev_communities:
            p["children_ids"] = sorted(children.get(p["community_id"], []))
    if levels:
        for c in levels[0]["communities"]:
            c.setdefault("parent_id", None)
            c.setdefault("parent_overlap_ratio", 1.0)
        for c in levels[-1]["communities"]:
            c.setdefault("children_ids", [])
    return {
        "min_parent_overlap": float(min_parent_overlap),
        "linked_children": int(linked),
        "unlinked_children": int(unlinked),
    }


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


def _build_summary_plan(
    levels: list[dict[str, Any]],
    summary_level_max: int,
    summary_min_size: int,
    summary_top_per_level: int,
    summary_min_per_level: int,
) -> tuple[set[str], dict[int, int], dict[int, int]]:
    """
    Build a summary plan by level.

    - Base rule: level <= summary_level_max AND size >= summary_min_size AND rank <= top_limit
    - Fallback: if planned summaries in an eligible level are fewer than summary_min_per_level,
      relax size constraint and backfill from largest remaining communities in the same level.
    """
    planned_ids: set[str] = set()
    base_counts: dict[int, int] = {}
    relaxed_counts: dict[int, int] = {}

    for lv in levels:
        level_idx = int(lv["level"])
        rows = lv["communities"]
        top_limit = int(summary_top_per_level) if int(summary_top_per_level) > 0 else len(rows)
        capped = rows[:top_limit]
        if level_idx > int(summary_level_max):
            base_counts[level_idx] = 0
            relaxed_counts[level_idx] = 0
            continue

        base = [
            c for c in capped
            if int(c.get("size", 0)) >= int(summary_min_size)
        ]
        chosen: list[dict[str, Any]] = list(base)
        if int(summary_min_per_level) > 0 and len(chosen) < int(summary_min_per_level):
            need = int(summary_min_per_level) - len(chosen)
            existing_ids = {str(c.get("community_id", "")) for c in chosen}
            pool = [c for c in capped if str(c.get("community_id", "")) not in existing_ids]
            chosen.extend(pool[:need])

        for c in chosen:
            cid = str(c.get("community_id", "")).strip()
            if cid:
                planned_ids.add(cid)

        base_counts[level_idx] = len(base)
        relaxed_counts[level_idx] = max(0, len(chosen) - len(base))

    return planned_ids, base_counts, relaxed_counts


def build_communities(
    graph_file: str,
    out_file: str,
    resolutions: list[float] | None = None,
    min_parent_overlap: float = 0.8,
    summary_level_max: int = 0,
    summary_min_size: int = 8,
    summary_top_per_level: int = 0,
    summary_min_per_level: int = 10,
    prune_hub_doc_threshold: int = 0,
    prune_hub_degree_threshold: int = 0,
    prune_generic_hubs: bool = False,
    prune_numeric_hubs: bool = False,
) -> dict:
    _assert_leiden_available()
    if resolutions is None:
        resolutions = [0.6, 1.0, 1.6]
    if float(min_parent_overlap) < 0 or float(min_parent_overlap) > 1:
        raise ValueError("min_parent_overlap must be in [0, 1]")

    _progress(f"start graph_file={graph_file}")
    graph = json.loads(Path(graph_file).read_text(encoding="utf-8"))
    raw_nodes = len(graph.get("nodes", []))
    raw_edges = len(graph.get("edges", []))
    graph_for_partition, prune_diag = _prune_graph_for_communities(
        graph,
        hub_doc_threshold=prune_hub_doc_threshold,
        hub_degree_threshold=prune_hub_degree_threshold,
        drop_generic_hubs=prune_generic_hubs,
        drop_numeric_hubs=prune_numeric_hubs,
    )
    edges = graph_for_partition.get("edges", [])
    _progress(
        f"graph loaded nodes={raw_nodes}, edges={raw_edges}; "
        f"partition_graph nodes={len(graph_for_partition.get('nodes', []))}, edges={len(edges)}"
    )
    g, _, _ = _build_igraph(graph_for_partition)
    _progress(f"igraph built vertices={g.vcount()} edges={g.ecount()}")
    levels = _run_leiden_levels(g, g.vs["name"], resolutions=resolutions)
    hierarchy_meta = _attach_hierarchy(levels, min_parent_overlap=min_parent_overlap)
    _progress(f"hierarchy attached levels={len(levels)}")

    telemetry = Telemetry()
    all_communities: list[dict[str, Any]] = []
    total_communities = sum(len(lv["communities"]) for lv in levels)
    summary_plan_ids, base_counts, relaxed_counts = _build_summary_plan(
        levels=levels,
        summary_level_max=summary_level_max,
        summary_min_size=summary_min_size,
        summary_top_per_level=summary_top_per_level,
        summary_min_per_level=summary_min_per_level,
    )
    plan_to_summarize = len(summary_plan_ids)
    relaxed_total = sum(relaxed_counts.values())
    _progress(
        f"summary policy: level_max={summary_level_max}, min_size={summary_min_size}, "
        f"top_per_level={summary_top_per_level}, min_per_level={summary_min_per_level}; "
        f"planned={plan_to_summarize}/{total_communities} (relaxed_backfill={relaxed_total})"
    )

    processed = 0
    summarized = 0
    for level in levels:
        _progress(f"summarizing level={level['level']} communities={len(level['communities'])}")
        level_rows = level["communities"]
        for idx, community in enumerate(level_rows, start=1):
            comp_set = set(community["nodes"])
            comp_edges = [
                e for e in graph_for_partition.get("edges", []) if e["source"] in comp_set and e["target"] in comp_set
            ]
            community_id = str(community.get("community_id", "")).strip()
            should_summarize = community_id in summary_plan_ids
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
                        "summary_min_per_level": int(summary_min_per_level),
                    },
                    "planned": False,
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
        "hierarchy_meta": hierarchy_meta,
        "summary_policy": {
            "summary_level_max": int(summary_level_max),
            "summary_min_size": int(summary_min_size),
            "summary_top_per_level": int(summary_top_per_level),
            "summary_min_per_level": int(summary_min_per_level),
            "planned_base_by_level": {str(k): int(v) for k, v in base_counts.items()},
            "planned_relaxed_backfill_by_level": {str(k): int(v) for k, v in relaxed_counts.items()},
        },
        "partition_pruning": prune_diag,
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
        "min_parent_overlap": float(min_parent_overlap),
        "linked_children": int(hierarchy_meta.get("linked_children", 0)),
        "unlinked_children": int(hierarchy_meta.get("unlinked_children", 0)),
        "num_levels": len(levels),
        "num_communities": len(all_communities),
        "summarized_communities": summarized_count,
        "partition_pruning": prune_diag,
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
        "--min-parent-overlap",
        type=float,
        default=0.8,
        help="Minimum child->parent node overlap ratio required to attach hierarchy parent_id.",
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
    parser.add_argument(
        "--summary-min-per-level",
        type=int,
        default=10,
        help=(
            "minimum summaries per eligible level. If base policy yields fewer summaries, "
            "auto-backfill from largest communities in that level."
        ),
    )
    parser.add_argument("--prune-hub-doc-threshold", type=int, default=0)
    parser.add_argument("--prune-hub-degree-threshold", type=int, default=0)
    parser.add_argument("--prune-generic-hubs", default="false")
    parser.add_argument("--prune-numeric-hubs", default="false")
    args = parser.parse_args()
    resolutions = [float(x.strip()) for x in args.resolutions.split(",") if x.strip()]
    metrics = build_communities(
        args.graph_file,
        args.out_file,
        resolutions=resolutions,
        min_parent_overlap=args.min_parent_overlap,
        summary_level_max=args.summary_level_max,
        summary_min_size=args.summary_min_size,
        summary_top_per_level=args.summary_top_per_level,
        summary_min_per_level=args.summary_min_per_level,
        prune_hub_doc_threshold=args.prune_hub_doc_threshold,
        prune_hub_degree_threshold=args.prune_hub_degree_threshold,
        prune_generic_hubs=str(args.prune_generic_hubs).strip().lower() in {"1", "true", "yes", "y", "on"},
        prune_numeric_hubs=str(args.prune_numeric_hubs).strip().lower() in {"1", "true", "yes", "y", "on"},
    )
    metrics_path = Path(args.metrics_file)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
