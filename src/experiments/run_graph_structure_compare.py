from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

try:
    import igraph as ig
except Exception as exc:  # pragma: no cover
    raise RuntimeError("python-igraph is required for graph structure comparison") from exc

try:
    import leidenalg
except Exception:  # pragma: no cover
    leidenalg = None


def _load_json(path: str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _stable_str(v: Any) -> str:
    return str(v or "").strip()


def _tokenize(text: str) -> set[str]:
    # Keep alnum tokens; good enough for mixed identifiers/domain terms.
    return {x for x in re.split(r"[^0-9A-Za-z]+", text.lower()) if x}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a.union(b)
    if not union:
        return 0.0
    return float(len(a.intersection(b)) / len(union))


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


def _stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "min": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "max": 0.0,
            "mean": 0.0,
        }
    vals = [float(x) for x in values]
    return {
        "count": int(len(vals)),
        "min": round(min(vals), 6),
        "p50": round(_percentile(vals, 0.5), 6),
        "p95": round(_percentile(vals, 0.95), 6),
        "max": round(max(vals), 6),
        "mean": round(sum(vals) / len(vals), 6),
    }


def _parse_graph_payload(payload: Any) -> tuple[list[str], list[dict[str, Any]]]:
    """
    Normalize graph payload into:
    - nodes: list[node_name]
    - edges: list[{source,target,relation,weight}]

    Supports two common schemas:
    1) enterprise-graphrag graph.json with {nodes:[{id}], edges:[{source,target,relation,weight}]}
    2) youtu style list of relationships with start_node/end_node blocks
    """
    if isinstance(payload, dict):
        raw_nodes = payload.get("nodes", [])
        raw_edges = payload.get("edges", [])

        nodes: list[str] = []
        for n in raw_nodes:
            if isinstance(n, dict):
                nid = _stable_str(n.get("id") or n.get("name") or n.get("label"))
            else:
                nid = _stable_str(n)
            if nid:
                nodes.append(nid)

        edges: list[dict[str, Any]] = []
        for e in raw_edges:
            if not isinstance(e, dict):
                continue
            s = _stable_str(e.get("source"))
            t = _stable_str(e.get("target"))
            if not s or not t:
                continue
            edges.append(
                {
                    "source": s,
                    "target": t,
                    "relation": _stable_str(e.get("relation") or e.get("label")),
                    "weight": float(e.get("weight", 1.0) or 1.0),
                }
            )
            nodes.append(s)
            nodes.append(t)

        return sorted(set(nodes)), edges

    if isinstance(payload, list):
        # youtu output/graphs/*_new.json style
        nodes: set[str] = set()
        edges: list[dict[str, Any]] = []
        for row in payload:
            if not isinstance(row, dict):
                continue
            s = _stable_str((((row.get("start_node") or {}).get("properties") or {}).get("name")))
            t = _stable_str((((row.get("end_node") or {}).get("properties") or {}).get("name")))
            if not s or not t:
                continue
            nodes.add(s)
            nodes.add(t)
            edges.append(
                {
                    "source": s,
                    "target": t,
                    "relation": _stable_str(row.get("relation")),
                    "weight": 1.0,
                }
            )
        return sorted(nodes), edges

    raise ValueError("Unsupported graph payload schema")


def _extract_communities(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        comms = payload.get("communities", [])
        return comms if isinstance(comms, list) else []
    if isinstance(payload, list):
        return payload
    return []


def _extract_embedded_communities_from_graph_payload(payload: Any) -> list[dict[str, Any]]:
    """
    Extract youtu-style embedded communities from graph relationship rows.

    Expected signals:
    - community nodes via start_node/end_node.label == "community"
    - member edges via relation in {"member_of", "keyword_of"} pointing to community
    - optional explicit member list in community node properties.members
    """
    if not isinstance(payload, list):
        return []

    comm_members: dict[str, set[str]] = {}
    comm_desc: dict[str, str] = {}

    def _node_name(node_obj: dict[str, Any]) -> str:
        props = (node_obj.get("properties") or {}) if isinstance(node_obj, dict) else {}
        return _stable_str(props.get("name"))

    for row in payload:
        if not isinstance(row, dict):
            continue
        s_node = row.get("start_node") if isinstance(row.get("start_node"), dict) else {}
        e_node = row.get("end_node") if isinstance(row.get("end_node"), dict) else {}
        s_label = _stable_str(s_node.get("label")).lower()
        e_label = _stable_str(e_node.get("label")).lower()
        s_name = _node_name(s_node)
        e_name = _node_name(e_node)
        relation = _stable_str(row.get("relation")).lower()

        # Register community node meta from either side.
        if s_label == "community" and s_name:
            comm_members.setdefault(s_name, set())
            desc = _stable_str(((s_node.get("properties") or {}).get("description")))
            if desc:
                comm_desc[s_name] = desc
            raw_members = ((s_node.get("properties") or {}).get("members"))
            if isinstance(raw_members, list):
                for m in raw_members:
                    mm = _stable_str(m)
                    if mm:
                        comm_members[s_name].add(mm)
            # Add community node itself to avoid fallback singleton.
            comm_members[s_name].add(s_name)

        if e_label == "community" and e_name:
            comm_members.setdefault(e_name, set())
            desc = _stable_str(((e_node.get("properties") or {}).get("description")))
            if desc:
                comm_desc[e_name] = desc
            raw_members = ((e_node.get("properties") or {}).get("members"))
            if isinstance(raw_members, list):
                for m in raw_members:
                    mm = _stable_str(m)
                    if mm:
                        comm_members[e_name].add(mm)
            comm_members[e_name].add(e_name)

        # Membership edges.
        if relation in {"member_of", "keyword_of"}:
            if e_label == "community" and e_name and s_name:
                comm_members.setdefault(e_name, set()).add(s_name)
            if s_label == "community" and s_name and e_name:
                comm_members.setdefault(s_name, set()).add(e_name)

    out: list[dict[str, Any]] = []
    for idx, (cid, members) in enumerate(sorted(comm_members.items(), key=lambda x: len(x[1]), reverse=True), start=1):
        member_list = sorted({m for m in members if _stable_str(m)})
        if not member_list:
            continue
        out.append(
            {
                "community_id": f"embedded_c{idx:04d}",
                "level": 0,
                "size": len(member_list),
                "nodes": member_list,
                "summary": comm_desc.get(cid, ""),
                "source_name": cid,
            }
        )
    return out


def _select_communities_for_level(communities: list[dict[str, Any]], level: int | None) -> list[dict[str, Any]]:
    if level is None:
        return [c for c in communities if isinstance(c, dict)]
    out = []
    for c in communities:
        if not isinstance(c, dict):
            continue
        raw = c.get("level")
        try:
            lv = int(raw)
        except (TypeError, ValueError):
            continue
        if lv == level:
            out.append(c)
    return out


def _first_available_level(communities: list[dict[str, Any]]) -> int | None:
    levels: list[int] = []
    for c in communities:
        if not isinstance(c, dict):
            continue
        try:
            levels.append(int(c.get("level")))
        except (TypeError, ValueError):
            continue
    if not levels:
        return None
    return min(levels)


def _build_igraph(nodes: list[str], edges: list[dict[str, Any]]) -> tuple[Any, list[dict[str, Any]]]:
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    g = ig.Graph(directed=False)
    g.add_vertices(len(nodes))
    g.vs["name"] = nodes

    edge_pairs: list[tuple[int, int]] = []
    weights: list[float] = []
    kept_edges: list[dict[str, Any]] = []
    for e in edges:
        s = _stable_str(e.get("source"))
        t = _stable_str(e.get("target"))
        if not s or not t or s == t:
            continue
        if s not in node_to_idx or t not in node_to_idx:
            continue
        edge_pairs.append((node_to_idx[s], node_to_idx[t]))
        weights.append(float(e.get("weight", 1.0) or 1.0))
        kept_edges.append(e)

    if edge_pairs:
        g.add_edges(edge_pairs)
        g.es["weight"] = weights
        g.es["relation"] = [_stable_str(x.get("relation")) for x in kept_edges]
    return g, kept_edges


def _collapsed_undirected_edge_count(kept_edges: list[dict[str, Any]]) -> int:
    pairs: set[tuple[str, str]] = set()
    for e in kept_edges:
        s = _stable_str(e.get("source"))
        t = _stable_str(e.get("target"))
        if not s or not t or s == t:
            continue
        a, b = sorted((s, t))
        pairs.add((a, b))
    return len(pairs)


def _build_membership(
    node_names: list[str],
    communities: list[dict[str, Any]],
) -> tuple[list[int], dict[str, Any], dict[int, int], int]:
    """Build non-overlapping membership for igraph modularity.

    If communities overlap, a node is assigned to the first community after
    sorting by size desc; unassigned nodes become singleton communities.
    """
    comm_rows: list[tuple[str, list[str]]] = []
    for idx, c in enumerate(communities, start=1):
        if not isinstance(c, dict):
            continue
        cid = _stable_str(c.get("community_id") or f"community_{idx}")
        members = c.get("nodes", [])
        if not isinstance(members, list):
            continue
        member_names = [_stable_str(x) for x in members if _stable_str(x)]
        if not member_names:
            continue
        comm_rows.append((cid, member_names))

    comm_rows.sort(key=lambda x: len(x[1]), reverse=True)

    node_to_group: dict[str, int] = {}
    overlap_nodes = 0
    next_group = 0
    for _, members in comm_rows:
        group = next_group
        assigned_any = False
        for n in members:
            if n in node_to_group:
                overlap_nodes += 1
                continue
            node_to_group[n] = group
            assigned_any = True
        if assigned_any:
            next_group += 1

    for n in node_names:
        if n not in node_to_group:
            node_to_group[n] = next_group
            next_group += 1

    membership = [node_to_group[n] for n in node_names]
    group_sizes: dict[int, int] = {}
    for gid in membership:
        group_sizes[gid] = group_sizes.get(gid, 0) + 1
    covered_nodes = len(node_to_group)
    meta = {
        "community_groups_used": int(len(set(membership))),
        "overlap_nodes_ignored": int(overlap_nodes),
    }
    return membership, meta, group_sizes, int(covered_nodes)


def _edge_partition_quality(
    g: Any,
    membership: list[int],
) -> dict[str, Any]:
    if g.ecount() == 0 or g.vcount() == 0:
        return {
            "intra_edges": 0,
            "inter_edges": 0,
            "intra_edge_ratio": 0.0,
            "inter_edge_ratio": 0.0,
        }
    intra = 0
    inter = 0
    for e in g.es:
        s_idx, t_idx = e.tuple
        if membership[s_idx] == membership[t_idx]:
            intra += 1
        else:
            inter += 1
    total = intra + inter
    return {
        "intra_edges": int(intra),
        "inter_edges": int(inter),
        "intra_edge_ratio": round(float(intra / total), 6) if total > 0 else 0.0,
        "inter_edge_ratio": round(float(inter / total), 6) if total > 0 else 0.0,
    }


def _community_size_quality(group_sizes: dict[int, int]) -> dict[str, Any]:
    sizes = [int(v) for v in group_sizes.values() if int(v) > 0]
    if not sizes:
        return {
            "size_stats": _stats([]),
            "singleton_community_ratio": 0.0,
            "largest_community_ratio": 0.0,
            "community_size_variance": 0.0,
        }
    total_nodes = sum(sizes)
    singleton = sum(1 for x in sizes if x == 1)
    largest = max(sizes)
    mean_size = float(total_nodes / len(sizes))
    variance = float(sum((float(x) - mean_size) ** 2 for x in sizes) / len(sizes))
    return {
        "size_stats": _stats([float(x) for x in sizes]),
        "singleton_community_ratio": round(float(singleton / len(sizes)), 6),
        "largest_community_ratio": round(float(largest / total_nodes), 6) if total_nodes > 0 else 0.0,
        "community_size_variance": round(variance, 6),
    }


def _groups_from_membership(node_names: list[str], membership: list[int]) -> dict[int, list[str]]:
    groups: dict[int, list[str]] = {}
    for idx, gid in enumerate(membership):
        groups.setdefault(int(gid), []).append(node_names[idx])
    return groups


def _semantic_quality_by_groups(groups: dict[int, list[str]]) -> dict[str, Any]:
    if not groups:
        return {
            "semantic_intra_similarity_mean": 0.0,
            "semantic_inter_similarity_mean": 0.0,
            "semantic_inter_similarity_reduction": 0.0,
            "semantic_inter_similarity_reduction_ratio": 0.0,
        }

    # Deterministic cap for efficiency on very large communities.
    max_nodes_per_comm = 30
    sampled_groups: dict[int, list[str]] = {}
    node_tokens: dict[str, set[str]] = {}
    for gid, members in groups.items():
        chosen = sorted(members)[:max_nodes_per_comm]
        sampled_groups[gid] = chosen
        for n in chosen:
            node_tokens[n] = _tokenize(n)

    intra_scores: list[float] = []
    centroid_tokens: dict[int, set[str]] = {}
    for gid, members in sampled_groups.items():
        if len(members) >= 2:
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    intra_scores.append(_jaccard(node_tokens[members[i]], node_tokens[members[j]]))
        # community text centroid represented by union of member tokens
        cset: set[str] = set()
        for n in members:
            cset.update(node_tokens[n])
        centroid_tokens[gid] = cset

    gids = sorted(centroid_tokens.keys())
    inter_scores: list[float] = []
    if len(gids) >= 2:
        for i in range(len(gids)):
            for j in range(i + 1, len(gids)):
                inter_scores.append(_jaccard(centroid_tokens[gids[i]], centroid_tokens[gids[j]]))

    intra_mean = float(sum(intra_scores) / len(intra_scores)) if intra_scores else 0.0
    inter_mean = float(sum(inter_scores) / len(inter_scores)) if inter_scores else 0.0
    reduction = intra_mean - inter_mean
    reduction_ratio = 0.0 if intra_mean <= 0 else (1.0 - (inter_mean / intra_mean))

    return {
        "semantic_intra_similarity_mean": round(intra_mean, 6),
        "semantic_inter_similarity_mean": round(inter_mean, 6),
        "semantic_inter_similarity_reduction": round(reduction, 6),
        "semantic_inter_similarity_reduction_ratio": round(max(-1.0, min(1.0, reduction_ratio)), 6),
    }


def _dedup_ratio_from_stats(stats_payload: Any, node_count_after: int) -> tuple[float | None, dict[str, Any]]:
    if not isinstance(stats_payload, dict):
        return None, {"available": False, "reason": "stats_file_missing_or_invalid"}

    if stats_payload.get("nodes_before") is not None and stats_payload.get("nodes_after") is not None:
        try:
            before = int(stats_payload.get("nodes_before"))
            after = int(stats_payload.get("nodes_after"))
            ratio = 0.0 if before <= 0 else max(0.0, min(1.0, 1.0 - (after / before)))
            return round(ratio, 6), {"available": True, "source": "stats.nodes_before_after", "before": before, "after": after}
        except (TypeError, ValueError):
            pass

    # Support nested build_stats conventions.
    build_stats = stats_payload.get("build_stats") if isinstance(stats_payload.get("build_stats"), dict) else None
    if build_stats and build_stats.get("nodes_before") is not None and build_stats.get("nodes_after") is not None:
        try:
            before = int(build_stats.get("nodes_before"))
            after = int(build_stats.get("nodes_after"))
            ratio = 0.0 if before <= 0 else max(0.0, min(1.0, 1.0 - (after / before)))
            return round(ratio, 6), {
                "available": True,
                "source": "stats.build_stats.nodes_before_after",
                "before": before,
                "after": after,
            }
        except (TypeError, ValueError):
            pass

    # If after-count exists in stats and differs from parsed graph, still return advisory only.
    return None, {
        "available": False,
        "reason": "nodes_before_after_not_found",
        "parsed_nodes_after": int(node_count_after),
    }


def _compute_side_metrics(
    graph_file: str,
    communities_file: str | None,
    name: str,
    community_level: int | None,
    stats_file: str | None,
    community_mode: str,
) -> dict[str, Any]:
    graph_payload = _load_json(graph_file)
    comm_payload = None
    if communities_file and Path(communities_file).exists():
        comm_payload = _load_json(communities_file)

    nodes, edges = _parse_graph_payload(graph_payload)
    g, kept_edges = _build_igraph(nodes, edges)

    n = int(g.vcount())
    m_raw = int(g.ecount())
    m_collapsed = _collapsed_undirected_edge_count(kept_edges)
    avg_degree_raw = round((2.0 * m_raw / n), 6) if n > 0 else 0.0
    avg_degree_collapsed = round((2.0 * m_collapsed / n), 6) if n > 0 else 0.0

    relations = {_stable_str(e.get("relation")) for e in kept_edges if _stable_str(e.get("relation"))}

    all_communities: list[dict[str, Any]] = []
    level_used = community_level
    selected: list[dict[str, Any]] = []
    modularity = None
    modularity_meta: dict[str, Any]
    community_source = "provided"
    community_quality: dict[str, Any] = {}

    use_provided_communities = community_mode == "provided" and comm_payload is not None

    if use_provided_communities:
        all_communities = _extract_communities(comm_payload)
        if level_used is None:
            level_used = _first_available_level(all_communities)
        selected = _select_communities_for_level(all_communities, level_used)
        if selected:
            membership, membership_meta, group_sizes, covered_nodes = _build_membership(nodes, selected)
            try:
                mod = g.modularity(membership, weights=(g.es["weight"] if m_raw > 0 else None)) if n > 0 else 0.0
                modularity = round(float(mod), 6)
                modularity_meta = {"available": True, "method": "provided_communities", **membership_meta}
                partition_quality = _edge_partition_quality(g, membership)
                size_quality = _community_size_quality(group_sizes)
                semantic_quality = _semantic_quality_by_groups(_groups_from_membership(nodes, membership))
                community_quality = {
                    "coverage_ratio": round(float(covered_nodes / n), 6) if n > 0 else 0.0,
                    **partition_quality,
                    **size_quality,
                    **semantic_quality,
                }
            except Exception as exc:  # noqa: BLE001
                modularity_meta = {"available": False, "reason": f"modularity_failed: {exc}"}
                community_quality = {}
        else:
            modularity_meta = {"available": False, "reason": "no_communities_for_selected_level"}
            community_quality = {}
    else:
        # For fair cross-pipeline comparison, fall back to a shared auto-partition path
        # unless both sides explicitly opt into provided communities.
        if n == 0:
            modularity = 0.0
            modularity_meta = {"available": True, "method": "empty_graph"}
            community_source = "auto_partition"
            community_quality = {
                "coverage_ratio": 1.0,
                "intra_edges": 0,
                "inter_edges": 0,
                "intra_edge_ratio": 0.0,
                "inter_edge_ratio": 0.0,
                "size_stats": _stats([]),
                "singleton_community_ratio": 0.0,
                "largest_community_ratio": 0.0,
                "community_size_variance": 0.0,
                "semantic_intra_similarity_mean": 0.0,
                "semantic_inter_similarity_mean": 0.0,
                "semantic_inter_similarity_reduction": 0.0,
                "semantic_inter_similarity_reduction_ratio": 0.0,
            }
        else:
            try:
                weights = g.es["weight"] if m_raw > 0 else None
                if leidenalg is not None:
                    part = leidenalg.find_partition(
                        g,
                        leidenalg.RBConfigurationVertexPartition,
                        weights=weights,
                        resolution_parameter=1.0,
                    )
                    membership = list(part.membership)
                    method = "leidenalg_rb_1.0"
                else:
                    # igraph fallback if leidenalg is unavailable
                    part = g.community_multilevel(weights=weights)
                    membership = list(part.membership)
                    method = "igraph_multilevel_fallback"
                mod = g.modularity(membership, weights=weights)
                modularity = round(float(mod), 6)
                community_source = "auto_partition"
                group_sizes: dict[int, int] = {}
                for gid in membership:
                    group_sizes[int(gid)] = group_sizes.get(int(gid), 0) + 1
                partition_quality = _edge_partition_quality(g, membership)
                size_quality = _community_size_quality(group_sizes)
                semantic_quality = _semantic_quality_by_groups(_groups_from_membership(nodes, membership))
                community_quality = {
                    "coverage_ratio": 1.0,
                    **partition_quality,
                    **size_quality,
                    **semantic_quality,
                }
                modularity_meta = {
                    "available": True,
                    "method": method,
                    "reason": "communities_file_missing_auto_partition",
                    "community_groups_used": int(len(set(membership))),
                }
            except Exception as exc:  # noqa: BLE001
                modularity_meta = {"available": False, "reason": f"auto_partition_failed: {exc}"}
                community_source = "auto_partition"
                community_quality = {}

    stats_payload = _load_json(stats_file) if stats_file and Path(stats_file).exists() else None
    dedup_ratio, dedup_meta = _dedup_ratio_from_stats(stats_payload, node_count_after=n)

    return {
        "name": name,
        "graph_file": graph_file,
        "communities_file": communities_file,
        "community_level_used": level_used,
        "num_nodes": n,
        "num_edges_raw": m_raw,
        "num_edges_collapsed_undirected": m_collapsed,
        "average_degree_raw": avg_degree_raw,
        "average_degree_collapsed_undirected": avg_degree_collapsed,
        "relation_type_count": int(len(relations)),
        "modularity": modularity,
        "modularity_meta": modularity_meta,
        "community_source": community_source,
        "community_quality": community_quality,
        "communities_total": int(len(all_communities)),
        "communities_selected": int(len(selected)),
        "dedup_node_merge_ratio": dedup_ratio,
        "dedup_meta": dedup_meta,
    }


def _diff_block(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    def _d(key: str) -> float | None:
        lv = left.get(key)
        rv = right.get(key)
        if isinstance(lv, (int, float)) and isinstance(rv, (int, float)):
            return round(float(lv) - float(rv), 6)
        return None

    def _cq_d(key: str) -> float | None:
        lq = left.get("community_quality")
        rq = right.get("community_quality")
        if not isinstance(lq, dict) or not isinstance(rq, dict):
            return None
        lv = lq.get(key)
        rv = rq.get(key)
        if isinstance(lv, (int, float)) and isinstance(rv, (int, float)):
            return round(float(lv) - float(rv), 6)
        return None

    return {
        "preferred_metrics_left_minus_right": {
            "num_nodes": _d("num_nodes"),
            "num_edges_collapsed_undirected": _d("num_edges_collapsed_undirected"),
            "average_degree_collapsed_undirected": _d("average_degree_collapsed_undirected"),
            "modularity": _d("modularity"),
            "dedup_node_merge_ratio": _d("dedup_node_merge_ratio"),
            "community_quality.intra_edge_ratio": _cq_d("intra_edge_ratio"),
            "community_quality.singleton_community_ratio": _cq_d("singleton_community_ratio"),
            "community_quality.largest_community_ratio": _cq_d("largest_community_ratio"),
            "community_quality.semantic_intra_similarity_mean": _cq_d("semantic_intra_similarity_mean"),
            "community_quality.semantic_inter_similarity_reduction": _cq_d("semantic_inter_similarity_reduction"),
            "community_quality.community_size_variance": _cq_d("community_size_variance"),
        },
        "left_minus_right": {
            "num_nodes": _d("num_nodes"),
            "num_edges_raw": _d("num_edges_raw"),
            "num_edges_collapsed_undirected": _d("num_edges_collapsed_undirected"),
            "average_degree_raw": _d("average_degree_raw"),
            "average_degree_collapsed_undirected": _d("average_degree_collapsed_undirected"),
            "relation_type_count": _d("relation_type_count"),
            "modularity": _d("modularity"),
            "dedup_node_merge_ratio": _d("dedup_node_merge_ratio"),
            "community_quality.intra_edge_ratio": _cq_d("intra_edge_ratio"),
            "community_quality.singleton_community_ratio": _cq_d("singleton_community_ratio"),
            "community_quality.largest_community_ratio": _cq_d("largest_community_ratio"),
            "community_quality.semantic_intra_similarity_mean": _cq_d("semantic_intra_similarity_mean"),
            "community_quality.semantic_inter_similarity_reduction": _cq_d("semantic_inter_similarity_reduction"),
            "community_quality.community_size_variance": _cq_d("community_size_variance"),
        }
    }


def run_compare(
    *,
    left_name: str,
    left_graph_file: str,
    left_communities_file: str | None,
    right_name: str,
    right_graph_file: str,
    right_communities_file: str | None,
    community_level: int | None,
    community_mode: str,
    left_stats_file: str | None,
    right_stats_file: str | None,
    out_json: str,
) -> dict[str, Any]:
    left_has_communities = bool(left_communities_file and Path(left_communities_file).exists())
    right_has_communities = bool(right_communities_file and Path(right_communities_file).exists())
    effective_community_mode = community_mode
    if community_mode == "consistent":
        effective_community_mode = "provided" if left_has_communities and right_has_communities else "auto"

    left = _compute_side_metrics(
        graph_file=left_graph_file,
        communities_file=left_communities_file,
        name=left_name,
        community_level=community_level,
        stats_file=left_stats_file,
        community_mode=effective_community_mode,
    )
    right = _compute_side_metrics(
        graph_file=right_graph_file,
        communities_file=right_communities_file,
        name=right_name,
        community_level=community_level,
        stats_file=right_stats_file,
        community_mode=effective_community_mode,
    )

    payload = {
        "left": left,
        "right": right,
        "comparison_policy": {
            "community_mode_requested": community_mode,
            "community_mode_effective": effective_community_mode,
            "preferred_edge_metric": "num_edges_collapsed_undirected",
            "preferred_degree_metric": "average_degree_collapsed_undirected",
            "relation_type_count_caution": "Compare only if both pipelines normalize relations to the same schema.",
        },
        "comparability": {
            "community_partition_source_match": left.get("community_source") == right.get("community_source"),
            "community_level_match": left.get("community_level_used") == right.get("community_level_used"),
            "community_mode_effective_match": True,
        },
        "comparison": _diff_block(left, right),
    }

    out_path = Path(out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare graph structure metrics across two graph pipelines")
    parser.add_argument("--left-name", default="current_project")
    parser.add_argument("--left-graph-file", required=True, default="outputs/graph/qa_aligned_graph.json")
    parser.add_argument("--left-communities-file", default="")
    parser.add_argument("--left-stats-file", default="")

    parser.add_argument("--right-name", default="youtu_graphrag")
    parser.add_argument("--right-graph-file", required=True,default='youtu-graphrag/output/graphs/cuad_new.json')
    parser.add_argument("--right-communities-file", default="")
    parser.add_argument("--right-stats-file", default="")

    parser.add_argument(
        "--community-level",
        default="auto",
        help="Community level used for modularity. Use integer or 'auto' (default: smallest available level)",
    )
    parser.add_argument(
        "--community-mode",
        default="consistent",
        choices=["consistent", "provided", "auto"],
        help=(
            "Community comparison mode. 'consistent' (default) uses provided communities only when both sides "
            "supply them; otherwise it forces auto partition on both sides for fairer comparison."
        ),
    )
    parser.add_argument("--out-json", default="outputs/results/graph_structure_compare.json")
    args = parser.parse_args()

    level: int | None
    if str(args.community_level).strip().lower() == "auto":
        level = None
    else:
        level = int(args.community_level)

    out = run_compare(
        left_name=args.left_name,
        left_graph_file=args.left_graph_file,
        left_communities_file=str(args.left_communities_file).strip() or None,
        right_name=args.right_name,
        right_graph_file=args.right_graph_file,
        right_communities_file=str(args.right_communities_file).strip() or None,
        community_level=level,
        community_mode=str(args.community_mode).strip().lower(),
        left_stats_file=str(args.left_stats_file).strip() or None,
        right_stats_file=str(args.right_stats_file).strip() or None,
        out_json=args.out_json,
    )
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
