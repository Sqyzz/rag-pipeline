from __future__ import annotations

import argparse
import json
import time
import re
from collections import Counter
from collections import defaultdict
from pathlib import Path
from typing import Any


def _progress(message: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[build_graph {ts}] {message}", flush=True)


def _doc_prefix(value: str | None) -> str:
    return str(value or "").split("#", 1)[0].strip()


def _normalize_entity_for_merge(value: str, mode: str) -> str:
    t = str(value or "").strip()
    if mode == "exact":
        return t
    t = re.sub(r"\s+", " ", t).strip().casefold()
    if mode == "normalized":
        t = t.strip("`\"' ")
        t = re.sub(r"^[,;:.()\[\]{}]+|[,;:.()\[\]{}]+$", "", t).strip()
    return t


def _entity_key(
    text: str,
    entity_type: str | None,
    node_merge_mode: str,
    node_scope_with_type: bool,
) -> tuple[str, str]:
    base = _normalize_entity_for_merge(text, mode=node_merge_mode)
    et = str(entity_type or "").strip().lower()
    if not node_scope_with_type or not et or et == "unknown":
        return (base, "")
    return (base, et)


def build_graph(
    triples_file: str,
    out_file: str,
    edge_merge_mode: str = "global",
    node_merge_mode: str = "normalized",
    node_scope_with_type: bool = True,
) -> dict[str, Any]:
    if edge_merge_mode not in {"global", "doc_scoped"}:
        raise ValueError("edge_merge_mode must be one of: global, doc_scoped")
    if node_merge_mode not in {"exact", "casefold", "normalized"}:
        raise ValueError("node_merge_mode must be one of: exact, casefold, normalized")
    raw_node_set = set()
    node_map: dict[tuple[str, str], dict[str, Any]] = {}
    edge_map: dict[tuple[Any, ...], dict[str, Any]] = {}
    adjacency = defaultdict(list)

    _progress(f"start triples_file={triples_file}")
    total_lines = 0
    with open(triples_file, encoding="utf-8") as f_count:
        for _ in f_count:
            total_lines += 1
    _progress(f"input_triples_lines={total_lines}")

    with open(triples_file, encoding="utf-8") as f:
        for line in f:
            t = json.loads(line)
            s = str(t.get("subject", "")).strip()
            r = str(t.get("relation", "")).strip()
            o = str(t.get("object", "")).strip()
            st = str(t.get("subject_type", "")).strip() or "unknown"
            ot = str(t.get("object_type", "")).strip() or "unknown"
            doc_key = _doc_prefix(t.get("doc_id"))
            if not (s and r and o):
                continue
            raw_node_set.add(s)
            raw_node_set.add(o)

            s_key = _entity_key(
                s,
                entity_type=st,
                node_merge_mode=node_merge_mode,
                node_scope_with_type=node_scope_with_type,
            )
            o_key = _entity_key(
                o,
                entity_type=ot,
                node_merge_mode=node_merge_mode,
                node_scope_with_type=node_scope_with_type,
            )
            if s_key not in node_map:
                node_map[s_key] = {
                    "id": s,
                    "label": s,
                    "_aliases": Counter(),
                    "_types": Counter(),
                    "mention_count": 0,
                }
            if o_key not in node_map:
                node_map[o_key] = {
                    "id": o,
                    "label": o,
                    "_aliases": Counter(),
                    "_types": Counter(),
                    "mention_count": 0,
                }
            s_node = node_map[s_key]
            o_node = node_map[o_key]
            s_node["_aliases"][s] += 1
            o_node["_aliases"][o] += 1
            s_node["_types"][st] += 1
            o_node["_types"][ot] += 1
            s_node["mention_count"] += 1
            o_node["mention_count"] += 1
            s_id = s_node["id"]
            o_id = o_node["id"]

            if edge_merge_mode == "doc_scoped":
                key = (s_id, r, o_id, doc_key)
            else:
                key = (s_id, r, o_id)
            if key not in edge_map:
                edge_map[key] = {
                    "edge_id": f"e{len(edge_map) + 1:06d}",
                    "source": s_id,
                    "relation": r,
                    "target": o_id,
                    "weight": 0,
                    "mentions": [],
                    "_source_types": Counter(),
                    "_target_types": Counter(),
                }
            edge = edge_map[key]
            edge["weight"] += 1
            edge["_source_types"][st] += 1
            edge["_target_types"][ot] += 1
            edge["mentions"].append(
                {
                    "chunk_id": t.get("chunk_id"),
                    "doc_id": t.get("doc_id"),
                    "evidence": t.get("evidence", ""),
                    "subject_type": st,
                    "object_type": ot,
                }
            )
            if total_lines > 0 and len(edge_map) % 500 == 0 and len(edge_map) > 0:
                _progress(f"unique_edges={len(edge_map)} / nodes={len(node_map)}")

    edges = sorted(edge_map.values(), key=lambda x: x["weight"], reverse=True)
    for edge in edges:
        edge["source_types"] = [k for k, _ in edge["_source_types"].most_common()]
        edge["target_types"] = [k for k, _ in edge["_target_types"].most_common()]
        edge.pop("_source_types", None)
        edge.pop("_target_types", None)
        adjacency[edge["source"]].append(
            {
                "edge_id": edge["edge_id"],
                "relation": edge["relation"],
                "target": edge["target"],
                "weight": edge["weight"],
            }
        )

    nodes = []
    for node in sorted(node_map.values(), key=lambda x: x["id"]):
        aliases = [k for k, _ in node["_aliases"].most_common()]
        types = [k for k, _ in node["_types"].most_common()]
        nodes.append(
            {
                "id": node["id"],
                "label": node["label"],
                "types": types,
                "mention_count": int(node["mention_count"]),
                "aliases": aliases[:10],
            }
        )

    graph = {"nodes": nodes, "edges": edges, "adjacency": dict(adjacency)}
    out_path = Path(out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
    _progress(f"done num_nodes={len(nodes)}, num_edges={len(edges)} -> {out_file}")
    merge_ratio = 0.0
    if raw_node_set:
        merge_ratio = round(1.0 - (len(nodes) / len(raw_node_set)), 6)
    return {
        "triples_file": triples_file,
        "graph_file": out_file,
        "edge_merge_mode": edge_merge_mode,
        "node_merge_mode": node_merge_mode,
        "node_scope_with_type": bool(node_scope_with_type),
        "raw_unique_nodes": len(raw_node_set),
        "num_nodes": len(nodes),
        "num_edges": len(edges),
        "node_merge_ratio": merge_ratio,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build graph json from triples jsonl")
    parser.add_argument("--triples-file", default="outputs/graph/triples.jsonl")
    parser.add_argument("--out-file", default="outputs/graph/graph.json")
    parser.add_argument("--edge-merge-mode", choices=["global", "doc_scoped"], default="global")
    parser.add_argument(
        "--node-merge-mode",
        choices=["exact", "casefold", "normalized"],
        default="normalized",
        help="Node merge strategy: exact text, case-insensitive, or light normalized text.",
    )
    parser.add_argument(
        "--node-scope-with-type",
        choices=["true", "false"],
        default="true",
        help="Whether node merge keys include entity type to reduce cross-type collisions.",
    )
    parser.add_argument("--metrics-file", default="outputs/results/graph_build_metrics.json")
    args = parser.parse_args()
    metrics = build_graph(
        args.triples_file,
        args.out_file,
        edge_merge_mode=args.edge_merge_mode,
        node_merge_mode=args.node_merge_mode,
        node_scope_with_type=str(args.node_scope_with_type).strip().lower() in {"1", "true", "yes", "y", "on"},
    )
    metrics_path = Path(args.metrics_file)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
if __name__ == "__main__":
    main()
