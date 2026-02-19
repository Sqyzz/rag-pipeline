from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


def _progress(message: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[build_graph {ts}] {message}", flush=True)


def build_graph(triples_file: str, out_file: str) -> dict[str, Any]:
    node_set = set()
    edge_map: dict[tuple[str, str, str], dict[str, Any]] = {}
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
            if not (s and r and o):
                continue
            key = (s, r, o)
            if key not in edge_map:
                edge_map[key] = {
                    "edge_id": f"e{len(edge_map) + 1:06d}",
                    "source": s,
                    "relation": r,
                    "target": o,
                    "weight": 0,
                    "mentions": [],
                }
            edge = edge_map[key]
            edge["weight"] += 1
            edge["mentions"].append(
                {
                    "chunk_id": t.get("chunk_id"),
                    "doc_id": t.get("doc_id"),
                    "evidence": t.get("evidence", ""),
                }
            )
            node_set.add(s)
            node_set.add(o)
            if total_lines > 0 and len(edge_map) % 500 == 0 and len(edge_map) > 0:
                _progress(f"unique_edges={len(edge_map)} / nodes={len(node_set)}")

    edges = sorted(edge_map.values(), key=lambda x: x["weight"], reverse=True)
    for edge in edges:
        adjacency[edge["source"]].append(
            {
                "edge_id": edge["edge_id"],
                "relation": edge["relation"],
                "target": edge["target"],
                "weight": edge["weight"],
            }
        )

    nodes = [{"id": n, "label": n} for n in sorted(node_set)]
    graph = {"nodes": nodes, "edges": edges, "adjacency": dict(adjacency)}
    out_path = Path(out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
    _progress(f"done num_nodes={len(nodes)}, num_edges={len(edges)} -> {out_file}")
    return {
        "triples_file": triples_file,
        "graph_file": out_file,
        "num_nodes": len(nodes),
        "num_edges": len(edges),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build graph json from triples jsonl")
    parser.add_argument("--triples-file", default="outputs/graph/triples.jsonl")
    parser.add_argument("--out-file", default="outputs/graph/graph.json")
    parser.add_argument("--metrics-file", default="outputs/results/graph_build_metrics.json")
    args = parser.parse_args()
    metrics = build_graph(args.triples_file, args.out_file)
    metrics_path = Path(args.metrics_file)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
if __name__ == "__main__":
    main()
