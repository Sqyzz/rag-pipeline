from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from baselines.graph_rag import answer_with_graphrag
from baselines.kg_rag import answer_with_kg


def _load_queries(path: str, limit: int | None = None) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def _write_json(path: str, payload: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_ablations(
    queries_file: str,
    graph_file: str,
    communities_file: str,
    store_file: str,
    out_file: str,
    limit: int | None = None,
) -> dict:
    queries = _load_queries(queries_file, limit=limit)

    kg_variants = {
        "kg_baseline": {
            "max_hops": 2,
            "max_start_entities": 5,
            "top_edges": 60,
            "use_entity_linking": True,
            "use_embedding_rerank": True,
        },
        "kg_hop_1": {"max_hops": 1, "use_entity_linking": True, "use_embedding_rerank": True},
        "kg_hop_2": {"max_hops": 2, "use_entity_linking": True, "use_embedding_rerank": True},
        "kg_hop_3": {"max_hops": 3, "use_entity_linking": True, "use_embedding_rerank": True},
        "kg_no_entity_linking": {"max_hops": 2, "use_entity_linking": False, "use_embedding_rerank": True},
        "kg_traversal_pruning": {
            "max_hops": 2,
            "top_edges": 20,
            "max_chunks": 6,
            "max_context_chars": 3600,
            "use_entity_linking": True,
            "use_embedding_rerank": True,
        },
    }

    graph_variants = {
        "graph_baseline": {
            "top_communities": 3,
            "query_level": -1,
            "use_hierarchy": True,
            "use_community_summaries": True,
            "shuffle_communities": True,
            "use_map_reduce": True,
        },
        "graph_no_community_summaries": {
            "top_communities": 3,
            "query_level": -1,
            "use_hierarchy": True,
            "use_community_summaries": False,
            "shuffle_communities": True,
            "use_map_reduce": True,
        },
        "graph_no_hierarchy": {
            "top_communities": 3,
            "query_level": -1,
            "use_hierarchy": False,
            "use_community_summaries": True,
            "shuffle_communities": True,
            "use_map_reduce": True,
        },
        "graph_no_shuffle": {
            "top_communities": 3,
            "query_level": -1,
            "use_hierarchy": True,
            "use_community_summaries": True,
            "shuffle_communities": False,
            "use_map_reduce": True,
        },
        "graph_no_map_reduce": {
            "top_communities": 3,
            "query_level": -1,
            "use_hierarchy": True,
            "use_community_summaries": True,
            "shuffle_communities": True,
            "use_map_reduce": False,
        },
    }

    results = []
    for q in queries:
        query = q["query"]
        qid = q.get("qid")
        qtype = q.get("type")
        row = {"qid": qid, "type": qtype, "query": query, "kg": {}, "graph": {}}

        for name, kwargs in kg_variants.items():
            out = answer_with_kg(
                query=query,
                graph_file=graph_file,
                store_file=store_file,
                **kwargs,
            )
            row["kg"][name] = {
                "answer": out["answer"],
                "telemetry": out["telemetry"],
                "max_hops": out["max_hops"],
                "linked_entities": out.get("linked_entities", []),
                "evidence_count": len(out.get("evidence", [])),
            }

        for name, kwargs in graph_variants.items():
            out = answer_with_graphrag(
                query=query,
                graph_file=graph_file,
                communities_file=communities_file,
                **kwargs,
            )
            row["graph"][name] = {
                "answer": out["answer"],
                "telemetry": out["telemetry"],
                "communities": out.get("communities", []),
                "evidence_count": len(out.get("evidence", [])),
                "map_partial_count": len(out.get("map_partial_answers", [])),
            }

        results.append(row)

    summary = {
        "queries_file": queries_file,
        "num_queries": len(results),
        "kg_variants": list(kg_variants.keys()),
        "graph_variants": list(graph_variants.keys()),
        "results": results,
    }
    _write_json(out_file, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run KG/Graph mechanism ablations")
    parser.add_argument("--queries-file", default="data/queries/queries.jsonl")
    parser.add_argument("--graph-file", default="outputs/graph/graph.json")
    parser.add_argument("--communities-file", default="outputs/graph/communities.json")
    parser.add_argument("--store-file", default="outputs/indexes/chunk_store_sampled.json")
    parser.add_argument("--out-file", default="outputs/results/ablation_results.json")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    summary = run_ablations(
        queries_file=args.queries_file,
        graph_file=args.graph_file,
        communities_file=args.communities_file,
        store_file=args.store_file,
        out_file=args.out_file,
        limit=args.limit,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
