from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

try:
    from adapters.youtu_graph_state import build_state_payload, decide_graph_reuse, save_graph_state
    from experiments.run_youtu_graphrag_test import _merge_telemetry
    from utils.telemetry import Telemetry
except ModuleNotFoundError:
    from src.adapters.youtu_graph_state import build_state_payload, decide_graph_reuse, save_graph_state
    from src.experiments.run_youtu_graphrag_test import _merge_telemetry
    from src.utils.telemetry import Telemetry

REQUIRED_GRAPH_KEYS = {
    "answer",
    "communities",
    "community_summaries",
    "query_level",
    "use_hierarchy",
    "use_community_summaries",
    "shuffle_communities",
    "use_map_reduce",
    "map_keypoints_limit",
    "map_partial_answers",
    "subgraph_edges",
    "evidence",
    "evidence_chunks",
    "telemetry",
}

TELEMETRY_KEYS = {
    "llm_calls",
    "embedding_calls",
    "llm_latency_ms",
    "embedding_latency_ms",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "extra",
}


def _read_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def validate_schema(results_file: str) -> dict[str, Any]:
    rows = _read_jsonl(results_file)
    errors: list[str] = []

    for i, row in enumerate(rows, start=1):
        regimes = row.get("regimes") or {}
        for rg, methods in regimes.items():
            graph = (methods or {}).get("graph_rag") or {}
            missing = sorted(list(REQUIRED_GRAPH_KEYS - set(graph.keys())))
            if missing:
                errors.append(f"row={i} regime={rg} missing keys: {missing}")
            telemetry = graph.get("telemetry") or {}
            missing_t = sorted(list(TELEMETRY_KEYS - set(telemetry.keys())))
            if missing_t:
                errors.append(f"row={i} regime={rg} telemetry missing keys: {missing_t}")

    return {
        "check": "schema",
        "ok": not errors,
        "num_rows": len(rows),
        "errors": errors,
    }


def validate_aggregation() -> dict[str, Any]:
    t = Telemetry()
    s1 = {
        "llm_calls": 1,
        "embedding_calls": 2,
        "llm_latency_ms": 10,
        "embedding_latency_ms": 20,
        "prompt_tokens": 30,
        "completion_tokens": 40,
        "total_tokens": 70,
    }
    s2 = {
        "llm_calls": 3,
        "embedding_calls": 4,
        "llm_latency_ms": 50,
        "embedding_latency_ms": 60,
        "prompt_tokens": 70,
        "completion_tokens": 80,
        "total_tokens": 150,
    }
    _merge_telemetry(t, s1)
    _merge_telemetry(t, s2)
    merged = t.to_dict()

    ok = (
        merged["llm_calls"] == 4
        and merged["embedding_calls"] == 6
        and merged["llm_latency_ms"] == 60
        and merged["embedding_latency_ms"] == 80
        and merged["prompt_tokens"] == 100
        and merged["completion_tokens"] == 120
        and merged["total_tokens"] == 220
    )
    return {
        "check": "aggregation",
        "ok": ok,
        "merged": merged,
    }


def validate_graph_reuse() -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        chunks = root / "chunks.jsonl"
        chunks.write_text('{"chunk_id":"c1","text":"hello"}\n', encoding="utf-8")
        graph_file = root / "graph.json"
        communities_file = root / "communities.json"
        graph_file.write_text("{}", encoding="utf-8")
        communities_file.write_text("{}", encoding="utf-8")
        state_file = root / "state.json"

        build_params = {"dataset": "enterprise", "sync_mode": "none"}
        first = decide_graph_reuse(
            graph_state_file=str(state_file),
            chunks_file=str(chunks),
            dataset="enterprise",
            build_params=build_params,
            reuse_graph=True,
            force_rebuild=False,
            require_local_assets=True,
            graph_file=str(graph_file),
            communities_file=str(communities_file),
        )
        save_graph_state(
            str(state_file),
            build_state_payload(
                dataset="enterprise",
                fingerprint=first["fingerprint"],
                build_params=build_params,
                graph_task_id="t-1",
            ),
        )
        second = decide_graph_reuse(
            graph_state_file=str(state_file),
            chunks_file=str(chunks),
            dataset="enterprise",
            build_params=build_params,
            reuse_graph=True,
            force_rebuild=False,
            require_local_assets=True,
            graph_file=str(graph_file),
            communities_file=str(communities_file),
        )

    ok = (not first["used_cached_graph"]) and second["used_cached_graph"] and second["reason"] == "fingerprint_match"
    return {
        "check": "graph_reuse",
        "ok": ok,
        "first": {
            "used_cached_graph": first["used_cached_graph"],
            "reason": first["reason"],
        },
        "second": {
            "used_cached_graph": second["used_cached_graph"],
            "reason": second["reason"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate youtu GraphRAG alignment")
    parser.add_argument("--results-file", default="outputs/results/youtu_compare_answers.jsonl")
    args = parser.parse_args()

    checks = [validate_aggregation(), validate_graph_reuse()]
    if Path(args.results_file).exists():
        checks.append(validate_schema(args.results_file))
    else:
        checks.append(
            {
                "check": "schema",
                "ok": False,
                "errors": [f"results file not found: {args.results_file}"],
            }
        )

    ok = all(c.get("ok") for c in checks)
    payload = {
        "ok": ok,
        "checks": checks,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
