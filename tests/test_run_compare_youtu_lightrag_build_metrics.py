import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for candidate in (ROOT, SRC):
    value = str(candidate)
    if value not in sys.path:
        sys.path.insert(0, value)

from src.experiments.run_compare_youtu_lightrag import _extract_graph_build_stats


def test_extract_graph_build_stats_exposes_construct_total_wall_time_and_graph_size():
    stats = _extract_graph_build_stats(
        {
            "triple_extract": {
                "llm_calls": 10,
                "wall_time_ms": 1200,
                "telemetry": {
                    "llm_latency_ms": 5000,
                    "embedding_latency_ms": 0,
                    "embedding_calls": 0,
                    "prompt_tokens": 100,
                    "completion_tokens": 20,
                    "total_tokens": 120,
                },
            },
            "graph_build": {
                "wall_time_ms": 40,
                "num_nodes": 12,
                "num_edges": 18,
            },
            "community_build": {
                "wall_time_ms": 300,
                "telemetry": {
                    "llm_calls": 2,
                    "llm_latency_ms": 700,
                    "embedding_latency_ms": 0,
                    "embedding_calls": 0,
                    "prompt_tokens": 30,
                    "completion_tokens": 10,
                    "total_tokens": 40,
                },
            },
        }
    )

    assert stats["total_tokens"] == 160
    assert stats["latency_ms_total"] == 5700
    assert stats["wall_time_ms_total"] == 1540
    assert stats["num_nodes"] == 12
    assert stats["num_edges"] == 18
    assert stats["phase_breakdown"]["construct_total"]["wall_time_ms"] == 1540
    assert stats["phase_breakdown"]["graph_build"]["wall_time_ms"] == 40
