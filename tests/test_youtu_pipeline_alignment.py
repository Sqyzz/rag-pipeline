from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.adapters.youtu_graph_state import build_state_payload, decide_graph_reuse, save_graph_state
from src.baselines.youtu_graph_rag_adapter import map_youtu_to_graph_payload


class TestYoutuPipelineAlignment(unittest.TestCase):
    def test_map_youtu_to_payload_has_required_keys(self) -> None:
        response = {
            "data": {
                "answer": "A",
                "retrieved_chunks": [{"chunk_id": "c1", "text": "chunk text"}],
                "retrieved_triples": [{"edge_id": "e1", "source": "s", "relation": "r", "target": "t"}],
                "communities": [{"community_id": "cm1", "summary": "sum"}],
            },
            "meta": {
                "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
                "llm_calls": [{"latency_ms": 10}],
                "embedding_calls": [{"latency_ms": 20}],
            },
        }
        payload = map_youtu_to_graph_payload(
            response,
            query_level=0,
            use_hierarchy=True,
            use_community_summaries=True,
            shuffle_communities=False,
            use_map_reduce=True,
            map_keypoints_limit=5,
            max_evidence=12,
        )

        for key in (
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
        ):
            self.assertIn(key, payload)

        self.assertEqual(payload["answer"], "A")
        self.assertEqual(payload["telemetry"]["total_tokens"], 3)
        self.assertEqual(payload["telemetry"]["llm_calls"], 1)

    def test_graph_reuse_second_run_hits_cache(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            chunks = root / "chunks.jsonl"
            graph = root / "graph.json"
            communities = root / "communities.json"
            state = root / "youtu_graph_state.json"

            chunks.write_text('{"chunk_id":"c1","text":"hello"}\n', encoding="utf-8")
            graph.write_text("{}", encoding="utf-8")
            communities.write_text("{}", encoding="utf-8")

            params = {"dataset": "enterprise", "sync_mode": "none"}
            first = decide_graph_reuse(
                graph_state_file=str(state),
                chunks_file=str(chunks),
                dataset="enterprise",
                build_params=params,
                reuse_graph=True,
                force_rebuild=False,
                require_local_assets=True,
                graph_file=str(graph),
                communities_file=str(communities),
            )
            self.assertFalse(first["used_cached_graph"])

            save_graph_state(
                str(state),
                build_state_payload(
                    dataset="enterprise",
                    fingerprint=first["fingerprint"],
                    build_params=params,
                    graph_task_id="task-1",
                ),
            )

            second = decide_graph_reuse(
                graph_state_file=str(state),
                chunks_file=str(chunks),
                dataset="enterprise",
                build_params=params,
                reuse_graph=True,
                force_rebuild=False,
                require_local_assets=True,
                graph_file=str(graph),
                communities_file=str(communities),
            )
            self.assertTrue(second["used_cached_graph"])
            self.assertEqual(second["reason"], "fingerprint_match")


if __name__ == "__main__":
    unittest.main()
