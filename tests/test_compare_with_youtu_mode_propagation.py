from __future__ import annotations

import json
import tempfile
import unittest
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from experiments import run_compare as run_compare_mod


class TestCompareWithYoutuModePropagation(unittest.TestCase):
    def test_compare_output_contains_mode_and_method_answer_mode(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            queries_file = root / "queries.jsonl"
            out_file = root / "compare_answers.jsonl"
            metrics_file = root / "compare_metrics.json"

            queries_file.write_text(
                json.dumps({"qid": "q1", "type": "local_factual", "query": "Q?"}, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            def _fake_retrieve(**kwargs):  # noqa: ANN001
                return ([{"doc_id": "DocA#p0", "chunk_id": "c1", "text": "ctx"}], {"embedding": {}})

            def _fake_answer_with_context(**kwargs):  # noqa: ANN001
                return "vec_answer", {}

            def _fake_kg(**kwargs):  # noqa: ANN001
                return {"answer": "kg_answer", "answer_mode": kwargs.get("answer_mode"), "telemetry": {}}

            def _fake_graph(**kwargs):  # noqa: ANN001
                return {"answer": "graph_answer", "answer_mode": kwargs.get("answer_mode"), "telemetry": {}}

            def _fake_youtu(**kwargs):  # noqa: ANN001
                return {"answer": "youtu_answer", "answer_mode": kwargs.get("answer_mode"), "telemetry": {}}

            with patch.object(run_compare_mod, "ensure_graph_assets", return_value={}):
                with patch.object(run_compare_mod, "_load_yaml", return_value={}):
                    with patch.object(run_compare_mod, "_detect_max_precomputed_summary_level", return_value=None):
                        with patch.object(run_compare_mod, "retrieve_with_evidence", side_effect=_fake_retrieve):
                            with patch.object(run_compare_mod, "answer_with_context", side_effect=_fake_answer_with_context):
                                with patch.object(run_compare_mod, "answer_with_kg", side_effect=_fake_kg):
                                    with patch.object(run_compare_mod, "answer_with_graphrag", side_effect=_fake_graph):
                                        with patch.object(run_compare_mod, "answer_with_youtu_graphrag", side_effect=_fake_youtu):
                                            with patch.object(run_compare_mod, "_ensure_youtu_graph_assets", return_value={}):
                                                with patch.object(run_compare_mod, "_YOUTU_AVAILABLE", True):
                                                    with patch.object(
                                                        run_compare_mod,
                                                        "compute_graph_structure_metrics",
                                                        return_value={},
                                                    ):
                                                        summary = run_compare_mod.run_compare(
                                                            queries_file=str(queries_file),
                                                            chunks_file="unused_chunks.jsonl",
                                                            idx_file="unused.idx",
                                                            store_file="unused_store.json",
                                                            triples_file="unused_triples.jsonl",
                                                            graph_file="unused_graph.json",
                                                            communities_file="unused_communities.json",
                                                            top_k=10,
                                                            out_file=str(out_file),
                                                            metrics_file=str(metrics_file),
                                                            regimes="best_effort",
                                                            budget_config_file="unused_budget.yaml",
                                                            warmup_graphrag=False,
                                                            include_youtu=True,
                                                            answer_mode="open",
                                                        )

            rows = [json.loads(x) for x in out_file.read_text(encoding="utf-8").splitlines() if x.strip()]
            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(row["mode"], "open")
            self.assertEqual(int(row["top_k"]), 10)
            regime_payload = row["regimes"]["best_effort"]
            self.assertEqual(regime_payload["vector_rag"]["answer_mode"], "open")
            self.assertEqual(regime_payload["kg_rag"]["answer_mode"], "open")
            self.assertEqual(regime_payload["graph_rag"]["answer_mode"], "open")
            self.assertEqual(regime_payload["youtu_graph_rag"]["answer_mode"], "open")
            self.assertEqual(summary["answer_mode"], "open")
            self.assertEqual(summary["answer_modes"], ["open"])


if __name__ == "__main__":
    unittest.main()
