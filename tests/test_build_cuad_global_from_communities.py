from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.evaluation.build_cuad_global_from_communities import build_cuad_global_from_communities


class TestBuildCuadGlobalFromCommunities(unittest.TestCase):
    def test_global_builder_outputs_grounded_global_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            graph_file = base / "graph.json"
            communities_file = base / "communities.json"
            queries_out = base / "queries.jsonl"
            gold_out = base / "gold.jsonl"

            graph = {
                "edges": [
                    {
                        "edge_id": "e1",
                        "source": "Buyer",
                        "relation": "inspects",
                        "target": "Product",
                        "weight": 3,
                        "mentions": [
                            {
                                "doc_id": "DOCX-TRANSPORT-AGREEMENT#p0",
                                "chunk_id": "c_good_1",
                                "evidence": (
                                    "Under this transport agreement, the buyer must inspect delivered products, pay invoices "
                                    "on time, maintain governance controls, and follow compliance obligations before acceptance."
                                ),
                            }
                        ],
                    },
                    {
                        "edge_id": "e2",
                        "source": "Buyer",
                        "relation": "pays",
                        "target": "Invoices",
                        "weight": 2,
                        "mentions": [
                            {
                                "doc_id": "DOCX-TRANSPORT-AGREEMENT#p0",
                                "chunk_id": "c_good_2",
                                "evidence": (
                                    "DOCX TRANSPORT AGREEMENT requires the buyer to pay invoices within ten days, dispute charges promptly, "
                                    "and preserve audit-ready billing records for risk review."
                                ),
                            }
                        ],
                    },
                ]
            }
            communities = {
                "communities": [
                    {
                        "community_id": "l0_c0001",
                        "summary": "Transport obligations and payment risks.",
                        "edges": ["e1", "e2"],
                    }
                ]
            }
            graph_file.write_text(json.dumps(graph, ensure_ascii=False), encoding="utf-8")
            communities_file.write_text(json.dumps(communities, ensure_ascii=False), encoding="utf-8")

            stats = build_cuad_global_from_communities(
                graph_file=str(graph_file),
                communities_file=str(communities_file),
                queries_out=str(queries_out),
                gold_out=str(gold_out),
                per_type=1,
                random_seed=7,
                question_style="template",
                global_summary_min_high_quality_chunks=2,
                global_summary_target_high_quality_chunks=2,
                global_summary_doc_consistency_min_hits=1,
            )

            self.assertEqual(stats["by_type"]["global_summary"], 1)
            query_rows = [json.loads(line) for line in queries_out.read_text(encoding="utf-8").splitlines() if line.strip()]
            gold_rows = [json.loads(line) for line in gold_out.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(query_rows), 1)
            self.assertEqual(len(gold_rows), 1)
            self.assertEqual(query_rows[0]["type"], "global_summary")
            self.assertTrue(gold_rows[0]["supporting_communities"])
            self.assertEqual(len(gold_rows[0]["supporting_chunks"]), 2)
            self.assertIn("buyer", gold_rows[0]["answer"].lower())


if __name__ == "__main__":
    unittest.main()
