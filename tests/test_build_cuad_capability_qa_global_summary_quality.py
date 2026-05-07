from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.evaluation.build_cuad_capability_qa import build_cuad_capability_qa


class TestBuildCuadCapabilityQaGlobalSummaryQuality(unittest.TestCase):
    def test_global_summary_requires_min_high_quality_chunks(self) -> None:
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
                                "chunk_id": "c_good",
                                "evidence": (
                                    "Under this transport agreement, the buyer must inspect delivered products, "
                                    "pay invoices on time, and follow compliance obligations for risk control and governance."
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
                                "chunk_id": "c_short",
                                "evidence": "Buyer pays.",
                            }
                        ],
                    },
                    {
                        "edge_id": "e3",
                        "source": "Seller",
                        "relation": "located_in",
                        "target": "Region",
                        "weight": 1,
                        "mentions": [
                            {
                                "doc_id": "DOCX-TRANSPORT-AGREEMENT#p0",
                                "chunk_id": "c_inconsistent",
                                "evidence": (
                                    "This biological assay protocol describes laboratory sequencing steps, "
                                    "microscopy setup, and reagent calibration in a clinical environment."
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
                        "edges": ["e1", "e2", "e3"],
                    }
                ]
            }
            graph_file.write_text(json.dumps(graph, ensure_ascii=False), encoding="utf-8")
            communities_file.write_text(json.dumps(communities, ensure_ascii=False), encoding="utf-8")

            stats_strict = build_cuad_capability_qa(
                graph_file=str(graph_file),
                communities_file=str(communities_file),
                queries_out=str(queries_out),
                gold_out=str(gold_out),
                per_type=1,
                random_seed=7,
                question_style="template",
                global_summary_min_high_quality_chunks=2,
                global_summary_target_high_quality_chunks=3,
                global_summary_doc_consistency_min_hits=1,
            )
            self.assertEqual(int((stats_strict.get("by_type") or {}).get("global_summary", 0)), 0)

            stats_relaxed = build_cuad_capability_qa(
                graph_file=str(graph_file),
                communities_file=str(communities_file),
                queries_out=str(queries_out),
                gold_out=str(gold_out),
                per_type=1,
                random_seed=7,
                question_style="template",
                global_summary_min_high_quality_chunks=1,
                global_summary_target_high_quality_chunks=3,
                global_summary_doc_consistency_min_hits=1,
            )
            self.assertEqual(int((stats_relaxed.get("by_type") or {}).get("global_summary", 0)), 1)


if __name__ == "__main__":
    unittest.main()
