from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.evaluation.build_cuad_capability_qa import build_cuad_capability_qa


class TestBuildCuadCapabilityQaLocalMultiTarget(unittest.TestCase):
    def test_local_factual_builds_list_question_for_multi_target_group(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            graph_file = base / "graph.json"
            queries_out = base / "queries.jsonl"
            gold_out = base / "gold.jsonl"

            graph = {
                "edges": [
                    {
                        "edge_id": "e1",
                        "source": "Acme",
                        "relation": "governing_law",
                        "target": "New York",
                        "weight": 3,
                        "mentions": [
                            {
                                "doc_id": "DocA#p0",
                                "chunk_id": "DocA#c1",
                                "evidence": "Acme governing law is New York under this contract.",
                            }
                        ],
                    },
                    {
                        "edge_id": "e2",
                        "source": "ICC",
                        "relation": "grants_license_to",
                        "target": "Beta Ltd",
                        "weight": 5,
                        "mentions": [
                            {
                                "doc_id": "DocB#p0",
                                "chunk_id": "DocB#c1",
                                "evidence": "ICC grants license to Beta Ltd for distribution rights.",
                            }
                        ],
                    },
                    {
                        "edge_id": "e3",
                        "source": "ICC",
                        "relation": "grants_license_to",
                        "target": "Gamma LLC",
                        "weight": 4,
                        "mentions": [
                            {
                                "doc_id": "DocB#p1",
                                "chunk_id": "DocB#c2",
                                "evidence": "ICC grants license to Gamma LLC under the same agreement.",
                            }
                        ],
                    },
                ]
            }
            graph_file.write_text(json.dumps(graph, ensure_ascii=False), encoding="utf-8")

            stats = build_cuad_capability_qa(
                graph_file=str(graph_file),
                communities_file=None,
                queries_out=str(queries_out),
                gold_out=str(gold_out),
                per_type=2,
                random_seed=7,
                question_style="template",
            )
            self.assertGreaterEqual(stats["num_queries"], 2)

            query_rows = [json.loads(x) for x in queries_out.read_text(encoding="utf-8").splitlines() if x.strip()]
            gold_rows = [json.loads(x) for x in gold_out.read_text(encoding="utf-8").splitlines() if x.strip()]
            local_pairs = [
                (q, g)
                for q, g in zip(query_rows, gold_rows)
                if str(q.get("type", "")).strip() == "local_factual"
            ]
            self.assertEqual(len(local_pairs), 2)

            list_rows = [(q, g) for q, g in local_pairs if "List all `grants_license_to` targets" in str(q.get("query", ""))]
            self.assertEqual(len(list_rows), 1)
            list_gold = list_rows[0][1]
            self.assertIsInstance(list_gold.get("answer"), list)
            self.assertEqual(set(list_gold["answer"]), {"Beta Ltd", "Gamma LLC"})

if __name__ == "__main__":
    unittest.main()
