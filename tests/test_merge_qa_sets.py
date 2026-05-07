from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.evaluation.merge_qa_sets import merge_qa_sets


class TestMergeQaSets(unittest.TestCase):
    def test_merge_qa_sets_merges_without_qid_conflict(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            local_queries = base / "local_queries.jsonl"
            local_gold = base / "local_gold.jsonl"
            global_queries = base / "global_queries.jsonl"
            global_gold = base / "global_gold.jsonl"
            out_queries = base / "queries.jsonl"
            out_gold = base / "gold.jsonl"

            local_queries.write_text(json.dumps({"qid": "q1", "type": "local_factual", "query": "Q1", "meta": {}}) + "\n", encoding="utf-8")
            local_gold.write_text(json.dumps({"qid": "q1", "type": "local_factual", "query": "Q1", "answer": "A1", "meta": {}}) + "\n", encoding="utf-8")
            global_queries.write_text(json.dumps({"qid": "q2", "type": "global_summary", "query": "Q2", "meta": {}}) + "\n", encoding="utf-8")
            global_gold.write_text(json.dumps({"qid": "q2", "type": "global_summary", "query": "Q2", "answer": "A2", "meta": {}}) + "\n", encoding="utf-8")

            stats = merge_qa_sets(
                query_files=[str(local_queries), str(global_queries)],
                gold_files=[str(local_gold), str(global_gold)],
                out_queries_file=str(out_queries),
                out_gold_file=str(out_gold),
            )

            self.assertEqual(stats["num_queries"], 2)
            self.assertEqual(stats["type_counts"], {"local_factual": 1, "global_summary": 1})

    def test_merge_qa_sets_rejects_qid_conflicts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            query_a = base / "query_a.jsonl"
            gold_a = base / "gold_a.jsonl"
            query_b = base / "query_b.jsonl"
            gold_b = base / "gold_b.jsonl"

            row_query = {"qid": "dup", "type": "local_factual", "query": "Q", "meta": {}}
            row_gold = {"qid": "dup", "type": "local_factual", "query": "Q", "answer": "A", "meta": {}}
            query_a.write_text(json.dumps(row_query) + "\n", encoding="utf-8")
            gold_a.write_text(json.dumps(row_gold) + "\n", encoding="utf-8")
            query_b.write_text(json.dumps({**row_query, "type": "global_summary"}) + "\n", encoding="utf-8")
            gold_b.write_text(json.dumps({**row_gold, "type": "global_summary"}) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Duplicate qid across QA sets"):
                merge_qa_sets(
                    query_files=[str(query_a), str(query_b)],
                    gold_files=[str(gold_a), str(gold_b)],
                    out_queries_file=str(base / "out_queries.jsonl"),
                    out_gold_file=str(base / "out_gold.jsonl"),
                )


if __name__ == "__main__":
    unittest.main()
