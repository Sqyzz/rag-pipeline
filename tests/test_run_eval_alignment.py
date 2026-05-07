from __future__ import annotations

import unittest

from src.evaluation.run_eval import (
    _build_alignment_preflight,
    _collect_gold_ids,
    _collect_pred_ids,
    _collect_pred_ranked_ids,
    _dedup_compare_eval_rows,
    _extract_pred_doc_ids,
    _hit_and_recall_at_k,
)


class TestRunEvalAlignment(unittest.TestCase):
    def test_extract_pred_doc_ids_ignores_none_like_values(self) -> None:
        payload = {
            "evidence": [
                {"doc_id": None},
                {"doc_id": "None"},
                {"doc_id": "null"},
                {"doc_id": "ContractA#p1"},
            ],
            "evidence_chunks": [
                {"doc_id": ""},
                {"doc_id": "ContractB#p0"},
            ],
        }
        got = _extract_pred_doc_ids(payload)
        self.assertEqual(got, ["ContractA", "ContractB"])

    def test_alignment_preflight_emits_youtu_alerts(self) -> None:
        rows = [
            {
                "method": "youtu_graph_rag",
                "raw_evidence_chunks": 10,
                "raw_evidence_chunks_empty_chunk_id": 10,
                "raw_subgraph_edges": 5,
                "raw_subgraph_edges_empty_edge_id": 5,
            },
            {
                "method": "vector_rag",
                "raw_evidence_chunks": 10,
                "raw_evidence_chunks_empty_chunk_id": 0,
                "raw_subgraph_edges": 0,
                "raw_subgraph_edges_empty_edge_id": 0,
            },
        ]
        out = _build_alignment_preflight(rows)
        self.assertTrue(any("youtu_graph_rag" in x for x in out.get("alerts", [])))

    def test_dedup_keeps_rows_when_mode_or_topk_differs(self) -> None:
        rows = [
            {"qid": "q1", "regime": "best_effort", "method": "vector_rag", "mode": "reject", "top_k": 10},
            {"qid": "q1", "regime": "best_effort", "method": "vector_rag", "mode": "open", "top_k": 10},
            {"qid": "q1", "regime": "best_effort", "method": "vector_rag", "mode": "open", "top_k": 20},
        ]
        deduped = _dedup_compare_eval_rows(rows)
        self.assertEqual(len(deduped), 3)

    def test_chunk_id_union_matches_any_original_chunk(self) -> None:
        payload = {
            "evidence_chunks": [
                {"chunk_id": "c1|c2|c3", "doc_id": "DocA#p0", "text": "ctx"},
            ]
        }
        pred_ids = _collect_pred_ids(payload)
        ranked_ids = _collect_pred_ranked_ids(payload)
        gold_ids = _collect_gold_ids({"supporting_chunks": [{"chunk_id": "c2", "doc_id": "DocA#p0"}]})

        self.assertEqual(pred_ids["chunks"], {"c1", "c2", "c3"})
        self.assertEqual(ranked_ids["chunks"], ["c1", "c2", "c3"])
        hit, recall = _hit_and_recall_at_k(ranked_ids["chunks"], gold_ids["chunks"], 20)
        self.assertEqual(hit, 1.0)
        self.assertEqual(recall, 1.0)

    def test_collect_gold_ids_falls_back_to_query_doc_key_for_doc_level(self) -> None:
        gold_ids = _collect_gold_ids(
            {
                "meta": {
                    "query_doc_key": "ContractA#p0",
                    "title": "IgnoredTitle",
                },
                "supporting_chunks": [],
            }
        )
        self.assertEqual(gold_ids["docs"], {"ContractA"})
        self.assertEqual(gold_ids["chunks"], set())


if __name__ == "__main__":
    unittest.main()
