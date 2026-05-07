from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for candidate in (ROOT, SRC):
    value = str(candidate)
    if value not in sys.path:
        sys.path.insert(0, value)

from src.experiments.run_compare_youtu_lightrag import (
    _build_paper_metrics_from_eval,
    _merge_retrieve_rows_with_existing,
    _run_pair_compare_once,
)


class TestRunCompareYoutuLightRagIncrementalRetrieve(unittest.TestCase):
    def test_build_paper_metrics_from_eval_uses_requested_metric_mapping(self) -> None:
        payload = {
            "summary": {
                "overall": [
                    {
                        "method": "graph_rag",
                        "chunk_recall_at_k": 0.68,
                        "chunk_hit_rate_at_k": 0.74,
                        "doc_recall_at_k": 0.61,
                        "doc_hit_rate_at_k": 0.79,
                    },
                    {
                        "method": "youtu_graph_rag",
                        "chunk_recall_at_k": 0.81,
                        "chunk_hit_rate_at_k": 0.89,
                        "doc_recall_at_k": 0.73,
                        "doc_hit_rate_at_k": 0.91,
                    },
                ],
                "by_type": [
                    {"type": "local_factual", "method": "graph_rag", "answer_token_f1": 0.63},
                    {"type": "local_factual", "method": "youtu_graph_rag", "answer_token_f1": 0.71},
                    {"type": "cross_clause", "method": "graph_rag", "answer_token_f1": 0.49},
                    {"type": "cross_clause", "method": "youtu_graph_rag", "answer_token_f1": 0.66},
                    {"type": "global_summary", "method": "graph_rag", "answer_token_f1": 0.57},
                    {"type": "global_summary", "method": "youtu_graph_rag", "answer_token_f1": 0.64},
                ],
            }
        }

        metrics = _build_paper_metrics_from_eval(payload, top_k=20)

        self.assertEqual(metrics["f1_metrics"]["local_f1"]["graph_rag"], 0.63)
        self.assertEqual(metrics["f1_metrics"]["structural_f1"]["youtu_graph_rag"], 0.66)
        self.assertEqual(metrics["f1_metrics"]["global_f1"]["graph_rag"], 0.57)
        self.assertEqual(metrics["retrieval_metrics"]["chunk_recall_at_20"]["graph_rag"], 0.68)
        self.assertEqual(metrics["retrieval_metrics"]["chunk_hit_at_20"]["youtu_graph_rag"], 0.89)
        self.assertEqual(metrics["retrieval_metrics"]["doc_recall_at_20"]["youtu_graph_rag"], 0.73)
        self.assertEqual(metrics["retrieval_metrics"]["doc_hit_at_20"]["graph_rag"], 0.79)

    def test_merge_retrieve_rows_preserves_unselected_methods(self) -> None:
        current_rows = [
            {
                "qid": "q1",
                "query": "What is Q1?",
                "type": "local_factual",
                "mode": "reject",
                "top_k": 20,
                "regimes": {
                    "best_effort": {
                        "graph_rag": {"answer": "new graph"},
                    }
                },
            }
        ]
        existing_rows = [
            {
                "qid": "q1",
                "query": "What is Q1?",
                "type": "local_factual",
                "mode": "reject",
                "top_k": 20,
                "regimes": {
                    "best_effort": {
                        "graph_rag": {"answer": "old graph"},
                        "lightrag": {"answer": "old light"},
                        "youtu_graph_rag": {"answer": "old youtu"},
                    }
                },
            }
        ]

        rows, incremental = _merge_retrieve_rows_with_existing(
            current_rows=current_rows,
            existing_rows=existing_rows,
            selected_methods={"graph_rag"},
        )

        self.assertTrue(incremental)
        payload = rows[0]["regimes"]["best_effort"]
        self.assertEqual(payload["graph_rag"]["answer"], "new graph")
        self.assertEqual(payload["lightrag"]["answer"], "old light")
        self.assertEqual(payload["youtu_graph_rag"]["answer"], "old youtu")

    def test_merge_retrieve_rows_skips_mismatched_query_shape(self) -> None:
        current_rows = [
            {
                "qid": "q1",
                "query": "Current query",
                "type": "local_factual",
                "mode": "reject",
                "top_k": 20,
                "regimes": {"best_effort": {"graph_rag": {"answer": "new graph"}}},
            }
        ]
        existing_rows = [
            {
                "qid": "q1",
                "query": "Old query",
                "type": "local_factual",
                "mode": "reject",
                "top_k": 20,
                "regimes": {"best_effort": {"lightrag": {"answer": "old light"}}},
            }
        ]

        rows, incremental = _merge_retrieve_rows_with_existing(
            current_rows=current_rows,
            existing_rows=existing_rows,
            selected_methods={"graph_rag"},
        )

        self.assertFalse(incremental)
        payload = rows[0]["regimes"]["best_effort"]
        self.assertEqual(set(payload.keys()), {"graph_rag"})

    def test_run_pair_compare_once_incremental_only_skips_completed_qids(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            queries_file = root / "queries.jsonl"
            out_file = root / "answers.jsonl"
            metrics_file = root / "metrics.json"
            budget_file = root / "budget.yaml"
            budget_file.write_text("", encoding="utf-8")

            queries = [
                {"qid": "q1", "query": "What is Q1?", "type": "local_factual"},
                {"qid": "q2", "query": "What is Q2?", "type": "local_factual"},
            ]
            with queries_file.open("w", encoding="utf-8") as f:
                for row in queries:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

            existing_rows = [
                {
                    "qid": "q1",
                    "query": "What is Q1?",
                    "type": "local_factual",
                    "mode": "reject",
                    "top_k": 20,
                    "regimes": {
                        "best_effort": {
                            "lightrag": {"answer": "cached q1"},
                        }
                    },
                }
            ]
            with out_file.open("w", encoding="utf-8") as f:
                for row in existing_rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

            calls: list[str] = []

            def _fake_answer_with_lightrag(**kwargs):
                calls.append(str(kwargs.get("query", "")))
                return {
                    "answer": f"fresh:{kwargs.get('query', '')}",
                    "telemetry": {"llm_calls": 1, "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }

            with patch("src.experiments.run_compare_youtu_lightrag.answer_with_lightrag", side_effect=_fake_answer_with_lightrag):
                summary = _run_pair_compare_once(
                    queries_file=str(queries_file),
                    chunks_file="data/processed/qa_aligned_chunks.jsonl",
                    graph_file="",
                    communities_file="",
                    top_k=20,
                    out_file=str(out_file),
                    metrics_file=str(metrics_file),
                    regimes="best_effort",
                    budget_config_file=str(budget_file),
                    max_queries=None,
                    answer_mode="reject",
                    selected_rags={"lightrag"},
                    youtu_base_url="http://127.0.0.1:8000",
                    youtu_dataset="cuad_v3",
                    youtu_route_type=None,
                    youtu_client_id=None,
                    youtu_graph_state_file="outputs/graph/youtu_graph_state.json",
                    youtu_reuse_graph=True,
                    youtu_force_rebuild=False,
                    youtu_sync_mode="none",
                    youtu_shared_corpus_dir="outputs/youtu_sync",
                    youtu_corpus_source_file=None,
                    youtu_construct_poll_sec=2,
                    youtu_construct_timeout_sec=1800,
                    youtu_require_fingerprint_match=True,
                    youtu_schema_file=None,
                    lightrag_mode="hybrid",
                    lightrag_working_dir=str(root / "lightrag"),
                    merge_chunks=1,
                    retrieve_reuse_cache=True,
                    cuad_doc_scope=False,
                    strict_doc_scope=True,
                    graph_query_level=None,
                    incremental_only=True,
                )

            self.assertEqual(calls, ["What is Q2?"])
            rows = [json.loads(line) for line in out_file.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual([row["qid"] for row in rows], ["q1", "q2"])
            self.assertTrue(summary["incremental_only"])
            self.assertEqual(summary["num_existing_answers"], 1)

    def test_incremental_only_completion_is_scoped_to_selected_rag_methods(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            queries_file = root / "queries.jsonl"
            out_file = root / "answers.jsonl"
            metrics_file = root / "metrics.json"
            budget_file = root / "budget.yaml"
            budget_file.write_text("", encoding="utf-8")

            with queries_file.open("w", encoding="utf-8") as f:
                f.write(json.dumps({"qid": "q1", "query": "What is Q1?", "type": "local_factual"}, ensure_ascii=False) + "\n")

            existing_row = {
                "qid": "q1",
                "query": "What is Q1?",
                "type": "local_factual",
                "mode": "reject",
                "top_k": 20,
                "regimes": {
                    "best_effort": {
                        "graph_rag": {"answer": "cached graph"},
                    }
                },
            }
            out_file.write_text(json.dumps(existing_row, ensure_ascii=False) + "\n", encoding="utf-8")

            with patch("src.experiments.run_compare_youtu_lightrag.answer_with_graphrag", side_effect=AssertionError("graph should be skipped")):
                summary_graph = _run_pair_compare_once(
                    queries_file=str(queries_file),
                    chunks_file="data/processed/qa_aligned_chunks.jsonl",
                    graph_file="outputs/graph/qa_aligned_graph_v3_docscoped.json",
                    communities_file="outputs/graph/qa_aligned_communities_v3_docscoped_pruned.json",
                    top_k=20,
                    out_file=str(out_file),
                    metrics_file=str(metrics_file),
                    regimes="best_effort",
                    budget_config_file=str(budget_file),
                    max_queries=None,
                    answer_mode="reject",
                    selected_rags={"graph_rag"},
                    youtu_base_url="http://127.0.0.1:8000",
                    youtu_dataset="cuad_v3",
                    youtu_route_type=None,
                    youtu_client_id=None,
                    youtu_graph_state_file="outputs/graph/youtu_graph_state.json",
                    youtu_reuse_graph=True,
                    youtu_force_rebuild=False,
                    youtu_sync_mode="none",
                    youtu_shared_corpus_dir="outputs/youtu_sync",
                    youtu_corpus_source_file=None,
                    youtu_construct_poll_sec=2,
                    youtu_construct_timeout_sec=1800,
                    youtu_require_fingerprint_match=True,
                    youtu_schema_file=None,
                    lightrag_mode="hybrid",
                    lightrag_working_dir=str(root / "lightrag"),
                    merge_chunks=1,
                    retrieve_reuse_cache=True,
                    cuad_doc_scope=False,
                    strict_doc_scope=True,
                    graph_query_level=None,
                    incremental_only=True,
                )

            self.assertEqual(summary_graph["query_consistency"]["effective"]["total_queries"], 0)

            lightrag_calls: list[str] = []

            def _fake_answer_with_lightrag(**kwargs):
                lightrag_calls.append(str(kwargs.get("query", "")))
                return {
                    "answer": "fresh light",
                    "telemetry": {"llm_calls": 1, "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }

            with patch("src.experiments.run_compare_youtu_lightrag.answer_with_lightrag", side_effect=_fake_answer_with_lightrag):
                summary_light = _run_pair_compare_once(
                    queries_file=str(queries_file),
                    chunks_file="data/processed/qa_aligned_chunks.jsonl",
                    graph_file="",
                    communities_file="",
                    top_k=20,
                    out_file=str(out_file),
                    metrics_file=str(metrics_file),
                    regimes="best_effort",
                    budget_config_file=str(budget_file),
                    max_queries=None,
                    answer_mode="reject",
                    selected_rags={"lightrag"},
                    youtu_base_url="http://127.0.0.1:8000",
                    youtu_dataset="cuad_v3",
                    youtu_route_type=None,
                    youtu_client_id=None,
                    youtu_graph_state_file="outputs/graph/youtu_graph_state.json",
                    youtu_reuse_graph=True,
                    youtu_force_rebuild=False,
                    youtu_sync_mode="none",
                    youtu_shared_corpus_dir="outputs/youtu_sync",
                    youtu_corpus_source_file=None,
                    youtu_construct_poll_sec=2,
                    youtu_construct_timeout_sec=1800,
                    youtu_require_fingerprint_match=True,
                    youtu_schema_file=None,
                    lightrag_mode="hybrid",
                    lightrag_working_dir=str(root / "lightrag"),
                    merge_chunks=1,
                    retrieve_reuse_cache=True,
                    cuad_doc_scope=False,
                    strict_doc_scope=True,
                    graph_query_level=None,
                    incremental_only=True,
                )

            self.assertEqual(lightrag_calls, ["What is Q1?"])
            self.assertEqual(summary_light["query_consistency"]["effective"]["total_queries"], 1)

    def test_incremental_only_does_not_fallback_to_query_text(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            queries_file = root / "queries.jsonl"
            out_file = root / "answers.jsonl"
            metrics_file = root / "metrics.json"
            budget_file = root / "budget.yaml"
            budget_file.write_text("", encoding="utf-8")

            repeated_query = 'Highlight the parts related to "Anti-Assignment".'
            with queries_file.open("w", encoding="utf-8") as f:
                f.write(json.dumps({"qid": "q1", "query": repeated_query, "type": "local_factual"}, ensure_ascii=False) + "\n")
                f.write(json.dumps({"qid": "q2", "query": repeated_query, "type": "local_factual"}, ensure_ascii=False) + "\n")

            existing_row = {
                "qid": "q1",
                "query": repeated_query,
                "type": "local_factual",
                "mode": "reject",
                "top_k": 20,
                "regimes": {
                    "best_effort": {
                        "lightrag": {"answer": "cached q1"},
                    }
                },
            }
            out_file.write_text(json.dumps(existing_row, ensure_ascii=False) + "\n", encoding="utf-8")

            lightrag_calls: list[str] = []

            def _fake_answer_with_lightrag(**kwargs):
                lightrag_calls.append(str(kwargs.get("query", "")))
                return {
                    "answer": "fresh",
                    "telemetry": {"llm_calls": 1, "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }

            with patch("src.experiments.run_compare_youtu_lightrag.answer_with_lightrag", side_effect=_fake_answer_with_lightrag):
                summary = _run_pair_compare_once(
                    queries_file=str(queries_file),
                    chunks_file="data/processed/qa_aligned_chunks.jsonl",
                    graph_file="",
                    communities_file="",
                    top_k=20,
                    out_file=str(out_file),
                    metrics_file=str(metrics_file),
                    regimes="best_effort",
                    budget_config_file=str(budget_file),
                    max_queries=None,
                    answer_mode="reject",
                    selected_rags={"lightrag"},
                    youtu_base_url="http://127.0.0.1:8000",
                    youtu_dataset="cuad_v3",
                    youtu_route_type=None,
                    youtu_client_id=None,
                    youtu_graph_state_file="outputs/graph/youtu_graph_state.json",
                    youtu_reuse_graph=True,
                    youtu_force_rebuild=False,
                    youtu_sync_mode="none",
                    youtu_shared_corpus_dir="outputs/youtu_sync",
                    youtu_corpus_source_file=None,
                    youtu_construct_poll_sec=2,
                    youtu_construct_timeout_sec=1800,
                    youtu_require_fingerprint_match=True,
                    youtu_schema_file=None,
                    lightrag_mode="hybrid",
                    lightrag_working_dir=str(root / "lightrag"),
                    merge_chunks=1,
                    retrieve_reuse_cache=True,
                    cuad_doc_scope=False,
                    strict_doc_scope=True,
                    graph_query_level=None,
                    incremental_only=True,
                )

            self.assertEqual(len(lightrag_calls), 1)
            self.assertEqual(summary["query_consistency"]["effective"]["total_queries"], 1)
            rows = [json.loads(line) for line in out_file.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual([row["qid"] for row in rows], ["q1", "q2"])

    def test_run_pair_compare_once_saves_progress_after_each_query(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            queries_file = root / "queries.jsonl"
            out_file = root / "answers.jsonl"
            metrics_file = root / "metrics.json"
            budget_file = root / "budget.yaml"
            budget_file.write_text("", encoding="utf-8")

            with queries_file.open("w", encoding="utf-8") as f:
                f.write(json.dumps({"qid": "q1", "query": "What is Q1?", "type": "local_factual"}, ensure_ascii=False) + "\n")
                f.write(json.dumps({"qid": "q2", "query": "What is Q2?", "type": "local_factual"}, ensure_ascii=False) + "\n")

            writes: list[list[str]] = []

            def _fake_write_jsonl(path, rows):
                writes.append([str(row.get("qid", "")) for row in rows])

            def _fake_answer_with_lightrag(**kwargs):
                return {
                    "answer": f"fresh:{kwargs.get('query', '')}",
                    "telemetry": {"llm_calls": 1, "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }

            with patch("src.experiments.run_compare_youtu_lightrag._write_jsonl", side_effect=_fake_write_jsonl):
                with patch("src.experiments.run_compare_youtu_lightrag.answer_with_lightrag", side_effect=_fake_answer_with_lightrag):
                    _run_pair_compare_once(
                        queries_file=str(queries_file),
                        chunks_file="data/processed/qa_aligned_chunks.jsonl",
                        graph_file="",
                        communities_file="",
                        top_k=20,
                        out_file=str(out_file),
                        metrics_file=str(metrics_file),
                        regimes="best_effort",
                        budget_config_file=str(budget_file),
                        max_queries=None,
                        answer_mode="reject",
                        selected_rags={"lightrag"},
                        youtu_base_url="http://127.0.0.1:8000",
                        youtu_dataset="cuad_v3",
                        youtu_route_type=None,
                        youtu_client_id=None,
                        youtu_graph_state_file="outputs/graph/youtu_graph_state.json",
                        youtu_reuse_graph=True,
                        youtu_force_rebuild=False,
                        youtu_sync_mode="none",
                        youtu_shared_corpus_dir="outputs/youtu_sync",
                        youtu_corpus_source_file=None,
                        youtu_construct_poll_sec=2,
                        youtu_construct_timeout_sec=1800,
                        youtu_require_fingerprint_match=True,
                        youtu_schema_file=None,
                        lightrag_mode="hybrid",
                        lightrag_working_dir=str(root / "lightrag"),
                        merge_chunks=1,
                        retrieve_reuse_cache=True,
                        cuad_doc_scope=False,
                        strict_doc_scope=True,
                        graph_query_level=None,
                        incremental_only=False,
                    )

            self.assertGreaterEqual(len(writes), 3)
            self.assertEqual(writes[0], ["q1"])
            self.assertEqual(writes[1], ["q1", "q2"])
            self.assertEqual(writes[-1], ["q1", "q2"])


if __name__ == "__main__":
    unittest.main()
