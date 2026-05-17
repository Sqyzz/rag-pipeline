from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for candidate in (ROOT, SRC):
    value = str(candidate)
    if value not in sys.path:
        sys.path.insert(0, value)


def _install_import_stubs() -> None:
    graph_rag = types.ModuleType("baselines.graph_rag")
    graph_rag.answer_with_graphrag = lambda **kwargs: {}  # noqa: ARG005
    ms_local_graphrag = types.ModuleType("baselines.ms_local_graphrag")
    ms_local_graphrag.answer_with_ms_local_graphrag = lambda **kwargs: {}  # noqa: ARG005
    lightrag_adapter = types.ModuleType("baselines.lightrag_adapter")
    lightrag_adapter.answer_with_lightrag = lambda **kwargs: {}  # noqa: ARG005
    vector_rag = types.ModuleType("baselines.vector_rag")
    vector_rag.retrieve_with_evidence = lambda **kwargs: ([], {})  # noqa: ARG005
    vector_rag.answer_with_context = lambda **kwargs: ("", {})  # noqa: ARG005

    config = types.ModuleType("utils.config")
    config.cfg = types.SimpleNamespace(retrieval=types.SimpleNamespace(top_k=8), comparison=None)

    ragas_converters = types.ModuleType("utils.ragas_converters")
    ragas_converters.infer_doc_prefix = lambda reference_doc_ids: str(reference_doc_ids[0]).split("#", 1)[0] if reference_doc_ids else ""
    ragas_converters.load_jsonl = lambda path: []  # noqa: ARG005
    ragas_converters.scope_cuad_question = lambda question, doc_prefix: f"[Document: {doc_prefix}] {question}" if doc_prefix else question
    ragas_converters.write_jsonl = lambda path, rows: None  # noqa: ARG005

    def _normalize_evidence_chunks(value):  # noqa: ANN001
        chunks = value if isinstance(value, list) else []
        return {
            "retrieved_contexts": [str(item.get("text") or "") for item in chunks if isinstance(item, dict)],
            "retrieved_context_ids": [str(item.get("chunk_id") or "") for item in chunks if isinstance(item, dict)],
            "retrieved_doc_ids": [str(item.get("doc_id") or "") for item in chunks if isinstance(item, dict)],
        }

    ragas_converters.normalize_evidence_chunks = _normalize_evidence_chunks

    sys.modules.setdefault("baselines.graph_rag", graph_rag)
    sys.modules.setdefault("baselines.ms_local_graphrag", ms_local_graphrag)
    sys.modules.setdefault("baselines.lightrag_adapter", lightrag_adapter)
    sys.modules.setdefault("baselines.vector_rag", vector_rag)
    sys.modules.setdefault("utils.config", config)
    sys.modules.setdefault("utils.ragas_converters", ragas_converters)


_install_import_stubs()

from src.experiments import run_ragas_cuad_compare as compare


class TestRagasCuadCompareVectorRag(unittest.TestCase):
    def test_filter_questions_by_qid_supports_comma_separated_list(self) -> None:
        rows = [
            {"qid": "ragas-cuad-0001"},
            {"qid": "ragas-cuad-0002"},
            {"qid": "ragas-cuad-0003"},
        ]

        filtered = compare._filter_questions_by_qid(rows, "ragas-cuad-0003, ragas-cuad-0001")

        self.assertEqual([row["qid"] for row in filtered], ["ragas-cuad-0001", "ragas-cuad-0003"])

    def test_filter_questions_by_qid_raises_for_missing_qid(self) -> None:
        rows = [{"qid": "ragas-cuad-0001"}]

        with self.assertRaises(SystemExit) as cm:
            compare._filter_questions_by_qid(rows, "ragas-cuad-9999")

        self.assertIn("ragas-cuad-9999", str(cm.exception))

    def test_vector_rag_payload_uses_raw_retrieval_and_unified_output_schema(self) -> None:
        captured: dict[str, object] = {}

        def _fake_retrieve_with_evidence(**kwargs):  # noqa: ANN001
            captured.update(kwargs)
            return [
                {
                    "rank": 1,
                    "score": 0.91,
                    "chunk_id": "doc-001#chunk-1",
                    "doc_id": "doc-001#p0",
                    "text": "The agreement terminates on December 31, 2025.",
                }
            ], {"embedding": {"total_tokens": 7}}

        def _fake_answer_with_context(**kwargs):  # noqa: ANN001
            captured["answer_kwargs"] = kwargs
            return "December 31, 2025", {"usage": {"total_tokens": 11}}

        test_row = {
            "qid": "ragas-cuad-test-001",
            "question": "When does the agreement terminate?",
            "reference": "December 31, 2025",
            "reference_contexts": ["reference context"],
            "reference_context_ids": ["doc-001#chunk-1"],
            "reference_doc_ids": ["doc-001#p0"],
            "synthesizer_name": "single_hop_specific_query_synthesizer",
        }

        with patch.object(compare, "retrieve_with_evidence", side_effect=_fake_retrieve_with_evidence):
            with patch.object(compare, "answer_with_context", side_effect=_fake_answer_with_context):
                result = compare._run_single_task(
                    0,
                    1,
                    1,
                    test_row,
                    "vector_rag",
                    graph_file="unused-graph.json",
                    communities_file="unused-communities.json",
                    chunks_file="unused-chunks.jsonl",
                    lightrag_working_dir=None,
                    vector_idx_file="outputs/indexes/faiss_cuad_sampled_ragas.idx",
                    vector_store_file="outputs/indexes/chunk_store_cuad_sampled_ragas.json",
                    youtu_base_url=None,
                    youtu_dataset=None,
                    answer_mode="reject",
                )

        row = result["row"]
        self.assertEqual(row["method"], "vector_rag")
        self.assertEqual(row["response"], "December 31, 2025")
        self.assertEqual(row["retrieved_context_ids"], ["doc-001#chunk-1"])
        self.assertEqual(row["retrieved_doc_ids"], ["doc-001#p0"])
        self.assertEqual(row["orchestration_mode"], "dense_vector_retrieval")
        self.assertEqual(row["retrieval_trace"]["retrieved_chunk_ids"], ["doc-001#chunk-1"])
        self.assertEqual(row["retrieval_trace"]["scores"], [0.91])
        self.assertEqual(captured["idx_file"], "outputs/indexes/faiss_cuad_sampled_ragas.idx")
        self.assertEqual(captured["store_file"], "outputs/indexes/chunk_store_cuad_sampled_ragas.json")
        self.assertIsNone(captured["doc_prefix_filter"])
        self.assertEqual(captured["answer_kwargs"]["answer_mode"], "reject")

    def test_ms_local_graphrag_does_not_receive_reference_doc_scope(self) -> None:
        captured: dict[str, object] = {}

        def _fake_answer_with_ms_local_graphrag(**kwargs):  # noqa: ANN001
            captured.update(kwargs)
            return {
                "answer": "answer",
                "answer_mode": "reject",
                "evaluation_payload": {
                    "response": "answer",
                    "retrieved_contexts": ["ctx"],
                    "retrieved_context_ids": ["doc-001#chunk-1"],
                    "retrieved_doc_ids": ["doc-001#p0"],
                },
                "retrieval_trace": {"orchestration_mode": "ms_graphrag_local_search_like"},
            }

        test_row = {
            "qid": "ragas-cuad-test-002",
            "question": "When does the agreement terminate?",
            "reference": "December 31, 2025",
            "reference_contexts": ["reference context"],
            "reference_context_ids": ["doc-001#chunk-1"],
            "reference_doc_ids": ["doc-001#p0"],
            "synthesizer_name": "single_hop_specific_query_synthesizer",
        }

        with patch.object(compare, "answer_with_ms_local_graphrag", side_effect=_fake_answer_with_ms_local_graphrag):
            result = compare._run_single_task(
                0,
                1,
                1,
                test_row,
                "ms_local_graphrag",
                graph_file="graph.json",
                communities_file="communities.json",
                chunks_file="chunks.jsonl",
                lightrag_working_dir=None,
                vector_idx_file=None,
                vector_store_file=None,
                youtu_base_url=None,
                youtu_dataset=None,
                answer_mode="reject",
            )

        self.assertEqual(result["row"]["method"], "ms_local_graphrag")
        self.assertEqual(captured["query"], "When does the agreement terminate?")
        self.assertIsNone(captured["doc_prefix_filter"])


if __name__ == "__main__":
    unittest.main()
