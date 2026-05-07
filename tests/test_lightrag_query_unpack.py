from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for candidate in (ROOT, SRC):
    value = str(candidate)
    if value not in sys.path:
        sys.path.insert(0, value)

from src.baselines import lightrag_adapter


class TestLightRAGQueryUnpack(unittest.TestCase):
    def test_answer_with_lightrag_accepts_import_tuple_of_seven(self) -> None:
        class DummyQueryParam:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class DummyRag:
            async def aquery_llm(self, query, param=None):
                return {
                    "llm_response": {"content": "ok"},
                    "data": {"chunks": [{"content": "Chunk one text", "file_path": "doc-1#p0"}]},
                }

            async def finalize_storages(self):
                return None

        class DummyTracker:
            def get_usage(self):
                return {"call_count": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        with patch.object(lightrag_adapter, "_ensure_indexed", return_value={"reused_cache": True}):
            with patch.object(
                lightrag_adapter,
                "_load_chunks_assets",
                return_value={
                    "rows": [{"chunk_id": "chunk-1", "doc_id": "doc-1#p0", "text": "Chunk one text"}],
                    "by_hash": {lightrag_adapter._hash_text("Chunk one text"): {"chunk_id": "chunk-1", "doc_id": "doc-1#p0", "text": "Chunk one text"}},
                    "fingerprint": "x",
                },
            ):
                with patch.object(
                    lightrag_adapter,
                    "_prepare_rows_for_index",
                    return_value=(
                        [{"chunk_id": "chunk-1", "doc_id": "doc-1#p0", "text": "Chunk one text"}],
                        {lightrag_adapter._hash_text("Chunk one text"): {"chunk_id": "chunk-1", "doc_id": "doc-1#p0", "text": "Chunk one text"}},
                    ),
                ):
                    with patch.object(
                        lightrag_adapter,
                        "_import_lightrag",
                        return_value=(object(), DummyQueryParam, object(), object(), object(), object(), object()),
                    ):
                        with patch.object(
                            lightrag_adapter,
                            "_create_rag",
                            return_value=(DummyRag(), DummyTracker(), DummyTracker(), {}),
                        ):
                            out = lightrag_adapter.answer_with_lightrag(
                                query="Q?",
                                chunks_file="data/processed/cuad_chunks_top10.jsonl",
                                working_dir="outputs/lightrag/cuad_chunks_top10",
                                force_rebuild=False,
                                merge_chunks=5,
                            )

        self.assertEqual(out["answer"], "ok")
        self.assertEqual(out["evaluation_payload"]["retrieved_context_ids"], ["chunk-1"])
        self.assertEqual(out["evaluation_payload"]["retrieved_doc_ids"], ["doc-1#p0"])
        self.assertEqual(out["evaluation_payload"]["response_for_eval_source"], "lightrag_answer")
        self.assertEqual(out["answer_scope_target_doc_id"], "doc-1#p0")
        self.assertEqual(out["answer_composition_mode"], "target_contract_primary")
        self.assertEqual(out["retrieval_trace"]["query"], "Q?")
        self.assertEqual(out["answer_trace"]["aggregation_strategy"], "lightrag_direct_generation_with_retrieved_chunk_backing")
        self.assertEqual(out["reasoning_steps"][0]["type"], "lightrag_query")

    def test_answer_with_lightrag_uses_shared_query_loop_runner(self) -> None:
        def _fake_runner(coro):
            coro.close()
            return {"answer": "ok"}

        with patch.object(lightrag_adapter, "_run_query_coro", side_effect=_fake_runner) as runner:
            with patch.object(lightrag_adapter, "_ensure_indexed", return_value={"reused_cache": True}):
                with patch.object(
                    lightrag_adapter,
                    "_load_chunks_assets",
                    return_value={"rows": [], "by_hash": {}, "fingerprint": "x"},
                ):
                    with patch.object(lightrag_adapter, "_prepare_rows_for_index", return_value=([], {})):
                        with patch.object(lightrag_adapter, "_import_lightrag", return_value=(None, None, None, None, None, None, None)):
                            with patch.object(lightrag_adapter, "_create_rag", return_value=(None, None, None, None)):
                                out = lightrag_adapter.answer_with_lightrag(
                                    query="Q?",
                                    chunks_file="data/processed/cuad_chunks_top10.jsonl",
                                    working_dir="outputs/lightrag/cuad_chunks_top10",
                                    force_rebuild=False,
                                    merge_chunks=5,
                                )

        self.assertEqual(out["answer"], "ok")
        runner.assert_called_once()


if __name__ == "__main__":
    unittest.main()
