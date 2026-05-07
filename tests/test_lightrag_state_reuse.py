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

from src.baselines import lightrag_adapter


class TestLightRAGStateReuse(unittest.TestCase):
    def test_save_state_persists_pipeline_file(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            workdir = Path(d) / "lightrag"
            lightrag_adapter._save_state(workdir, {"ok": True})
            state_file = workdir / "_pipeline_state.json"
            self.assertTrue(state_file.exists())
            self.assertEqual(json.loads(state_file.read_text(encoding="utf-8")), {"ok": True})

    def test_ensure_indexed_rejects_reusable_artifacts_without_state(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            chunks_file = root / "chunks.jsonl"
            chunks_file.write_text('{"chunk_id":"c1","doc_id":"d1","text":"hello"}\n', encoding="utf-8")
            workdir = root / "lightrag"
            workdir.mkdir(parents=True, exist_ok=True)
            for name in (
                "graph_chunk_entity_relation.graphml",
                "kv_store_doc_status.json",
                "kv_store_full_docs.json",
                "kv_store_text_chunks.json",
                "vdb_chunks.json",
                "vdb_entities.json",
                "vdb_relationships.json",
            ):
                (workdir / name).write_text("{}", encoding="utf-8")

            with patch.object(lightrag_adapter, "_load_chunks_assets", side_effect=AssertionError("should not rebuild")):
                with self.assertRaisesRegex(RuntimeError, "missing _pipeline_state.json"):
                    lightrag_adapter._ensure_indexed(
                        chunks_file=str(chunks_file),
                        working_dir=workdir,
                        force_rebuild=False,
                        merge_chunks=5,
                    )

    def test_ensure_indexed_rejects_state_mismatch_without_rebuild(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            chunks_file = root / "chunks.jsonl"
            chunks_file.write_text('{"chunk_id":"c1","doc_id":"d1","text":"hello"}\n', encoding="utf-8")
            other_chunks_file = root / "other_chunks.jsonl"
            other_chunks_file.write_text('{"chunk_id":"c2","doc_id":"d2","text":"world"}\n', encoding="utf-8")
            workdir = root / "lightrag"
            workdir.mkdir(parents=True, exist_ok=True)
            lightrag_adapter._save_state(
                workdir,
                lightrag_adapter._target_state(str(other_chunks_file), merge_chunks=3),
            )

            with patch.object(lightrag_adapter, "_load_chunks_assets", side_effect=AssertionError("should not rebuild")):
                with self.assertRaisesRegex(RuntimeError, "different configuration"):
                    lightrag_adapter._ensure_indexed(
                        chunks_file=str(chunks_file),
                        working_dir=workdir,
                        force_rebuild=False,
                        merge_chunks=5,
                    )


if __name__ == "__main__":
    unittest.main()
