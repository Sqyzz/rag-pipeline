from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for candidate in (ROOT, SRC):
    value = str(candidate)
    if value not in sys.path:
        sys.path.insert(0, value)

from src.ingestion.build_cuad_aligned_docs import build_cuad_aligned_docs


class TestBuildCuadAlignedDocs(unittest.TestCase):
    def test_build_aligned_docs_keeps_all_matched_and_samples_unmatched(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            cuad_file = base / "CUADv1.json"
            qa_file = base / "queries.jsonl"
            out_docs = base / "docs.jsonl"

            cuad_payload = {
                "version": "1.0",
                "data": [
                    {"title": "DocA", "paragraphs": [{"context": "A0", "qas": []}]},
                    {"title": "DocB", "paragraphs": [{"context": "B0", "qas": []}]},
                    {"title": "DocC", "paragraphs": [{"context": "C0", "qas": []}]},
                    {"title": "DocD", "paragraphs": [{"context": "D0", "qas": []}]},
                ],
            }
            cuad_file.write_text(json.dumps(cuad_payload, ensure_ascii=False), encoding="utf-8")

            rows = [
                {"qid": "q1", "meta": {"query_doc_key": "DocA", "title": "DocA"}},
                {"qid": "q2", "meta": {"query_doc_key": "DocB", "title": "DocB"}},
            ]
            qa_file.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8")

            summary = build_cuad_aligned_docs(
                cuad_file=str(cuad_file),
                qa_file=str(qa_file),
                out_docs_file=str(out_docs),
                matched_doc_ratio=0.5,
                random_seed=7,
                split_name="aligned_test",
            )

            self.assertEqual(summary["num_selected_matched_docs"], 2)
            self.assertEqual(summary["num_selected_unmatched_docs"], 2)
            self.assertEqual(summary["num_selected_docs"], 4)

            written = [json.loads(x) for x in out_docs.read_text(encoding="utf-8").splitlines() if x.strip()]
            titles = {str(x.get("meta", {}).get("title", "")) for x in written}
            self.assertIn("DocA", titles)
            self.assertIn("DocB", titles)
            self.assertEqual(len(titles), 4)

    def test_build_aligned_docs_prefers_query_doc_key_over_title(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            cuad_file = base / "CUADv1.json"
            qa_file = base / "queries.jsonl"
            out_docs = base / "docs.jsonl"

            cuad_payload = {
                "version": "1.0",
                "data": [
                    {"title": "DocA", "paragraphs": [{"context": "A0", "qas": []}]},
                    {"title": "DocB", "paragraphs": [{"context": "B0", "qas": []}]},
                ],
            }
            cuad_file.write_text(json.dumps(cuad_payload, ensure_ascii=False), encoding="utf-8")
            qa_file.write_text(
                json.dumps({"qid": "q1", "meta": {"query_doc_key": "DocA", "title": "DocB"}}, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            summary = build_cuad_aligned_docs(
                cuad_file=str(cuad_file),
                qa_file=str(qa_file),
                out_docs_file=str(out_docs),
                matched_doc_ratio=1.0,
                random_seed=7,
                split_name="aligned_test",
            )

            self.assertEqual(summary["num_selected_docs"], 1)
            written = [json.loads(x) for x in out_docs.read_text(encoding="utf-8").splitlines() if x.strip()]
            self.assertEqual({str(x.get("meta", {}).get("title", "")) for x in written}, {"DocA"})

    def test_build_aligned_docs_uses_title_when_query_doc_key_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            cuad_file = base / "CUADv1.json"
            qa_file = base / "queries.jsonl"
            out_docs = base / "docs.jsonl"

            cuad_payload = {
                "version": "1.0",
                "data": [
                    {"title": "DocA", "paragraphs": [{"context": "A0", "qas": []}]},
                    {"title": "DocB", "paragraphs": [{"context": "B0", "qas": []}]},
                ],
            }
            cuad_file.write_text(json.dumps(cuad_payload, ensure_ascii=False), encoding="utf-8")
            qa_file.write_text(
                json.dumps({"qid": "q1", "meta": {"title": "DocB"}}, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            summary = build_cuad_aligned_docs(
                cuad_file=str(cuad_file),
                qa_file=str(qa_file),
                out_docs_file=str(out_docs),
                matched_doc_ratio=1.0,
                random_seed=7,
                split_name="aligned_test",
            )

            self.assertEqual(summary["num_selected_docs"], 1)
            written = [json.loads(x) for x in out_docs.read_text(encoding="utf-8").splitlines() if x.strip()]
            self.assertEqual({str(x.get("meta", {}).get("title", "")) for x in written}, {"DocB"})


if __name__ == "__main__":
    unittest.main()
