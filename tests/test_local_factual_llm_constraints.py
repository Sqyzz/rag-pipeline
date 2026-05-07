from __future__ import annotations

import unittest

from src.evaluation.build_cuad_capability_qa import (
    _answer_len_ok,
    _local_query_allowed,
    _resolve_local_evidence_span,
)


class TestLocalFactualLlmConstraints(unittest.TestCase):
    def test_answer_length_rule(self) -> None:
        self.assertTrue(_answer_len_ok("New York"))
        self.assertTrue(_answer_len_ok("违约责任"))
        self.assertFalse(_answer_len_ok(""))
        self.assertFalse(_answer_len_ok("this is a very long answer with too many english tokens over limit"))

    def test_query_rejects_explanatory_and_copy_like(self) -> None:
        chunk = "ICC grants license to Beta Ltd for distribution rights effective from Jan 1, 2025."
        self.assertFalse(_local_query_allowed("Why does ICC grant license to Beta Ltd?", chunk))
        self.assertFalse(_local_query_allowed(chunk, chunk))
        self.assertTrue(_local_query_allowed("Who receives the license granted by ICC?", chunk))

    def test_span_resolves_by_quote_and_char_range(self) -> None:
        chunk = "The governing law is New York and disputes are resolved by ICC arbitration."
        span = _resolve_local_evidence_span(
            chunk_text=chunk,
            answer="New York",
            evidence_obj={"quote": "New York", "start_char": 21, "end_char": 29},
        )
        self.assertIsNotNone(span)
        assert span is not None
        self.assertEqual(span["quote"], "New York")
        self.assertEqual(span["answer_grounding"], "extractive")

    def test_span_rejects_unlocatable_abstractive_answer(self) -> None:
        chunk = "The governing law is New York and disputes are resolved by ICC arbitration."
        span = _resolve_local_evidence_span(
            chunk_text=chunk,
            answer="United States law",
            evidence_obj={"quote": "New York", "start_char": 21, "end_char": 29},
        )
        self.assertIsNone(span)


if __name__ == "__main__":
    unittest.main()
