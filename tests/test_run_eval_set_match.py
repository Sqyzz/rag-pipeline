from __future__ import annotations

import unittest

from src.evaluation.run_eval import _match_scores


class TestRunEvalSetMatch(unittest.TestCase):
    def test_multi_answer_exact_match_with_json_array_prediction(self) -> None:
        scores = _match_scores('["beta", "alpha"]', ["Alpha", "Beta"])
        self.assertEqual(scores["answer_exact"], 1)
        self.assertEqual(scores["answer_exact_relaxed"], 1)
        self.assertEqual(scores["answer_similarity"], 1.0)

    def test_multi_answer_subset_is_not_exact(self) -> None:
        scores = _match_scores("Alpha", ["Alpha", "Beta"])
        self.assertEqual(scores["answer_exact"], 0)
        self.assertEqual(scores["answer_exact_relaxed"], 0)
        self.assertEqual(scores["answer_contains"], 1)

    def test_single_answer_still_works(self) -> None:
        scores = _match_scores("New York", "new york")
        self.assertEqual(scores["answer_exact"], 1)
        self.assertEqual(scores["answer_exact_relaxed"], 1)

    def test_multi_answer_similarity_uses_set_f1(self) -> None:
        scores = _match_scores("Alpha; Beta", "Beta; Alpha")
        self.assertEqual(scores["answer_similarity"], 1.0)

    def test_single_answer_phrase_vs_sentence_gets_relaxed_credit(self) -> None:
        pred = "The Buyer must inspect and test the product for damage, defects, or shortages immediately upon receipt."
        gold = "amage, defect or shortage"
        scores = _match_scores(pred, gold)
        self.assertGreaterEqual(scores["answer_contains_relaxed"], 1)
        self.assertGreaterEqual(scores["answer_similarity"], 0.67)


if __name__ == "__main__":
    unittest.main()
