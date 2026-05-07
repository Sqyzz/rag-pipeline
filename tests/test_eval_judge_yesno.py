from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.evaluation.run_eval import run_eval


class TestEvalJudgeYesNo(unittest.TestCase):
    def test_llm_yesno_writes_answer_semantic_column_and_topk_fields(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            pred_file = root / "pred.jsonl"
            gold_file = root / "gold.jsonl"
            out_csv = root / "eval.csv"
            out_summary = root / "eval_summary.json"

            pred_row = {
                "qid": "q1",
                "type": "local_factual",
                "query": "Q?",
                "mode": "open",
                "top_k": 10,
                "regimes": {
                    "best_effort": {
                        "vector_rag": {
                            "answer": "pred different",
                            "evidence": [{"doc_id": "DocA#p0", "chunk_id": "c1", "text": "x"}],
                            "telemetry": {"aggregate": {}},
                        }
                    }
                },
            }
            gold_row = {
                "qid": "q1",
                "type": "local_factual",
                "query": "Q?",
                "answer": "gold answer",
                "supporting_chunks": [{"doc_id": "DocA#p9", "chunk_id": "c9"}],
            }
            pred_file.write_text(json.dumps(pred_row, ensure_ascii=False) + "\n", encoding="utf-8")
            gold_file.write_text(json.dumps(gold_row, ensure_ascii=False) + "\n", encoding="utf-8")

            with patch("src.evaluation.run_eval.semantic_equivalent_yes_no", return_value=1):
                summary = run_eval(
                    pred_file=str(pred_file),
                    gold_file=str(gold_file),
                    out_csv=str(out_csv),
                    out_summary=str(out_summary),
                    make_plots=False,
                    semantic_similarity=False,
                    judge_mode="llm_yesno",
                    judge_model="qwen-flash",
                )

            self.assertEqual(summary["judge"]["enabled"], True)
            self.assertEqual(summary["judge"]["judge_mode"], "llm_yesno")
            df = pd.read_csv(out_csv)
            self.assertIn("answer_semantic_yesno", df.columns)
            self.assertIn("topk_support", df.columns)
            self.assertIn("topk_correct", df.columns)
            self.assertIn("topk_accuracy", df.columns)
            self.assertEqual(int(df.loc[0, "answer_semantic_yesno"]), 1)
            self.assertEqual(int(df.loc[0, "topk_support"]), 1)
            self.assertEqual(int(df.loc[0, "topk_correct"]), 1)
            self.assertEqual(int(df.loc[0, "topk_accuracy"]), 1)


if __name__ == "__main__":
    unittest.main()
