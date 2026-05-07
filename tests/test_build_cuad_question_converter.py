from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.evaluation.build_cuad_question_converter import convert_cuad_questions


class TestBuildCuadQuestionConverter(unittest.TestCase):
    def test_converter_can_emit_only_local_and_cross_capability_types(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            cuad_file = base / "train_separate_questions.json"
            queries_out = base / "queries.jsonl"
            gold_out = base / "gold.jsonl"

            cuad_payload = {
                "data": [
                    {
                        "title": "DocA",
                        "paragraphs": [
                            {
                                "context": "Contract text.",
                                "qas": [
                                    {
                                        "question": 'What is the clause related to "License Grant"?',
                                        "answers": [{"text": "License is granted."}],
                                        "is_impossible": False,
                                    },
                                    {
                                        "question": 'What is the clause related to "Change Of Control"?',
                                        "answers": [{"text": "Consent is required."}],
                                        "is_impossible": False,
                                    },
                                ],
                            }
                        ],
                    }
                ]
            }
            cuad_file.write_text(json.dumps(cuad_payload, ensure_ascii=False), encoding="utf-8")

            stats = convert_cuad_questions(
                cuad_train_file=str(cuad_file),
                out_queries_file=str(queries_out),
                out_gold_file=str(gold_out),
                qas_per_type=1,
                total_per_type=0,
                mode="template",
                max_docs=0,
                min_theme_labels=2,
                max_label_refs_in_query=4,
                max_completion_tokens=64,
                temperature=0.0,
                seed=7,
                answer_mode="weak_reference",
                include_empty_support=True,
                progress_every=0,
                progress_every_qa=0,
                type_name_scheme="capability",
                llm_types=set(),
                global_context_max_chars=1000,
                impossible_keep_ratio=0.0,
                selected_types={"local_retrieval", "structural_reasoning"},
            )

            query_rows = [json.loads(line) for line in queries_out.read_text(encoding="utf-8").splitlines() if line.strip()]
            gold_rows = [json.loads(line) for line in gold_out.read_text(encoding="utf-8").splitlines() if line.strip()]

            self.assertEqual(len(query_rows), 2)
            self.assertEqual(len(gold_rows), 2)
            self.assertEqual({row["type"] for row in query_rows}, {"local_factual", "cross_clause"})
            self.assertEqual({row["qid"] for row in query_rows}, {row["qid"] for row in gold_rows})
            self.assertEqual(stats["counts_by_type"], {"local_factual": 1, "cross_clause": 1})


if __name__ == "__main__":
    unittest.main()
