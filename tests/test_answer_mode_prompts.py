from __future__ import annotations

import unittest
from unittest.mock import patch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from baselines import graph_rag, kg_rag, vector_rag


class _DummyLLM:
    def __init__(self) -> None:
        self.last_prompt = ""

    def chat(self, prompt: str, max_tokens=None, return_meta: bool = False, model=None):  # noqa: ANN001
        self.last_prompt = str(prompt)
        if return_meta:
            return "OK", {"usage": {}}
        return "OK"


class TestAnswerModePrompts(unittest.TestCase):
    def test_vector_prompt_switches_by_answer_mode(self) -> None:
        dummy = _DummyLLM()
        with patch.object(vector_rag, "llm", dummy):
            vector_rag.answer_with_context(
                query="Q",
                contexts=["C1"],
                return_meta=False,
                answer_mode="reject",
            )
            self.assertIn("return exactly: NOT_FOUND", dummy.last_prompt)
            self.assertNotIn("OUTSIDE_EVIDENCE:", dummy.last_prompt)

            vector_rag.answer_with_context(
                query="Q",
                contexts=["C1"],
                return_meta=False,
                answer_mode="open",
            )
            self.assertIn("OUTSIDE_EVIDENCE:", dummy.last_prompt)

    def test_kg_prompt_and_payload_switch_by_answer_mode(self) -> None:
        captured = {"prompt": ""}

        def _fake_llm_chat(messages, **kwargs):  # noqa: ANN001
            captured["prompt"] = str(messages[0]["content"])
            return "OK", {"usage": {}}

        with patch.object(kg_rag, "_load_json", return_value={"nodes": [], "edges": []}):
            with patch.object(kg_rag, "llm_chat", side_effect=_fake_llm_chat):
                out_reject = kg_rag.answer_with_kg(
                    query="Q",
                    graph_file="unused.json",
                    store_file=None,
                    answer_mode="reject",
                )
                self.assertEqual(out_reject["answer_mode"], "reject")
                self.assertIn("return exactly: NOT_FOUND", captured["prompt"])

                out_open = kg_rag.answer_with_kg(
                    query="Q",
                    graph_file="unused.json",
                    store_file=None,
                    answer_mode="open",
                )
                self.assertEqual(out_open["answer_mode"], "open")
                self.assertIn("OUTSIDE_EVIDENCE:", captured["prompt"])

    def test_graph_reduce_prompt_switch_by_answer_mode(self) -> None:
        captured = {"prompt": ""}

        def _fake_llm_chat(messages, **kwargs):  # noqa: ANN001
            captured["prompt"] = str(messages[0]["content"])
            return "OK", {"usage": {}}

        with patch.object(graph_rag, "llm_chat", side_effect=_fake_llm_chat):
            answer, _ = graph_rag._reduce_answers(
                query="Q",
                partials=[],
                answer_mode="reject",
            )
            self.assertEqual(answer, "OK")
            self.assertIn("return exactly: NOT_FOUND", captured["prompt"])

            answer, _ = graph_rag._reduce_answers(
                query="Q",
                partials=[],
                answer_mode="open",
            )
            self.assertEqual(answer, "OK")
            self.assertIn("OUTSIDE_EVIDENCE:", captured["prompt"])


if __name__ == "__main__":
    unittest.main()
