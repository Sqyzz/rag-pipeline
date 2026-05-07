from __future__ import annotations

import ast
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

from src.adapters.youtu_graph_state import build_state_payload, decide_graph_reuse, save_graph_state
from src.baselines.youtu_graph_rag_adapter import map_youtu_to_graph_payload


def _extract_compare_functions(*function_names: str):
    path = SRC / "experiments" / "run_ragas_cuad_compare.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    selected = []
    wanted = set(function_names)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            selected.append(node)
    assert len(selected) == len(wanted), f"missing compare functions: {wanted - {node.name for node in selected}}"
    module = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(module)

    def _normalize_evidence_chunks(value):
        chunks = value if isinstance(value, list) else []
        return {
            "retrieved_contexts": [str(item.get("text") or "") for item in chunks if isinstance(item, dict)],
            "retrieved_context_ids": [str(item.get("chunk_id") or "") for item in chunks if isinstance(item, dict)],
            "retrieved_doc_ids": [str(item.get("doc_id") or "") for item in chunks if isinstance(item, dict)],
        }

    namespace = {
        "Any": object,
        "normalize_evidence_chunks": _normalize_evidence_chunks,
    }
    exec(compile(module, str(path), "exec"), namespace)
    return {name: namespace[name] for name in function_names}


_COMPARE_HELPERS = _extract_compare_functions("_normalize_answer_mode", "_extract_response_for_eval", "_build_output_row")
_build_output_row = _COMPARE_HELPERS["_build_output_row"]


class TestYoutuPipelineAlignment(unittest.TestCase):
    def test_map_youtu_to_payload_has_required_keys(self) -> None:
        response = {
            "data": {
                "answer": "A",
                "retrieved_chunks": [{"chunk_id": "c1", "text": "chunk text"}],
                "retrieved_triples": [{"edge_id": "e1", "source": "s", "relation": "r", "target": "t"}],
                "communities": [{"community_id": "cm1", "summary": "sum"}],
            },
            "meta": {
                "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
                "llm_calls": [{"latency_ms": 10}],
                "embedding_calls": [{"latency_ms": 20}],
            },
        }
        payload = map_youtu_to_graph_payload(
            response,
            query_level=0,
            use_hierarchy=True,
            use_community_summaries=True,
            shuffle_communities=False,
            use_map_reduce=True,
            map_keypoints_limit=5,
            max_evidence=12,
        )

        for key in (
            "answer",
            "communities",
            "community_summaries",
            "query_level",
            "use_hierarchy",
            "use_community_summaries",
            "shuffle_communities",
            "use_map_reduce",
            "map_keypoints_limit",
            "map_partial_answers",
            "subgraph_edges",
            "evidence",
            "evidence_chunks",
            "telemetry",
        ):
            self.assertIn(key, payload)

        self.assertEqual(payload["answer"], "A")
        self.assertEqual(payload["telemetry"]["total_tokens"], 3)
        self.assertEqual(payload["telemetry"]["llm_calls"], 1)

    def test_map_youtu_payload_alignment_fields_and_inline_chunk_id(self) -> None:
        response = {
            "data": {
                "answer": "A",
                "retrieved_chunk_items": [
                    {"chunk_id": "chunk-001", "doc_id": "doc-001#p0", "text": "first chunk"},
                    {"text": "[chunk_id=inline-123] second chunk"},
                ],
                "retrieved_triples_struct": [
                    {"source": "s", "relation": "r", "target": "t", "edge_id": "edge-1"},
                ],
            },
            "meta": {},
        }
        payload = map_youtu_to_graph_payload(
            response,
            query_level=0,
            use_hierarchy=True,
            use_community_summaries=True,
            shuffle_communities=False,
            use_map_reduce=True,
            map_keypoints_limit=5,
            max_evidence=12,
        )
        self.assertEqual(payload["evidence_chunks"][0]["chunk_id"], "chunk-001")
        self.assertEqual(payload["evidence_chunks"][0]["doc_id"], "doc-001#p0")
        self.assertEqual(payload["evidence_chunks"][0]["alignment_source"], "backend_structured")
        self.assertEqual(payload["evidence_chunks"][1]["chunk_id"], "inline-123")
        self.assertEqual(payload["evidence_chunks"][1]["alignment_source"], "inline_chunk_id")
        self.assertEqual(payload["subgraph_edges"][0]["edge_id"], "edge-1")

    def test_map_youtu_payload_filters_virtual_community_context_chunk(self) -> None:
        response = {
            "data": {
                "answer": "A",
                "retrieved_chunk_items": [
                    {"chunk_id": "community_context", "text": "summary text"},
                    {"chunk_id": "chunk-001", "doc_id": "doc-001#p0", "text": "first chunk"},
                ],
            },
            "meta": {},
        }
        payload = map_youtu_to_graph_payload(
            response,
            query_level=0,
            use_hierarchy=True,
            use_community_summaries=True,
            shuffle_communities=False,
            use_map_reduce=True,
            map_keypoints_limit=5,
            max_evidence=12,
        )
        self.assertEqual(len(payload["evidence_chunks"]), 1)
        self.assertEqual(payload["evidence_chunks"][0]["chunk_id"], "chunk-001")

    def test_map_youtu_payload_rejects_text_hash_fallback_when_doc_scope_conflicts(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            store_file = Path(d) / "chunks.json"
            store_file.write_text(
                json.dumps(
                    [
                        {
                            "chunk_id": "chunk-a",
                            "doc_id": "doc-a#p0",
                            "text": "same repeated clause text",
                        }
                    ]
                ),
                encoding="utf-8",
            )
            response = {
                "data": {
                    "answer": "A",
                    "retrieved_chunk_items": [
                        {"doc_id": "doc-b#p0", "text": "same repeated clause text"},
                    ],
                },
                "meta": {},
            }
            payload = map_youtu_to_graph_payload(
                response,
                query_level=0,
                use_hierarchy=True,
                use_community_summaries=True,
                shuffle_communities=False,
                use_map_reduce=True,
                map_keypoints_limit=5,
                max_evidence=12,
                store_file=str(store_file),
            )

        self.assertEqual(payload["evidence_chunks"][0]["chunk_id"], "")
        self.assertEqual(payload["evidence_chunks"][0]["doc_id"], "doc-b#p0")
        self.assertEqual(payload["evidence_chunks"][0]["alignment_source"], "text_hash_doc_conflict")

    def test_map_youtu_payload_parses_textual_triples(self) -> None:
        response = {
            "data": {
                "answer": "A",
                "retrieved_triples": [
                    "([Unknown Node: Exhibit B] , contains, [Unknown Node: Operational terms for purchase of Xplore Products] ) [score: 0.442]",
                ],
            },
            "meta": {},
        }
        payload = map_youtu_to_graph_payload(
            response,
            query_level=0,
            use_hierarchy=True,
            use_community_summaries=True,
            shuffle_communities=False,
            use_map_reduce=True,
            map_keypoints_limit=5,
            max_evidence=12,
        )
        edge = payload["subgraph_edges"][0]
        self.assertEqual(edge["source"], "Exhibit B")
        self.assertEqual(edge["relation"], "contains")
        self.assertEqual(edge["target"], "Operational terms for purchase of Xplore Products")
        self.assertTrue(str(edge["edge_id"]).strip())

    def test_map_youtu_payload_parses_textual_triples_from_dict_text(self) -> None:
        response = {
            "data": {
                "answer": "A",
                "retrieved_triples": [
                    {
                        "edge_id": "",
                        "source": None,
                        "relation": None,
                        "target": None,
                        "text": "([Unknown Node: Exhibit B] , governs, [Unknown Node: Delivery of Products] ) [score: 0.361]",
                    }
                ],
            },
            "meta": {},
        }
        payload = map_youtu_to_graph_payload(
            response,
            query_level=0,
            use_hierarchy=True,
            use_community_summaries=True,
            shuffle_communities=False,
            use_map_reduce=True,
            map_keypoints_limit=5,
            max_evidence=12,
        )
        edge = payload["subgraph_edges"][0]
        self.assertEqual(edge["source"], "Exhibit B")
        self.assertEqual(edge["relation"], "governs")
        self.assertEqual(edge["target"], "Delivery of Products")
        self.assertTrue(str(edge["edge_id"]).strip())

    def test_map_youtu_payload_preserves_dual_surface_payloads(self) -> None:
        response = {
            "data": {
                "answer": "A",
                "retrieval_trace": {"query": "q"},
                "reasoning_trace": {"sub_question_answers": [{"sub_question_id": "sq_1"}]},
                "answer_trace": {"aggregation_strategy": "sub_answer_reason_first_with_chunk_backing"},
                "reasoning_steps": [{"type": "sub_question_ircot", "sub_question_id": "sq_1", "step": 1}],
                "evaluation_payload": {
                    "response": "A",
                    "retrieved_contexts": ["raw chunk"],
                    "retrieved_context_ids": ["chunk-1"],
                    "retrieved_doc_ids": ["doc-1#p0"],
                },
            },
            "meta": {},
        }
        payload = map_youtu_to_graph_payload(
            response,
            query_level=0,
            use_hierarchy=True,
            use_community_summaries=True,
            shuffle_communities=False,
            use_map_reduce=True,
            map_keypoints_limit=5,
            max_evidence=12,
        )
        self.assertEqual(payload["retrieval_trace"]["query"], "q")
        self.assertEqual(payload["reasoning_trace"]["sub_question_answers"][0]["sub_question_id"], "sq_1")
        self.assertEqual(payload["answer_trace"]["aggregation_strategy"], "sub_answer_reason_first_with_chunk_backing")
        self.assertEqual(payload["reasoning_steps"][0]["type"], "sub_question_ircot")
        self.assertEqual(payload["evaluation_payload"]["retrieved_context_ids"], ["chunk-1"])

    def test_compare_output_row_prefers_evaluation_payload_contexts(self) -> None:
        row = _build_output_row(
            {
                "qid": "q1",
                "question": "What is the term?",
                "reference": "The term is two years.",
                "reference_contexts": ["gold ctx"],
                "reference_context_ids": ["gold-1"],
                "reference_doc_ids": ["doc-1#p0"],
            },
            "youtu_graph_rag",
            {
                "answer": "generated answer",
                "evidence_chunks": [{"chunk_id": "fallback-1", "doc_id": "doc-x#p0", "text": "fallback ctx"}],
                "evaluation_payload": {
                    "response": "eval answer",
                    "retrieved_contexts": ["raw eval ctx"],
                    "retrieved_context_ids": ["eval-1"],
                    "retrieved_doc_ids": ["doc-1#p0"],
                    "external_related_context_ids": ["ext-1"],
                    "external_related_doc_ids": ["doc-2#p0"],
                },
                "retrieval_trace": {
                    "query": "What is the term?",
                    "answer_scope_target_doc_id": "doc-1#p0",
                    "answer_composition_mode": "cross_document_bridge",
                    "semantic_alignment": {
                        "alignment_type": "conceptual_overlap",
                        "shared_concepts": ["third-party", "costs/expenses"],
                    },
                },
                "reasoning_trace": {"sub_question_answers": []},
                "answer_trace": {"aggregation_strategy": "sub_answer_reason_first_with_chunk_backing"},
                "reasoning_steps": [{"type": "sub_question_ircot", "step": 1}],
            },
        )
        self.assertEqual(row["response"], "eval answer")
        self.assertEqual(row["retrieved_contexts"], ["raw eval ctx"])
        self.assertEqual(row["retrieved_context_ids"], ["eval-1"])
        self.assertEqual(row["external_related_context_ids"], ["ext-1"])
        self.assertEqual(row["external_related_doc_ids"], ["doc-2#p0"])
        self.assertEqual(row["answer_scope_target_doc_id"], "doc-1#p0")
        self.assertEqual(row["answer_composition_mode"], "cross_document_bridge")
        self.assertEqual(row["semantic_alignment"]["alignment_type"], "conceptual_overlap")
        self.assertEqual(row["retrieval_trace"]["query"], "What is the term?")
        self.assertEqual(row["reasoning_steps"][0]["type"], "sub_question_ircot")

    def test_compare_output_row_extracts_eval_ready_response_tail(self) -> None:
        row = _build_output_row(
            {
                "qid": "q2",
                "question": "How do these clauses relate?",
                "reference": "They are related but distinct.",
                "reference_contexts": [],
                "reference_context_ids": [],
                "reference_doc_ids": [],
            },
            "youtu_graph_rag",
            {
                "evaluation_payload": {
                    "response": (
                        "**Reasoning over Facts**\n\n"
                        "**Grounded Facts**\n\n"
                        "Fact block.\n\n"
                        "**Inference and Conclusion:**\n\n"
                        "The clauses are related but distinct."
                    ),
                    "retrieved_contexts": [],
                    "retrieved_context_ids": [],
                    "retrieved_doc_ids": [],
                },
                "retrieval_trace": {},
                "reasoning_trace": {},
                "answer_trace": {},
            },
        )
        self.assertEqual(row["response_for_eval"], "The clauses are related but distinct.")
        self.assertEqual(row["response_for_eval_source"], "inference_and_conclusion")

    def test_compare_output_row_prefers_final_answer_marker_over_other_sections(self) -> None:
        row = _build_output_row(
            {
                "qid": "q2b",
                "question": "How do these clauses relate?",
                "reference": "They are related but distinct.",
                "reference_contexts": [],
                "reference_context_ids": [],
                "reference_doc_ids": [],
            },
            "youtu_graph_rag",
            {
                "evaluation_payload": {
                    "response": (
                        "**Reasoning over Facts**\n\n"
                        "Fact block.\n\n"
                        "**Inference and Conclusion:**\n\n"
                        "Older concise answer.\n\n"
                        "**Final Answer:**\n\n"
                        "Canonical concise answer."
                    ),
                    "retrieved_contexts": [],
                    "retrieved_context_ids": [],
                    "retrieved_doc_ids": [],
                },
                "retrieval_trace": {},
                "reasoning_trace": {},
                "answer_trace": {},
            },
        )
        self.assertEqual(row["response_for_eval"], "Canonical concise answer.")
        self.assertEqual(row["response_for_eval_source"], "final_answer")

    def test_compare_output_row_prefers_explicit_response_for_eval_field(self) -> None:
        row = _build_output_row(
            {
                "qid": "q2c",
                "question": "How do these clauses relate?",
                "reference": "They are related but distinct.",
                "reference_contexts": [],
                "reference_context_ids": [],
                "reference_doc_ids": [],
            },
            "youtu_graph_rag",
            {
                "evaluation_payload": {
                    "response": (
                        "Grounded Facts from the Target Contract:\n\n"
                        "Fact block.\n\n"
                        "Final Answer:\n\n"
                        "Canonical concise answer."
                    ),
                    "response_for_eval": "Preparsed concise answer.",
                    "response_for_eval_source": "evaluation_payload",
                    "retrieved_contexts": [],
                    "retrieved_context_ids": [],
                    "retrieved_doc_ids": [],
                },
                "retrieval_trace": {},
                "reasoning_trace": {},
                "answer_trace": {},
            },
        )
        self.assertEqual(row["response_for_eval"], "Preparsed concise answer.")
        self.assertEqual(row["response_for_eval_source"], "evaluation_payload")

    def test_compare_output_row_falls_back_to_evidence_when_eval_context_lists_are_empty(self) -> None:
        row = _build_output_row(
            {
                "qid": "q3",
                "question": "What is the term?",
                "reference": "The term is two years.",
                "reference_contexts": [],
                "reference_context_ids": [],
                "reference_doc_ids": [],
            },
            "youtu_graph_rag",
            {
                "answer": "generated answer",
                "evidence_chunks": [{"chunk_id": "fallback-1", "doc_id": "doc-x#p0", "text": "fallback ctx"}],
                "evaluation_payload": {
                    "response": "eval answer",
                    "retrieved_contexts": [],
                    "retrieved_context_ids": [],
                    "retrieved_doc_ids": [],
                },
                "retrieval_trace": {},
                "reasoning_trace": {},
                "answer_trace": {},
            },
        )
        self.assertEqual(row["retrieved_contexts"], ["fallback ctx"])
        self.assertEqual(row["retrieved_context_ids"], ["fallback-1"])
        self.assertEqual(row["retrieved_doc_ids"], ["doc-x#p0"])

    def test_map_youtu_payload_marks_chunk_id_pairing_mismatch_as_unaligned(self) -> None:
        response = {
            "data": {
                "answer": "A",
                "retrieved_chunks": ["first chunk", "second chunk"],
                "retrieved_chunk_ids": ["chunk-001"],
            },
            "meta": {},
        }
        payload = map_youtu_to_graph_payload(
            response,
            query_level=0,
            use_hierarchy=True,
            use_community_summaries=True,
            shuffle_communities=False,
            use_map_reduce=True,
            map_keypoints_limit=5,
            max_evidence=12,
        )
        self.assertEqual(payload["evidence_chunks"][0]["chunk_id"], "")
        self.assertEqual(payload["evidence_chunks"][0]["alignment_source"], "paired_chunk_ids_length_mismatch")
        self.assertEqual(payload["evidence_chunks"][1]["chunk_id"], "")
        self.assertEqual(payload["evidence_chunks"][1]["alignment_source"], "paired_chunk_ids_length_mismatch")

    def test_graph_reuse_second_run_hits_cache(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            chunks = root / "chunks.jsonl"
            graph = root / "graph.json"
            communities = root / "communities.json"
            state = root / "youtu_graph_state.json"

            chunks.write_text('{"chunk_id":"c1","text":"hello"}\n', encoding="utf-8")
            graph.write_text("{}", encoding="utf-8")
            communities.write_text("{}", encoding="utf-8")

            params = {"dataset": "enterprise", "sync_mode": "none"}
            first = decide_graph_reuse(
                graph_state_file=str(state),
                chunks_file=str(chunks),
                dataset="enterprise",
                build_params=params,
                reuse_graph=True,
                force_rebuild=False,
                require_local_assets=True,
                graph_file=str(graph),
                communities_file=str(communities),
            )
            self.assertFalse(first["used_cached_graph"])

            save_graph_state(
                str(state),
                build_state_payload(
                    dataset="enterprise",
                    fingerprint=first["fingerprint"],
                    build_params=params,
                    graph_task_id="task-1",
                ),
            )

            second = decide_graph_reuse(
                graph_state_file=str(state),
                chunks_file=str(chunks),
                dataset="enterprise",
                build_params=params,
                reuse_graph=True,
                force_rebuild=False,
                require_local_assets=True,
                graph_file=str(graph),
                communities_file=str(communities),
            )
            self.assertTrue(second["used_cached_graph"])
            self.assertEqual(second["reason"], "fingerprint_match")


if __name__ == "__main__":
    unittest.main()
