from __future__ import annotations

import ast
import concurrent
import concurrent.futures
import time
import types
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


ROOT = Path(__file__).resolve().parents[1]
YOUTU_ROOT = ROOT / "youtu-graphrag"


logger_stub = types.SimpleNamespace(
    info=lambda *a, **k: None,
    warning=lambda *a, **k: None,
    error=lambda *a, **k: None,
    debug=lambda *a, **k: None,
)


def _extract_top_level_functions(path: Path, *function_names: str, extra_namespace: Optional[Dict[str, Any]] = None):
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    selected = []
    wanted = set(function_names)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            selected.append(node)
    assert len(selected) == len(wanted), f"missing functions: {wanted - {node.name for node in selected}}"

    module = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "Any": Any,
        "Dict": Dict,
        "List": List,
        "logger": logger_stub,
    }
    if extra_namespace:
        namespace.update(extra_namespace)
    exec(compile(module, str(path), "exec"), namespace)
    return {name: namespace[name] for name in function_names}


def _extract_class_method(path: Path, class_name: str, method_name: str, extra_namespace: Optional[Dict[str, Any]] = None):
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    target = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == method_name:
                    target = child
                    break
    assert target is not None, f"{class_name}.{method_name} not found"

    module = ast.Module(body=[target], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "Any": Any,
        "Dict": Dict,
        "List": List,
        "Optional": Optional,
        "Tuple": Tuple,
        "concurrent": concurrent,
        "logger": logger_stub,
        "time": time,
    }
    if extra_namespace:
        namespace.update(extra_namespace)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[method_name]


def test_initial_question_decomposition_consumes_retrieval_requirements() -> None:
    helpers = _extract_top_level_functions(
        YOUTU_ROOT / "main.py",
        "_dedupe_preserve_order",
        "deduplicate_triples",
        "merge_chunk_contents",
        "rerank_chunks_by_keywords",
        "_normalize_route_type",
        "_coerce_str_list",
        "_build_requirement_queries",
        "_build_requirement_sub_question",
        "_build_retrieval_work_items",
        "initial_question_decomposition",
        extra_namespace={
            "config": types.SimpleNamespace(
                retrieval=types.SimpleNamespace(top_k_filter=5)
            ),
        },
    )

    initial_question_decomposition = helpers["initial_question_decomposition"]
    seen: Dict[str, Any] = {}

    graphq = types.SimpleNamespace(
        decompose=lambda question, schema_path: {
            "sub_questions": [],
            "retrieval_requirements": [
                {
                    "route_type": "local",
                    "route_reason": "single clause fact lookup",
                    "entities": ["Supplier"],
                    "terms": ["notice address"],
                    "anchors": ["Section 5.1"],
                    "query_keywords": ["notice address", "supplier"],
                    "target_patterns": ["which clause states"],
                }
            ],
            "involved_types": {"nodes": ["clause"], "relations": [], "attributes": []},
        }
    )

    def _process_subquestions_parallel(sub_questions, top_k=10, involved_types=None, original_question=None):
        seen["sub_questions"] = sub_questions
        seen["top_k"] = top_k
        seen["involved_types"] = involved_types
        seen["original_question"] = original_question
        return (
            {
                "triples": ["triple_1"],
                "chunk_ids": ["chunk_1"],
                "chunk_contents": {"chunk_1": "chunk body"},
                "sub_question_results": [
                    {
                        "sub_question": sub_questions[0]["sub-question"],
                        "route_type": sub_questions[0]["route_type"],
                        "triples_count": 1,
                        "chunk_ids_count": 1,
                        "time_taken": 0.1,
                    }
                ],
            },
            0.1,
        )

    kt_retriever = types.SimpleNamespace(
        _infer_target_doc_id_from_question=lambda question: "DOC_1",
        process_subquestions_parallel=_process_subquestions_parallel,
        generate_prompt=lambda question, context: f"{question}\n{context}",
        generate_answer=lambda prompt: "answer",
    )

    result = initial_question_decomposition(graphq, kt_retriever, "Original question", "schema.json")

    assert seen["original_question"] == "Original question"
    assert len(seen["sub_questions"]) == 1
    assert seen["sub_questions"][0]["route_type"] == "local"
    assert seen["sub_questions"][0]["retrieval_requirement"]["terms"] == ["notice address"]
    assert seen["sub_questions"][0]["target_doc_id"] == "DOC_1"
    assert result["sub_questions"][0]["route_type"] == "local"
    assert result["decomposition_result"]["compiled_sub_questions"][0]["route_type"] == "local"
    assert result["chunk_ids"] == ["chunk_1"]


def test_process_single_subquestion_merges_retrieval_queries_with_route_metadata() -> None:
    method = _extract_class_method(
        YOUTU_ROOT / "models" / "retriever" / "enhanced_kt_retriever.py",
        "KTRetriever",
        "_process_single_subquestion",
    )

    calls: List[Dict[str, Any]] = []

    def _dedupe(items: List[str]) -> List[str]:
        ordered: List[str] = []
        seen: set[str] = set()
        for item in items or []:
            text = str(item or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            ordered.append(text)
        return ordered

    def _process_retrieval_results(
        query,
        top_k,
        involved_types,
        route_type=None,
        original_question=None,
        target_doc_id=None,
        retrieval_requirement=None,
    ):
        calls.append(
            {
                "query": query,
                "top_k": top_k,
                "involved_types": involved_types,
                "route_type": route_type,
                "original_question": original_question,
                "target_doc_id": target_doc_id,
                "retrieval_requirement": retrieval_requirement,
            }
        )
        return (
            {
                "triples": [f"triple:{query}"],
                "chunk_ids": [f"chunk:{query}"],
                "chunk_contents": [f"content:{query}"],
                "route_type": route_type,
                "route_fallback": "",
                "target_doc_id": target_doc_id,
                "scope_plan": {},
            },
            0.1,
        )

    def _merge_payloads(payloads, fallback_route):
        triples: List[str] = []
        chunk_ids: List[str] = []
        chunk_contents: List[str] = []
        for payload in payloads:
            triples.extend(payload.get("triples", []) or [])
            chunk_ids.extend(payload.get("chunk_ids", []) or [])
            chunk_contents.extend(payload.get("chunk_contents", []) or [])
        return {
            "triples": triples,
            "chunk_ids": chunk_ids,
            "chunk_contents": chunk_contents,
            "route_type": fallback_route,
            "route_fallback": "",
            "target_doc_id": "DOC_1",
            "scope_plan": {},
        }

    fake_self = types.SimpleNamespace(
        _normalize_route_type=lambda route: route or "structural",
        _dedupe_preserve_order=_dedupe,
        _merge_retrieval_payloads=_merge_payloads,
        process_retrieval_results=_process_retrieval_results,
    )

    result = method(
        fake_self,
        {
            "sub-question": "What does the agreement say?",
            "route_type": "local",
            "retrieval_queries": ["supplier notice", "section 5.1 notice"],
            "retrieval_requirement": {"terms": ["notice address"]},
            "target_doc_id": "DOC_1",
        },
        6,
        {"nodes": ["clause"], "relations": [], "attributes": []},
        "Original question",
    )

    assert [call["query"] for call in calls] == ["supplier notice", "section 5.1 notice", "What does the agreement say?"]
    assert all(call["route_type"] == "local" for call in calls)
    assert all(call["original_question"] == "Original question" for call in calls)
    assert all(call["target_doc_id"] == "DOC_1" for call in calls)
    assert result["triples"] == [
        "triple:supplier notice",
        "triple:section 5.1 notice",
        "triple:What does the agreement say?",
    ]
    assert result["chunk_ids"] == [
        "chunk:supplier notice",
        "chunk:section 5.1 notice",
        "chunk:What does the agreement say?",
    ]
    assert result["sub_result"]["route_type"] == "local"


def test_process_subquestions_parallel_preserves_input_order() -> None:
    method = _extract_class_method(
        YOUTU_ROOT / "models" / "retriever" / "enhanced_kt_retriever.py",
        "KTRetriever",
        "process_subquestions_parallel",
    )

    def _dedupe(items: List[str]) -> List[str]:
        ordered: List[str] = []
        seen: set[str] = set()
        for item in items or []:
            text = str(item or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            ordered.append(text)
        return ordered

    def _process_single_subquestion(sub_question, top_k, involved_types=None, original_question=None):
        if sub_question["sub-question"] == "q1":
            time.sleep(0.05)
        else:
            time.sleep(0.01)
        chunk_id = f"chunk:{sub_question['sub-question']}"
        return {
            "triples": [f"triple:{sub_question['sub-question']}"],
            "chunk_ids": [chunk_id],
            "chunk_contents": {chunk_id: f"content:{sub_question['sub-question']}"},
            "sub_result": {
                "sub_question": sub_question["sub-question"],
                "route_type": sub_question.get("route_type", "structural"),
                "triples_count": 1,
                "chunk_ids_count": 1,
                "time_taken": 0.01,
            },
        }

    fake_self = types.SimpleNamespace(
        config=types.SimpleNamespace(
            retrieval=types.SimpleNamespace(
                faiss=types.SimpleNamespace(max_workers=2)
            )
        ),
        _dedupe_preserve_order=_dedupe,
        _process_single_subquestion=_process_single_subquestion,
    )

    aggregated, _ = method(
        fake_self,
        [
            {"sub-question": "q1", "route_type": "local"},
            {"sub-question": "q2", "route_type": "structural"},
        ],
        top_k=4,
        involved_types={},
        original_question="Original question",
    )

    assert aggregated["triples"] == ["triple:q1", "triple:q2"]
    assert aggregated["chunk_ids"] == ["chunk:q1", "chunk:q2"]
    assert aggregated["sub_question_results"][0]["sub_question"] == "q1"
    assert aggregated["sub_question_results"][1]["sub_question"] == "q2"
