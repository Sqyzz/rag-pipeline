from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Dict, List, Optional


def _load_backend_module():
    path = Path(__file__).resolve().parents[1] / "youtu-graphrag" / "backend.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    targets = []
    wanted = {"_select_chunk_ids_with_subquestion_guarantee", "_prioritize_chunk_ids_by_preferred_docs"}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            targets.append(node)
    assert {node.name for node in targets} == wanted
    module = ast.Module(body=targets, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"List": List, "Dict": Dict, "Any": Any, "Optional": Optional}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


def test_select_chunk_ids_with_subquestion_guarantee_reserves_each_hop() -> None:
    backend = _load_backend_module()
    selected = backend["_select_chunk_ids_with_subquestion_guarantee"](
        [
            "a1",
            "a2",
            "a3",
            "a4",
            "b1",
            "b2",
            "b3",
            "c1",
            "c2",
        ],
        [
            {"retrieved_chunk_ids_all": ["a1", "a2", "a3", "a4"]},
            {"retrieved_chunk_ids_all": ["b1", "b2", "b3"]},
            {"retrieved_chunk_ids_all": ["c1", "c2"]},
        ],
        total_limit=6,
        per_subquestion_min=2,
    )

    assert selected == ["a1", "a2", "b1", "b2", "c1", "c2"]


def test_select_chunk_ids_with_subquestion_guarantee_prefers_same_doc_softly() -> None:
    backend = _load_backend_module()
    selected = backend["_select_chunk_ids_with_subquestion_guarantee"](
        [
            "x1",
            "a1",
            "x2",
            "b1",
            "a2",
            "b2",
            "x3",
        ],
        [
            {"retrieved_chunk_ids_all": ["a1", "a2"]},
            {"retrieved_chunk_ids_all": ["b1", "b2"]},
        ],
        total_limit=5,
        per_subquestion_min=1,
        chunk_id_to_doc_id={
            "x1": "doc_x",
            "x2": "doc_x",
            "x3": "doc_x",
            "a1": "doc_a",
            "a2": "doc_a",
            "b1": "doc_b",
            "b2": "doc_b",
        },
        preferred_doc_ids=["doc_a", "doc_b"],
        same_doc_bonus=2.0,
    )

    assert selected[:4] == ["a1", "b1", "a2", "b2"]
    assert selected[-1] == "x1"


def test_select_chunk_ids_with_subquestion_guarantee_does_not_overreserve_weak_not_found_queries() -> None:
    backend = _load_backend_module()
    selected = backend["_select_chunk_ids_with_subquestion_guarantee"](
        [
            "strong1",
            "strong2",
            "strong3",
            "weak1",
            "weak2",
            "weak3",
        ],
        [
            {
                "retrieved_chunk_ids_all": ["weak1", "weak2", "weak3"],
                "query_text": "what clause states clause{name or identifier referencing 'Section 504'}",
                "retrieval_queries": ["what clause states clause{name or identifier referencing 'Section 504'}"],
                "status": "not_found",
                "confidence": 0.0,
            },
            {
                "retrieved_chunk_ids_all": ["strong1", "strong2", "strong3"],
                "query_text": "section 8 section 10.3 indemnification uncapped liability",
                "retrieval_queries": ["section 8 section 10.3 indemnification uncapped liability"],
                "status": "answered",
                "confidence": 0.84,
            },
        ],
        total_limit=4,
        per_subquestion_min=2,
    )

    assert selected == ["strong1", "strong2", "strong3", "weak1"]


def test_prioritize_chunk_ids_by_preferred_docs_drops_cross_doc_fill_when_same_doc_supply_is_sufficient() -> None:
    backend = _load_backend_module()
    selected = backend["_prioritize_chunk_ids_by_preferred_docs"](
        ["a1", "a2", "x1", "a3", "x2", "a4", "x3"],
        chunk_id_to_doc_id={
            "a1": "doc_a",
            "a2": "doc_a",
            "a3": "doc_a",
            "a4": "doc_a",
            "x1": "doc_x",
            "x2": "doc_x",
            "x3": "doc_x",
        },
        preferred_doc_ids=["doc_a"],
        total_limit=4,
        strict_if_enough=True,
    )

    assert selected == ["a1", "a2", "a3", "a4"]
