from __future__ import annotations

import ast
import math
import importlib.util
import json
import re
import sys
import time
import textwrap
import types
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
YOUTU_ROOT = ROOT / "youtu-graphrag"

for candidate in (YOUTU_ROOT, ROOT):
    value = str(candidate)
    if value not in sys.path:
        sys.path.insert(0, value)


logger_stub = types.SimpleNamespace(
    info=lambda *a, **k: None,
    warning=lambda *a, **k: None,
    error=lambda *a, **k: None,
)
fake_torch = types.SimpleNamespace(Tensor=object)


class FakeTensor:
    def __init__(self, data):
        self.data = np.array(data, dtype=np.float32)

    def cpu(self):
        return self

    def numpy(self):
        return self.data

    def unsqueeze(self, axis: int):
        return FakeTensor(np.expand_dims(self.data, axis=axis))


class FakeFunctional:
    @staticmethod
    def cosine_similarity(left, right, dim=1):
        left_arr = np.array(left.data if isinstance(left, FakeTensor) else left, dtype=np.float32)
        right_arr = np.array(right.data if isinstance(right, FakeTensor) else right, dtype=np.float32)
        numerator = np.sum(left_arr * right_arr, axis=dim)
        denominator = np.linalg.norm(left_arr, axis=dim) * np.linalg.norm(right_arr, axis=dim)
        denominator = np.maximum(denominator, 1e-8)
        return numerator / denominator


sys.modules.setdefault("utils.logger", types.SimpleNamespace(logger=logger_stub))
sys.modules.setdefault("json_repair", types.SimpleNamespace(loads=json.loads))
sys.modules.setdefault(
    "utils.call_llm_api",
    types.SimpleNamespace(LLMCompletionCall=lambda *a, **k: None),
)


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _extract_class_method(path: Path, class_name: str, method_name: str):
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
        "Counter": Counter,
        "Dict": Dict,
        "List": List,
        "Optional": Optional,
        "Set": Set,
        "Tuple": Tuple,
        "F": FakeFunctional,
        "defaultdict": defaultdict,
        "logger": logger_stub,
        "math": math,
        "np": np,
        "re": re,
        "time": time,
        "torch": fake_torch,
    }
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[method_name]


def _load_ktretriever_process_retrieval_results():
    return _extract_class_method(
        YOUTU_ROOT / "models" / "retriever" / "enhanced_kt_retriever.py",
        "KTRetriever",
        "process_retrieval_results",
    )


def _extract_backend_functions(*function_names: str):
    path = YOUTU_ROOT / "backend.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    selected = []
    wanted = set(function_names)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            selected.append(node)
    assert len(selected) == len(wanted), f"missing backend functions: {wanted - {node.name for node in selected}}"
    module = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "Any": Any,
        "Dict": Dict,
        "List": List,
        "Optional": Optional,
        "Tuple": Tuple,
        "defaultdict": defaultdict,
        "EvidenceItem": dataclass(
            type(
                "EvidenceItem",
                (),
                {
                    "__annotations__": {
                        "origin_sub_question_id": str,
                        "origin_route": str,
                        "origin_query": str,
                        "doc_id": str,
                        "chunk_id": str,
                        "evidence_role": str,
                        "score": float,
                        "scope_label": str,
                    }
                },
            )
        ),
        "re": re,
    }
    exec(compile(module, str(path), "exec"), namespace)
    return {name: namespace[name] for name in function_names}


def _load_backend_ast() -> ast.Module:
    path = YOUTU_ROOT / "backend.py"
    source = path.read_text(encoding="utf-8")
    return ast.parse(source, filename=str(path))


def _load_backend_compile_namespace():
    path = YOUTU_ROOT / "backend.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    start = source.index("        def _extract_phrase_hints")
    end = source.index("        def _merge_retrieval_payloads")
    snippet = textwrap.dedent(source[start:end])
    top_level_helpers = _extract_backend_functions(
        "_clean_prompt_field",
        "_score_retrieval_query_specificity",
        "_prioritize_retrieval_queries",
        "_strip_question_prompt_prefix",
        "_is_generic_scope_phrase",
        "_is_low_value_subject_hint",
        "_extract_question_subject_overrides",
        "_normalize_semantic_need",
        "_extract_compare_requirement_parts",
        "_extract_role_requirement_parts",
        "_extract_bridge_requirement_parts",
        "_estimate_evidence_weight",
        "_serialize_evidence_requirement",
        "_serialize_sub_question_plan",
    )
    selected_nodes = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name in {"EvidenceRequirement", "SubQuestionPlan"}:
            selected_nodes.append(node)
    ir_module = ast.Module(body=selected_nodes, type_ignores=[])
    ast.fix_missing_locations(ir_module)
    namespace = {
        "Any": Any,
        "Dict": Dict,
        "List": List,
        "Optional": Optional,
        "dataclass": dataclass,
        "field": field,
        "ast": ast,
        "generic_query_terms": {
            "party", "clause", "agreement", "obligation", "payment", "event", "document",
            "location", "duration", "amount", "date", "insurance", "jurisdiction",
            "remedy", "right", "service", "product", "notice",
        },
        "max_query_variants": 4,
        "orchestration_mode": "legacy",
        "_clean_str_list": lambda values: [
            str(value or "").strip()
            for value in (values or [])
            if str(value or "").strip()
        ],
        "re": re,
        **top_level_helpers,
    }
    exec(compile(ir_module, str(path), "exec"), namespace)
    exec(compile(snippet, str(path), "exec"), namespace)
    return namespace


def _load_backend_compile_requirement_queries():
    namespace = _load_backend_compile_namespace()
    return namespace["_compile_requirement_queries"]


def _load_backend_resolve_target_doc_id():
    return _extract_backend_functions("_resolve_target_doc_id_from_question")["_resolve_target_doc_id_from_question"]


def _load_backend_grounding_validator():
    return _extract_backend_functions(
        "_semantic_text_overlap_ratio",
        "_normalize_prompt_text",
        "_validate_sub_answer_grounding",
    )["_validate_sub_answer_grounding"]


def _load_backend_requirements_to_sub_questions_namespace():
    namespace = _load_backend_compile_namespace()
    namespace["enable_query_compilation"] = True
    return namespace


def _load_backend_refactor_scope_namespace():
    path = YOUTU_ROOT / "backend.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    selected = []
    wanted = {
        "_clean_prompt_field",
        "_normalize_prompt_text",
        "_trim_prompt_excerpt",
        "_collect_section_anchor_rescue_candidates",
        "_resolve_target_doc_id_from_question",
        "_resolve_refactor_subquestion_scope",
    }
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            selected.append(node)
    assert len(selected) == len(wanted), f"missing scope helpers: {wanted - {node.name for node in selected}}"
    module = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "Any": Any,
        "Dict": Dict,
        "List": List,
        "Optional": Optional,
        "Tuple": Tuple,
        "re": re,
    }
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


def _load_backend_support_lane_namespace():
    path = YOUTU_ROOT / "backend.py"
    source = path.read_text(encoding="utf-8")
    start = source.index("        def _normalize_chunk_payload")
    end = source.index("        async def _run_subquestion_ircot")
    snippet = textwrap.dedent(source[start:end])
    namespace = {
        "Any": Any,
        "Dict": Dict,
        "List": List,
        "Optional": Optional,
        "Tuple": Tuple,
        "re": re,
        "_score_retrieval_query_specificity": lambda text, route_type="": len(str(text or "")),
        "_rerank_support_pairs_for_subquestion": lambda pairs, **kwargs: list(pairs),
        "_strong_rerank_support_pairs_for_subquestion": lambda pairs, **kwargs: list(pairs),
    }
    exec(compile(snippet, str(path), "exec"), namespace)
    return namespace


agentic_decomposer = _load_module(
    "youtu_agentic_decomposer_p0_test",
    YOUTU_ROOT / "models" / "retriever" / "agentic_decomposer.py",
)


def test_decomposer_drops_generic_requirement_endpoints() -> None:
    config = types.SimpleNamespace(
        retrieval=types.SimpleNamespace(
            decomposition=types.SimpleNamespace(mode="retrieval_requirements")
        )
    )
    decomposer = agentic_decomposer.GraphQ("cuad_v3", config=config)
    decomposer.read_schema = lambda _: "{}"
    decomposer.llm_client = types.SimpleNamespace(
        call_api=lambda _prompt: json.dumps(
            {
                "retrieval_requirements": [
                    {
                        "route_type": "structural",
                        "intent": "bridge_lookup",
                        "route_reason": "bridge",
                        "entities": ["party", "Section 8"],
                        "terms": ["breach consequences"],
                        "anchors": ["Section 8"],
                        "query_keywords": ["breach consequences", "indemnification procedures"],
                        "target_patterns": ["which clause explicitly connects"],
                        "left_endpoint": "party",
                        "right_endpoint": "clause",
                        "bridge_relation": "explicit legal connection",
                        "scope": "",
                    }
                ],
                "involved_types": {"nodes": [], "relations": [], "attributes": []},
            }
        )
    )

    result = decomposer.decompose("q", "schema.json")
    req = result["retrieval_requirements"][0]

    assert req["left_endpoint"] == ""
    assert req["right_endpoint"] == ""
    assert req["entities"] == ["Section 8"]


def test_compile_requirement_queries_avoids_generic_endpoint_queries() -> None:
    compile_requirement_queries = _load_backend_compile_requirement_queries()

    queries = compile_requirement_queries(
        {
            "route_type": "structural",
            "intent": "bridge_lookup",
            "entities": ["Section 8"],
            "terms": ["breach consequences", "indemnification procedures"],
            "anchors": ["Section 8"],
            "query_keywords": ["breach consequences", "indemnification procedures"],
            "target_patterns": ["which clause explicitly connects"],
            "left_endpoint": "party",
            "right_endpoint": "clause",
            "bridge_relation": "explicit legal connection",
            "scope": "",
        }
    )

    lowered = {query.lower() for query in queries}
    assert "which clause defines party" not in lowered
    assert "which clause defines clause" not in lowered
    assert any("section 8" in query.lower() for query in queries)
    assert any("breach consequences" in query.lower() for query in queries)


def test_compile_requirement_queries_sanitizes_structured_artifacts_and_generic_wrappers() -> None:
    compile_requirement_queries = _load_backend_compile_requirement_queries()

    queries = compile_requirement_queries(
        {
            "route_type": "structural",
            "intent": "bridge_lookup",
            "entities": ["party (supplier)", "food safety audits", "insurance"],
            "terms": ["audit rights", "supplier obligations"],
            "anchors": ["clause{name or identifier referencing 'Section 504' or 'Privacy Regulations'}"],
            "query_keywords": ["food safety audits", "insurance", "supplier obligations"],
            "target_patterns": ["which clause explicitly connects"],
            "left_endpoint": "party (supplier)",
            "right_endpoint": "clause{name or identifier referencing 'Section 504' or 'Privacy Regulations'}",
            "bridge_relation": "audit rights",
            "scope": "",
        }
    )

    lowered = [query.lower() for query in queries]
    assert queries, "expected compiled structural queries"
    assert all("{" not in query and "}" not in query for query in queries)
    assert all("name or identifier" not in query.lower() for query in queries)
    assert any("supplier" in query.lower() for query in queries)
    assert any("section 504" in query.lower() for query in queries)
    assert any("privacy regulations" in query.lower() for query in queries)
    assert "party (supplier)" not in lowered[0]


def test_compile_requirement_queries_local_location_becomes_explicit_lookup() -> None:
    compile_requirement_queries = _load_backend_compile_requirement_queries()

    queries = compile_requirement_queries(
        {
            "route_type": "local",
            "intent": "fact_lookup",
            "entities": ["Go Call, Inc."],
            "terms": ["location"],
            "anchors": [],
            "query_keywords": ["Cambridge Ontario", "Go Call, Inc.", "location"],
            "target_patterns": [],
            "left_endpoint": "",
            "right_endpoint": "",
            "bridge_relation": "",
            "scope": "",
        }
    )

    assert queries
    assert queries[0].lower() == "what is go call, inc. address"
    assert any("which clause lists go call, inc. address" == query.lower() for query in queries)
    assert any("cambridge ontario" in query.lower() for query in queries)


def test_grounding_validator_rejects_unsupported_generated_definition_claim() -> None:
    validate_sub_answer_grounding = _load_backend_grounding_validator()

    result = validate_sub_answer_grounding(
        sub_question_text="What types of Expenses incurred by Metavante are defined in the agreement?",
        parsed_sub_answer={
            "sub_answer": (
                "Expenses are defined as any and all reasonable and direct expenses paid by "
                "Metavante in connection with services provided to or on behalf of the Customer."
            ),
            "reason": "The agreement defines reimbursable Metavante expenses and ties them to third-party reimbursement.",
            "confidence": 0.88,
        },
        support_pairs=[
            {
                "chunk_id": "neoforma-deductible",
                "doc_id": "Neoforma#p0",
                "text": "DEDUCTIBLES. Neoforma shall be responsible for all deductibles under the insurance policies.",
            },
            {
                "chunk_id": "neoforma-revenue",
                "doc_id": "Neoforma#p0",
                "text": "Neoforma Site means the co-branded site used for product listing and net advertising revenue.",
            },
        ],
        retrieval_requirement={
            "intent": "definition_lookup",
            "semantic_need": "definition",
        },
    )

    assert result["should_invalidate"] is True
    assert result["validated_sub_answer"]["sub_answer"] == "NOT_FOUND"
    assert result["diagnostics"]["validation_mode"] == "unsupported_generated_answer"
    assert result["diagnostics"]["grounded_support_chunk_ids"] == []
    assert "metavante" in result["diagnostics"]["unmatched_answer_terms"]


def test_grounding_validator_keeps_supported_definition_claim() -> None:
    validate_sub_answer_grounding = _load_backend_grounding_validator()

    result = validate_sub_answer_grounding(
        sub_question_text="What types of Expenses incurred by Metavante are defined in the agreement?",
        parsed_sub_answer={
            "sub_answer": (
                "Expenses are defined as reasonable and direct expenses paid by Metavante "
                "that are properly reimbursable from a third party."
            ),
            "reason": "The clause states that Expenses shall mean reasonable and direct expenses paid by Metavante and reimbursable from a third party.",
            "confidence": 0.83,
        },
        support_pairs=[
            {
                "chunk_id": "ofg-expenses",
                "doc_id": "OFGBANCORP#p0",
                "text": (
                    "Expenses shall mean any and all reasonable and direct expenses paid by Metavante, "
                    "or for which Metavante becomes liable, that are properly reimbursable by Metavante from a third party."
                ),
            }
        ],
        retrieval_requirement={
            "intent": "definition_lookup",
            "semantic_need": "definition",
        },
    )

    assert result["should_invalidate"] is False
    assert result["validated_sub_answer"]["sub_answer"].startswith("Expenses are defined as")
    assert result["diagnostics"]["validation_mode"] in {"pass_through", "grounded"}
    assert result["diagnostics"]["answer_token_coverage"] > 0.5
    assert result["diagnostics"]["grounded_support_doc_ids"] == ["OFGBANCORP#p0"]


def test_grounding_validator_attributes_mixed_support_to_grounding_chunk_only() -> None:
    validate_sub_answer_grounding = _load_backend_grounding_validator()

    result = validate_sub_answer_grounding(
        sub_question_text="What types of Expenses incurred by Metavante are defined in the agreement?",
        parsed_sub_answer={
            "sub_answer": (
                "Expenses are defined as reasonable and direct expenses paid by Metavante "
                "that are properly reimbursable from a third party."
            ),
            "reason": "The definition states that Metavante's reimbursable expenses are third-party charges connected to services.",
            "confidence": 0.81,
        },
        support_pairs=[
            {
                "chunk_id": "neoforma-deductible",
                "doc_id": "Neoforma#p0",
                "text": "DEDUCTIBLES. Neoforma shall be responsible for all deductibles under the insurance policies.",
            },
            {
                "chunk_id": "ofg-expenses",
                "doc_id": "OFGBANCORP#p0",
                "text": (
                    "Expenses shall mean any and all reasonable and direct expenses paid by Metavante, "
                    "or for which Metavante becomes liable, that are properly reimbursable by Metavante from a third party."
                ),
            },
        ],
        retrieval_requirement={
            "intent": "definition_lookup",
            "semantic_need": "definition",
        },
    )

    assert result["should_invalidate"] is False
    assert result["diagnostics"]["grounded_support_chunk_ids"] == ["ofg-expenses"]
    assert result["diagnostics"]["grounded_support_doc_ids"] == ["OFGBANCORP#p0"]


def test_normalize_ircot_followup_queries_turns_instructional_text_into_short_anchor_queries() -> None:
    namespace = _load_backend_compile_namespace()
    normalize_ircot_followup_queries = namespace["_normalize_ircot_followup_queries"]

    queries = normalize_ircot_followup_queries(
        'Please retrieve the full text of Section 504, specifically titled "Privacy Regulations," and the full text of Section 10.5.2, which provides the definition for "Sensitive Customer Information," from the contract document "NeoformaInc_19991202_S-1A_EX-10.26_5224521_EX-10.26_Co-Branding Agreement". Ensure the retrieval focuses on the precise sections and their content.',
        route_type="structural",
        original_sub_question="What is the significance of Section 504 in relation to Sensitive Customer Information?",
        max_items=3,
    )

    assert queries
    assert queries[0].lower() == "section 504 privacy regulations"
    assert "section 10.5.2 sensitive customer information" in [query.lower() for query in queries]
    assert all("please retrieve" not in query.lower() for query in queries)
    assert all("ensure the retrieval focuses" not in query.lower() for query in queries)
    assert all("neoformainc_19991202" not in query.lower() for query in queries)


def test_normalize_ircot_followup_queries_strips_provide_imperative_and_splits_section_anchors() -> None:
    namespace = _load_backend_compile_namespace()
    normalize_ircot_followup_queries = namespace["_normalize_ircot_followup_queries"]

    queries = normalize_ircot_followup_queries(
        "** Provide Section 10.2 (Indemnification) and Section 10.3 (Indemnification Procedures) of the OFGBANCORP_03_28_2007-EX-10.23-OUTSOURCING AGREEMENT.",
        route_type="structural",
        original_sub_question="whats the impact of breaching section 8 if customer fails to perform under section 10.3 regarding indemnification obligations?",
        max_items=3,
    )

    lowered = [query.lower() for query in queries]
    assert queries
    assert all("provide section" not in query.lower() for query in queries)


def test_normalize_ircot_followup_queries_can_preserve_natural_question_for_subquestion_ircot() -> None:
    namespace = _load_backend_compile_namespace()
    normalize_ircot_followup_queries = namespace["_normalize_ircot_followup_queries"]

    queries = normalize_ircot_followup_queries(
        "What does Section 10.5.2 describe about the co-branded training and education center?",
        route_type="structural",
        original_sub_question="What business context does Section 10.5.2 describe in the agreement?",
        max_items=3,
        preserve_natural_question=True,
    )

    assert queries
    assert queries[0] == "What does Section 10.5.2 describe about the co-branded training and education center?"
    assert any("section 10.5.2" in query.lower() for query in queries[1:])
    assert all("ofgbancorp_03_28_2007" not in query.lower() for query in queries)


def test_all_followup_queries_are_redundant_detects_same_subquestion_query() -> None:
    namespace = _load_backend_compile_namespace()
    redundant = namespace["_all_followup_queries_are_redundant"]

    assert redundant(
        ["What clause, if any, states whether right:audit right and third-party are connected?"],
        current_query="What clause, if any, states whether right:audit right and third-party are connected?",
        original_sub_question="What clause, if any, states whether right:audit right and third-party are connected?",
    )


def test_all_followup_queries_are_redundant_detects_equivalent_query_after_normalization() -> None:
    namespace = _load_backend_compile_namespace()
    redundant = namespace["_all_followup_queries_are_redundant"]

    assert redundant(
        ["what clause if any states whether right audit right and third-party are connected"],
        current_query="What clause, if any, states whether right:audit right and third-party are connected?",
        original_sub_question="What clause, if any, states whether right:audit right and third-party are connected?",
    )


def test_build_structural_evidence_summary_surfaces_bridge_fields_and_cleans_metadata() -> None:
    helpers = _extract_backend_functions(
        "_normalize_prompt_text",
        "_clean_prompt_field",
        "_humanize_relation_label",
        "_trim_prompt_excerpt",
        "_extract_structural_focus",
        "_build_structural_evidence_summary",
    )

    summary = helpers["_build_structural_evidence_summary"](
        [
            {
                "sub_question_id": "sq_1",
                "route_type": "structural",
                "sub_answer": "Motorola is connected through trademark licensing.",
                "retrieval_requirement": {
                    "left_endpoint": "Motorola",
                    "bridge_relation": "grants_license_to",
                    "right_endpoint": "PageMaster promotion",
                    "anchors": ["Section 8"],
                },
                "support_spans": [
                    {
                        "chunk_id": "c1",
                        "text": "[META] source: cuad\nPageMaster Corporation warrants and represents that it has a license to advertise and use the trademarks, logos, etc. of Motorola, Inc. for the promotion.",
                    }
                ],
            }
        ]
    )

    assert "left_endpoint=Motorola" in summary
    assert "bridge_relation=grants license to" in summary
    assert "right_endpoint=PageMaster promotion" in summary
    assert "anchor=Section 8" in summary
    assert "[META]" not in summary
    assert "doc_id:" not in summary


def test_build_subquestion_triple_evidence_compacts_metadata_without_touching_retrieval_outputs() -> None:
    helpers = _extract_backend_functions(
        "_normalize_prompt_text",
        "_clean_prompt_field",
        "_humanize_relation_label",
        "_trim_prompt_excerpt",
        "_split_top_level_csv",
        "_compact_relation_line",
        "_build_subquestion_triple_evidence",
    )

    lines = helpers["_build_subquestion_triple_evidence"](
        [
            "(GO CALL, INC. [doc_id: contract, schema_type: party, mention_count: 14], has_attribute, address: 15 Queen Street East [key: address, chunk ids: abc]) [score: 0.70]",
            "([Unknown Node: PAGEMASTER CORPORATION] , notice_to, [Unknown Node: Go Call] ) [score: 0.52]",
        ],
        max_items=4,
    )

    assert lines == ["GO CALL, INC. -- has attribute -> address: 15 Queen Street East"]
    assert all("doc_id:" not in line for line in lines)
    assert all("schema_type:" not in line for line in lines)
    assert all("[score:" not in line for line in lines)


def test_build_final_answer_knowledge_package_replaces_global_triple_dump() -> None:
    helpers = _extract_backend_functions(
        "_normalize_prompt_text",
        "_clean_prompt_field",
        "_humanize_relation_label",
        "_trim_prompt_excerpt",
        "_extract_structural_focus",
        "_build_structural_evidence_summary",
        "_split_top_level_csv",
        "_compact_relation_line",
        "_build_supporting_relations",
        "_build_findings_block",
        "_format_refactor_reasoning_plan",
        "_build_final_answer_knowledge_package",
    )

    package = helpers["_build_final_answer_knowledge_package"](
        question="What is Motorola's role in the PageMaster promotion?",
        subquestion_context="[Sub-question 1] route=structural",
        sub_question_answers=[
            {
                "sub_question_id": "sq_1",
                "route_type": "structural",
                "query_text": "Motorola role in promotion",
                "sub_answer": "Motorola is referenced through trademark licensing.",
                "reason": "Clause 8 says PageMaster has a license to use Motorola trademarks for the promotion.",
                "support_chunk_ids": ["chunk_1"],
                "retrieval_requirement": {
                    "left_endpoint": "Motorola",
                    "bridge_relation": "grants_license_to",
                    "right_endpoint": "PageMaster promotion",
                    "anchors": ["Section 8"],
                },
                "support_spans": [
                    {
                        "chunk_id": "chunk_1",
                        "text": "[META] source: cuad\nPageMaster Corporation warrants and represents that it has a license to advertise and use the trademarks, logos, etc. of Motorola, Inc.",
                    }
                ],
            }
        ],
        backing_chunk_contents=["Chunk A"],
        triples=[
            "(PAGEMASTER CORPORATION [doc_id: contract, schema_type: party], grants_license_to, Motorola, Inc. [doc_id: contract]) [score: 0.91]"
        ],
        fallback_chunk_contents=["Fallback Chunk"],
    )

    assert "=== Structural Evidence Summary ===" in package
    assert "left_endpoint=Motorola" in package
    assert "=== Global Triple Pool ===" not in package
    assert "doc_id:" not in package


def test_infer_answer_scope_target_doc_id_from_sub_questions_prefers_claim_bearing_contract() -> None:
    helpers = _extract_backend_functions(
        "_strip_question_prompt_prefix",
        "_is_generic_scope_phrase",
        "_is_low_value_subject_hint",
        "_extract_question_subject_overrides",
        "_infer_answer_scope_target_doc_id_from_sub_questions",
    )

    answer_scope_target_doc_id = helpers["_infer_answer_scope_target_doc_id_from_sub_questions"](
        [
            {
                "route_type": "local",
                "retrieval_requirement": {
                    "route_type": "local",
                    "entities": ["Metavante"],
                    "terms": ["reimbursable expenses"],
                    "query_keywords": ["reimbursable from third parties"],
                },
            },
            {
                "route_type": "local",
                "retrieval_requirement": {
                    "route_type": "local",
                    "entities": ["Neoforma"],
                    "terms": ["indemnification obligations", "third-party claims"],
                    "query_keywords": ["indemnify", "claim"],
                },
            },
        ],
        fallback_target_doc_id="",
        normalized_entity_doc_ids={
            "metavante": {"OFGBANCORP#p0"},
            "neoforma": {"Neoforma#p0"},
        },
        normalize_doc_hint=lambda text: re.sub(r"[^a-z0-9]+", "", str(text or "").lower()),
    )

    assert answer_scope_target_doc_id == "Neoforma#p0"


def test_infer_answer_scope_target_doc_id_prefers_question_claim_anchor_over_customer_noise() -> None:
    helpers = _extract_backend_functions(
        "_strip_question_prompt_prefix",
        "_is_generic_scope_phrase",
        "_is_low_value_subject_hint",
        "_extract_question_subject_overrides",
        "_infer_answer_scope_target_doc_id_from_sub_questions",
    )

    answer_scope_target_doc_id = helpers["_infer_answer_scope_target_doc_id_from_sub_questions"](
        [
            {
                "route_type": "local",
                "retrieval_requirement": {
                    "route_type": "local",
                    "entities": ["Metavante"],
                    "terms": ["reimbursable from third parties", "expenses"],
                    "query_keywords": ["reimbursable expenses"],
                },
            },
            {
                "route_type": "local",
                "retrieval_requirement": {
                    "route_type": "local",
                    "entities": ["Customer"],
                    "terms": ["third-party claims", "indemnification obligations"],
                    "query_keywords": ["indemnification obligations", "third-party claims"],
                },
            },
            {
                "route_type": "local",
                "retrieval_requirement": {
                    "route_type": "local",
                    "entities": ["Neoforma", "indemnification", "third-party claims"],
                    "terms": ["indemnification obligations", "third-party claims"],
                    "query_keywords": ["Neoforma", "indemnification obligations"],
                },
            },
        ],
        question=(
            "Under the agreement, what types of expenses incurred by Metavante in connection with services "
            "provided to or on behalf of the Customer are considered reimbursable from third parties, and "
            "how does this relate to Neoforma's indemnification obligations regarding third-party claims?"
        ),
        fallback_target_doc_id="",
        normalized_entity_doc_ids={
            "metavante": {"OFGBANCORP#p0"},
            "customer": {"OFGBANCORP#p0"},
            "neoforma": {"Neoforma#p0"},
        },
        normalize_doc_hint=lambda text: re.sub(r"[^a-z0-9]+", "", str(text or "").lower()),
    )

    assert answer_scope_target_doc_id == "Neoforma#p0"


def test_infer_answer_composition_mode_enables_cross_document_bridge_for_0041_shape() -> None:
    helpers = _extract_backend_functions(
        "_strip_question_prompt_prefix",
        "_infer_answer_composition_mode",
    )

    mode = helpers["_infer_answer_composition_mode"](
        question=(
            "Under the agreement, what types of expenses incurred by Metavante in connection with services "
            "provided to or on behalf of the Customer are considered reimbursable from third parties, and "
            "how does this relate to Neoforma's indemnification obligations regarding third-party claims?"
        ),
        sub_question_answers=[
            {
                "retrieval_requirement": {"intent": "bridge_lookup"},
                "evidence_scope_label": "external_related",
                "support_doc_ids": ["OFGBANCORP#p0"],
            },
            {
                "retrieval_requirement": {"intent": "bridge_lookup"},
                "evidence_scope_label": "target_contract",
                "support_doc_ids": ["Neoforma#p0"],
            },
        ],
        answer_scope_target_doc_id="Neoforma#p0",
    )

    assert mode == "cross_document_bridge"


def test_infer_answer_composition_mode_treats_context_lookup_as_bridge_signal() -> None:
    helpers = _extract_backend_functions(
        "_strip_question_prompt_prefix",
        "_infer_answer_composition_mode",
    )

    mode = helpers["_infer_answer_composition_mode"](
        question=(
            "What is the significance of Section 504 in relation to the definition of Privacy Regulations "
            "and how does it connect to the concept of Sensitive Customer Information as defined under Section 10.5.2?"
        ),
        sub_question_answers=[
            {
                "retrieval_requirement": {"intent": "definition_lookup"},
                "evidence_scope_label": "target_contract",
                "support_doc_ids": ["OFGBANCORP#p0"],
            },
            {
                "retrieval_requirement": {"intent": "context_lookup"},
                "evidence_scope_label": "external_related",
                "support_doc_ids": ["Neoforma#p0"],
            },
        ],
        answer_scope_target_doc_id="OFGBANCORP#p0",
    )

    assert mode == "cross_document_bridge"


def test_build_answer_scope_prompt_rules_allows_cross_document_bridge_synthesis() -> None:
    helpers = _extract_backend_functions(
        "_build_answer_scope_prompt_rules",
    )

    rules = helpers["_build_answer_scope_prompt_rules"](
        "Neoforma#p0",
        answer_composition_mode="cross_document_bridge",
    )

    assert "primary contract" in rules.lower()
    assert "cross-document bridge questions" in rules.lower()
    assert "attribute each material claim to its source agreement" in rules.lower()
    assert "semantic alignment" in rules.lower()


def test_question_response_exposes_top_level_scope_and_alignment_fields() -> None:
    tree = _load_backend_ast()

    question_response_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "QuestionResponse"
    )
    field_names = [
        node.target.id
        for node in question_response_class.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    ]

    assert "answer_scope_target_doc_id" in field_names
    assert "answer_composition_mode" in field_names
    assert "semantic_alignment" in field_names

    response_calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "QuestionResponse"
    ]
    assert response_calls, "QuestionResponse constructor call not found"
    keyword_names = [kw.arg for kw in response_calls[-1].keywords if kw.arg]

    assert "answer_scope_target_doc_id" in keyword_names
    assert "answer_composition_mode" in keyword_names
    assert "semantic_alignment" in keyword_names


def test_infer_semantic_alignment_recognizes_shared_third_party_cost_concept() -> None:
    helpers = _extract_backend_functions(
        "_strip_question_prompt_prefix",
        "_infer_semantic_alignment",
    )

    alignment = helpers["_infer_semantic_alignment"](
        question=(
            "Under the agreement, what types of expenses incurred by Metavante in connection with services "
            "provided to or on behalf of the Customer are considered reimbursable from third parties, and "
            "how does this relate to Neoforma's indemnification obligations regarding third-party claims?"
        ),
        sub_question_answers=[
            {
                "sub_question_id": "sq_1",
                "query_text": "Metavante Expenses definition",
                "sub_answer": (
                    "Expenses are defined as reasonable and direct expenses paid by Metavante to Third Parties, "
                    "including postage, supplies, materials, travel and lodging, and telecommunication fees."
                ),
                "reason": "The clause defines Expenses with an itemized list of third-party costs.",
                "evidence_scope_label": "external_related",
                "support_doc_ids": ["OFGBANCORP#p0"],
            },
            {
                "sub_question_id": "sq_2",
                "query_text": "Neoforma indemnification obligations",
                "sub_answer": (
                    "Neoforma must indemnify and hold harmless VerticalNet against third-party claims, losses, "
                    "damages, liabilities, costs, and expenses arising from specified misconduct."
                ),
                "reason": "Section 15.4 covers losses, claims, damages, liabilities, costs, and expenses.",
                "evidence_scope_label": "target_contract",
                "support_doc_ids": ["Neoforma#p0"],
            },
        ],
        answer_composition_mode="cross_document_bridge",
        backing_chunk_contents=[
            "Neoforma shall indemnify and hold harmless VerticalNet against any and all losses, claims, damages, liabilities, costs, expenses and disbursements asserted by a third party."
        ],
        external_related_chunk_contents=[
            "\"Expenses\" shall mean any and all reasonable and direct expenses paid by Metavante to Third Parties, including postage, supplies, materials, travel and lodging, and telecommunication fees."
        ],
    )

    assert alignment["alignment_type"] == "conceptual_overlap"
    assert alignment["explicit_link_present"] is False
    assert alignment["shared_concepts"] == ["third-party", "costs/expenses"]
    assert "defines the categories of third-party expenses" in alignment["left_role"]
    assert "third-party claims" in alignment["right_role"]
    assert "do not expressly cross-reference each other" in alignment["safe_summary"]


def test_infer_semantic_alignment_allows_privacy_definition_and_business_context_synthesis() -> None:
    helpers = _extract_backend_functions(
        "_strip_question_prompt_prefix",
        "_infer_semantic_alignment",
    )

    alignment = helpers["_infer_semantic_alignment"](
        question=(
            "What is the significance of Section 504 in relation to the definition of Privacy Regulations "
            "and how does it connect to the concept of Sensitive Customer Information as defined under Section 10.5.2?"
        ),
        sub_question_answers=[
            {
                "sub_question_id": "sq_1",
                "query_text": "How is Sensitive Customer Information defined in the agreement corpus?",
                "sub_answer": (
                    "Sensitive Customer Information means customer data tied to identifiers such as Social Security numbers, "
                    "account numbers, card numbers, and passwords that permit access to an account."
                ),
                "reason": "The clause defines categories of sensitive customer data subject to privacy protection.",
                "evidence_scope_label": "target_contract",
                "support_doc_ids": ["OFGBANCORP#p0"],
                "retrieval_requirement": {"intent": "definition_lookup"},
            },
            {
                "sub_question_id": "sq_2",
                "query_text": "What business context does Section 10.5.2 describe in the agreement?",
                "sub_answer": (
                    "Section 10.5.2 describes the Co-Branded Training and Education Center and the Training and Education Gross Margin, "
                    "including listing fees and e-commerce revenue derived from that center."
                ),
                "reason": "The clause provides the business and revenue context rather than a privacy definition.",
                "evidence_scope_label": "external_related",
                "support_doc_ids": ["Neoforma#p0"],
                "retrieval_requirement": {"intent": "context_lookup"},
            },
        ],
        answer_composition_mode="cross_document_bridge",
        backing_chunk_contents=[
            "\"Sensitive Customer Information\" shall mean Customer Data with respect to a Consumer tied to Social Security numbers, account numbers, card numbers, or passwords."
        ],
        external_related_chunk_contents=[
            "Section 10.5.2 Co-Branded Training and Education Center. Training and Education Gross Margin shall mean listing fees and e-commerce revenue derived during the Term from users of the Co-Branded Training and Education Center."
        ],
    )

    assert alignment["alignment_type"] == "conceptual_overlap"
    assert "privacy-sensitive-data" in alignment["shared_concepts"]
    assert "sensitive customer data" in alignment["left_role"]
    assert "operational or business context" in alignment["right_role"]


def test_classify_sub_question_evidence_scope_marks_external_related_when_support_doc_differs_from_answer_scope() -> None:
    helpers = _extract_backend_functions(
        "_classify_sub_question_evidence_scope",
    )

    meta = helpers["_classify_sub_question_evidence_scope"](
        {
            "target_doc_id": "OFGBANCORP#p0",
            "scope_plan": {"primary_doc_id": "OFGBANCORP#p0"},
            "support_chunk_ids": ["c1"],
            "retrieved_chunk_ids_all": ["c1", "c2"],
            "first_stage_chunk_ids": ["c1", "c2"],
        },
        answer_scope_target_doc_id="Neoforma#p0",
        chunk_id_to_doc_id={
            "c1": "OFGBANCORP#p0",
            "c2": "OFGBANCORP#p0",
        },
    )

    assert meta["evidence_scope_label"] == "external_related"
    assert meta["answer_scope_target_doc_id"] == "Neoforma#p0"
    assert meta["support_doc_ids"] == ["OFGBANCORP#p0"]
    assert meta["retrieval_doc_ids"] == ["OFGBANCORP#p0"]


def test_classify_sub_question_evidence_scope_prefers_support_chunk_docs_over_first_stage_noise() -> None:
    helpers = _extract_backend_functions(
        "_classify_sub_question_evidence_scope",
    )

    meta = helpers["_classify_sub_question_evidence_scope"](
        {
            "target_doc_id": "Neoforma#p0",
            "scope_plan": {"primary_doc_id": "Neoforma#p0"},
            "support_chunk_ids": ["neo-1"],
            "retrieved_chunk_ids_all": ["neo-1", "ofg-1"],
            "first_stage_chunk_ids": ["neo-1", "ofg-1"],
        },
        answer_scope_target_doc_id="Neoforma#p0",
        chunk_id_to_doc_id={
            "neo-1": "Neoforma#p0",
            "ofg-1": "OFGBANCORP#p0",
        },
    )

    assert meta["evidence_scope_label"] == "target_contract"
    assert meta["support_doc_ids"] == ["Neoforma#p0"]
    assert meta["retrieval_doc_ids"] == ["Neoforma#p0", "OFGBANCORP#p0"]


def test_partition_final_chunk_context_by_ownership_keeps_target_primary_and_external_separate() -> None:
    helpers = _extract_backend_functions(
        "_collect_ranked_chunk_ids_from_answer_item",
        "_partition_final_chunk_context_by_ownership",
    )

    partition = helpers["_partition_final_chunk_context_by_ownership"](
        ["ofg-1", "neo-1", "ofg-2", "neo-2"],
        sub_question_answers=[
            {
                "evidence_scope_label": "external_related",
                "support_chunk_ids": ["ofg-1"],
                "retrieved_chunk_ids_all": ["ofg-1", "ofg-2"],
            },
            {
                "evidence_scope_label": "target_contract",
                "support_chunk_ids": ["neo-1"],
                "retrieved_chunk_ids_all": ["neo-1", "neo-2"],
            },
        ],
        chunk_id_to_doc_id={
            "ofg-1": "OFGBANCORP#p0",
            "ofg-2": "OFGBANCORP#p0",
            "neo-1": "Neoforma#p0",
            "neo-2": "Neoforma#p0",
        },
        answer_scope_target_doc_id="Neoforma#p0",
        total_limit=3,
        external_limit=2,
    )

    assert partition["primary_chunk_ids"] == ["neo-1", "neo-2"]
    assert partition["external_related_chunk_ids"] == ["ofg-1", "ofg-2"]


def test_build_final_answer_knowledge_package_separates_target_and_external_findings() -> None:
    helpers = _extract_backend_functions(
        "_normalize_prompt_text",
        "_clean_prompt_field",
        "_humanize_relation_label",
        "_trim_prompt_excerpt",
        "_extract_structural_focus",
        "_build_structural_evidence_summary",
        "_split_top_level_csv",
        "_compact_relation_line",
        "_build_supporting_relations",
        "_build_findings_block",
        "_format_refactor_reasoning_plan",
        "_build_final_answer_knowledge_package",
    )

    package = helpers["_build_final_answer_knowledge_package"](
        question="How does Metavante reimbursement relate to Neoforma indemnification?",
        subquestion_context="[Sub-question 1] route=local\n[Sub-question 2] route=local",
        sub_question_answers=[
            {
                "sub_question_id": "sq_1",
                "route_type": "local",
                "query_text": "Metavante expenses",
                "sub_answer": "Metavante expenses are reimbursable by Customer.",
                "reason": "OFG defines Expenses and assigns them to Customer.",
                "support_chunk_ids": ["c1"],
                "support_doc_ids": ["OFGBANCORP#p0"],
                "evidence_scope_label": "external_related",
                "evidence_scope_reason": "Different contract from the target answer scope.",
                "retrieval_requirement": {},
            },
            {
                "sub_question_id": "sq_2",
                "route_type": "local",
                "query_text": "Neoforma indemnification",
                "sub_answer": "Neoforma indemnifies VerticalNet against third-party claims.",
                "reason": "Section 15.4 states the indemnification obligation.",
                "support_chunk_ids": ["c2"],
                "support_doc_ids": ["Neoforma#p0"],
                "evidence_scope_label": "target_contract",
                "evidence_scope_reason": "Aligned with the target contract.",
                "retrieval_requirement": {},
            },
        ],
        backing_chunk_contents=["Chunk A", "Chunk B"],
        external_related_chunk_contents=["External Chunk"],
        triples=[],
        fallback_chunk_contents=[],
        answer_scope_target_doc_id="Neoforma#p0",
    )

    assert "=== Answer Scope Contract ===\nNeoforma#p0" in package
    assert "=== Target Contract Findings ===" in package
    assert "=== External Related Findings (Do not treat these as target-contract facts) ===" in package
    assert "=== External Related Chunks (Context only; not target-contract support) ===" in package


def test_build_final_answer_knowledge_package_bridge_mode_promotes_cross_document_synthesis() -> None:
    helpers = _extract_backend_functions(
        "_normalize_prompt_text",
        "_clean_prompt_field",
        "_humanize_relation_label",
        "_trim_prompt_excerpt",
        "_extract_structural_focus",
        "_build_structural_evidence_summary",
        "_split_top_level_csv",
        "_compact_relation_line",
        "_build_supporting_relations",
        "_build_findings_block",
        "_format_refactor_reasoning_plan",
        "_build_final_answer_knowledge_package",
    )

    package = helpers["_build_final_answer_knowledge_package"](
        question="How do Metavante expenses relate to Neoforma indemnification obligations?",
        subquestion_context="[Sub-question 1] route=local\n[Sub-question 2] route=local",
        sub_question_answers=[
            {
                "sub_question_id": "sq_1",
                "route_type": "local",
                "query_text": "Metavante expenses",
                "sub_answer": "NOT_FOUND",
                "reason": "The evidence defines Expenses as costs paid by Metavante to third parties.",
                "support_chunk_ids": ["c1"],
                "support_doc_ids": ["OFGBANCORP#p0"],
                "evidence_scope_label": "external_related",
                "evidence_scope_reason": "Related agreement evidence for a different component.",
                "retrieval_requirement": {"intent": "bridge_lookup"},
            },
            {
                "sub_question_id": "sq_2",
                "route_type": "local",
                "query_text": "Neoforma indemnification",
                "sub_answer": "Neoforma indemnifies VerticalNet against third-party claims.",
                "reason": "Section 15.4 states the indemnification obligation.",
                "support_chunk_ids": ["c2"],
                "support_doc_ids": ["Neoforma#p0"],
                "evidence_scope_label": "target_contract",
                "evidence_scope_reason": "Aligned with the primary contract.",
                "retrieval_requirement": {"intent": "bridge_lookup"},
            },
        ],
        backing_chunk_contents=["Neo Chunk"],
        external_related_chunk_contents=["OFG Chunk"],
        triples=[],
        fallback_chunk_contents=[],
        answer_scope_target_doc_id="Neoforma#p0",
        answer_composition_mode="cross_document_bridge",
        semantic_alignment={
            "alignment_type": "conceptual_overlap",
            "confidence": 0.88,
            "shared_concepts": ["third-party", "costs/expenses"],
            "left_role": "defines the categories of third-party expenses and the types of reimbursable costs",
            "right_role": "defines the scope of responsibility for losses, claims, costs, and expenses arising from third-party claims",
            "safe_summary": "Although the agreements do not expressly cross-reference each other, they align on the concept of third-party and costs/expenses.",
        },
    )

    assert "=== Primary Answer Contract ===\nNeoforma#p0" in package
    assert "=== Answer Composition Mode ===" in package
    assert "cross_document_bridge" in package
    assert "=== Primary Contract Findings ===" in package
    assert "=== Cross-Document Related Findings (Use with source attribution) ===" in package
    assert "=== Cross-Document Evidence Chunks (Use with source attribution; not sole primary-contract proof) ===" in package
    assert "=== Semantic Alignment ===" in package
    assert "shared_concepts=third-party, costs/expenses" in package


def test_prioritize_retrieval_queries_prefers_clean_local_style_queries() -> None:
    helpers = _extract_backend_functions(
        "_score_retrieval_query_specificity",
        "_prioritize_retrieval_queries",
    )

    ranked = helpers["_prioritize_retrieval_queries"](
        [
            "specifically the supplier party obligation, insurance, audit reports, certs maintains_insurance, confidentiality_applies_to, audit_right_over, survives_for supplier food safety compliance third-party audit regulatory",
            "Section 2.2 supplier food safety audits",
            "supplier insurance records audit rights",
        ],
        route_type="structural",
        limit=2,
    )

    assert ranked == [
        "Section 2.2 supplier food safety audits",
        "supplier insurance records audit rights",
    ]
    noisy_score = helpers["_score_retrieval_query_specificity"](
        "supplier party obligation maintains_insurance audit_right_over",
        route_type="structural",
    )
    clean_score = helpers["_score_retrieval_query_specificity"](
        "Section 2.2 supplier food safety audits",
        route_type="structural",
    )
    assert clean_score > noisy_score


def test_rerank_support_pairs_prefers_clause_excerpt_over_entity_dump() -> None:
    helpers = _extract_backend_functions(
        "_normalize_prompt_text",
        "_clean_prompt_field",
        "_score_retrieval_query_specificity",
        "_prioritize_retrieval_queries",
        "_collect_retrieval_focus_terms",
        "_rerank_support_pairs_for_subquestion",
        "_collect_structural_anchor_terms",
        "_strong_rerank_support_pairs_for_subquestion",
    )

    ranked = helpers["_rerank_support_pairs_for_subquestion"](
        [
            {
                "chunk_id": "noise",
                "text": "=== Entity Information ===\n• Supplier\n• Confidentiality\n• audit reports\n• insurance records\n• regulatory requirements",
            },
            {
                "chunk_id": "gold",
                "text": "[META] source: cuad\n2.2 Supplier will conduct third-party food safety audits and submit summaries of audit reports to PNC upon request.",
            },
            {
                "chunk_id": "secondary",
                "text": "[META] source: cuad\nSupplier shall maintain insurance records and make them available for audit upon request.",
            },
        ],
        query_text="specifically the supplier party obligation, insurance, audit reports, certs maintains_insurance, confidentiality_applies_to",
        retrieval_queries=[
            "Section 2.2 supplier food safety audits",
            "supplier insurance records audit rights",
        ],
        retrieval_requirement={
            "anchors": ["Section 2.2"],
            "terms": ["food safety audits", "audit reports", "insurance records"],
            "left_endpoint": "Supplier",
        },
    )

    assert [item["chunk_id"] for item in ranked[:2]] == ["gold", "secondary"]


def test_compile_requirement_queries_narrows_specific_section_queries() -> None:
    compile_requirement_queries = _load_backend_compile_requirement_queries()

    queries = compile_requirement_queries(
        {
            "route_type": "structural",
            "intent": "specific_bridge_lookup",
            "anchors": ["Section 8", "Section 10.3"],
            "terms": ["termination fee", "indemnification procedures"],
            "entities": ["Customer", "Metavante"],
            "query_keywords": [],
            "target_patterns": [],
            "left_endpoint": "Section 8",
            "right_endpoint": "Section 10.3",
            "bridge_relation": "indemnification obligations",
        }
    )

    lowered = [query.lower() for query in queries]
    assert any("section 8" == query.lower() for query in queries)
    assert any("section 10.3" == query.lower() for query in queries)
    assert any("which clause connects section 8 and section 10.3" == query.lower() for query in queries)
    assert "section 8 section 10.3" in lowered


def test_compile_requirement_queries_adds_soft_contract_binding_without_dropping_soft_queries() -> None:
    namespace = _load_backend_compile_namespace()
    namespace["question"] = (
        'Contract title: "OFGBANCORP_03_28_2007-EX-10.23-OUTSOURCING AGREEMENT"\n'
        "Question: whats the impact of breaching section 8 if customer fails to perform under section 10.3 regarding indemnification obligations?"
    )
    compile_requirement_queries = namespace["_compile_requirement_queries"]

    queries = compile_requirement_queries(
        {
            "route_type": "structural",
            "intent": "specific_bridge_lookup",
            "anchors": ["Section 8", "Section 10.3"],
            "terms": ["termination fee", "indemnification procedures"],
            "entities": ["Customer", "Metavante"],
            "query_keywords": [],
            "target_patterns": [],
            "left_endpoint": "Section 8",
            "right_endpoint": "Section 10.3",
            "bridge_relation": "indemnification obligations",
        }
    )

    lowered = [query.lower() for query in queries]
    assert any("outsourcing agreement" in query.lower() and "section 8" in query.lower() for query in queries)
    assert "section 8" in lowered
    assert "section 10.3" in lowered
    assert any("which clause connects section 8 and section 10.3" == query.lower() for query in queries)


def test_compile_requirement_queries_keeps_exact_and_fallback_structural_queries() -> None:
    compile_requirement_queries = _load_backend_compile_requirement_queries()

    queries = compile_requirement_queries(
        {
            "route_type": "structural",
            "intent": "impact_lookup",
            "anchors": ["Section 8", "Section 10.3"],
            "terms": ["reimbursable expenses", "third-party claims"],
            "entities": ["Metavante", "Neoforma"],
            "query_keywords": [],
            "target_patterns": [],
            "left_endpoint": "Section 8",
            "right_endpoint": "Section 10.3",
            "bridge_relation": "indemnification obligations",
        }
    )

    lowered = [query.lower() for query in queries]
    assert any(query.lower() == "section 8" for query in queries)
    assert any("which clause connects section 8 and section 10.3" == query.lower() for query in queries)
    assert any(query.lower() == "section 8 section 10.3" for query in queries)


def test_compile_requirement_queries_preserves_specific_relation_terms_for_multi_hop() -> None:
    namespace = _load_backend_compile_namespace()
    namespace["question"] = (
        "Under the agreement, what types of expenses incurred by Metavante in connection with services provided "
        "to or on behalf of the Customer are considered reimbursable from third parties, and how does this "
        "relate to Neoforma's indemnification obligations regarding third-party claims?"
    )
    compile_requirement_queries = namespace["_compile_requirement_queries"]

    queries = compile_requirement_queries(
        {
            "route_type": "structural",
            "intent": "specific_bridge_lookup",
            "entities": ["Metavante", "Neoforma"],
            "terms": ["reimbursable expenses", "third-party claims"],
            "query_keywords": ["third-party reimbursement"],
            "target_patterns": [],
            "left_endpoint": "Metavante",
            "right_endpoint": "Neoforma",
            "bridge_relation": "indemnification obligations",
        }
    )

    lowered = [query.lower() for query in queries]
    assert any("reimbursable from third parties" in query for query in lowered)
    assert any("third-party claims" in query or "third party claims" in query for query in lowered)
    assert not any(query == "payment amount" for query in lowered)


def test_compile_requirement_queries_preserves_complete_relation_phrases_for_0041_style_query() -> None:
    namespace = _load_backend_compile_namespace()
    namespace["question"] = (
        "Under the agreement, what types of expenses incurred by Metavante are reimbursable from third parties, "
        "and how does this relate to Neoforma's indemnification obligations regarding third-party claims?"
    )
    compile_requirement_queries = namespace["_compile_requirement_queries"]

    queries = compile_requirement_queries(
        {
            "route_type": "structural",
            "intent": "specific_bridge_lookup",
            "entities": ["Metavante", "Neoforma"],
            "terms": ["reimbursable expenses", "third-party claims"],
            "query_keywords": [],
            "target_patterns": [],
            "left_endpoint": "Metavante",
            "right_endpoint": "Neoforma",
            "bridge_relation": "indemnification obligations",
        }
    )

    lowered = [query.lower() for query in queries]
    assert any("reimbursable from third parties" in query for query in lowered)
    assert any("third-party claims" in query or "third party claims" in query for query in lowered)
    assert not any(query == "metavante reimbursable" for query in lowered)
    assert not any("payment amount" in query for query in lowered)


def test_compile_requirement_queries_filters_over_generic_relation_queries_when_protected_terms_exist() -> None:
    namespace = _load_backend_compile_namespace()
    namespace["question"] = (
        "How do reimbursable expenses from third parties relate to Neoforma's indemnification obligations?"
    )
    compile_requirement_queries = namespace["_compile_requirement_queries"]

    queries = compile_requirement_queries(
        {
            "route_type": "local",
            "intent": "fact_lookup",
            "entities": ["Metavante", "Customer"],
            "terms": ["reimbursable expenses", "third-party claims"],
            "query_keywords": [],
            "target_patterns": [],
            "left_endpoint": "",
            "right_endpoint": "",
            "bridge_relation": "",
        }
    )

    lowered = [query.lower() for query in queries]
    assert not any("payment amount" in query for query in lowered)
    assert any("reimbursable expenses" in query for query in lowered)
    assert not any(query in {"metavante customer", "co-branding agreement metavante customer"} for query in lowered)


def test_compile_requirement_queries_filters_over_broad_local_entity_queries_when_protected_terms_exist() -> None:
    namespace = _load_backend_compile_namespace()
    namespace["question"] = (
        "Contract title: \"NeoformaInc_19991202_S-1A_EX-10.26_5224521_EX-10.26_Co-Branding Agreement\"\n"
        "Question: How do reimbursable expenses from third parties relate to Neoforma's indemnification obligations?"
    )
    compile_requirement_queries = namespace["_compile_requirement_queries"]

    queries = compile_requirement_queries(
        {
            "route_type": "local",
            "intent": "fact_lookup",
            "entities": ["Neoforma"],
            "terms": ["third-party claims", "indemnification obligations"],
            "anchors": ["Co-Branding Agreement"],
            "query_keywords": [],
            "target_patterns": [],
            "left_endpoint": "",
            "right_endpoint": "",
            "bridge_relation": "",
            "scope": "",
        }
    )

    lowered = [query.lower() for query in queries]
    assert any("third-party claims" in query or "third party claims" in query for query in lowered)
    assert "co-branding agreement neoforma" not in lowered


def test_compile_requirement_queries_local_keeps_subject_fallback_query() -> None:
    compile_requirement_queries = _load_backend_compile_requirement_queries()

    queries = compile_requirement_queries(
        {
            "route_type": "local",
            "intent": "fact_lookup",
            "entities": ["Go Call, Inc."],
            "terms": ["notice address"],
            "anchors": [],
            "query_keywords": ["Cambridge Ontario"],
            "target_patterns": [],
            "left_endpoint": "",
            "right_endpoint": "",
            "bridge_relation": "",
            "scope": "",
        }
    )

    lowered = [query.lower() for query in queries]
    assert "what is go call, inc. notice address" in lowered
    assert any(query.lower() == "go call, inc." for query in queries)


def test_build_claim_oriented_sub_question_local_prefers_natural_question_over_keyword_query() -> None:
    namespace = _load_backend_compile_namespace()
    build_claim_oriented_sub_question = namespace["_build_claim_oriented_sub_question"]

    question = build_claim_oriented_sub_question(
        {
            "route_type": "local",
            "intent": "fact_lookup",
            "entities": ["Go Call, Inc."],
            "terms": ["notice address"],
            "query_keywords": ["Cambridge Ontario"],
        },
        [
            "go call inc notice address",
            "what is go call, inc. notice address",
            "cambridge ontario",
        ],
        "What is Go Call, Inc. notice address?",
    )

    assert question.lower().startswith("what is go call, inc. notice address")


def test_compile_requirement_queries_local_role_lookup_prefers_promotion_description_queries() -> None:
    namespace = _load_backend_compile_namespace()
    namespace["question"] = (
        'Contract title: "GOCALLINC_03_30_2000-EX-10.7-Promotion Agreement"\n'
        "Question: What is Motorola's role in the PageMaster Corporation promotion for paging services?"
    )
    compile_requirement_queries = namespace["_compile_requirement_queries"]

    queries = compile_requirement_queries(
        {
            "route_type": "local",
            "intent": "fact_lookup",
            "entities": ["Motorola", "PageMaster Corporation"],
            "terms": ["role", "promotion", "paging services"],
            "anchors": ["Motorola", "PageMaster Corporation"],
            "query_keywords": ["Motorola role", "PageMaster promotion", "paging services"],
            "target_patterns": ["party_to", "grants_right_to", "grants_license_to", "pays"],
            "left_endpoint": "",
            "right_endpoint": "",
            "bridge_relation": "",
            "scope": "single_agreement",
        }
    )

    lowered = [query.lower() for query in queries]
    assert any("description of the promotion" in query and "motorola" in query for query in lowered)
    assert any("free new pagers" in query or "no activation fee" in query for query in lowered)
    assert all("party to" not in query for query in lowered)
    assert all("grants license to" not in query for query in lowered)


def test_build_claim_oriented_sub_question_for_promotion_role_avoids_direct_connection_wording() -> None:
    namespace = _load_backend_compile_namespace()
    build_claim_oriented_sub_question = namespace["_build_claim_oriented_sub_question"]

    sub_question = build_claim_oriented_sub_question(
        {
            "route_type": "structural",
            "intent": "bridge_lookup",
            "entities": ["Motorola", "PageMaster Corporation", "promotion agreement"],
            "terms": ["promotion", "paging services", "duties", "responsibilities", "rights"],
            "anchors": ["Motorola", "PageMaster Corporation", "promotion agreement"],
            "query_keywords": ["Motorola rights", "Motorola obligations", "promotion duties", "service agreement"],
            "target_patterns": ["grants_right_to", "grants_license_to", "pays", "exclusive_to"],
            "left_endpoint": "Motorola",
            "right_endpoint": "PageMaster Corporation",
            "bridge_relation": "grants_right_to",
            "scope": "agreement_context",
        },
        [
            "Description of the Promotion Motorola PageMaster Corporation",
            "which clause describes Motorola role in the promotion",
        ],
        "What is Motorola's role in the PageMaster Corporation promotion for paging services?",
    )

    assert "role in the promotion" in sub_question.lower()
    assert "connected" not in sub_question.lower()
    assert "motorola's" in sub_question.lower()


def test_build_bridge_evidence_sub_questions_splits_0043_style_definition_bridge() -> None:
    namespace = _load_backend_compile_namespace()
    compile_requirement_queries = namespace["_compile_requirement_queries"]
    build_bridge_evidence_sub_questions = namespace["_build_bridge_evidence_sub_questions"]

    question = (
        "What is the significance of Section 504 in relation to the definition of Privacy Regulations "
        "and how does it connect to the concept of Sensitive Customer Information as defined under Section 10.5.2?"
    )
    requirement = {
        "route_type": "structural",
        "intent": "definition_lookup",
        "route_reason": "Cross-clause definitional bridge",
        "entities": ["law"],
        "terms": [
            "Section 504",
            "Privacy Regulations",
            "Sensitive Customer Information",
            "Section 10.5.2",
            "definition",
        ],
        "anchors": ["Section 504", "Section 10.5.2"],
        "query_keywords": ["significance", "relation", "definition", "connect", "concept", "defined under"],
        "target_patterns": [
            "clause-[504]-named_as->document-[Privacy Regulations Definition]",
            "clause-[10.5.2]-named_as->document-[Sensitive Customer Information Definition]",
        ],
        "left_endpoint": "Section 504",
        "right_endpoint": "Section 10.5.2",
        "bridge_relation": "confidentiality_applies_to, governed_by",
        "scope": "Cross-clause definitional and applicability analysis within a single agreement or related document set.",
    }

    queries = compile_requirement_queries(requirement)
    sub_questions = build_bridge_evidence_sub_questions(requirement, queries, question)

    assert len(sub_questions) == 3
    assert all(item["route_type"] == "local" for item in sub_questions)
    assert all(item["allow_original_question_target_inference"] is False for item in sub_questions)

    left_question = sub_questions[0]["sub-question"].lower()
    middle_question = sub_questions[1]["sub-question"].lower()
    right_question = sub_questions[2]["sub-question"].lower()
    assert "section 504" in left_question
    assert "privacy regulations" in left_question
    assert "connected" not in left_question
    assert "sensitive customer information" in middle_question
    assert "defined" in middle_question
    assert "section 10.5.2" in right_question
    assert "connected" not in right_question
    assert "business context" in right_question

    middle_queries = [query.lower() for query in sub_questions[1]["retrieval_queries"]]
    assert "sensitive customer information definition" in middle_queries
    assert any("which clause defines sensitive customer information" == query for query in middle_queries)
    assert sub_questions[1]["retrieval_requirement"]["scope_inference_mode"] == "open"

    right_queries = [query.lower() for query in sub_questions[2]["retrieval_queries"]]
    assert "section 10.5.2 business context" in right_queries
    assert "section 10.5.2" in right_queries
    assert sub_questions[2]["retrieval_requirement"]["intent"] == "context_lookup"
    assert sub_questions[2]["retrieval_requirement"]["scope_inference_mode"] == "open"


def test_requirements_to_sub_questions_prefers_definition_bridge_split_over_structural_connection_for_0043() -> None:
    namespace = _load_backend_requirements_to_sub_questions_namespace()
    requirements_to_sub_questions = namespace["_requirements_to_sub_questions"]

    compiled = requirements_to_sub_questions(
        [
            {
                "route_type": "structural",
                "intent": "definition_lookup",
                "route_reason": "Cross-clause definitional bridge",
                "entities": ["law"],
                "terms": [
                    "Section 504",
                    "Privacy Regulations",
                    "Sensitive Customer Information",
                    "Section 10.5.2",
                    "definition",
                ],
                "anchors": ["Section 504", "Section 10.5.2"],
                "query_keywords": ["significance", "relation", "definition", "connect", "concept", "defined under"],
                "target_patterns": [
                    "clause-[504]-named_as->document-[Privacy Regulations Definition]",
                    "clause-[10.5.2]-named_as->document-[Sensitive Customer Information Definition]",
                ],
                "left_endpoint": "Section 504",
                "right_endpoint": "Section 10.5.2",
                "bridge_relation": "confidentiality_applies_to, governed_by",
                "scope": "Cross-clause definitional and applicability analysis within a single agreement or related document set.",
            }
        ],
        "What is the significance of Section 504 in relation to the definition of Privacy Regulations and how does it connect to the concept of Sensitive Customer Information as defined under Section 10.5.2?",
    )

    assert len(compiled) == 3
    assert all(item["route_type"] == "local" for item in compiled)
    assert all("connected" not in item["sub-question"].lower() for item in compiled)
    assert compiled[0]["sub_question_id"] == "rq_1"
    assert compiled[1]["sub_question_id"] == "rq_2"
    assert compiled[2]["sub_question_id"] == "rq_3"


def test_refactor_v1_requirement_pipeline_emits_requirement_and_plan_metadata() -> None:
    namespace = _load_backend_requirements_to_sub_questions_namespace()
    namespace["orchestration_mode"] = "refactor_v1"
    build_requirements = namespace["_build_evidence_requirements"]
    build_plans = namespace["_requirements_to_sub_question_plans"]

    evidence_requirements = build_requirements(
        [
            {
                "route_type": "structural",
                "intent": "definition_lookup",
                "route_reason": "Cross-clause definitional bridge",
                "entities": ["law"],
                "terms": [
                    "Section 504",
                    "Privacy Regulations",
                    "Sensitive Customer Information",
                    "Section 10.5.2",
                    "definition",
                ],
                "anchors": ["Section 504", "Section 10.5.2"],
                "query_keywords": ["significance", "relation", "definition", "connect", "concept", "defined under"],
                "left_endpoint": "Section 504",
                "right_endpoint": "Section 10.5.2",
                "bridge_relation": "confidentiality_applies_to, governed_by",
                "scope": "Cross-clause definitional and applicability analysis within a single agreement or related document set.",
            }
        ],
        "What is the significance of Section 504 in relation to the definition of Privacy Regulations and how does it connect to the concept of Sensitive Customer Information as defined under Section 10.5.2?",
    )

    compiled, serialized_requirements, serialized_plans = build_plans(
        evidence_requirements,
        "What is the significance of Section 504 in relation to the definition of Privacy Regulations and how does it connect to the concept of Sensitive Customer Information as defined under Section 10.5.2?",
    )

    assert len(serialized_requirements) == 3
    assert serialized_requirements[0]["requirement_id"] == "er_1_1"
    assert serialized_requirements[0]["semantic_need"] == "definition"
    assert len(compiled) == 3
    assert len(serialized_plans) == 3
    assert compiled[0]["sub_question_id"] == "rq_1"
    assert compiled[0]["evidence_requirement_id"] == "er_1_1"
    assert compiled[0]["semantic_need"] == "definition"
    assert compiled[0]["sub_question_plan"]["requirement_id"] == "er_1_1"
    assert compiled[0]["sub_question_plan"]["route_type"] == "local"


def test_refactor_v1_requirement_pipeline_splits_0022_compare_question_into_two_fact_requirements() -> None:
    namespace = _load_backend_requirements_to_sub_questions_namespace()
    namespace["orchestration_mode"] = "refactor_v1"
    build_requirements = namespace["_build_evidence_requirements"]
    build_plans = namespace["_requirements_to_sub_question_plans"]
    question = "What triggers a Service Level Credit Event for the CMS Electronic Funds Delivery Service Level, and how does it relate to the Core Systems' Availability requirements?"

    evidence_requirements = build_requirements(
        [
            {
                "route_type": "structural",
                "intent": "bridge_lookup",
                "route_reason": "compare service-level clauses",
                "entities": ["CMS Electronic Funds Delivery Service Level", "Core Systems"],
                "terms": [
                    "Service Level Credit Event",
                    "CMS Electronic Funds Delivery Service Level",
                    "Core Systems' Availability requirements",
                ],
                "left_endpoint": "CMS Electronic Funds Delivery Service Level",
                "right_endpoint": "Core Systems' Availability requirements",
                "bridge_relation": "relates_to",
            }
        ],
        question,
    )

    compiled, serialized_requirements, serialized_plans = build_plans(evidence_requirements, question)

    assert len(serialized_requirements) == 2
    assert all(item["semantic_need"] == "fact" for item in serialized_requirements)
    assert len(compiled) == 2
    assert compiled[0]["route_type"] == "local"
    assert compiled[1]["route_type"] == "local"
    assert compiled[0]["sub-question"].lower() == "what triggers a service level credit event for the cms electronic funds delivery service level?"
    assert compiled[1]["sub-question"].lower() == "what are the core systems' availability requirements?"
    assert compiled[0]["retrieval_queries"][0].lower().startswith("what triggers")
    assert "availability requirements" in compiled[1]["retrieval_queries"][0].lower()
    assert len(serialized_plans) == 2


def test_refactor_v1_requirement_pipeline_deduplicates_duplicate_compare_requirements() -> None:
    namespace = _load_backend_requirements_to_sub_questions_namespace()
    namespace["orchestration_mode"] = "refactor_v1"
    build_requirements = namespace["_build_evidence_requirements"]
    question = "What triggers a Service Level Credit Event for the CMS Electronic Funds Delivery Service Level, and how does it relate to the Core Systems' Availability requirements?"

    duplicated_requirements = [
        {
            "route_type": "structural",
            "intent": "bridge_lookup",
            "route_reason": "compare service-level clauses",
            "entities": ["CMS Electronic Funds Delivery Service Level", "Core Systems"],
            "terms": [
                "Service Level Credit Event",
                "CMS Electronic Funds Delivery Service Level",
                "Core Systems' Availability requirements",
            ],
            "left_endpoint": "CMS Electronic Funds Delivery Service Level",
            "right_endpoint": "Core Systems' Availability requirements",
            "bridge_relation": "relates_to",
        }
        for _ in range(3)
    ]

    evidence_requirements = build_requirements(duplicated_requirements, question)

    assert len(evidence_requirements) == 2
    assert [item.requirement_id for item in evidence_requirements] == ["er_1_1", "er_1_2"]


def test_refactor_v1_requirement_pipeline_builds_role_requirement_for_0002_shape() -> None:
    namespace = _load_backend_requirements_to_sub_questions_namespace()
    namespace["orchestration_mode"] = "refactor_v1"
    build_requirements = namespace["_build_evidence_requirements"]
    build_plans = namespace["_requirements_to_sub_question_plans"]
    question = "What is Motorola's role in the PageMaster Corporation promotion for paging services?"

    evidence_requirements = build_requirements(
        [
            {
                "route_type": "local",
                "intent": "fact_lookup",
                "entities": ["Motorola", "PageMaster Corporation"],
                "terms": ["role", "promotion", "paging services"],
                "anchors": ["Motorola", "PageMaster Corporation"],
                "query_keywords": ["Motorola role", "PageMaster promotion", "paging services"],
                "target_patterns": ["party_to", "grants_right_to", "grants_license_to", "pays"],
                "scope": "single_agreement",
            }
        ],
        question,
    )
    compiled, serialized_requirements, serialized_plans = build_plans(evidence_requirements, question)

    assert len(serialized_requirements) == 1
    assert serialized_requirements[0]["semantic_need"] == "fact"
    assert compiled[0]["sub-question"].lower() == "which clause describes motorola's role in the pagemaster corporation promotion for paging services?"
    lowered = [query.lower() for query in compiled[0]["retrieval_queries"]]
    assert any("description of the promotion" in query for query in lowered)
    assert any("free new pagers" in query for query in lowered)
    assert len(serialized_plans) == 1


def test_refactor_v1_requirement_pipeline_splits_0041_bridge_into_two_requirements() -> None:
    namespace = _load_backend_requirements_to_sub_questions_namespace()
    namespace["orchestration_mode"] = "refactor_v1"
    build_requirements = namespace["_build_evidence_requirements"]
    build_plans = namespace["_requirements_to_sub_question_plans"]
    question = (
        "Under the agreement, what types of expenses incurred by Metavante in connection with services provided "
        "to or on behalf of the Customer are considered reimbursable from third parties, and how does this relate "
        "to Neoforma's indemnification obligations regarding third-party claims?"
    )

    evidence_requirements = build_requirements(
        [
            {
                "route_type": "structural",
                "intent": "bridge_lookup",
                "entities": ["Metavante", "Neoforma"],
                "terms": ["reimbursable expenses", "third-party claims"],
                "left_endpoint": "Metavante",
                "right_endpoint": "Neoforma",
                "bridge_relation": "indemnification obligations",
            }
        ],
        question,
    )
    compiled, serialized_requirements, serialized_plans = build_plans(evidence_requirements, question)

    assert len(serialized_requirements) == 2
    assert [item["semantic_need"] for item in serialized_requirements] == ["definition", "fact"]
    assert len(compiled) == 2
    assert compiled[0]["sub-question"].lower() == "what types of expenses incurred by metavante are defined in the agreement?"
    assert "neoforma's indemnification obligations" in compiled[1]["sub-question"].lower()
    assert len(serialized_plans) == 2


def test_refactor_v1_requirement_pipeline_deduplicates_0041_semantic_targets_across_bridge_and_local_requirements() -> None:
    namespace = _load_backend_requirements_to_sub_questions_namespace()
    namespace["orchestration_mode"] = "refactor_v1"
    build_requirements = namespace["_build_evidence_requirements"]
    build_plans = namespace["_requirements_to_sub_question_plans"]
    question = (
        "Under the agreement, what types of expenses incurred by Metavante in connection with services provided "
        "to or on behalf of the Customer are considered reimbursable from third parties, and how does this relate "
        "to Neoforma's indemnification obligations regarding third-party claims?"
    )

    evidence_requirements = build_requirements(
        [
            {
                "route_type": "structural",
                "intent": "bridge_lookup",
                "route_reason": "Bridge reimbursable expenses to indemnification obligations",
                "entities": ["Metavante", "Customer", "Neoforma"],
                "terms": ["expenses", "reimbursable", "services", "indemnification", "third-party claims"],
                "anchors": ["Metavante", "Customer", "Neoforma"],
                "query_keywords": ["expenses incurred", "third parties", "indemnification obligations"],
                "left_endpoint": "Metavante",
                "right_endpoint": "indemnification",
                "bridge_relation": "pays / obligation / grants_right_to",
                "scope": "agreement",
            },
            {
                "route_type": "local",
                "intent": "definition_lookup",
                "route_reason": "First-part local fact",
                "entities": ["Metavante", "reimbursable", "expenses"],
                "terms": ["expenses", "reimbursable", "Metavante", "Customer", "services"],
                "anchors": ["Metavante", "reimbursable", "expenses"],
                "query_keywords": ["reimbursable expenses", "expenses incurred", "services provided"],
                "left_endpoint": "expense_reimbursement",
                "right_endpoint": "expense",
                "bridge_relation": "grants_right_to / pays",
                "scope": "clause",
            },
            {
                "route_type": "local",
                "intent": "fact_lookup",
                "route_reason": "Second-part local fact",
                "entities": ["Neoforma", "indemnification", "third-party"],
                "terms": ["indemnification", "obligations", "Neoforma", "third-party claims"],
                "anchors": ["Neoforma", "indemnification", "third-party"],
                "query_keywords": ["indemnification obligations", "third-party claims", "Neoforma"],
                "left_endpoint": "indemnification",
                "right_endpoint": "third_party_claim",
                "bridge_relation": "obligation / survives_for / terminates_on_event",
                "scope": "clause",
            },
        ],
        question,
    )
    compiled, serialized_requirements, serialized_plans = build_plans(evidence_requirements, question)

    assert len(serialized_requirements) == 2
    assert len(compiled) == 2
    assert len(serialized_plans) == 2
    assert compiled[0]["sub-question"].lower() == "what types of expenses incurred by metavante are defined in the agreement?"
    assert "neoforma's indemnification obligations" in compiled[1]["sub-question"].lower()
    assert compiled[0]["sub_question_plan"]["metadata"]["target_scope_hint"] == "agreement"
    assert compiled[1]["sub_question_plan"]["metadata"]["target_scope_hint"] == "agreement"


def test_refactor_v1_requirement_pipeline_splits_0043_bridge_into_three_requirements() -> None:
    namespace = _load_backend_requirements_to_sub_questions_namespace()
    namespace["orchestration_mode"] = "refactor_v1"
    build_requirements = namespace["_build_evidence_requirements"]
    build_plans = namespace["_requirements_to_sub_question_plans"]
    question = "What is the significance of Section 504 in relation to the definition of Privacy Regulations and how does it connect to the concept of Sensitive Customer Information as defined under Section 10.5.2?"

    evidence_requirements = build_requirements(
        [
            {
                "route_type": "structural",
                "intent": "definition_lookup",
                "route_reason": "Cross-clause definitional bridge",
                "entities": ["law"],
                "terms": [
                    "Section 504",
                    "Privacy Regulations",
                    "Sensitive Customer Information",
                    "Section 10.5.2",
                    "definition",
                ],
                "anchors": ["Section 504", "Section 10.5.2"],
                "left_endpoint": "Section 504",
                "right_endpoint": "Section 10.5.2",
                "bridge_relation": "confidentiality_applies_to, governed_by",
            }
        ],
        question,
    )
    compiled, serialized_requirements, serialized_plans = build_plans(evidence_requirements, question)

    assert len(serialized_requirements) == 3
    assert [item["semantic_need"] for item in serialized_requirements] == ["definition", "definition", "context"]
    assert len(compiled) == 3
    assert compiled[0]["sub-question"].lower() == "what does section 504 say about privacy regulations?"
    assert compiled[1]["sub-question"].lower() == "how is sensitive customer information defined?"


def test_resolve_refactor_subquestion_scope_routes_0041_neoforma_fact_leg_to_neoforma() -> None:
    namespace = _load_backend_refactor_scope_namespace()
    resolve_scope = namespace["_resolve_refactor_subquestion_scope"]

    result = resolve_scope(
        sub_question_text="What does the agreement say about Neoforma's indemnification obligations regarding third-party claims?",
        sub_question_plan={
            "semantic_need": "fact",
            "bridge_scope": "",
            "metadata": {
                "anchor_terms": ["Neoforma", "indemnification", "third-party claims"],
                "target_entity": "Neoforma",
                "target_scope_hint": "",
            },
        },
        retrieval_requirement={
            "semantic_need": "fact",
            "intent": "fact_lookup",
            "entities": ["Neoforma"],
            "anchors": ["Neoforma", "indemnification", "third-party claims"],
            "terms": ["indemnification obligations", "third-party claims"],
            "retrieval_queries_override": [
                "What does the agreement say about Neoforma's indemnification obligations regarding third-party claims?",
                "Neoforma indemnification obligations third-party claims",
            ],
        },
        original_question="Under the agreement, what types of expenses incurred by Metavante are considered reimbursable from third parties, and how does this relate to Neoforma's indemnification obligations regarding third-party claims?",
        explicit_question_target_doc_id="",
        global_target_doc_id="OFGBANCORP#p0",
        known_doc_ids=["OFGBANCORP#p0", "Neoforma#p0"],
        normalized_entity_doc_ids={
            "neoforma": {"Neoforma#p0"},
            "indemnification": {"Neoforma#p0"},
            "thirdpartyclaims": {"Neoforma#p0"},
        },
        normalize_doc_hint=lambda text: re.sub(r"[^a-z0-9]+", "", str(text or "").lower()),
        infer_target_doc_id=lambda text: "",
        doc_id_aliases=lambda doc_id: [doc_id],
        chunk_text_map={},
        chunk_id_to_doc_id={},
    )

    assert result["target_doc_id"] == "Neoforma#p0"
    assert result["scope_plan"]["primary_doc_id"] == "Neoforma#p0"
    assert result["scope_plan"]["scope_type"] == "single_doc"


def test_resolve_refactor_subquestion_scope_prefers_explicit_entity_over_generic_phrase_on_tie() -> None:
    namespace = _load_backend_refactor_scope_namespace()
    resolve_scope = namespace["_resolve_refactor_subquestion_scope"]

    result = resolve_scope(
        sub_question_text="What does the agreement say about Neoforma's indemnification obligations regarding third-party claims?",
        sub_question_plan={
            "semantic_need": "fact",
            "bridge_scope": "",
            "metadata": {
                "anchor_terms": ["Neoforma", "third-party claims"],
                "target_entity": "Neoforma",
                "target_scope_hint": "",
            },
        },
        retrieval_requirement={
            "semantic_need": "fact",
            "intent": "fact_lookup",
            "entities": ["Neoforma"],
            "anchors": ["Neoforma", "third-party claims"],
            "terms": ["third-party claims"],
            "retrieval_queries_override": [
                "What does the agreement say about Neoforma's indemnification obligations regarding third-party claims?",
            ],
        },
        original_question="Under the agreement, what types of expenses incurred by Metavante are considered reimbursable from third parties, and how does this relate to Neoforma's indemnification obligations regarding third-party claims?",
        explicit_question_target_doc_id="Neoforma#p0",
        global_target_doc_id="GOCALL#p0",
        known_doc_ids=["GOCALL#p0", "Neoforma#p0"],
        normalized_entity_doc_ids={
            "neoforma": {"Neoforma#p0"},
            "thirdpartyclaims": {"GOCALL#p0"},
        },
        normalize_doc_hint=lambda text: re.sub(r"[^a-z0-9]+", "", str(text or "").lower()),
        infer_target_doc_id=lambda text: "",
        doc_id_aliases=lambda doc_id: [doc_id],
        chunk_text_map={},
        chunk_id_to_doc_id={},
    )

    assert result["target_doc_id"] == "Neoforma#p0"
    assert result["scope_plan"]["primary_doc_id"] == "Neoforma#p0"


def test_resolve_refactor_subquestion_scope_uses_section_anchor_chunk_hits_for_0043_context_leg() -> None:
    namespace = _load_backend_refactor_scope_namespace()
    resolve_scope = namespace["_resolve_refactor_subquestion_scope"]

    result = resolve_scope(
        sub_question_text="What business context does Section 10.5.2 describe in the agreement?",
        sub_question_plan={
            "semantic_need": "context",
            "bridge_scope": "open",
            "metadata": {
                "anchor_terms": ["Section 10.5.2"],
                "target_entity": "Section 10.5.2",
                "target_scope_hint": "open",
            },
        },
        retrieval_requirement={
            "semantic_need": "context",
            "intent": "context_lookup",
            "anchors": ["Section 10.5.2"],
            "entities": ["law"],
            "terms": ["Section 504", "Privacy Regulations", "Sensitive Customer Information", "Section 10.5.2", "definition"],
            "query_keywords": ["significance", "relation to", "definition", "connects to", "concept", "defined under"],
            "retrieval_queries_override": [
                "What business context does Section 10.5.2 describe in the agreement?",
                "Section 10.5.2 business context",
                "Section 10.5.2",
            ],
            "scope_inference_mode": "open",
        },
        original_question="What is the significance of Section 504 in relation to the definition of Privacy Regulations and how does it connect to the concept of Sensitive Customer Information as defined under Section 10.5.2?",
        explicit_question_target_doc_id="",
        global_target_doc_id="OFGBANCORP#p0",
        known_doc_ids=["OFGBANCORP#p0", "Neoforma#p0", "WORLD#p0"],
        normalized_entity_doc_ids={},
        normalize_doc_hint=lambda text: re.sub(r"[^a-z0-9]+", "", str(text or "").lower()),
        infer_target_doc_id=lambda text: "",
        doc_id_aliases=lambda doc_id: [doc_id],
        chunk_text_map={
            "c1": "Section 10.5.2 [CO-BRANDED TRAINING AND EDUCATION CENTER] and Training and Education Gross Margin.",
            "c2": "Section 504 defines Privacy Regulations.",
        },
        chunk_id_to_doc_id={
            "c1": "Neoforma#p0",
            "c2": "OFGBANCORP#p0",
        },
    )

    assert result["target_doc_id"] == "Neoforma#p0"
    assert result["scope_plan"]["primary_doc_id"] == "Neoforma#p0"
    assert result["scope_plan"]["scope_type"] == "single_doc"


def test_resolve_refactor_subquestion_scope_does_not_hard_bind_0041_expense_leg_to_parent_target() -> None:
    namespace = _load_backend_refactor_scope_namespace()
    resolve_scope = namespace["_resolve_refactor_subquestion_scope"]

    result = resolve_scope(
        sub_question_text="What types of Expenses incurred by Metavante are defined in the agreement?",
        sub_question_plan={
            "semantic_need": "definition",
            "bridge_scope": "",
            "metadata": {
                "anchor_terms": ["Metavante", "Expenses"],
                "target_entity": "Metavante",
                "target_scope_hint": "",
            },
        },
        retrieval_requirement={
            "semantic_need": "definition",
            "intent": "definition_lookup",
            "entities": ["Metavante"],
            "anchors": ["Metavante", "Expenses"],
            "terms": ["Expenses", "reasonable and direct expenses"],
            "query_keywords": ["Metavante Expenses definition", "paid by Metavante to Third Parties"],
            "target_doc_id": "Neoforma#p0",
            "allow_original_question_target_inference": False,
            "retrieval_queries_override": [
                "What types of Expenses incurred by Metavante are defined in the agreement?",
                "what clause defines Metavante Expenses",
                "Metavante Expenses definition",
            ],
        },
        original_question="Under the agreement, what types of expenses incurred by Metavante are considered reimbursable from third parties, and how does this relate to Neoforma's indemnification obligations regarding third-party claims?",
        explicit_question_target_doc_id="Neoforma#p0",
        global_target_doc_id="Neoforma#p0",
        known_doc_ids=["OFGBANCORP#p0", "Neoforma#p0"],
        normalized_entity_doc_ids={
            "metavante": {"OFGBANCORP#p0"},
            "expenses": {"OFGBANCORP#p0"},
            "neoforma": {"Neoforma#p0"},
        },
        normalize_doc_hint=lambda text: re.sub(r"[^a-z0-9]+", "", str(text or "").lower()),
        infer_target_doc_id=lambda text: "",
        doc_id_aliases=lambda doc_id: [doc_id],
        chunk_text_map={},
        chunk_id_to_doc_id={},
    )

    assert result["target_doc_id"] == "OFGBANCORP#p0"
    assert result["scope_plan"]["primary_doc_id"] == "OFGBANCORP#p0"
    assert result["scope_plan"]["scope_type"] == "single_doc"


def test_collect_section_anchor_rescue_candidates_prefers_clause_body_over_ascribed_reference() -> None:
    helpers = _extract_backend_functions("_collect_section_anchor_rescue_candidates")
    method = helpers["_collect_section_anchor_rescue_candidates"]

    rescue = method(
        "What business context does Section 10.5.2 describe in the agreement?",
        ["Section 10.5.2 business context", "Section 10.5.2"],
        {"anchors": ["Section 10.5.2"], "terms": ["Section 10.5.2"]},
        chunk_text_map={
            "neo_chunk": "Section 10.5.2 [CO-BRANDED TRAINING AND EDUCATION CENTER]. Training and Education Gross Margin shall have the meaning in this section.",
            "ofg_chunk": "TRAINING AND EDUCATION GROSS MARGIN shall have the meaning ascribed thereto in Section 10.5.2.",
        },
        chunk_id_to_doc_id={
            "neo_chunk": "Neoforma#p0",
            "ofg_chunk": "OFGBANCORP#p0",
        },
        limit=4,
    )

    assert rescue["doc_ids"][0] == "Neoforma#p0"
    assert rescue["chunk_ids"][0] == "neo_chunk"
    assert rescue["doc_hits"]["Neoforma#p0"]["best_hit_type"] == "clause_body"
    assert "OFGBANCORP#p0" not in rescue["doc_hits"] or rescue["doc_hits"]["OFGBANCORP#p0"]["best_hit_type"] == "cross_reference_only"


def test_classify_sub_question_evidence_scope_marks_mixed_support_as_external_related() -> None:
    helpers = _extract_backend_functions(
        "_classify_sub_question_evidence_scope",
    )

    meta = helpers["_classify_sub_question_evidence_scope"](
        {
            "target_doc_id": "",
            "scope_plan": {"primary_doc_id": "", "secondary_doc_ids": []},
            "support_chunk_ids": ["neo-1", "ofg-1"],
            "retrieved_chunk_ids_all": ["neo-1", "ofg-1"],
            "first_stage_chunk_ids": ["neo-1", "ofg-1"],
        },
        answer_scope_target_doc_id="OFGBANCORP#p0",
        chunk_id_to_doc_id={
            "neo-1": "Neoforma#p0",
            "ofg-1": "OFGBANCORP#p0",
        },
    )

    assert meta["evidence_scope_label"] == "external_related"
    assert meta["support_doc_ids"] == ["Neoforma#p0", "OFGBANCORP#p0"]


def test_classify_sub_question_evidence_scope_does_not_backfill_target_ownership_without_observed_support() -> None:
    helpers = _extract_backend_functions(
        "_classify_sub_question_evidence_scope",
    )

    meta = helpers["_classify_sub_question_evidence_scope"](
        {
            "target_doc_id": "Neoforma#p0",
            "scope_plan": {"primary_doc_id": "Neoforma#p0"},
            "support_chunk_ids": [],
            "retrieved_chunk_ids_all": ["ofg-1"],
            "first_stage_chunk_ids": ["ofg-1"],
        },
        answer_scope_target_doc_id="Neoforma#p0",
        chunk_id_to_doc_id={
            "ofg-1": "OFGBANCORP#p0",
        },
    )

    assert meta["support_doc_ids"] == []
    assert meta["observed_support_doc_ids"] == []
    assert meta["inferred_target_doc_id"] == "Neoforma#p0"
    assert meta["grounding_status"] == "insufficient_support"
    assert meta["evidence_scope_label"] == "insufficient_support"
    assert meta["retrieval_doc_ids"] == ["OFGBANCORP#p0"]


def test_select_support_lane_from_query_payloads_prefers_single_doc_lane_over_mixed_pool() -> None:
    namespace = _load_backend_support_lane_namespace()
    select_support_lane = namespace["_select_support_lane_from_query_payloads"]

    lane = select_support_lane(
        [
            {
                "query_text": "Section 10.5.2 definition",
                "chunk_ids": ["neo-1", "neo-2"],
                "chunk_contents": {"neo-1": "Neo chunk 1", "neo-2": "Neo chunk 2"},
                "chunk_stage_trace": {"first_stage_chunk_ids": ["neo-1", "neo-2"]},
                "target_doc_id": "",
                "scope_plan": {"scope_type": "global_open", "primary_doc_id": "", "secondary_doc_ids": []},
            },
            {
                "query_text": "Section 10.5.2",
                "chunk_ids": ["ofg-1", "neo-1"],
                "chunk_contents": {"ofg-1": "OFG chunk", "neo-1": "Neo chunk 1"},
                "chunk_stage_trace": {"first_stage_chunk_ids": ["ofg-1", "neo-1"]},
                "target_doc_id": "",
                "scope_plan": {"scope_type": "global_open", "primary_doc_id": "", "secondary_doc_ids": []},
            },
        ],
        sub_question_text="What does Section 10.5.2 define in the agreement?",
        retrieval_queries=["Section 10.5.2 definition", "Section 10.5.2"],
        retrieval_requirement={"intent": "definition_lookup"},
        route_type="local",
        target_doc_id="",
        scope_plan={"scope_type": "global_open", "primary_doc_id": "", "secondary_doc_ids": []},
        chunk_id_to_doc_id={
            "neo-1": "Neoforma#p0",
            "neo-2": "Neoforma#p0",
            "ofg-1": "OFGBANCORP#p0",
        },
    )

    assert lane["query_text"] == "Section 10.5.2 definition"
    assert [item["chunk_id"] for item in lane["support_pairs"][:2]] == ["neo-1", "neo-2"]


def test_select_support_lane_from_query_payloads_prefers_anchor_doc_lane_for_0043_shape() -> None:
    namespace = _load_backend_support_lane_namespace()
    select_support_lane = namespace["_select_support_lane_from_query_payloads"]

    lane = select_support_lane(
        [
            {
                "query_text": "Section 10.5.2 definition",
                "chunk_ids": ["neo-1", "neo-2"],
                "chunk_contents": {"neo-1": "Neo chunk 1", "neo-2": "Neo chunk 2"},
                "chunk_stage_trace": {"first_stage_chunk_ids": ["neo-1", "neo-2"]},
                "target_doc_id": "",
                "scope_plan": {"scope_type": "global_open", "primary_doc_id": "", "secondary_doc_ids": []},
            },
            {
                "query_text": "which clause defines Sensitive Customer Information",
                "chunk_ids": ["ofg-1", "ofg-2"],
                "chunk_contents": {"ofg-1": "OFG chunk 1", "ofg-2": "OFG chunk 2"},
                "chunk_stage_trace": {"first_stage_chunk_ids": ["ofg-1", "ofg-2"]},
                "target_doc_id": "",
                "scope_plan": {"scope_type": "global_open", "primary_doc_id": "", "secondary_doc_ids": []},
            },
        ],
        sub_question_text="What does Section 10.5.2 define in the agreement?",
        retrieval_queries=[
            "Section 10.5.2 definition",
            "which clause defines Sensitive Customer Information",
        ],
        retrieval_requirement={
            "intent": "definition_lookup",
            "anchors": ["Section 10.5.2"],
            "query_keywords": ["which clause defines Sensitive Customer Information"],
        },
        route_type="local",
        target_doc_id="",
        scope_plan={"scope_type": "global_open", "primary_doc_id": "", "secondary_doc_ids": []},
        chunk_id_to_doc_id={
            "neo-1": "Neoforma#p0",
            "neo-2": "Neoforma#p0",
            "ofg-1": "OFGBANCORP#p0",
            "ofg-2": "OFGBANCORP#p0",
        },
        normalized_entity_doc_ids={
            "section1052": {"Neoforma#p0"},
            "sensitivecustomerinformation": {"OFGBANCORP#p0"},
        },
        normalize_doc_hint=lambda text: re.sub(r"[^a-z0-9]+", "", str(text or "").lower()),
    )

    assert lane["query_text"] == "Section 10.5.2 definition"
    assert lane["anchor_doc_ids"] == ["Neoforma#p0"]
    assert [item["chunk_id"] for item in lane["support_pairs"][:2]] == ["neo-1", "neo-2"]


def test_select_support_lane_from_query_payloads_prefers_answer_scope_anchor_lane_when_primary_conflicts() -> None:
    namespace = _load_backend_support_lane_namespace()
    select_support_lane = namespace["_select_support_lane_from_query_payloads"]

    lane = select_support_lane(
        [
            {
                "query_text": "third-party claims indemnification",
                "chunk_ids": ["gocall-1", "gocall-2"],
                "chunk_contents": {"gocall-1": "GOCALL chunk 1", "gocall-2": "GOCALL chunk 2"},
                "chunk_stage_trace": {"first_stage_chunk_ids": ["gocall-1", "gocall-2"]},
                "target_doc_id": "GOCALL#p0",
                "scope_plan": {"scope_type": "single_doc", "primary_doc_id": "GOCALL#p0", "secondary_doc_ids": []},
            },
            {
                "query_text": "Section 15.4 Neoforma indemnification",
                "chunk_ids": ["neo-15-4", "neo-2"],
                "chunk_contents": {"neo-15-4": "Neo chunk 15.4", "neo-2": "Neo chunk 2"},
                "chunk_stage_trace": {"first_stage_chunk_ids": ["neo-15-4", "neo-2"]},
                "target_doc_id": "GOCALL#p0",
                "scope_plan": {"scope_type": "single_doc", "primary_doc_id": "GOCALL#p0", "secondary_doc_ids": []},
            },
        ],
        sub_question_text="What does the agreement say about Neoforma's indemnification obligations regarding third-party claims?",
        retrieval_queries=[
            "third-party claims indemnification",
            "Section 15.4 Neoforma indemnification",
        ],
        retrieval_requirement={
            "intent": "bridge_lookup",
            "anchors": ["Section 15.4"],
            "terms": ["Neoforma", "third-party claims"],
        },
        route_type="structural",
        target_doc_id="GOCALL#p0",
        answer_scope_target_doc_id="Neoforma#p0",
        scope_plan={"scope_type": "single_doc", "primary_doc_id": "GOCALL#p0", "secondary_doc_ids": []},
        chunk_id_to_doc_id={
            "gocall-1": "GOCALL#p0",
            "gocall-2": "GOCALL#p0",
            "neo-15-4": "Neoforma#p0",
            "neo-2": "Neoforma#p0",
        },
        normalized_entity_doc_ids={
            "section154": {"Neoforma#p0"},
            "thirdpartyclaims": {"GOCALL#p0"},
        },
        normalize_doc_hint=lambda text: re.sub(r"[^a-z0-9]+", "", str(text or "").lower()),
    )

    assert lane["query_text"] == "Section 15.4 Neoforma indemnification"
    assert lane["anchor_doc_ids"] == ["Neoforma#p0"]


def test_collect_section_anchor_rescue_candidates_prefers_exact_section_chunks() -> None:
    helpers = _extract_backend_functions(
        "_collect_section_anchor_rescue_candidates",
    )

    rescue = helpers["_collect_section_anchor_rescue_candidates"](
        "What business context does Section 10.5.2 describe in the agreement?",
        ["Section 10.5.2 business context", "Section 10.5.2"],
        {"anchors": ["Section 10.5.2"], "intent": "context_lookup"},
        chunk_text_map={
            "neo-1": "Section 10.5.2 Co-Branded Training and Education Center. Training and Education Gross Margin shall mean listing fees and e-commerce revenue.",
            "ofg-1": "\"Sensitive Customer Information\" shall mean Customer Data tied to account numbers and passwords.",
            "wsi-1": "Section 10.12(b) Confidential Information means proprietary data.",
        },
        chunk_id_to_doc_id={
            "neo-1": "Neoforma#p0",
            "ofg-1": "OFGBANCORP#p0",
            "wsi-1": "WORLDWIDE#p0",
        },
        limit=5,
    )

    assert rescue["anchors"] == ["Section 10.5.2"]
    assert rescue["chunk_ids"][0] == "neo-1"
    assert rescue["doc_ids"] == ["Neoforma#p0"]
    assert rescue["doc_hits"]["Neoforma#p0"]["best_hit_type"] == "clause_body"


def test_collect_section_anchor_rescue_candidates_matches_bare_section_number_clause_body() -> None:
    helpers = _extract_backend_functions(
        "_collect_section_anchor_rescue_candidates",
    )

    rescue = helpers["_collect_section_anchor_rescue_candidates"](
        "What business context does Section 10.5.2 describe in the agreement?",
        ["Section 10.5.2 business context", "Section 10.5.2"],
        {"anchors": ["Section 10.5.2"], "intent": "context_lookup"},
        chunk_text_map={
            "neo-ctx": "10.5.2 CO-BRANDED TRAINING AND EDUCATION CENTER. VerticalNet will pay Neoforma [*] of the Training and Education Gross Margin.",
            "ofg-ref": "TRAINING AND EDUCATION GROSS MARGIN shall have the meaning ascribed thereto in Section 10.5.2.",
        },
        chunk_id_to_doc_id={
            "neo-ctx": "Neoforma#p0",
            "ofg-ref": "OFGBANCORP#p0",
        },
        limit=5,
    )

    assert rescue["chunk_ids"][0] == "neo-ctx"
    assert rescue["doc_ids"][0] == "Neoforma#p0"
    assert rescue["doc_hits"]["Neoforma#p0"]["best_hit_type"] == "clause_body"


def test_resolve_refactor_subquestion_scope_context_section_anchor_collapses_to_single_doc() -> None:
    helpers = _extract_backend_functions("_resolve_refactor_subquestion_scope", "_collect_section_anchor_rescue_candidates")
    method = helpers["_resolve_refactor_subquestion_scope"]

    result = method(
        sub_question_text="What business context does Section 10.5.2 describe in the agreement?",
        sub_question_plan={
            "semantic_need": "context",
            "bridge_scope": "open",
            "metadata": {
                "anchor_terms": ["Section 10.5.2"],
                "target_scope_hint": "open",
            },
        },
        retrieval_requirement={
            "semantic_need": "context",
            "anchors": ["Section 10.5.2"],
            "retrieval_queries_override": [
                "What business context does Section 10.5.2 describe in the agreement?",
                "Section 10.5.2 business context",
                "Section 10.5.2",
            ],
            "scope_inference_mode": "open",
        },
        original_question="What is the significance of Section 504 in relation to the definition of Privacy Regulations and how does it connect to the concept of Sensitive Customer Information as defined under Section 10.5.2?",
        explicit_question_target_doc_id="",
        global_target_doc_id="OFGBANCORP#p0",
        known_doc_ids=["OFGBANCORP#p0", "Neoforma#p0"],
        normalized_entity_doc_ids={},
        normalize_doc_hint=lambda text: re.sub(r"[^a-z0-9]+", "", str(text or "").lower()),
        infer_target_doc_id=lambda text: "",
        doc_id_aliases=lambda doc_id: [doc_id],
        chunk_text_map={
            "neo-ctx": "10.5.2 CO-BRANDED TRAINING AND EDUCATION CENTER. VerticalNet will pay Neoforma [*] of the Training and Education Gross Margin.",
            "ofg-ref": "TRAINING AND EDUCATION GROSS MARGIN shall have the meaning ascribed thereto in Section 10.5.2.",
        },
        chunk_id_to_doc_id={
            "neo-ctx": "Neoforma#p0",
            "ofg-ref": "OFGBANCORP#p0",
        },
    )

    assert result["target_doc_id"] == "Neoforma#p0"
    assert result["scope_plan"]["primary_doc_id"] == "Neoforma#p0"
    assert result["scope_plan"]["scope_type"] == "single_doc"
    assert result["scope_plan"]["secondary_doc_ids"] == []


def test_compute_refactor_subquestion_budget_prioritizes_definition_over_context() -> None:
    helpers = _extract_backend_functions("_compute_refactor_subquestion_budget")
    compute_budget = helpers["_compute_refactor_subquestion_budget"]

    high = compute_budget(
        {"semantic_need": "definition", "priority": 0.8},
        {"semantic_need": "definition", "evidence_weight": 0.8},
        top_k_high=12,
        top_k_medium=10,
        top_k_low=8,
    )
    low = compute_budget(
        {"semantic_need": "context", "priority": 0.3},
        {"semantic_need": "context", "evidence_weight": 0.3},
        top_k_high=12,
        top_k_medium=10,
        top_k_low=8,
    )

    assert high["tier"] == "high"
    assert high["top_k"] == 12
    assert low["tier"] == "low"
    assert low["top_k"] == 8


def test_build_refactor_evidence_matrix_cross_subquestion_dedups_semantic_overlap() -> None:
    helpers = _extract_backend_functions(
        "_serialize_evidence_item",
        "_semantic_text_overlap_ratio",
        "_build_refactor_evidence_matrix",
    )
    build_matrix = helpers["_build_refactor_evidence_matrix"]

    matrix = build_matrix(
        [
            {
                "sub_question_id": "sq_1",
                "route_type": "local",
                "query_text": "q1",
                "evidence_scope_label": "target_contract",
                "semantic_need": "definition",
                "evidence_weight": 0.8,
                "retrieved_chunks_all": [
                    {"chunk_id": "c1", "text": "Section 10.5.2 defines the Co-Branded Training and Education Center.", "doc_id": "neo", "score": 4.0},
                ],
            },
            {
                "sub_question_id": "sq_2",
                "route_type": "local",
                "query_text": "q2",
                "evidence_scope_label": "external_related",
                "semantic_need": "context",
                "evidence_weight": 0.4,
                "retrieved_chunks_all": [
                    {"chunk_id": "c2", "text": "Section 10.5.2 defines the Co-Branded Training and Education Center and related margin.", "doc_id": "neo", "score": 3.0},
                ],
            },
        ],
        max_items_per_subquestion=4,
        dedup_threshold=0.7,
    )

    assert len(matrix) == 1
    assert matrix[0]["chunk_id"] == "c1"


def test_select_refactor_context_from_evidence_matrix_prefers_high_score_primary_and_caps_external() -> None:
    helpers = _extract_backend_functions("_select_refactor_context_from_evidence_matrix")
    select_context = helpers["_select_refactor_context_from_evidence_matrix"]

    selected = select_context(
        [
            {"origin_sub_question_id": "sq_1", "chunk_id": "p1", "score": 4.0, "scope_label": "target_contract"},
            {"origin_sub_question_id": "sq_1", "chunk_id": "p2", "score": 3.0, "scope_label": "target_contract"},
            {"origin_sub_question_id": "sq_2", "chunk_id": "e1", "score": 5.0, "scope_label": "external_related"},
            {"origin_sub_question_id": "sq_3", "chunk_id": "p3", "score": 2.0, "scope_label": "target_contract"},
            {"origin_sub_question_id": "sq_4", "chunk_id": "e2", "score": 4.5, "scope_label": "external_related"},
        ],
        total_limit=2,
        external_limit=1,
    )

    assert selected["primary_chunk_ids"] == ["p1", "p3"]
    assert selected["external_chunk_ids"] == ["e1"]


def test_context_lookup_rescue_fallback_uses_rescued_chunk_when_llm_returns_not_found() -> None:
    source = (YOUTU_ROOT / "backend.py").read_text(encoding="utf-8")

    assert "_build_structured_clause_summary_from_chunk(" in source


def test_build_structured_clause_summary_from_chunk_preserves_section_context() -> None:
    helpers = _extract_backend_functions(
        "_normalize_prompt_text",
        "_clean_prompt_field",
        "_trim_prompt_excerpt",
        "_build_structured_clause_summary_from_chunk",
    )

    summary = helpers["_build_structured_clause_summary_from_chunk"](
        "1.44 TRAINING AND EDUCATION GROSS MARGIN shall have the meaning ascribed thereto in Section 10.5.2 [CO-BRANDED TRAINING AND EDUCATION CENTER].",
        anchors=["Section 10.5.2"],
        intent="context_lookup",
        query_text="What business context does Section 10.5.2 describe in the agreement?",
    )

    assert "Section 10.5.2 is identified as CO-BRANDED TRAINING AND EDUCATION CENTER." in summary["sub_answer"]
    assert "It supplies the agreement context for TRAINING AND EDUCATION GROSS MARGIN." in summary["sub_answer"]
    assert summary["confidence"] >= 0.7


def test_cross_document_bridge_prompt_rules_preserve_grounded_section_findings() -> None:
    helpers = _extract_backend_functions("_build_answer_scope_prompt_rules")

    rules = helpers["_build_answer_scope_prompt_rules"](
        "OFGBANCORP#p0",
        answer_composition_mode="cross_document_bridge",
    )

    assert "If a grounded sub-question comes from an explicit Section or Article anchor" in rules
    assert "Do not discard a grounded section-anchored clause as noise" in rules


def test_cross_document_bridge_prompt_rules_forbid_clause_absence_when_anchor_exists() -> None:
    helpers = _extract_backend_functions("_build_answer_scope_prompt_rules")

    rules = helpers["_build_answer_scope_prompt_rules"](
        "Neoforma#p0",
        answer_composition_mode="cross_document_bridge",
    )

    assert "do not rewrite that as the contract not containing the clause" in rules


def test_final_answer_knowledge_package_surfaces_explicit_section_findings() -> None:
    helpers = _extract_backend_functions(
        "_build_findings_block",
        "_format_refactor_reasoning_plan",
        "_build_supporting_relations",
        "_build_structural_evidence_summary",
        "_build_final_answer_knowledge_package",
    )

    package = helpers["_build_final_answer_knowledge_package"](
        question="How does Section 504 connect to Section 10.5.2?",
        subquestion_context="sq_1\nsq_2",
        sub_question_answers=[
            {
                "sub_question_id": "sq_3",
                "route_type": "local",
                "query_text": "What business context does Section 10.5.2 describe in the agreement?",
                "anchors": ["Section 10.5.2"],
                "evidence_scope_label": "external_related",
                "evidence_scope_reason": "rescued via explicit section anchor",
                "support_doc_ids": ["Neoforma#p0"],
                "support_chunk_ids": ["neo-1"],
                "sub_answer": "Section 10.5.2 is identified as CO-BRANDED TRAINING AND EDUCATION CENTER. It supplies the agreement context for Training and Education Gross Margin.",
                "reason": "Recovered from a section-anchor rescue pass.",
            }
        ],
        backing_chunk_contents=[],
        triples=[],
        answer_scope_target_doc_id="OFGBANCORP#p0",
        answer_composition_mode="cross_document_bridge",
        semantic_alignment={},
    )

    assert "=== Explicit Section Findings (Must be preserved in the final answer) ===" in package
    assert "Section 10.5.2" in package


def test_build_refactor_reasoning_plan_structures_compare_facts() -> None:
    helpers = _extract_backend_functions(
        "_strip_question_prompt_prefix",
        "_build_refactor_reasoning_plan",
    )

    plan = helpers["_build_refactor_reasoning_plan"](
        question="What triggers the CMS Electronic Funds Delivery Service Level Credit Event, and how does it relate to Core Systems' Availability requirements?",
        sub_question_answers=[
            {
                "sub_question_id": "sq_1",
                "semantic_need": "fact",
                "query_text": "What triggers a Service Level Credit Event for the CMS Electronic Funds Delivery Service Level?",
                "sub_answer": "Availability is 98% or less three times in any consecutive six-month period.",
                "reason": "Clause 2.9.A defines the event.",
                "support_doc_ids": ["A#p0"],
                "support_chunk_ids": ["a1"],
                "evidence_scope_label": "target_contract",
            },
            {
                "sub_question_id": "sq_2",
                "semantic_need": "fact",
                "query_text": "What are the Core Systems' Availability requirements?",
                "sub_answer": "Core Systems must maintain 99% availability during scheduled hours.",
                "reason": "Clause 2.9.B defines the requirement.",
                "support_doc_ids": ["A#p0"],
                "support_chunk_ids": ["a2"],
                "evidence_scope_label": "target_contract",
            },
        ],
        evidence_matrix=[
            {"scope_label": "target_contract"},
            {"scope_label": "target_contract"},
        ],
        answer_composition_mode="target_contract_primary",
        semantic_alignment={},
    )

    assert plan["mode"] == "compare"
    assert len(plan["facts"]) == 2
    assert plan["facts"][0]["reasoning_stage"] == "fact_presentation"
    assert any("Present each grounded fact separately" in step for step in plan["inference_steps"])


def test_build_refactor_reasoning_plan_recovers_supported_definition_from_not_found_bridge_leg() -> None:
    helpers = _extract_backend_functions(
        "_strip_question_prompt_prefix",
        "_build_refactor_reasoning_plan",
    )

    plan = helpers["_build_refactor_reasoning_plan"](
        question="What is the significance of Section 504 in relation to the definition of Privacy Regulations and how does it connect to Sensitive Customer Information under Section 10.5.2?",
        sub_question_answers=[
            {
                "sub_question_id": "sq_1",
                "semantic_need": "definition",
                "query_text": "What does Section 504 say about Privacy Regulations?",
                "sub_answer": "NOT_FOUND",
                "reason": "The provided evidence defines 'Privacy Regulations' as the regulations promulgated under Section 504 of the Gramm-Leach-Bliley Act, but it does not contain any information about what Section 504 itself says regarding those regulations.",
                "support_doc_ids": ["OFG#p0"],
                "support_chunk_ids": ["c1"],
                "evidence_scope_label": "target_contract",
            },
            {
                "sub_question_id": "sq_2",
                "semantic_need": "context",
                "query_text": "What business context does Section 10.5.2 describe in the agreement?",
                "sub_answer": "A co-branded training and education center where revenue is shared from listing fees and e-commerce.",
                "reason": "The evidence directly states Section 10.5.2 is titled CO-BRANDED TRAINING AND EDUCATION CENTER.",
                "support_doc_ids": ["Neoforma#p0"],
                "support_chunk_ids": ["c2"],
                "evidence_scope_label": "external_related",
            },
        ],
        evidence_matrix=[
            {"scope_label": "target_contract"},
            {"scope_label": "external_related"},
        ],
        answer_composition_mode="cross_document_bridge",
        semantic_alignment={"shared_concepts": ["privacy-sensitive-data"], "safe_summary": "aligned"},
    )

    assert len(plan["facts"]) == 2
    assert any(
        "Privacy Regulations are defined by reference to the regulations promulgated under Section 504" in fact["summary"]
        for fact in plan["facts"]
    )


def test_build_refactor_reasoning_plan_prefers_bridge_mode_when_requirements_indicate_bridge() -> None:
    helpers = _extract_backend_functions(
        "_strip_question_prompt_prefix",
        "_build_refactor_reasoning_plan",
    )

    plan = helpers["_build_refactor_reasoning_plan"](
        question="Under the agreement, what types of expenses incurred by Metavante are considered reimbursable from third parties, and how does this relate to Neoforma's indemnification obligations regarding third-party claims?",
        sub_question_answers=[
            {
                "sub_question_id": "sq_1",
                "semantic_need": "definition",
                "query_text": "What types of Expenses incurred by Metavante are defined in the agreement?",
                "sub_answer": "Expenses are defined as reasonable and direct expenses paid by Metavante to third parties.",
                "reason": "The evidence provides the definition.",
                "support_doc_ids": ["OFG#p0"],
                "support_chunk_ids": ["c1"],
                "evidence_scope_label": "external_related",
            },
            {
                "sub_question_id": "sq_2",
                "semantic_need": "fact",
                "query_text": "What does the agreement say about Neoforma's indemnification obligations regarding third-party claims?",
                "sub_answer": "NOT_FOUND",
                "reason": "The target contract evidence is insufficient.",
                "support_doc_ids": ["Neoforma#p0"],
                "support_chunk_ids": ["c2"],
                "evidence_scope_label": "external_related",
            },
        ],
        evidence_matrix=[
            {"scope_label": "external_related"},
            {"scope_label": "external_related"},
        ],
        answer_composition_mode="target_contract_primary",
        semantic_alignment={},
        evidence_requirements=[
            {"semantic_need": "definition", "route_reason": "bridge_requirement_split"},
            {"semantic_need": "fact", "route_reason": "bridge_requirement_split"},
        ],
    )

    assert plan["mode"] == "bridge"


def test_build_refactor_reasoning_plan_marks_external_claims_as_forbidden_target_facts() -> None:
    helpers = _extract_backend_functions(
        "_strip_question_prompt_prefix",
        "_build_refactor_reasoning_plan",
    )

    plan = helpers["_build_refactor_reasoning_plan"](
        question="How does this relate to Neoforma's indemnification obligations?",
        sub_question_answers=[
            {
                "sub_question_id": "sq_1",
                "semantic_need": "definition",
                "query_text": "What types of Expenses incurred by Metavante are defined in the agreement?",
                "sub_answer": "Expenses are defined as reasonable and direct expenses paid by Metavante to third parties.",
                "reason": "The evidence provides the definition.",
                "support_doc_ids": ["OFG#p0"],
                "observed_support_doc_ids": ["OFG#p0"],
                "support_chunk_ids": ["c1"],
                "observed_support_chunk_ids": ["c1"],
                "evidence_scope_label": "external_related",
                "grounding_status": "external_only_for_target_scope",
                "inferred_target_doc_id": "Neoforma#p0",
            },
        ],
        evidence_matrix=[
            {"scope_label": "external_related"},
        ],
        answer_composition_mode="target_contract_primary",
        semantic_alignment={},
    )

    assert plan["claims"][0]["claim_scope"] == "external_related"
    assert plan["claims"][0]["forbidden_as_target_fact"] is True
    assert plan["claims"][0]["support_doc_ids"] == ["OFG#p0"]


def test_build_refactor_reasoning_plan_does_not_promote_retrieval_docs_into_support_ownership() -> None:
    helpers = _extract_backend_functions(
        "_strip_question_prompt_prefix",
        "_build_refactor_reasoning_plan",
    )

    plan = helpers["_build_refactor_reasoning_plan"](
        question="How does this relate to Neoforma's indemnification obligations?",
        sub_question_answers=[
            {
                "sub_question_id": "sq_1",
                "semantic_need": "definition",
                "query_text": "What types of Expenses incurred by Metavante are defined in the agreement?",
                "sub_answer": "Expenses are defined as reasonable and direct expenses paid by Metavante to third parties.",
                "reason": "The evidence provides the definition.",
                "support_doc_ids": [],
                "observed_support_doc_ids": [],
                "support_chunk_ids": [],
                "observed_support_chunk_ids": [],
                "retrieval_doc_ids": ["OFG#p0"],
                "evidence_scope_label": "insufficient_support",
                "grounding_status": "insufficient_support",
                "inferred_target_doc_id": "Neoforma#p0",
            },
        ],
        evidence_matrix=[
            {"scope_label": "insufficient_support"},
        ],
        answer_composition_mode="target_contract_primary",
        semantic_alignment={},
    )

    assert plan["facts"][0]["support_doc_ids"] == []
    assert plan["claims"][0]["support_doc_ids"] == []
    assert plan["claims"][0]["claim_scope"] == "insufficient_support"


def test_select_provenance_support_from_grounding_validation_requires_grounded_chunks() -> None:
    helpers = _extract_backend_functions(
        "_select_provenance_support_from_grounding_validation",
    )

    support_chunk_ids, support_spans = helpers["_select_provenance_support_from_grounding_validation"](
        parsed_sub_answer={
            "sub_answer": "Expenses are defined as reasonable and direct expenses paid by Metavante to third parties.",
            "reason": "The evidence provides the definition.",
        },
        grounding_validation={
            "should_invalidate": False,
            "diagnostics": {
                "grounded_support_chunk_ids": [],
            },
        },
        support_pairs=[
            {"chunk_id": "c1", "text": "Chunk 1"},
            {"chunk_id": "c2", "text": "Chunk 2"},
        ],
    )

    assert support_chunk_ids == []
    assert support_spans == []


def test_split_answer_and_reasoning_surfaces_prefers_inference_section() -> None:
    helpers = _extract_backend_functions(
        "_normalize_answer_section_heading",
        "_extract_answer_sections",
        "_split_answer_and_reasoning_surfaces",
    )

    answer, reasoning, source = helpers["_split_answer_and_reasoning_surfaces"](
        "**Reasoning over Facts**\n\n"
        "**Grounded Facts**\n\n"
        "Fact block.\n\n"
        "**Inference and Conclusion:**\n\n"
        "This is the concise answer.",
    )

    assert answer == "This is the concise answer."
    assert "**Reasoning over Facts**" in reasoning
    assert source == "inference_and_conclusion"


def test_split_answer_and_reasoning_surfaces_prefers_final_answer_section() -> None:
    helpers = _extract_backend_functions(
        "_normalize_answer_section_heading",
        "_extract_answer_sections",
        "_split_answer_and_reasoning_surfaces",
    )

    answer, reasoning, source = helpers["_split_answer_and_reasoning_surfaces"](
        "Grounded Facts from the Target Contract:\n\n"
        "Fact block.\n\n"
        "Inference and Bridge Conclusion:\n\n"
        "Intermediate answer.\n\n"
        "Final Answer:\n\n"
        "Canonical concise answer."
    )

    assert answer == "Canonical concise answer."
    assert "Grounded Facts from the Target Contract" in reasoning
    assert source == "final_answer"


def test_split_answer_and_reasoning_surfaces_uses_fallback_reasoning_when_rewritten_answer_is_plain() -> None:
    helpers = _extract_backend_functions(
        "_normalize_answer_section_heading",
        "_extract_answer_sections",
        "_split_answer_and_reasoning_surfaces",
    )

    answer, reasoning, source = helpers["_split_answer_and_reasoning_surfaces"](
        "This is the concise rewritten answer.",
        fallback_reasoning_text=(
            "**Reasoning over Facts**\n\n"
            "**Bridge Conclusion:**\n\n"
            "This is the concise rewritten answer."
        ),
    )

    assert answer == "This is the concise rewritten answer."
    assert "**Reasoning over Facts**" in reasoning
    assert source == "fallback_bridge_conclusion"


def test_split_answer_and_reasoning_surfaces_returns_plain_answer_without_reasoning() -> None:
    helpers = _extract_backend_functions(
        "_normalize_answer_section_heading",
        "_extract_answer_sections",
        "_split_answer_and_reasoning_surfaces",
    )

    answer, reasoning, source = helpers["_split_answer_and_reasoning_surfaces"](
        "This is already a concise final answer."
    )

    assert answer == "This is already a concise final answer."
    assert reasoning == ""
    assert source == "plain_answer"


def test_final_answer_knowledge_package_includes_reasoning_over_facts_block() -> None:
    helpers = _extract_backend_functions(
        "_build_findings_block",
        "_format_refactor_reasoning_plan",
        "_build_supporting_relations",
        "_build_structural_evidence_summary",
        "_build_final_answer_knowledge_package",
    )

    package = helpers["_build_final_answer_knowledge_package"](
        question="How do the agreements relate?",
        subquestion_context="sq_1\nsq_2",
        sub_question_answers=[],
        backing_chunk_contents=["Chunk A"],
        triples=[],
        answer_scope_target_doc_id="Neoforma#p0",
        answer_composition_mode="cross_document_bridge",
        semantic_alignment={},
        reasoning_plan={
            "mode": "bridge",
            "facts": [
                {
                    "sub_question_id": "sq_1",
                    "scope_label": "external_related",
                    "support_doc_ids": ["OFG#p0"],
                    "summary": "OFG defines the relevant expense categories.",
                },
                {
                    "sub_question_id": "sq_2",
                    "scope_label": "target_contract",
                    "support_doc_ids": ["Neoforma#p0"],
                    "summary": "Neoforma defines the related indemnification obligations.",
                },
            ],
            "shared_concepts": ["third-party", "costs/expenses"],
            "alignment_summary": "The findings align on third-party cost responsibility.",
            "inference_steps": ["Present facts", "Preserve alignment", "Infer bridge conclusion"],
            "evidence_counts": {"target_contract": 1, "external_related": 1},
        },
    )

    assert "=== Reasoning over Facts ===" in package
    assert "mode=bridge" in package
    assert "alignment_summary=The findings align on third-party cost responsibility." in package


def test_final_answer_knowledge_package_includes_claim_grounding_ledger() -> None:
    helpers = _extract_backend_functions(
        "_build_findings_block",
        "_format_refactor_reasoning_plan",
        "_build_supporting_relations",
        "_build_structural_evidence_summary",
        "_build_final_answer_knowledge_package",
    )

    package = helpers["_build_final_answer_knowledge_package"](
        question="How does this relate to Neoforma's indemnification obligations?",
        subquestion_context="sq_1",
        sub_question_answers=[],
        backing_chunk_contents=["Chunk A"],
        triples=[],
        answer_scope_target_doc_id="Neoforma#p0",
        answer_composition_mode="target_contract_primary",
        semantic_alignment={},
        reasoning_plan={
            "mode": "fact_set",
            "facts": [],
            "claims": [
                {
                    "sub_question_id": "sq_1",
                    "claim_text": "Expenses are defined as reasonable and direct expenses paid by Metavante to third parties.",
                    "claim_scope": "external_related",
                    "support_chunk_ids": ["c1"],
                    "support_doc_ids": ["OFG#p0"],
                    "grounding_status": "external_only_for_target_scope",
                    "forbidden_as_target_fact": True,
                }
            ],
            "shared_concepts": [],
            "alignment_summary": "",
            "inference_steps": [],
            "evidence_counts": {},
        },
    )

    assert "=== Claim-Level Grounding Ledger ===" in package
    assert "forbidden_as_target_fact=True" in package
    assert "claim_scope=external_related" in package


def test_bridge_subquestions_disable_global_target_fallback_in_execution_path() -> None:
    source = (YOUTU_ROOT / "backend.py").read_text(encoding="utf-8")

    assert "target_doc_id=retrieval_target_doc_id" in source
    assert 'elif allow_original_question_target_inference and explicit_question_target_doc_id:' in source
    assert 'elif allow_original_question_target_inference and callable(infer_target_doc_id):' in source


def test_infer_scope_plan_respects_open_scope_inference_mode_for_bridge_leg() -> None:
    method = _extract_class_method(
        YOUTU_ROOT / "models" / "retriever" / "enhanced_kt_retriever.py",
        "KTRetriever",
        "_infer_scope_plan",
    )

    fake_self = types.SimpleNamespace(
        _normalize_route_type=lambda route: str(route or "").strip().lower(),
        _scope_signal_phrases=lambda question, requirement=None: ["Section 10.5.2"],
        _is_high_signal_scope_phrase=lambda text: True,
        _normalize_doc_hint=lambda text: str(text or "").strip().lower(),
        _dedupe_preserve_order=lambda values: list(dict.fromkeys(values)),
        _normalized_entity_doc_ids=defaultdict(set, {"section 10.5.2": {"OFGBANCORP#p0"}}),
    )

    scope = method(
        fake_self,
        "What does Section 10.5.2 define in the agreement?",
        retrieval_requirement={"scope_inference_mode": "open"},
        target_doc_id="",
        route_type="local",
    )

    assert scope["scope_type"] == "global_open"
    assert scope["primary_doc_id"] == ""


def test_resolve_target_doc_id_from_question_matches_contract_title_without_page_suffix() -> None:
    resolve_target_doc_id = _load_backend_resolve_target_doc_id()

    question = (
        'Contract title: "NeoformaInc_19991202_S-1A_EX-10.26_5224521_EX-10.26_Co-Branding Agreement"\n'
        "Question: test"
    )
    doc_id = resolve_target_doc_id(
        question,
        [
            "NeoformaInc_19991202_S-1A_EX-10.26_5224521_EX-10.26_Co-Branding Agreement#p0",
            "OFGBANCORP_03_28_2007-EX-10.23-OUTSOURCING AGREEMENT#p0",
        ],
    )

    assert doc_id == "NeoformaInc_19991202_S-1A_EX-10.26_5224521_EX-10.26_Co-Branding Agreement#p0"


def test_requirements_to_sub_questions_splits_0041_style_bridge_into_two_local_evidence_questions() -> None:
    namespace = _load_backend_requirements_to_sub_questions_namespace()
    namespace["question"] = (
        'Contract title: "NeoformaInc_19991202_S-1A_EX-10.26_5224521_EX-10.26_Co-Branding Agreement"\n'
        "Question: Under the agreement, what types of expenses incurred by Metavante are reimbursable from third parties, and how does this relate to Neoforma's indemnification obligations regarding third-party claims?"
    )
    requirements_to_sub_questions = namespace["_requirements_to_sub_questions"]

    sub_questions = requirements_to_sub_questions(
        [
            {
                "route_type": "structural",
                "intent": "bridge_lookup",
                "entities": ["Metavante", "Neoforma"],
                "terms": ["reimbursable expenses", "third-party claims"],
                "query_keywords": ["reimbursable from third parties", "third-party claims"],
                "target_patterns": [],
                "left_endpoint": "Metavante",
                "right_endpoint": "Neoforma",
                "bridge_relation": "indemnification obligations",
                "route_reason": "bridge",
            }
        ],
        namespace["question"],
    )

    assert len(sub_questions) == 2
    assert [item["route_type"] for item in sub_questions] == ["local", "local"]
    assert all(item.get("allow_original_question_target_inference") is False for item in sub_questions)
    assert all(
        item["retrieval_requirement"].get("allow_original_question_target_inference") is False
        for item in sub_questions
    )

    left_question = sub_questions[0]["sub-question"].lower()
    right_question = sub_questions[1]["sub-question"].lower()
    assert "metavante" in left_question
    assert "types of expenses" in left_question or "expenses incurred by metavante" in left_question
    assert sub_questions[0]["retrieval_requirement"].get("intent") == "definition_lookup"
    assert "neoforma" in right_question
    assert "indemnification" in right_question or "third-party claims" in right_question or "third party claims" in right_question


def test_requirements_to_sub_questions_bridge_prefers_explicit_claim_subject_from_question() -> None:
    namespace = _load_backend_requirements_to_sub_questions_namespace()
    namespace["question"] = (
        "Question: Under the agreement, what types of expenses incurred by Metavante are reimbursable from "
        "third parties, and how does this relate to Neoforma's indemnification obligations regarding "
        "third-party claims?"
    )
    requirements_to_sub_questions = namespace["_requirements_to_sub_questions"]

    sub_questions = requirements_to_sub_questions(
        [
            {
                "route_type": "structural",
                "intent": "bridge_lookup",
                "entities": ["Metavante", "Customer"],
                "terms": ["reimbursable expenses", "third-party claims"],
                "query_keywords": ["reimbursable from third parties", "indemnification obligations"],
                "target_patterns": [],
                "left_endpoint": "Metavante",
                "right_endpoint": "Customer",
                "bridge_relation": "indemnification obligations",
                "route_reason": "bridge",
            }
        ],
        namespace["question"],
    )

    assert len(sub_questions) == 2
    assert "metavante" in sub_questions[0]["sub-question"].lower()
    assert sub_questions[0]["retrieval_requirement"].get("intent") == "definition_lookup"
    assert "neoforma" in sub_questions[1]["sub-question"].lower()
    assert "customer's third party claims" not in sub_questions[1]["sub-question"].lower()


def test_requirements_to_sub_questions_preserves_structural_clause_bridge_when_question_has_explicit_anchors() -> None:
    namespace = _load_backend_requirements_to_sub_questions_namespace()
    namespace["question"] = (
        'Contract title: "OFGBANCORP_03_28_2007-EX-10.23-OUTSOURCING AGREEMENT"\n'
        "Question: whats the impact of breaching section 8 if customer fails to perform under section 10.3 regarding indemnification obligations?"
    )
    requirements_to_sub_questions = namespace["_requirements_to_sub_questions"]

    sub_questions = requirements_to_sub_questions(
        [
            {
                "route_type": "structural",
                "intent": "specific_bridge_lookup",
                "entities": ["Customer", "Metavante"],
                "terms": ["termination fee", "indemnification procedures"],
                "query_keywords": [],
                "target_patterns": [],
                "left_endpoint": "Section 8",
                "right_endpoint": "Section 10.3",
                "bridge_relation": "indemnification obligations",
                "route_reason": "bridge",
            }
        ],
        namespace["question"],
    )

    assert len(sub_questions) == 1
    assert sub_questions[0]["route_type"] == "structural"
    sub_question = sub_questions[0]["sub-question"].lower()
    assert "clause" in sub_question
    assert "section 8" in sub_question
    assert "section 10.3" in sub_question
    assert "allow_original_question_target_inference" not in sub_questions[0]


def test_requirements_to_sub_questions_builds_claim_oriented_local_reimbursable_question() -> None:
    namespace = _load_backend_requirements_to_sub_questions_namespace()
    namespace["question"] = "Question: what is reimbursable from third parties for Metavante?"
    requirements_to_sub_questions = namespace["_requirements_to_sub_questions"]

    sub_questions = requirements_to_sub_questions(
        [
            {
                "route_type": "local",
                "intent": "fact_lookup",
                "entities": ["Metavante"],
                "terms": ["reimbursable from third parties"],
                "query_keywords": ["reimbursable from third parties"],
                "target_patterns": [],
                "left_endpoint": "Metavante",
                "right_endpoint": "",
                "bridge_relation": "",
                "route_reason": "fact",
            }
        ],
        namespace["question"],
    )

    assert len(sub_questions) == 1
    sub_question = sub_questions[0]["sub-question"].lower()
    assert "what does the agreement say about whether metavante expenses are reimbursable from third parties" in sub_question


def test_requirements_to_sub_questions_local_fact_lookup_ignores_graphish_endpoint_subjects() -> None:
    namespace = _load_backend_requirements_to_sub_questions_namespace()
    namespace["question"] = (
        "Question: Under the agreement, what types of expenses incurred by Metavante are reimbursable from "
        "third parties, and how does this relate to Neoforma's indemnification obligations regarding "
        "third-party claims?"
    )
    requirements_to_sub_questions = namespace["_requirements_to_sub_questions"]

    sub_questions = requirements_to_sub_questions(
        [
            {
                "route_type": "local",
                "intent": "fact_lookup",
                "entities": ["Metavante", "Customer", "reimbursable"],
                "terms": ["reimbursable expenses", "services", "Customer", "Metavante"],
                "query_keywords": ["reimbursable expenses", "services provided", "Customer"],
                "target_patterns": ["which clause grants right to", "what clause states payment amount"],
                "left_endpoint": "expense_reimbursement",
                "right_endpoint": "reimbursable_amount",
                "bridge_relation": "payment_amount / grants_right_to",
                "scope": "clause",
                "route_reason": "fact",
            },
            {
                "route_type": "local",
                "intent": "fact_lookup",
                "entities": ["Neoforma", "indemnification", "third-party claims"],
                "terms": ["indemnification", "Neoforma", "third-party claims"],
                "query_keywords": ["indemnification obligations", "third-party claims", "Neoforma"],
                "target_patterns": ["which clause grants right to", "what clause survives after termination"],
                "left_endpoint": "indemnification",
                "right_endpoint": "indemnify",
                "bridge_relation": "grants_right_to / survives_for",
                "scope": "clause",
                "route_reason": "fact",
            },
        ],
        namespace["question"],
    )

    lowered = [item["sub-question"].lower() for item in sub_questions]
    assert any("metavante" in item and "expense reimbursement expenses" not in item for item in lowered)
    assert any("neoforma" in item and "indemnification expenses" not in item for item in lowered)


def test_strong_rerank_support_pairs_prefers_target_doc_anchor_coverage() -> None:
    helpers = _extract_backend_functions(
        "_normalize_prompt_text",
        "_clean_prompt_field",
        "_collect_retrieval_focus_terms",
        "_collect_structural_anchor_terms",
        "_strong_rerank_support_pairs_for_subquestion",
    )

    ranked = helpers["_strong_rerank_support_pairs_for_subquestion"](
        [
            {
                "chunk_id": "wrong-doc",
                "text": "[META] source: cuad\nSection 10.3 Indemnification Procedures. Customer shall give prompt notice of any indemnification claim.",
            },
            {
                "chunk_id": "section-8",
                "text": "[META] source: cuad\nSection 8. If Customer breaches this Agreement, Metavante may partially terminate the affected Service and Customer shall pay the termination fee.",
            },
            {
                "chunk_id": "section-10-3",
                "text": "[META] source: cuad\nSection 10.3 Indemnification Procedures. The indemnified party must promptly notify the indemnifying party and permit control of the defense.",
            },
        ],
        query_text="section 8 section 10.3",
        retrieval_queries=[
            "Section 8 breach",
            "Section 10.3 Indemnification Procedures",
            "which clause connects Section 8 and Section 10.3",
        ],
        retrieval_requirement={
            "anchors": ["Section 8", "Section 10.3"],
            "terms": ["termination fee", "indemnification procedures"],
            "left_endpoint": "Customer",
            "right_endpoint": "Metavante",
            "bridge_relation": "indemnification obligations",
            "intent": "bridge_lookup",
        },
        route_type="structural",
        target_doc_id="OFGBANCORP#p0",
        chunk_id_to_doc_id={
            "wrong-doc": "WORLDWIDESTRATEGIES#p0",
            "section-8": "OFGBANCORP#p0",
            "section-10-3": "OFGBANCORP#p0",
        },
    )

    assert [item["chunk_id"] for item in ranked[:2]] == ["section-8", "section-10-3"]


def test_strong_rerank_support_pairs_rescues_answer_scope_anchor_against_wrong_target_doc() -> None:
    helpers = _extract_backend_functions(
        "_normalize_prompt_text",
        "_clean_prompt_field",
        "_collect_retrieval_focus_terms",
        "_collect_structural_anchor_terms",
        "_strong_rerank_support_pairs_for_subquestion",
    )

    ranked = helpers["_strong_rerank_support_pairs_for_subquestion"](
        [
            {
                "chunk_id": "gocall-indemnity",
                "text": "[META] source: cuad\nThird-party claims against Customer shall be defended by Go Call under the indemnification procedures.",
            },
            {
                "chunk_id": "neo-15-4",
                "text": "[META] source: cuad\nSection 15.4 Indemnification by Neoforma. Neoforma shall indemnify the counterparty against third-party claims.",
            },
            {
                "chunk_id": "gocall-general",
                "text": "[META] source: cuad\nGo Call indemnification obligations are described generally elsewhere in the agreement.",
            },
        ],
        query_text="What does the agreement say about Neoforma's indemnification obligations regarding third-party claims?",
        retrieval_queries=[
            "Neoforma indemnification obligations third-party claims",
            "Section 15.4 Neoforma indemnification",
        ],
        retrieval_requirement={
            "anchors": ["Section 15.4"],
            "terms": ["Neoforma", "indemnification obligations", "third-party claims"],
            "intent": "bridge_lookup",
        },
        route_type="structural",
        target_doc_id="GOCALL#p0",
        answer_scope_target_doc_id="Neoforma#p0",
        chunk_id_to_doc_id={
            "gocall-indemnity": "GOCALL#p0",
            "neo-15-4": "Neoforma#p0",
            "gocall-general": "GOCALL#p0",
        },
        limit=2,
    )

    assert ranked[0]["chunk_id"] == "neo-15-4"
    assert "Neoforma#p0" in [item["doc_id"] for item in ranked[:2]]


def test_strong_rerank_support_pairs_does_not_force_low_relevance_anchor_slot() -> None:
    helpers = _extract_backend_functions(
        "_normalize_prompt_text",
        "_clean_prompt_field",
        "_collect_retrieval_focus_terms",
        "_collect_structural_anchor_terms",
        "_strong_rerank_support_pairs_for_subquestion",
    )

    ranked = helpers["_strong_rerank_support_pairs_for_subquestion"](
        [
            {
                "chunk_id": "section-8-primary",
                "text": "[META] source: cuad\nSection 8. Customer breach triggers termination fee and partial termination.",
            },
            {
                "chunk_id": "section-8-secondary",
                "text": "[META] source: cuad\nSection 8. Upon breach, the service may be terminated.",
            },
            {
                "chunk_id": "weak-section-10-3",
                "text": "10.3",
            },
        ],
        query_text="section 8 section 10.3",
        retrieval_queries=[
            "Section 8 breach",
            "Section 10.3 Indemnification Procedures",
            "which clause connects Section 8 and Section 10.3",
        ],
        retrieval_requirement={
            "anchors": ["Section 8", "Section 10.3"],
            "terms": ["termination fee", "indemnification procedures"],
            "bridge_relation": "indemnification obligations",
            "intent": "bridge_lookup",
        },
        route_type="structural",
        target_doc_id="OFGBANCORP#p0",
        chunk_id_to_doc_id={
            "section-8-primary": "OFGBANCORP#p0",
            "section-8-secondary": "OFGBANCORP#p0",
            "weak-section-10-3": "OFGBANCORP#p0",
        },
    )

    assert [item["chunk_id"] for item in ranked[:2]] == ["section-8-primary", "section-8-secondary"]


def test_strong_rerank_support_pairs_uses_exact_single_clause_mode_for_local_queries() -> None:
    helpers = _extract_backend_functions(
        "_normalize_prompt_text",
        "_clean_prompt_field",
        "_collect_retrieval_focus_terms",
        "_collect_structural_anchor_terms",
        "_strong_rerank_support_pairs_for_subquestion",
    )

    ranked = helpers["_strong_rerank_support_pairs_for_subquestion"](
        [
            {
                "chunk_id": "wrong-doc-address",
                "text": "[META] source: cuad\nGO CALL, INC. 15 Queen Street East Cambridge Ontario, Canada N3C2A7 ATTN: Ian Smith, President",
            },
            {
                "chunk_id": "target-address",
                "text": "[META] source: cuad\nGo Call, Inc. located at 15 Queen Street East, Cambridge Ontario, Canada N3C2A7.",
            },
            {
                "chunk_id": "target-name-only",
                "text": "[META] source: cuad\nPromotion Agreement between PageMaster Corporation and Go Call, Inc.",
            },
        ],
        query_text="what is Go Call Inc address",
        retrieval_queries=["what is Go Call Inc address", "which clause lists Go Call Inc address"],
        retrieval_requirement={
            "intent": "fact_lookup",
            "terms": ["address"],
            "entities": ["Go Call, Inc."],
        },
        route_type="local",
        target_doc_id="GOCALLINC#p0",
        chunk_id_to_doc_id={
            "wrong-doc-address": "METAVANTE#p0",
            "target-address": "GOCALLINC#p0",
            "target-name-only": "GOCALLINC#p0",
        },
        limit=2,
    )

    assert [item["chunk_id"] for item in ranked[:2]] == ["target-address", "target-name-only"]


def test_strong_rerank_support_pairs_prefers_expense_definition_clause_for_definition_lookup() -> None:
    helpers = _extract_backend_functions(
        "_normalize_prompt_text",
        "_clean_prompt_field",
        "_collect_retrieval_focus_terms",
        "_collect_structural_anchor_terms",
        "_strong_rerank_support_pairs_for_subquestion",
    )

    ranked = helpers["_strong_rerank_support_pairs_for_subquestion"](
        [
            {
                "chunk_id": "invoice-costs",
                "text": "[META] source: cuad\nCustomer shall, within 30 days of Metavante's invoice, pay any and all costs and expenses incurred by Metavante for such conversion efforts.",
            },
            {
                "chunk_id": "training-expenses",
                "text": "[META] source: cuad\nCustomer shall be responsible for all Expenses incurred by the participants and Metavante's training personnel.",
            },
            {
                "chunk_id": "expense-definition",
                "text": "[META] source: cuad\n\"Expenses\" shall mean any and all reasonable and direct expenses paid by Metavante to Third Parties in connection with Services provided to or on behalf of Customer under this Agreement, including any postage, supplies, materials, travel and lodging, and telecommunication fees.",
            },
        ],
        query_text="What types of Expenses incurred by Metavante are defined in the agreement?",
        retrieval_queries=[
            "What types of Expenses incurred by Metavante are defined in the agreement?",
            "what clause defines Metavante Expenses",
            "Metavante Expenses definition",
        ],
        retrieval_requirement={
            "intent": "definition_lookup",
            "terms": ["Expenses", "types of expenses", "reasonable and direct expenses"],
            "query_keywords": ["Metavante Expenses definition", "paid by Metavante to Third Parties"],
            "entities": ["Metavante"],
        },
        route_type="local",
        target_doc_id="OFGBANCORP#p0",
        chunk_id_to_doc_id={
            "invoice-costs": "OFGBANCORP#p0",
            "training-expenses": "OFGBANCORP#p0",
            "expense-definition": "OFGBANCORP#p0",
        },
        limit=2,
    )

    assert ranked[0]["chunk_id"] == "expense-definition"


def test_strong_rerank_support_pairs_prefers_promotion_description_clause_for_role_lookup() -> None:
    helpers = _extract_backend_functions(
        "_normalize_prompt_text",
        "_clean_prompt_field",
        "_collect_retrieval_focus_terms",
        "_collect_structural_anchor_terms",
        "_strong_rerank_support_pairs_for_subquestion",
    )

    ranked = helpers["_strong_rerank_support_pairs_for_subquestion"](
        [
            {
                "chunk_id": "license-clause",
                "text": "[META] source: cuad\nRepresentation and Warranties. PageMaster Corporation warrants and represents that it has a license to advertise and use the trademarks, logos, etc. of Motorola, Inc. for the promotion.",
            },
            {
                "chunk_id": "go-call-responsibilities",
                "text": "[META] source: cuad\nResponsibilities of Go Call. Go Call shall prepare and distribute advertising materials to be used for this promotion.",
            },
            {
                "chunk_id": "promotion-description",
                "text": "[META] source: cuad\n1. Description of the Promotion. PageMaster Corporation in conjunction with Go Call, shall offer free new Motorola Wordline Alphanumeric pagers with no activation fee to customers who purchase twelve months of numeric paging and airtime products and services from PageMaster Corporation.",
            },
        ],
        query_text="What is Motorola's role in the PageMaster Corporation promotion for paging services?",
        retrieval_queries=[
            "Description of the Promotion Motorola promotion",
            "which clause describes Motorola role in the promotion",
            "Motorola free new pagers no activation fee",
        ],
        retrieval_requirement={
            "intent": "fact_lookup",
            "terms": ["role", "promotion", "paging services"],
            "query_keywords": ["Motorola role", "PageMaster promotion", "paging services"],
            "entities": ["Motorola", "PageMaster Corporation"],
        },
        route_type="local",
        target_doc_id="GOCALLINC#p0",
        chunk_id_to_doc_id={
            "license-clause": "GOCALLINC#p0",
            "go-call-responsibilities": "GOCALLINC#p0",
            "promotion-description": "GOCALLINC#p0",
        },
        limit=2,
    )

    assert ranked[0]["chunk_id"] == "promotion-description"


def test_strong_rerank_support_pairs_emits_score_breakdown_diagnostics() -> None:
    helpers = _extract_backend_functions(
        "_normalize_prompt_text",
        "_clean_prompt_field",
        "_collect_retrieval_focus_terms",
        "_collect_structural_anchor_terms",
        "_strong_rerank_support_pairs_for_subquestion",
    )

    ranked = helpers["_strong_rerank_support_pairs_for_subquestion"](
        [
            {
                "chunk_id": "section-8",
                "text": "[META] source: cuad\nSection 8. Customer breach triggers termination fee and partial termination.",
            },
            {
                "chunk_id": "section-10-3",
                "text": "[META] source: cuad\nSection 10.3 Indemnification Procedures. The indemnified party must promptly notify the indemnifying party.",
            },
        ],
        query_text="section 8 section 10.3",
        retrieval_queries=[
            "Section 8 breach",
            "Section 10.3 Indemnification Procedures",
        ],
        retrieval_requirement={
            "anchors": ["Section 8", "Section 10.3"],
            "terms": ["termination fee", "indemnification procedures"],
            "bridge_relation": "indemnification obligations",
            "intent": "bridge_lookup",
        },
        route_type="structural",
        target_doc_id="OFGBANCORP#p0",
        chunk_id_to_doc_id={
            "section-8": "OFGBANCORP#p0",
            "section-10-3": "OFGBANCORP#p0",
        },
        limit=2,
    )

    assert set(ranked[0]["score_breakdown"].keys()) == {
        "prior_score",
        "doc_score",
        "term_score",
        "anchor_score",
        "bridge_score",
        "exactness_score",
        "redundancy_penalty",
        "noise_penalty",
    }
    assert isinstance(ranked[0]["score"], float)


def test_type_filtered_node_relation_retrieval_preserves_chunk_first_stage_results() -> None:
    method = _extract_class_method(
        YOUTU_ROOT / "models" / "retriever" / "enhanced_kt_retriever.py",
        "KTRetriever",
        "_type_filtered_node_relation_retrieval",
    )

    fake_self = types.SimpleNamespace(
        top_k=5,
        _filter_nodes_by_schema_type=lambda target_types: ["node_1"] if target_types else [],
        _similarity_search_on_filtered_nodes=lambda question_embed, nodes: {"top_nodes": ["node_1"]},
        _get_one_hop_triples_from_nodes=lambda nodes: [("node_1", "mentions", "node_2")],
        _hybrid_chunk_retrieval=lambda question_embed, question, top_k, target_doc_id, strict_target_doc_mode=False, scope_plan=None: {
            "chunk_ids": ["chunk_dense", "chunk_sparse"],
            "scores": [0.9, 0.8],
            "chunk_contents": ["chunk a", "chunk b"],
        },
        _extract_ordered_chunk_ids_from_nodes=lambda nodes: ["node_chunk"],
        _merge_ranked_chunk_ids=lambda *args, **kwargs: ["chunk_dense", "chunk_sparse", "node_chunk"],
        _node_relation_retrieval=lambda question_embed, question, target_doc_id=None, strict_target_doc_mode=False, scope_plan=None: {"fallback": True},
    )

    result = method(
        fake_self,
        FakeTensor([1.0, 0.0]),
        "section 2.2 supplier audits",
        {"nodes": ["clause"], "relations": [], "attributes": []},
        target_doc_id="doc_1",
    )

    assert result["path1_results"]["chunk_results"]["chunk_ids"] == ["chunk_dense", "chunk_sparse"]
    assert result["chunk_ids"] == ["chunk_dense", "chunk_sparse", "node_chunk"]


def test_get_node_chunk_id_supports_new_graph_evidence_fields() -> None:
    method = _extract_class_method(
        YOUTU_ROOT / "models" / "retriever" / "enhanced_kt_retriever.py",
        "KTRetriever",
        "_get_node_chunk_id",
    )

    fake_self = types.SimpleNamespace()

    assert method(fake_self, {"properties": {"chunk id": "legacy-chunk"}}) == "legacy-chunk"
    assert method(fake_self, {"properties": {"evidence_chunk_ids": ["entity-chunk", "backup"]}}) == "entity-chunk"
    assert method(fake_self, {"properties": {"chunk ids": ["attr-chunk", "backup"]}}) == "attr-chunk"


def test_extract_ordered_chunk_ids_from_nodes_ignores_keyword_nodes_without_chunk_warnings() -> None:
    get_chunk_id_method = _extract_class_method(
        YOUTU_ROOT / "models" / "retriever" / "enhanced_kt_retriever.py",
        "KTRetriever",
        "_get_node_chunk_id",
    )
    extract_method = _extract_class_method(
        YOUTU_ROOT / "models" / "retriever" / "enhanced_kt_retriever.py",
        "KTRetriever",
        "_extract_ordered_chunk_ids_from_nodes",
    )

    warnings: List[str] = []
    original_warning = logger_stub.warning
    logger_stub.warning = lambda message, *a, **k: warnings.append(str(message))
    try:
        fake_self = types.SimpleNamespace(
            graph=types.SimpleNamespace(
                nodes={
                    "keyword_1": {
                        "label": "keyword",
                        "properties": {"name": "motorola"},
                    },
                    "attribute_1": {
                        "label": "attribute",
                        "properties": {"name": "address: 15 Queen Street", "chunk ids": ["attr-chunk"]},
                    },
                    "entity_1": {
                        "label": "entity",
                        "properties": {"name": "Go Call, Inc.", "evidence_chunk_ids": ["entity-chunk"]},
                    },
                    "broken_1": {
                        "label": "entity",
                        "properties": {"name": "Broken Entity"},
                    },
                }
            ),
        )
        fake_self._get_node_chunk_id = lambda node_data: get_chunk_id_method(fake_self, node_data)

        chunk_ids = extract_method(fake_self, ["keyword_1", "attribute_1", "entity_1", "broken_1"])
    finally:
        logger_stub.warning = original_warning

    assert chunk_ids == ["attr-chunk", "entity-chunk"]
    assert all("keyword_1" not in message for message in warnings)
    assert any("broken_1" in message for message in warnings)


def test_merge_target_doc_candidates_strict_mode_reserves_slots_without_dropping_global_fill() -> None:
    reserved_slots_method = _extract_class_method(
        YOUTU_ROOT / "models" / "retriever" / "enhanced_kt_retriever.py",
        "KTRetriever",
        "_target_doc_reserved_slots",
    )
    merge_method = _extract_class_method(
        YOUTU_ROOT / "models" / "retriever" / "enhanced_kt_retriever.py",
        "KTRetriever",
        "_merge_target_doc_candidates",
    )
    priority_slots_method = _extract_class_method(
        YOUTU_ROOT / "models" / "retriever" / "enhanced_kt_retriever.py",
        "KTRetriever",
        "_target_doc_priority_slots",
    )

    fake_self = types.SimpleNamespace(
        strict_target_doc_mode="auto",
        _target_doc_reserved_slots=lambda top_k: reserved_slots_method(
            types.SimpleNamespace(strict_target_doc_mode="auto"),
            top_k,
        ),
        _scope_doc_slot_plan=lambda top_k, scope_plan=None, strict_target_doc_mode=False: {
            "primary_slots": 2,
            "secondary_slots": 0,
        },
        _target_doc_priority_slots=lambda top_k, target_count: priority_slots_method(
            types.SimpleNamespace(
                strict_target_doc_mode="auto",
                _target_doc_reserved_slots=lambda limit: reserved_slots_method(
                    types.SimpleNamespace(strict_target_doc_mode="auto"),
                    limit,
                ),
            ),
            top_k,
            target_count,
        ),
    )

    merged = merge_method(
        fake_self,
        [
            ("other-a", "DOC_B", 0.95),
            ("target-a", "DOC_A", 0.90),
            ("other-b", "DOC_C", 0.85),
            ("target-b", "DOC_A", 0.80),
        ],
        top_k=3,
        target_doc_id="DOC_A",
        strict_target_doc_mode=True,
    )

    assert [chunk_id for chunk_id, _ in merged] == ["target-a", "target-b", "other-a"]


def test_chunk_embedding_retrieval_strict_target_doc_mode_keeps_target_doc_priority_without_full_gating() -> None:
    method = _extract_class_method(
        YOUTU_ROOT / "models" / "retriever" / "enhanced_kt_retriever.py",
        "KTRetriever",
        "_chunk_embedding_retrieval",
    )
    reserved_slots_method = _extract_class_method(
        YOUTU_ROOT / "models" / "retriever" / "enhanced_kt_retriever.py",
        "KTRetriever",
        "_target_doc_reserved_slots",
    )
    merge_method = _extract_class_method(
        YOUTU_ROOT / "models" / "retriever" / "enhanced_kt_retriever.py",
        "KTRetriever",
        "_merge_target_doc_candidates",
    )
    priority_slots_method = _extract_class_method(
        YOUTU_ROOT / "models" / "retriever" / "enhanced_kt_retriever.py",
        "KTRetriever",
        "_target_doc_priority_slots",
    )

    class FakeIndex:
        ntotal = 3

        def search(self, _query, _k):
            return np.array([[0.9, 0.8, 0.7]], dtype=np.float32), np.array([[0, 1, 2]], dtype=np.int64)

    fake_self = types.SimpleNamespace(
        chunk_embeddings_precomputed=True,
        chunk_faiss_index=FakeIndex(),
        chunk_id_to_doc_id={"target-a": "DOC_A", "other-a": "DOC_B", "target-b": "DOC_A"},
        chunk2id={"target-a": "A", "other-a": "B", "target-b": "C"},
        _collect_global_dense_ranked_candidates=lambda question_embed, search_k: [
            ("target-a", "DOC_A", 0.9),
            ("other-a", "DOC_B", 0.8),
            ("target-b", "DOC_A", 0.7),
        ],
        _build_chunk_results_from_pairs=lambda pairs: {
            "chunk_ids": [chunk_id for chunk_id, _ in pairs],
            "scores": [score for _, score in pairs],
            "chunk_contents": [{"target-a": "A", "other-a": "B", "target-b": "C"}[chunk_id] for chunk_id, _ in pairs],
        },
        _select_diverse_global_candidates=lambda ranked_candidates, top_k: [(chunk_id, score) for chunk_id, _, score in ranked_candidates[:top_k]],
        _target_doc_reserved_slots=lambda top_k: reserved_slots_method(
            types.SimpleNamespace(strict_target_doc_mode="auto"),
            top_k,
        ),
        _merge_target_doc_candidates=lambda ranked_candidates, top_k, target_doc_id=None, strict_target_doc_mode=False, scope_plan=None: merge_method(
            types.SimpleNamespace(
                strict_target_doc_mode="auto",
                _scope_doc_slot_plan=lambda top_k, scope_plan=None, strict_target_doc_mode=False: {
                    "primary_slots": 2,
                    "secondary_slots": 0,
                },
                _target_doc_reserved_slots=lambda limit: reserved_slots_method(
                    types.SimpleNamespace(strict_target_doc_mode="auto"),
                    limit,
                ),
                _target_doc_priority_slots=lambda limit, target_count: priority_slots_method(
                    types.SimpleNamespace(
                        strict_target_doc_mode="auto",
                        _target_doc_reserved_slots=lambda inner_limit: reserved_slots_method(
                            types.SimpleNamespace(strict_target_doc_mode="auto"),
                            inner_limit,
                        ),
                    ),
                    limit,
                    target_count,
                ),
            ),
            ranked_candidates,
            top_k=top_k,
            target_doc_id=target_doc_id,
            strict_target_doc_mode=strict_target_doc_mode,
            scope_plan=scope_plan,
        ),
    )

    result = method(
        fake_self,
        FakeTensor([1.0, 0.0]),
        top_k=2,
        target_doc_id="DOC_A",
        strict_target_doc_mode=True,
        scope_context={"scope_decision": "open", "probe_trace": {}},
    )

    assert result["chunk_ids"] == ["target-a", "target-b"]


def test_chunk_embedding_retrieval_strict_target_doc_mode_preserves_global_fill_when_target_doc_is_sparse() -> None:
    method = _extract_class_method(
        YOUTU_ROOT / "models" / "retriever" / "enhanced_kt_retriever.py",
        "KTRetriever",
        "_chunk_embedding_retrieval",
    )
    reserved_slots_method = _extract_class_method(
        YOUTU_ROOT / "models" / "retriever" / "enhanced_kt_retriever.py",
        "KTRetriever",
        "_target_doc_reserved_slots",
    )
    merge_method = _extract_class_method(
        YOUTU_ROOT / "models" / "retriever" / "enhanced_kt_retriever.py",
        "KTRetriever",
        "_merge_target_doc_candidates",
    )
    priority_slots_method = _extract_class_method(
        YOUTU_ROOT / "models" / "retriever" / "enhanced_kt_retriever.py",
        "KTRetriever",
        "_target_doc_priority_slots",
    )

    class FakeIndex:
        ntotal = 4

        def search(self, _query, _k):
            return np.array([[0.95, 0.90, 0.85, 0.80]], dtype=np.float32), np.array([[0, 1, 2, 3]], dtype=np.int64)

    fake_self = types.SimpleNamespace(
        chunk_embeddings_precomputed=True,
        chunk_faiss_index=FakeIndex(),
        chunk_id_to_doc_id={"other-a": "DOC_B", "target-a": "DOC_A", "other-b": "DOC_C", "other-c": "DOC_D"},
        chunk2id={"other-a": "A", "target-a": "B", "other-b": "C", "other-c": "D"},
        _collect_global_dense_ranked_candidates=lambda question_embed, search_k: [
            ("other-a", "DOC_B", 0.95),
            ("target-a", "DOC_A", 0.90),
            ("other-b", "DOC_C", 0.85),
            ("other-c", "DOC_D", 0.80),
        ],
        _build_chunk_results_from_pairs=lambda pairs: {
            "chunk_ids": [chunk_id for chunk_id, _ in pairs],
            "scores": [score for _, score in pairs],
            "chunk_contents": [{"other-a": "A", "target-a": "B", "other-b": "C", "other-c": "D"}[chunk_id] for chunk_id, _ in pairs],
        },
        _select_diverse_global_candidates=lambda ranked_candidates, top_k: [(chunk_id, score) for chunk_id, _, score in ranked_candidates[:top_k]],
        _target_doc_reserved_slots=lambda top_k: reserved_slots_method(
            types.SimpleNamespace(strict_target_doc_mode="auto"),
            top_k,
        ),
        _merge_target_doc_candidates=lambda ranked_candidates, top_k, target_doc_id=None, strict_target_doc_mode=False, scope_plan=None: merge_method(
            types.SimpleNamespace(
                strict_target_doc_mode="auto",
                _scope_doc_slot_plan=lambda top_k, scope_plan=None, strict_target_doc_mode=False: {
                    "primary_slots": 1,
                    "secondary_slots": 0,
                },
                _target_doc_reserved_slots=lambda limit: reserved_slots_method(
                    types.SimpleNamespace(strict_target_doc_mode="auto"),
                    limit,
                ),
                _target_doc_priority_slots=lambda limit, target_count: priority_slots_method(
                    types.SimpleNamespace(
                        strict_target_doc_mode="auto",
                        _target_doc_reserved_slots=lambda inner_limit: reserved_slots_method(
                            types.SimpleNamespace(strict_target_doc_mode="auto"),
                            inner_limit,
                        ),
                    ),
                    limit,
                    target_count,
                ),
            ),
            ranked_candidates,
            top_k=top_k,
            target_doc_id=target_doc_id,
            strict_target_doc_mode=strict_target_doc_mode,
            scope_plan=scope_plan,
        ),
    )

    result = method(
        fake_self,
        FakeTensor([1.0, 0.0]),
        top_k=3,
        target_doc_id="DOC_A",
        strict_target_doc_mode=True,
        scope_context={"scope_decision": "open", "probe_trace": {}},
    )

    assert result["chunk_ids"] == ["target-a", "other-a", "other-b"]


def test_should_use_strict_target_doc_mode_respects_off_and_force_modes() -> None:
    method = _extract_class_method(
        YOUTU_ROOT / "models" / "retriever" / "enhanced_kt_retriever.py",
        "KTRetriever",
        "_should_use_strict_target_doc_mode",
    )

    fake_self_off = types.SimpleNamespace(
        strict_target_doc_mode="off",
        _normalized_doc_ids={"DOC_A": "contracttitledoca"},
        _normalize_doc_hint=lambda text: re.sub(r"[^a-z0-9]+", "", text.lower()),
    )
    assert method(
        fake_self_off,
        'Contract title: "DOC_A"',
        None,
        "DOC_A",
    ) is False

    fake_self_force = types.SimpleNamespace(
        strict_target_doc_mode="force",
        _normalized_doc_ids={"DOC_A": "contracttitledoca"},
        _normalize_doc_hint=lambda text: re.sub(r"[^a-z0-9]+", "", text.lower()),
    )
    assert method(
        fake_self_force,
        "ambiguous question",
        None,
        "DOC_A",
    ) is True


def test_infer_scope_plan_detects_cross_doc_bridge_from_high_signal_entities() -> None:
    method = _extract_class_method(
        YOUTU_ROOT / "models" / "retriever" / "enhanced_kt_retriever.py",
        "KTRetriever",
        "_infer_scope_plan",
    )

    fake_self = types.SimpleNamespace(
        _normalize_route_type=lambda route: route or "structural",
        _scope_signal_phrases=lambda question, retrieval_requirement=None: [
            "Metavante",
            "Neoforma",
            "third-party claims",
        ],
        _is_high_signal_scope_phrase=lambda phrase: phrase.lower() != "third-party claims",
        _normalize_doc_hint=lambda text: re.sub(r"[^a-z0-9]+", "", str(text or "").lower()),
        _normalized_entity_doc_ids={
            "metavante": {"OFGBANCORP#p0"},
            "neoforma": {"Neoforma#p0"},
        },
        _dedupe_preserve_order=lambda items: list(dict.fromkeys(items)),
    )

    scope_plan = method(
        fake_self,
        "Question about Metavante and Neoforma",
        retrieval_requirement={"entities": ["Metavante", "Neoforma"]},
        target_doc_id="Neoforma#p0",
        route_type="structural",
    )

    assert scope_plan["scope_type"] == "cross_doc_bridge"
    assert scope_plan["primary_doc_id"] == "Neoforma#p0"
    assert scope_plan["secondary_doc_ids"] == ["OFGBANCORP#p0"]


def test_infer_scope_plan_without_explicit_target_doc_id_promotes_first_matched_doc_and_bridge_scope() -> None:
    method = _extract_class_method(
        YOUTU_ROOT / "models" / "retriever" / "enhanced_kt_retriever.py",
        "KTRetriever",
        "_infer_scope_plan",
    )

    fake_self = types.SimpleNamespace(
        _normalize_route_type=lambda route: route or "structural",
        _scope_signal_phrases=lambda question, retrieval_requirement=None: [
            "Metavante",
            "Neoforma",
            "third-party claims",
        ],
        _is_high_signal_scope_phrase=lambda phrase: phrase.lower() != "third-party claims",
        _normalize_doc_hint=lambda text: re.sub(r"[^a-z0-9]+", "", str(text or "").lower()),
        _normalized_entity_doc_ids={
            "metavante": {"OFGBANCORP#p0"},
            "neoforma": {"Neoforma#p0"},
        },
        _dedupe_preserve_order=lambda items: list(dict.fromkeys(items)),
    )

    scope_plan = method(
        fake_self,
        "Question about Metavante and Neoforma",
        retrieval_requirement={"entities": ["Metavante", "Neoforma"]},
        target_doc_id="",
        route_type="structural",
    )

    assert scope_plan["scope_type"] == "cross_doc_bridge"
    assert scope_plan["primary_doc_id"] == "OFGBANCORP#p0"
    assert scope_plan["secondary_doc_ids"] == ["Neoforma#p0"]


def test_strong_rerank_support_pairs_cross_doc_bridge_keeps_secondary_doc_competitive() -> None:
    helpers = _extract_backend_functions(
        "_normalize_prompt_text",
        "_clean_prompt_field",
        "_collect_retrieval_focus_terms",
        "_collect_structural_anchor_terms",
        "_strong_rerank_support_pairs_for_subquestion",
    )

    ranked = helpers["_strong_rerank_support_pairs_for_subquestion"](
        [
            {
                "chunk_id": "primary",
                "text": "[META] source: cuad\nNeoforma shall indemnify Customer against third-party claims.",
            },
            {
                "chunk_id": "secondary",
                "text": "[META] source: cuad\nExpenses incurred by Metavante in connection with services are reimbursable by Customer.",
            },
            {
                "chunk_id": "other",
                "text": "[META] source: cuad\nUnrelated indemnification clause from another agreement.",
            },
        ],
        query_text="Metavante reimbursable from third parties and Neoforma third-party claims",
        retrieval_queries=[
            "Metavante reimbursable from third parties",
            "Neoforma third-party claims",
        ],
        retrieval_requirement={
            "intent": "bridge_lookup",
            "terms": ["reimbursable from third parties", "third-party claims"],
        },
        route_type="structural",
        target_doc_id="Neoforma#p0",
        scope_type="cross_doc_bridge",
        secondary_doc_ids=["OFGBANCORP#p0"],
        chunk_id_to_doc_id={
            "primary": "Neoforma#p0",
            "secondary": "OFGBANCORP#p0",
            "other": "OTHER#p0",
        },
        limit=3,
    )

    ranked_ids = [item["chunk_id"] for item in ranked[:2]]
    assert "primary" in ranked_ids
    assert "secondary" in ranked_ids


def test_infer_target_doc_id_matches_question_contract_title_without_page_suffix() -> None:
    method = _extract_class_method(
        YOUTU_ROOT / "models" / "retriever" / "enhanced_kt_retriever.py",
        "KTRetriever",
        "_infer_target_doc_id_from_question",
    )

    fake_self = types.SimpleNamespace(
        _normalize_doc_hint=lambda text: re.sub(r"[^a-z0-9]+", "", str(text or "").lower()),
        _normalized_doc_ids={
            "NeoformaInc_19991202_S-1A_EX-10.26_5224521_EX-10.26_Co-Branding Agreement#p0":
                "neoformainc19991202s1aex10265224521ex1026cobrandingagreement",
        },
    )

    inferred = method(
        fake_self,
        'Contract title: "NeoformaInc_19991202_S-1A_EX-10.26_5224521_EX-10.26_Co-Branding Agreement"',
    )

    assert inferred == "NeoformaInc_19991202_S-1A_EX-10.26_5224521_EX-10.26_Co-Branding Agreement#p0"


def test_retrieve_by_route_prefers_explicit_target_doc_id_over_inference() -> None:
    method = _extract_class_method(
        YOUTU_ROOT / "models" / "retriever" / "enhanced_kt_retriever.py",
        "KTRetriever",
        "retrieve_by_route",
    )

    seen: Dict[str, Any] = {}

    def _node_relation_retrieval(question_embed, question, target_doc_id=None, strict_target_doc_mode=False, scope_plan=None):
        seen["node_target_doc_id"] = target_doc_id
        seen["scope_plan"] = scope_plan
        return {
            "top_nodes": [],
            "one_hop_triples": [],
            "chunk_results": {"chunk_ids": [], "scores": [], "chunk_contents": []},
        }

    fake_self = types.SimpleNamespace(
        default_route="structural",
        top_k=5,
        use_edge_layer=False,
        _active_edge_layers=None,
        _normalize_route_type=lambda route: route,
        _normalize_involved_types=lambda involved: involved or {},
        _get_query_embedding=lambda question: FakeTensor([1.0, 0.0]),
        _resolve_route_top_k=lambda route, top_k=None: top_k or 5,
        _infer_target_doc_id_from_question=lambda question: "DOC_INFERRED",
        _infer_scope_plan=lambda question, retrieval_requirement=None, target_doc_id=None, route_type=None: {
            "scope_type": "single_doc",
            "primary_doc_id": target_doc_id,
            "secondary_doc_ids": [],
        },
        _should_use_strict_target_doc_mode=lambda question, original_question=None, target_doc_id=None: seen.setdefault("strict_target_doc_id", target_doc_id) == "DOC_EXPLICIT",
        _node_relation_retrieval=_node_relation_retrieval,
        _triple_only_retrieval=lambda question_embed: {"scored_triples": []},
        _extract_ordered_chunk_ids_from_nodes=lambda nodes: [],
        _merge_ranked_chunk_ids=lambda *args, **kwargs: [],
        _ensure_retrieval_result_shape=lambda results: results,
    )

    _, results = method(
        fake_self,
        "q",
        route_type="local",
        involved_types={},
        top_k=5,
        original_question="orig",
        target_doc_id="DOC_EXPLICIT",
    )

    assert seen["node_target_doc_id"] == "DOC_EXPLICIT"
    assert seen["strict_target_doc_id"] == "DOC_EXPLICIT"
    assert seen["scope_plan"]["scope_type"] == "single_doc"
    assert results["route_type"] == "local"


def test_retrieve_by_route_backfills_target_doc_id_from_scope_plan_primary_doc() -> None:
    method = _extract_class_method(
        YOUTU_ROOT / "models" / "retriever" / "enhanced_kt_retriever.py",
        "KTRetriever",
        "retrieve_by_route",
    )

    seen: Dict[str, Any] = {}

    def _node_relation_retrieval(question_embed, question, target_doc_id=None, strict_target_doc_mode=False, scope_plan=None):
        seen["node_target_doc_id"] = target_doc_id
        seen["scope_plan"] = scope_plan
        return {
            "top_nodes": [],
            "one_hop_triples": [],
            "chunk_results": {"chunk_ids": [], "scores": [], "chunk_contents": []},
        }

    fake_self = types.SimpleNamespace(
        default_route="local",
        top_k=5,
        use_edge_layer=False,
        _active_edge_layers=None,
        _normalize_route_type=lambda route: route,
        _normalize_involved_types=lambda involved: involved or {},
        _get_query_embedding=lambda question: FakeTensor([1.0, 0.0]),
        _resolve_route_top_k=lambda route, top_k=None: top_k or 5,
        _infer_target_doc_id_from_question=lambda question: "",
        _infer_scope_plan=lambda question, retrieval_requirement=None, target_doc_id=None, route_type=None: {
            "scope_type": "single_doc",
            "primary_doc_id": "DOC_SCOPE",
            "secondary_doc_ids": [],
        },
        _should_use_strict_target_doc_mode=lambda question, original_question=None, target_doc_id=None: seen.setdefault("strict_target_doc_id", target_doc_id) == "DOC_SCOPE",
        _node_relation_retrieval=_node_relation_retrieval,
        _triple_only_retrieval=lambda question_embed: {"scored_triples": []},
        _extract_ordered_chunk_ids_from_nodes=lambda nodes: [],
        _merge_ranked_chunk_ids=lambda *args, **kwargs: [],
        _ensure_retrieval_result_shape=lambda results: results,
    )

    _, results = method(
        fake_self,
        "q",
        route_type="local",
        involved_types={},
        top_k=5,
        original_question="orig",
        target_doc_id="",
        retrieval_requirement={"allow_original_question_target_inference": False},
    )

    assert seen["node_target_doc_id"] == "DOC_SCOPE"
    assert seen["strict_target_doc_id"] == "DOC_SCOPE"
    assert seen["scope_plan"]["primary_doc_id"] == "DOC_SCOPE"
    assert results["route_type"] == "local"


def test_process_retrieval_results_preserves_explicit_target_doc_id_for_chunk_processing() -> None:
    method = _extract_class_method(
        YOUTU_ROOT / "models" / "retriever" / "enhanced_kt_retriever.py",
        "KTRetriever",
        "process_retrieval_results",
    )

    seen: Dict[str, Any] = {}
    def _process_chunk_results(chunk_results, question_embed, top_k, target_doc_id=None, anchor_terms=None, scope_plan=None):
        seen["target_doc_id"] = target_doc_id
        seen["scope_plan"] = scope_plan
        return [], ["c1"]

    fake_self = types.SimpleNamespace(
        _normalize_involved_types=lambda involved: involved or {},
        _infer_scope_plan=lambda question, retrieval_requirement=None, target_doc_id=None, route_type=None: {
            "scope_type": "single_doc",
            "primary_doc_id": target_doc_id,
            "secondary_doc_ids": [],
        },
        retrieve_by_route=lambda question, route_type, involved_types=None, top_k=None, original_question=None, target_doc_id=None, retrieval_requirement=None, scope_plan=None: (
            FakeTensor([1.0, 0.0]),
            {
                "path1_results": {
                    "chunk_results": {"chunk_ids": ["c1"], "scores": [0.9], "chunk_contents": ["chunk"]},
                    "one_hop_triples": [],
                },
                "path2_results": {"scored_triples": []},
                "chunk_ids": ["c1"],
                "route_type": route_type,
                "route_fallback": "",
                "scope_plan": scope_plan or {},
                "target_doc_id": target_doc_id,
            },
        ),
        _ensure_retrieval_result_shape=lambda results: results,
        _infer_target_doc_id_from_question=lambda question: "DOC_INFERRED",
        _build_query_anchor_terms=lambda question, original_question=None: [],
        _process_chunk_results=_process_chunk_results,
        _collect_all_scored_triples=lambda results, question_embed, anchor_terms=None, target_doc_id=None, scope_plan=None: [],
        _format_scored_triples=lambda triples: [],
        _extract_chunk_ids_from_triples=lambda triples: [],
        _infer_preferred_doc_ids_from_chunks=lambda chunk_ids, target_doc_id=None: [target_doc_id] if target_doc_id else [],
        _apply_doc_consistency_to_chunk_order=lambda chunk_ids, preferred_doc_ids=None: chunk_ids,
        _get_matching_chunks=lambda chunk_ids: ["chunk"],
        _housekeep_runtime_caches=lambda: None,
        chunk2id={"c1": "chunk"},
    )

    method(
        fake_self,
        "q",
        top_k=5,
        involved_types={},
        route_type="local",
        original_question="orig",
        target_doc_id="DOC_EXPLICIT",
    )

    assert seen["target_doc_id"] == "DOC_EXPLICIT"


def test_process_retrieval_results_uses_scope_primary_doc_when_parent_target_inference_is_disabled() -> None:
    method = _extract_class_method(
        YOUTU_ROOT / "models" / "retriever" / "enhanced_kt_retriever.py",
        "KTRetriever",
        "process_retrieval_results",
    )

    seen: Dict[str, Any] = {}

    def _process_chunk_results(chunk_results, question_embed, top_k, target_doc_id=None, anchor_terms=None, scope_plan=None):
        seen["chunk_target_doc_id"] = target_doc_id
        seen["chunk_scope_plan"] = scope_plan
        return [], ["c1"]

    def _infer_scope_plan(question, retrieval_requirement=None, target_doc_id=None, route_type=None):
        seen["scope_input_target_doc_id"] = target_doc_id
        return {
            "scope_type": "single_doc",
            "primary_doc_id": "DOC_SCOPE",
            "secondary_doc_ids": [],
        }

    def _retrieve_by_route(question, route_type, involved_types=None, top_k=None, original_question=None, target_doc_id=None, retrieval_requirement=None, scope_plan=None):
        seen["retrieve_target_doc_id"] = target_doc_id
        seen["retrieve_scope_plan"] = scope_plan
        return (
            FakeTensor([1.0, 0.0]),
            {
                "path1_results": {
                    "chunk_results": {"chunk_ids": ["c1"], "scores": [0.9], "chunk_contents": ["chunk"]},
                    "one_hop_triples": [],
                },
                "path2_results": {"scored_triples": []},
                "chunk_ids": ["c1"],
                "route_type": route_type,
                "route_fallback": "",
                "scope_plan": scope_plan or {},
                "target_doc_id": target_doc_id,
            },
        )

    fake_self = types.SimpleNamespace(
        _normalize_involved_types=lambda involved: involved or {},
        _infer_scope_plan=_infer_scope_plan,
        retrieve_by_route=_retrieve_by_route,
        _ensure_retrieval_result_shape=lambda results: results,
        _infer_target_doc_id_from_question=lambda question: "DOC_FROM_ORIGINAL" if "Contract title" in str(question or "") else "",
        _build_query_anchor_terms=lambda question, original_question=None: [],
        _process_chunk_results=_process_chunk_results,
        _collect_all_scored_triples=lambda results, question_embed, anchor_terms=None, target_doc_id=None, scope_plan=None: [],
        _format_scored_triples=lambda triples: [],
        _extract_chunk_ids_from_triples=lambda triples: [],
        _infer_preferred_doc_ids_from_chunks=lambda chunk_ids, target_doc_id=None: [target_doc_id] if target_doc_id else [],
        _apply_doc_consistency_to_chunk_order=lambda chunk_ids, preferred_doc_ids=None: chunk_ids,
        _get_matching_chunks=lambda chunk_ids: ["chunk"],
        _housekeep_runtime_caches=lambda: None,
        chunk2id={"c1": "chunk"},
    )

    method(
        fake_self,
        "q",
        top_k=5,
        involved_types={},
        route_type="local",
        original_question='Contract title: "Neoforma Contract"\nQuestion: q',
        target_doc_id="",
        retrieval_requirement={"allow_original_question_target_inference": False},
    )

    assert seen["scope_input_target_doc_id"] == ""
    assert seen["retrieve_target_doc_id"] == "DOC_SCOPE"
    assert seen["chunk_target_doc_id"] == "DOC_SCOPE"


def test_hybrid_type_filtered_retrieval_merges_path1_chunk_results_first() -> None:
    method = _extract_class_method(
        YOUTU_ROOT / "models" / "retriever" / "enhanced_kt_retriever.py",
        "KTRetriever",
        "_hybrid_type_filtered_retrieval",
    )

    fake_self = types.SimpleNamespace(
        _filter_nodes_by_schema_type=lambda target_types: ["node_1"] if target_types else [],
        _type_filtered_node_relation_path=lambda question_embed, question, filtered_nodes, target_doc_id=None, strict_target_doc_mode=False, scope_plan=None: {
            "top_nodes": ["node_1"],
            "one_hop_triples": [("node_1", "related_to", "node_2")],
            "chunk_results": {"chunk_ids": ["chunk_path1"], "scores": [0.8], "chunk_contents": ["path1 chunk"]},
        },
        _node_relation_retrieval=lambda question_embed, question, target_doc_id=None, strict_target_doc_mode=False, scope_plan=None: {
            "top_nodes": [],
            "one_hop_triples": [],
            "chunk_results": {"chunk_ids": [], "scores": [], "chunk_contents": []},
        },
        _triple_only_retrieval=lambda question_embed: {
            "scored_triples": [("h", "rel", "t", 0.9)],
            "comm_nodes": ["comm_1"],
        },
        _extract_ordered_chunk_ids_from_nodes=lambda nodes: ["node_chunk"] if nodes == ["node_1"] else ["comm_chunk"],
        _extract_ordered_chunk_ids_from_triple_nodes=lambda triples: ["triple_chunk"],
        _merge_ranked_chunk_ids=lambda *args, **kwargs: [cid for group in args for cid in group],
    )

    result = method(
        fake_self,
        FakeTensor([1.0, 0.0]),
        "section 10.3 indemnification procedures",
        {"nodes": ["clause"], "relations": [], "attributes": []},
        target_doc_id="doc_1",
    )

    assert result["path1_results"]["chunk_results"]["chunk_ids"] == ["chunk_path1"]
    assert result["chunk_ids"][:4] == ["chunk_path1", "node_chunk", "triple_chunk", "comm_chunk"]


def test_build_indices_precomputes_chunk_embeddings() -> None:
    method = _extract_class_method(
        YOUTU_ROOT / "models" / "retriever" / "enhanced_kt_retriever.py",
        "KTRetriever",
        "build_indices",
    )

    calls: List[str] = []
    fake_self = types.SimpleNamespace(
        faiss_retriever=types.SimpleNamespace(build_indices=lambda: calls.append("faiss")),
        _precompute_sparse_chunk_index=lambda: calls.append("sparse"),
        _precompute_chunk_embeddings=lambda: calls.append("chunk"),
        _precompute_node_embeddings=lambda: calls.append("node"),
    )

    method(fake_self)

    assert calls == ["faiss", "sparse", "chunk", "node"]


def test_merge_ranked_chunk_ids_preserves_cross_path_signal() -> None:
    merge_ranked_chunk_ids = _extract_class_method(
        YOUTU_ROOT / "models" / "retriever" / "enhanced_kt_retriever.py",
        "KTRetriever",
        "_merge_ranked_chunk_ids",
    )

    merged = merge_ranked_chunk_ids(
        object(),
        ["chunk_a", "chunk_b", "chunk_c"],
        ["chunk_c", "chunk_a"],
        ["chunk_d"],
        top_k=3,
    )

    assert merged == ["chunk_a", "chunk_c", "chunk_b"]


def test_chunk_embedding_retrieval_keeps_global_score_order_before_doc_diversity() -> None:
    chunk_embedding_retrieval = _extract_class_method(
        YOUTU_ROOT / "models" / "retriever" / "enhanced_kt_retriever.py",
        "KTRetriever",
        "_chunk_embedding_retrieval",
    )

    class FakeIndex:
        ntotal = 4

        def search(self, _query_embed_np, _search_k):
            return (
                np.array([[0.95, 0.94, 0.93, 0.92]], dtype="float32"),
                np.array([[0, 1, 2, 3]], dtype="int64"),
            )

    fake_self = types.SimpleNamespace(
        chunk_embeddings_precomputed=True,
        chunk_faiss_index=FakeIndex(),
        chunk_id_to_doc_id={
            "chunk_z1": "doc_z",
            "chunk_z2": "doc_z",
            "chunk_a1": "doc_a",
            "chunk_b1": "doc_b",
        },
        chunk2id={
            "chunk_z1": "doc_z top chunk",
            "chunk_z2": "doc_z second chunk",
            "chunk_a1": "doc_a top chunk",
            "chunk_b1": "doc_b top chunk",
        },
        _collect_global_dense_ranked_candidates=lambda question_embed, search_k: [
            ("chunk_z1", "doc_z", 0.95),
            ("chunk_z2", "doc_z", 0.94),
            ("chunk_a1", "doc_a", 0.93),
            ("chunk_b1", "doc_b", 0.92),
        ],
        _build_chunk_results_from_pairs=lambda pairs: {
            "chunk_ids": [chunk_id for chunk_id, _ in pairs],
            "scores": [score for _, score in pairs],
            "chunk_contents": [
                {
                    "chunk_z1": "doc_z top chunk",
                    "chunk_z2": "doc_z second chunk",
                    "chunk_a1": "doc_a top chunk",
                    "chunk_b1": "doc_b top chunk",
                }[chunk_id]
                for chunk_id, _ in pairs
            ],
        },
        _select_diverse_global_candidates=lambda ranked_candidates, top_k: [
            ("chunk_z1", 0.95),
            ("chunk_a1", 0.93),
            ("chunk_b1", 0.92),
        ][:top_k],
    )

    result = chunk_embedding_retrieval(
        fake_self,
        FakeTensor([1.0, 0.0]),
        top_k=3,
        target_doc_id=None,
        scope_context={"scope_decision": "open", "probe_trace": {}},
    )

    assert result["chunk_ids"] == ["chunk_z1", "chunk_a1", "chunk_b1"]
    assert [round(score, 2) for score in result["scores"]] == [0.95, 0.93, 0.92]


def test_fuse_chunk_results_rrf_combines_dense_and_sparse_hits() -> None:
    fuse_chunk_results_rrf = _extract_class_method(
        YOUTU_ROOT / "models" / "retriever" / "enhanced_kt_retriever.py",
        "KTRetriever",
        "_fuse_chunk_results_rrf",
    )

    fake_self = types.SimpleNamespace(
        hybrid_rrf_k=60,
        chunk2id={
            "dense_only": "dense only chunk",
            "shared": "shared chunk",
            "sparse_only": "sparse only chunk",
        },
    )

    fused = fuse_chunk_results_rrf(
        fake_self,
        {
            "chunk_ids": ["dense_only", "shared"],
            "scores": [0.95, 0.90],
            "chunk_contents": ["dense only chunk", "shared chunk"],
        },
        {
            "chunk_ids": ["shared", "sparse_only"],
            "scores": [9.0, 8.0],
            "chunk_contents": ["shared chunk", "sparse only chunk"],
        },
        top_k=3,
    )

    assert fused["chunk_ids"] == ["shared", "dense_only", "sparse_only"]
    assert fused["hybrid_sources"]["shared"] == ["dense", "sparse"]
    assert fused["hybrid_sources"]["dense_only"] == ["dense"]
    assert fused["hybrid_sources"]["sparse_only"] == ["sparse"]


def test_hybrid_chunk_retrieval_falls_back_to_dense_when_sparse_empty() -> None:
    hybrid_chunk_retrieval = _extract_class_method(
        YOUTU_ROOT / "models" / "retriever" / "enhanced_kt_retriever.py",
        "KTRetriever",
        "_hybrid_chunk_retrieval",
    )

    dense_result = {
        "chunk_ids": ["dense_a", "dense_b"],
        "scores": [0.9, 0.8],
        "chunk_contents": ["A", "B"],
    }
    fake_self = types.SimpleNamespace(
        hybrid_chunk_retrieval_enabled=True,
        hybrid_sparse_search_multiplier=4,
        _resolve_scope_retrieval_context=lambda *a, **k: {"scope_decision": "open", "probe_trace": {}},
        _chunk_embedding_retrieval=lambda question_embed, top_k, target_doc_id=None, strict_target_doc_mode=False, scope_plan=None, scope_context=None: dense_result,
        _sparse_chunk_retrieval=lambda question, top_k, target_doc_id=None, strict_target_doc_mode=False, scope_plan=None, scope_context=None: {
            "chunk_ids": [],
            "scores": [],
            "chunk_contents": [],
        },
        _fuse_chunk_results_rrf=lambda *a, **k: {"chunk_ids": ["should_not_happen"]},
    )

    result = hybrid_chunk_retrieval(
        fake_self,
        FakeTensor([1.0, 0.0]),
        "Section 504 Privacy Regulations",
        top_k=5,
        target_doc_id=None,
    )

    assert result == dense_result


def test_chunk_embedding_retrieval_scope_decision_strict_uses_scoped_candidates() -> None:
    method = _extract_class_method(
        YOUTU_ROOT / "models" / "retriever" / "enhanced_kt_retriever.py",
        "KTRetriever",
        "_chunk_embedding_retrieval",
    )

    fake_self = types.SimpleNamespace(
        chunk_embeddings_precomputed=True,
        chunk_faiss_index=types.SimpleNamespace(ntotal=4),
        chunk2id={
            "noise-a": "noise a",
            "noise-b": "noise b",
            "target-a": "target a",
            "target-b": "target b",
        },
        _collect_global_dense_ranked_candidates=lambda question_embed, search_k: [
            ("noise-a", "DOC_B", 0.99),
            ("noise-b", "DOC_C", 0.98),
            ("target-a", "DOC_A", 0.50),
            ("target-b", "DOC_A", 0.49),
        ],
        _collect_scoped_dense_ranked_candidates=lambda question_embed, top_k, target_doc_id=None, scope_plan=None, scope_decision="open": [
            ("target-a", "DOC_A", 0.50),
            ("target-b", "DOC_A", 0.49),
        ],
        _build_chunk_results_from_pairs=lambda pairs: {
            "chunk_ids": [chunk_id for chunk_id, _ in pairs],
            "scores": [score for _, score in pairs],
            "chunk_contents": [{"target-a": "target a", "target-b": "target b"}[chunk_id] for chunk_id, _ in pairs],
        },
        _merge_doc_scoped_results=lambda *a, **k: [],
        _merge_target_doc_candidates=lambda *a, **k: [],
        _select_diverse_global_candidates=lambda *a, **k: [],
    )

    result = method(
        fake_self,
        FakeTensor([1.0, 0.0]),
        top_k=2,
        target_doc_id="DOC_A",
        strict_target_doc_mode=True,
        scope_context={"scope_decision": "strict", "probe_trace": {"passed": True}},
    )

    assert result["chunk_ids"] == ["target-a", "target-b"]
    assert result["scope_decision"] == "strict"
    assert result["scoped_chunk_ids"] == ["target-a", "target-b"]


def test_sparse_chunk_retrieval_scope_decision_strict_masks_allowed_docs() -> None:
    method = _extract_class_method(
        YOUTU_ROOT / "models" / "retriever" / "enhanced_kt_retriever.py",
        "KTRetriever",
        "_sparse_chunk_retrieval",
    )

    fake_self = types.SimpleNamespace(
        _collect_sparse_ranked_candidates=lambda question, allowed_doc_ids=None: (
            [
                ("target-a", "DOC_A", 8.0),
                ("target-b", "DOC_A", 7.0),
            ]
            if allowed_doc_ids
            else [
                ("noise-a", "DOC_B", 20.0),
                ("target-a", "DOC_A", 8.0),
                ("noise-b", "DOC_C", 7.5),
                ("target-b", "DOC_A", 7.0),
            ]
        ),
        _merge_doc_scoped_results=lambda *a, **k: [],
        _merge_target_doc_candidates=lambda *a, **k: [],
        _select_diverse_global_candidates=lambda *a, **k: [],
        _build_chunk_results_from_pairs=lambda pairs: {
            "chunk_ids": [chunk_id for chunk_id, _ in pairs],
            "scores": [score for _, score in pairs],
            "chunk_contents": [chunk_id for chunk_id, _ in pairs],
        },
    )

    result = method(
        fake_self,
        "what is the fee",
        top_k=2,
        target_doc_id="DOC_A",
        strict_target_doc_mode=True,
        scope_context={
            "scope_decision": "strict",
            "probe_trace": {"passed": True},
            "allowed_doc_ids": {"DOC_A"},
        },
    )

    assert result["chunk_ids"] == ["target-a", "target-b"]
    assert result["scoped_chunk_ids"] == ["target-a", "target-b"]


def test_rerank_triples_uses_current_triple_text_for_lexical_bonus() -> None:
    rerank_triples_by_relevance = _extract_class_method(
        YOUTU_ROOT / "models" / "retriever" / "enhanced_kt_retriever.py",
        "KTRetriever",
        "_rerank_triples_by_relevance",
    )

    node_text = {
        "h1": "alpha head",
        "t1": "beta tail",
        "h2": "gamma head",
        "t2": "delta tail",
    }

    def _encode_to_tensor(texts, convert_to_tensor=True):
        def _embed(text: str) -> np.ndarray:
            if "alpha" in text:
                return np.array([0.1, 1.0], dtype=np.float32)
            return np.array([0.8, 0.6], dtype=np.float32)

        if isinstance(texts, list):
            return np.stack([_embed(text) for text in texts])
        return _embed(texts)

    fake_self = types.SimpleNamespace(
        _get_node_text=lambda node: node_text[node],
        _encode_to_tensor=_encode_to_tensor,
        _lexical_overlap_score=lambda text, _anchors: 5.0 if "alpha" in text else 0.0,
        _relation_intent_bonus=lambda _relation, _anchors: 0.0,
        _get_node_doc_id=lambda _node: None,
        _rerank_triples_individual=lambda *a, **k: [],
    )

    ranked = rerank_triples_by_relevance(
        fake_self,
        [("h1", "related_to", "t1"), ("h2", "related_to", "t2")],
        FakeTensor([1.0, 0.0]),
        anchor_terms=["alpha"],
        target_doc_id=None,
    )

    assert ranked[0][:3] == ("h1", "related_to", "t1")
    assert ranked[0][3] > ranked[1][3]
