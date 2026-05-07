from __future__ import annotations

import ast
import math
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for candidate in (ROOT, SRC):
    value = str(candidate)
    if value not in sys.path:
        sys.path.insert(0, value)


def _extract_eval_functions(*function_names: str):
    path = SRC / "evaluation" / "run_ragas_eval.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    selected = []
    wanted = set(function_names)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            selected.append(node)
    assert len(selected) == len(wanted), f"missing eval functions: {wanted - {node.name for node in selected}}"
    module = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "Any": Any,
        "math": math,
    }
    exec(compile(module, str(path), "exec"), namespace)
    return {name: namespace[name] for name in function_names}


def test_compute_diagnostic_metrics_reads_current_trace_layers() -> None:
    helpers = _extract_eval_functions(
        "_safe_mean",
        "_coverage",
        "_normalize_string_ids",
        "_normalize_string_list",
        "_flatten_trace_chunk_ids",
        "_trace_field_present",
        "_overlap_metrics",
        "_compute_diagnostic_metrics",
    )

    metrics = helpers["_compute_diagnostic_metrics"](
        {
            "reference_context_ids": ["gold-1", "gold-2"],
            "reference_doc_ids": ["doc-a#p0"],
            "retrieved_context_ids": ["gold-1", "noise-1", "noise-2"],
            "retrieved_doc_ids": ["doc-a#p0", "doc-a#p0", "doc-b#p0"],
            "retrieval_trace": {
                "sub_questions": [
                    {
                        "first_stage_chunk_ids": ["gold-1", "noise-0"],
                        "lightweight_reranked_chunk_ids": ["gold-1", "noise-2"],
                        "strong_reranked_chunk_ids": ["gold-1", "gold-2"],
                        "support_chunk_ids": ["gold-1", "noise-1"],
                        "final_selected_chunk_ids": ["gold-1", "noise-2"],
                    },
                    {
                        "first_stage_chunk_ids": ["gold-2"],
                        "lightweight_reranked_chunk_ids": ["gold-2"],
                        "strong_reranked_chunk_ids": ["gold-2"],
                        "support_chunk_ids": ["noise-3"],
                        "final_selected_chunk_ids": ["gold-2"],
                    },
                ]
            },
        }
    )

    assert metrics["doc_hit"] == 1.0
    assert metrics["doc_purity"] == 0.666667
    assert metrics["first_stage_chunk_hit"] == 1.0
    assert metrics["first_stage_chunk_precision"] == 0.666667
    assert metrics["first_stage_chunk_recall"] == 1.0
    assert metrics["lightweight_reranked_chunk_hit"] == 1.0
    assert metrics["lightweight_reranked_chunk_precision"] == 0.666667
    assert metrics["lightweight_reranked_chunk_recall"] == 1.0
    assert metrics["strong_reranked_chunk_hit"] == 1.0
    assert metrics["strong_reranked_chunk_precision"] == 1.0
    assert metrics["strong_reranked_chunk_recall"] == 1.0
    assert metrics["support_chunk_hit"] == 1.0
    assert metrics["support_chunk_precision"] == 0.333333
    assert metrics["support_chunk_recall"] == 0.5
    assert metrics["final_selected_chunk_hit"] == 1.0
    assert metrics["final_selected_chunk_precision"] == 0.666667
    assert metrics["final_selected_chunk_recall"] == 1.0
    assert metrics["final_chunk_hit"] == 1.0
    assert metrics["final_chunk_precision"] == 0.333333
    assert metrics["final_chunk_recall"] == 0.5


def test_summarize_group_averages_diagnostic_metrics() -> None:
    helpers = _extract_eval_functions(
        "_safe_mean",
        "_coverage",
        "_summarize_group",
    )

    summary = helpers["_summarize_group"](
        [
            {"response": "a", "retrieved_contexts": ["ctx"], "retrieved_context_ids": ["c1"], "reference_context_ids": ["g1"], "doc_hit": 1.0, "support_chunk_hit": 1.0},
            {"response": "b", "retrieved_contexts": ["ctx"], "retrieved_context_ids": ["c2"], "reference_context_ids": ["g2"], "doc_hit": 0.0, "support_chunk_hit": 0.0},
        ],
        ["doc_hit", "support_chunk_hit"],
    )

    assert summary["num_samples"] == 2
    assert summary["with_response"]["count"] == 2
    assert summary["doc_hit"] == 0.5
    assert summary["support_chunk_hit"] == 0.5


def test_compute_diagnostic_metrics_leaves_unavailable_stage_metrics_empty() -> None:
    helpers = _extract_eval_functions(
        "_safe_mean",
        "_coverage",
        "_normalize_string_ids",
        "_normalize_string_list",
        "_flatten_trace_chunk_ids",
        "_trace_field_present",
        "_overlap_metrics",
        "_compute_diagnostic_metrics",
    )

    metrics = helpers["_compute_diagnostic_metrics"](
        {
            "reference_context_ids": ["gold-1"],
            "reference_doc_ids": ["doc-a#p0"],
            "retrieved_context_ids": ["gold-1"],
            "retrieved_doc_ids": ["doc-a#p0"],
            "retrieval_trace": {
                "sub_questions": [
                    {
                        "support_chunk_ids": ["gold-1"],
                        "final_selected_chunk_ids": ["gold-1"],
                    }
                ]
            },
        }
    )

    assert metrics["first_stage_chunk_hit"] is None
    assert metrics["lightweight_reranked_chunk_hit"] is None
    assert metrics["strong_reranked_chunk_hit"] is None
    assert metrics["support_chunk_hit"] == 1.0


def test_extract_response_for_eval_prefers_inference_section() -> None:
    helpers = _extract_eval_functions("_extract_response_for_eval")

    extracted = helpers["_extract_response_for_eval"](
        "**Reasoning over Facts**\n\n"
        "**Grounded Facts**\n\n"
        "Fact block.\n\n"
        "**Inference:**\n\n"
        "This is the concise answer."
    )

    assert extracted == "This is the concise answer."


def test_extract_response_for_eval_prefers_final_answer_marker() -> None:
    helpers = _extract_eval_functions("_extract_response_for_eval")

    extracted = helpers["_extract_response_for_eval"](
        "**Reasoning over Facts**\n\n"
        "Fact block.\n\n"
        "**Inference and Conclusion:**\n\n"
        "Older concise answer.\n\n"
        "**Final Answer:**\n\n"
        "Canonical concise answer."
    )

    assert extracted == "Canonical concise answer."


def test_extract_response_for_eval_parses_plain_final_answer_heading() -> None:
    helpers = _extract_eval_functions("_extract_response_for_eval")

    extracted = helpers["_extract_response_for_eval"](
        "Grounded Facts from the Target Contract:\n\n"
        "Fact block.\n\n"
        "Final Answer:\n\n"
        "Canonical concise answer."
    )

    assert extracted == "Canonical concise answer."
