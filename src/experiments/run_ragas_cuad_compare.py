from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from baselines.graph_rag import answer_with_graphrag
from baselines.lightrag_adapter import answer_with_lightrag
from utils.config import cfg
from utils.ragas_converters import (
    infer_doc_prefix,
    load_jsonl,
    normalize_evidence_chunks,
    scope_cuad_question,
    write_jsonl,
)

try:
    from baselines.youtu_graph_rag_adapter import answer_with_youtu_graphrag

    _YOUTU_AVAILABLE = True
except ImportError:
    _YOUTU_AVAILABLE = False


def _normalize_answer_mode(answer_mode: str | None) -> str:
    mode = str(answer_mode or "reject").strip().lower()
    return mode if mode in {"reject", "open"} else "reject"


def _extract_response_for_eval(response: str | None) -> tuple[str, str]:
    text = str(response or "").strip()
    if not text:
        return "", ""

    def _normalize_heading(line: str) -> str:
        cleaned = str(line or "").strip()
        if not cleaned:
            return ""
        cleaned = cleaned.lstrip("#*- ").rstrip("*: ").strip().lower()
        cleaned = " ".join(cleaned.split())
        aliases = {
            "final answer": "final_answer",
            "inference and bridge conclusion": "inference_and_bridge_conclusion",
            "inference and conclusion": "inference_and_conclusion",
            "bridge conclusion": "bridge_conclusion",
            "inference": "inference",
        }
        return aliases.get(cleaned, "")

    sections: dict[str, str] = {}
    current = ""
    buffer: list[str] = []
    for raw_line in text.splitlines():
        section = _normalize_heading(raw_line)
        if section:
            if current:
                content = "\n".join(buffer).strip()
                if content and current not in sections:
                    sections[current] = content
            current = section
            buffer = []
            continue
        if current:
            buffer.append(raw_line)
    if current:
        content = "\n".join(buffer).strip()
        if content and current not in sections:
            sections[current] = content

    if sections.get("final_answer"):
        return sections["final_answer"], "final_answer"
    if sections.get("inference_and_conclusion"):
        return sections["inference_and_conclusion"], "inference_and_conclusion"
    if sections.get("bridge_conclusion"):
        return sections["bridge_conclusion"], "bridge_conclusion"
    if sections.get("inference_and_bridge_conclusion"):
        return sections["inference_and_bridge_conclusion"], "inference_and_bridge_conclusion"
    if sections.get("inference"):
        return sections["inference"], "inference"
    return text, "full_response"


def _default_graph_cfg() -> dict[str, Any]:
    graph_cfg = getattr(getattr(cfg, "comparison", None), "best_effort", None)
    graph = getattr(graph_cfg, "graph", None)
    return {
        "top_communities": int(getattr(graph, "top_communities", 3)),
        "max_evidence": int(getattr(graph, "max_evidence", 12)),
        "query_level": int(getattr(graph, "query_level", 0)),
        "use_hierarchy": bool(getattr(graph, "use_hierarchy", True)),
        "use_community_summaries": bool(getattr(graph, "use_community_summaries", True)),
        "shuffle_communities": bool(getattr(graph, "shuffle_communities", True)),
        "use_map_reduce": bool(getattr(graph, "use_map_reduce", True)),
        "max_summary_chars": int(getattr(graph, "max_summary_chars", 1800)),
    }


def _run_method(
    method: str,
    question: str,
    *,
    scoped_question: str,
    graph_file: str,
    communities_file: str,
    chunks_file: str,
    lightrag_working_dir: str | None,
    youtu_base_url: str | None,
    youtu_dataset: str | None,
    answer_mode: str,
    doc_prefix: str,
) -> dict[str, Any]:
    graph_cfg = _default_graph_cfg()
    if method == "graph_rag":
        return answer_with_graphrag(
            query=scoped_question,
            graph_file=graph_file,
            communities_file=communities_file,
            answer_mode=answer_mode,
            doc_prefix_filter=doc_prefix or None,
            strict_doc_scope=False,
            **graph_cfg,
        )
    if method == "lightrag":
        return answer_with_lightrag(
            query=scoped_question,
            chunks_file=chunks_file,
            working_dir=lightrag_working_dir,
            answer_mode=answer_mode,
            top_k=int(getattr(getattr(cfg, "retrieval", None), "top_k", 8)),
            doc_prefix_filter=doc_prefix or None,
        )
    if method == "youtu_graph_rag":
        if not _YOUTU_AVAILABLE:
            raise RuntimeError("youtu_graph_rag adapter is not available in this environment")
        return answer_with_youtu_graphrag(
            query=scoped_question,
            graph_file=graph_file,
            communities_file=communities_file,
            store_file=chunks_file,
            youtu_base_url=youtu_base_url,
            youtu_dataset=youtu_dataset,
            answer_mode=answer_mode,
            doc_prefix_filter=doc_prefix or None,
            strict_doc_scope=False,
            **graph_cfg,
        )
    raise ValueError(f"Unsupported method: {method}")


def _build_output_row(test_row: dict[str, Any], method: str, payload: dict[str, Any]) -> dict[str, Any]:
    evaluation_payload = payload.get("evaluation_payload") if isinstance(payload.get("evaluation_payload"), dict) else {}
    normalized = normalize_evidence_chunks(payload.get("evidence_chunks"))
    retrieved_contexts = evaluation_payload.get("retrieved_contexts")
    retrieved_context_ids = evaluation_payload.get("retrieved_context_ids")
    retrieved_doc_ids = evaluation_payload.get("retrieved_doc_ids")
    external_related_context_ids = evaluation_payload.get("external_related_context_ids")
    external_related_doc_ids = evaluation_payload.get("external_related_doc_ids")
    retrieval_trace = payload.get("retrieval_trace") if isinstance(payload.get("retrieval_trace"), dict) else {}
    answer_scope_target_doc_id = str(
        payload.get("answer_scope_target_doc_id")
        or retrieval_trace.get("answer_scope_target_doc_id")
        or ""
    ).strip()
    answer_composition_mode = str(
        payload.get("answer_composition_mode")
        or retrieval_trace.get("answer_composition_mode")
        or ""
    ).strip()
    semantic_alignment = payload.get("semantic_alignment")
    if not isinstance(semantic_alignment, dict):
        semantic_alignment = retrieval_trace.get("semantic_alignment") if isinstance(retrieval_trace.get("semantic_alignment"), dict) else {}
    reasoning_trace = payload.get("reasoning_trace") if isinstance(payload.get("reasoning_trace"), dict) else {}
    answer_trace = payload.get("answer_trace") if isinstance(payload.get("answer_trace"), dict) else {}
    reasoning_plan = retrieval_trace.get("reasoning_plan") if isinstance(retrieval_trace.get("reasoning_plan"), dict) else {}
    evidence_matrix = retrieval_trace.get("evidence_matrix") if isinstance(retrieval_trace.get("evidence_matrix"), list) else []
    final_chunk_selection = retrieval_trace.get("final_chunk_selection") if isinstance(retrieval_trace.get("final_chunk_selection"), dict) else {}
    final_answer_prompt_inputs = answer_trace.get("final_answer_prompt_inputs") if isinstance(answer_trace.get("final_answer_prompt_inputs"), dict) else {}
    orchestration_mode = str(retrieval_trace.get("orchestration_mode") or "").strip()
    final_chunk_selection_strategy = str(final_chunk_selection.get("strategy") or "").strip()
    reasoning_plan_mode = str(reasoning_plan.get("mode") or "").strip()
    response = str(evaluation_payload.get("response", payload.get("answer", "")) or "").strip()
    response_for_eval = str(evaluation_payload.get("response_for_eval") or "").strip()
    response_for_eval_source = str(evaluation_payload.get("response_for_eval_source") or "").strip()
    if not response_for_eval:
        response_for_eval, response_for_eval_source = _extract_response_for_eval(response)

    def _prefer_non_empty_eval_list(value: Any, fallback: list[Any]) -> list[Any]:
        if isinstance(value, list) and value:
            return list(value)
        return list(fallback)

    return {
        "qid": str(test_row.get("qid", "")).strip(),
        "method": method,
        "question": str(test_row.get("question", "")).strip(),
        "response": response,
        "response_for_eval": response_for_eval,
        "response_for_eval_source": response_for_eval_source,
        "reference": str(test_row.get("reference", "") or "").strip(),
        "retrieved_contexts": _prefer_non_empty_eval_list(retrieved_contexts, normalized["retrieved_contexts"]),
        "retrieved_context_ids": _prefer_non_empty_eval_list(retrieved_context_ids, normalized["retrieved_context_ids"]),
        "retrieved_doc_ids": _prefer_non_empty_eval_list(retrieved_doc_ids, normalized["retrieved_doc_ids"]),
        "external_related_context_ids": list(external_related_context_ids) if isinstance(external_related_context_ids, list) else [],
        "external_related_doc_ids": list(external_related_doc_ids) if isinstance(external_related_doc_ids, list) else [],
        "reference_contexts": list(test_row.get("reference_contexts") or []),
        "reference_context_ids": list(test_row.get("reference_context_ids") or []),
        "reference_doc_ids": list(test_row.get("reference_doc_ids") or []),
        "synthesizer_name": test_row.get("synthesizer_name"),
        "persona_name": test_row.get("persona_name"),
        "query_style": test_row.get("query_style"),
        "query_length": test_row.get("query_length"),
        "answer_mode": payload.get("answer_mode", _normalize_answer_mode(None)),
        "answer_scope_target_doc_id": answer_scope_target_doc_id,
        "answer_composition_mode": answer_composition_mode,
        "semantic_alignment": semantic_alignment,
        "orchestration_mode": orchestration_mode,
        "final_chunk_selection_strategy": final_chunk_selection_strategy,
        "evidence_matrix_size": len(evidence_matrix),
        "reasoning_plan_mode": reasoning_plan_mode,
        "reasoning_plan": reasoning_plan,
        "phase4_budget": retrieval_trace.get("phase4_budget") if isinstance(retrieval_trace.get("phase4_budget"), dict) else {},
        "sub_question_plan_count": len(retrieval_trace.get("sub_question_plans") or []) if isinstance(retrieval_trace.get("sub_question_plans"), list) else 0,
        "evidence_requirement_count": len(retrieval_trace.get("evidence_requirements") or []) if isinstance(retrieval_trace.get("evidence_requirements"), list) else 0,
        "reasoning_plan_fact_count": len(reasoning_plan.get("facts") or []) if isinstance(reasoning_plan.get("facts"), list) else 0,
        "semantic_alignment_shared_concepts": list(semantic_alignment.get("shared_concepts") or []) if isinstance(semantic_alignment, dict) else [],
        "answer_prompt_reasoning_plan_mode": str((final_answer_prompt_inputs.get("reasoning_plan") or {}).get("mode") or "").strip()
        if isinstance(final_answer_prompt_inputs.get("reasoning_plan"), dict)
        else "",
        "telemetry": payload.get("telemetry") if isinstance(payload.get("telemetry"), dict) else {},
        "retrieval_trace": retrieval_trace,
        "reasoning_trace": reasoning_trace,
        "answer_trace": answer_trace,
        "reasoning_steps": payload.get("reasoning_steps") if isinstance(payload.get("reasoning_steps"), list) else [],
    }


def _print_progress(
    completed: int,
    total: int,
    *,
    qid: str,
    method: str,
    question_index: int,
    num_questions: int,
    elapsed_sec: float,
) -> None:
    print(
        json.dumps(
            {
                "type": "progress",
                "completed": completed,
                "total": total,
                "ratio": round((completed / total), 6) if total else 0.0,
                "question_index": question_index,
                "num_questions": num_questions,
                "qid": qid,
                "method": method,
                "elapsed_sec": round(elapsed_sec, 2),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


def _limit_questions_per_type(rows: list[dict[str, Any]], max_per_type: int) -> list[dict[str, Any]]:
    limit = int(max_per_type)
    if limit <= 0:
        return list(rows)
    kept: list[dict[str, Any]] = []
    seen_per_type: dict[str, int] = defaultdict(int)
    for row in rows:
        synth = str(row.get("synthesizer_name", "") or "unknown").strip() or "unknown"
        if seen_per_type[synth] >= limit:
            continue
        kept.append(row)
        seen_per_type[synth] += 1
    return kept


def _run_single_task(
    task_index: int,
    question_index: int,
    num_questions: int,
    test_row: dict[str, Any],
    method: str,
    *,
    graph_file: str,
    communities_file: str,
    chunks_file: str,
    lightrag_working_dir: str | None,
    youtu_base_url: str | None,
    youtu_dataset: str | None,
    answer_mode: str,
) -> dict[str, Any]:
    question = str(test_row.get("question", "")).strip()
    doc_prefix = infer_doc_prefix(test_row.get("reference_doc_ids"))
    scoped_question = scope_cuad_question(question, doc_prefix)
    payload = _run_method(
        method,
        question,
        scoped_question=scoped_question,
        graph_file=graph_file,
        communities_file=communities_file,
        chunks_file=chunks_file,
        lightrag_working_dir=lightrag_working_dir,
        youtu_base_url=youtu_base_url,
        youtu_dataset=youtu_dataset,
        answer_mode=answer_mode,
        doc_prefix=doc_prefix,
    )
    row = _build_output_row(test_row, method, payload)
    return {
        "task_index": task_index,
        "question_index": question_index,
        "num_questions": num_questions,
        "qid": str(test_row.get("qid", "")).strip(),
        "method": method,
        "row": row,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run unified CUAD ragas compare across graph_rag/lightrag/youtu.")
    parser.add_argument("--testset-file", required=True)
    parser.add_argument("--chunks-file", required=True)
    parser.add_argument("--graph-file", required=True)
    parser.add_argument("--communities-file", required=True)
    parser.add_argument("--lightrag-working-dir", default="")
    parser.add_argument("--youtu-base-url", default="")
    parser.add_argument("--youtu-dataset", default="")
    parser.add_argument("--methods", default="graph_rag,lightrag,youtu_graph_rag")
    parser.add_argument("--answer-mode", default="reject")
    parser.add_argument(
        "--max-questions-per-type",
        type=int,
        default=0,
        help="Run at most N questions for each synthesizer_name; set 0 to use the full testset.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1,
        help="Print progress every N completed question-method tasks; set 0 to disable.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Maximum concurrent question-method tasks. Set 1 to keep sequential execution.",
    )
    parser.add_argument("--out-file", required=True)
    args = parser.parse_args()

    testset_rows = load_jsonl(args.testset_file)
    testset_rows = _limit_questions_per_type(testset_rows, int(args.max_questions_per_type))
    methods = [part.strip() for part in str(args.methods).split(",") if part.strip()]
    answer_mode = _normalize_answer_mode(args.answer_mode)
    progress_every = max(0, int(args.progress_every))
    max_workers = max(1, int(args.max_workers))
    total_tasks = len(testset_rows) * len(methods)
    completed_tasks = 0
    started_at = time.perf_counter()

    tasks: list[tuple[int, int, dict[str, Any], str]] = []
    task_index = 0
    for question_index, test_row in enumerate(testset_rows, start=1):
        for method in methods:
            tasks.append((task_index, question_index, test_row, method))
            task_index += 1

    completed_results: list[dict[str, Any]] = []
    common_kwargs = {
        "graph_file": args.graph_file,
        "communities_file": args.communities_file,
        "chunks_file": args.chunks_file,
        "lightrag_working_dir": (str(args.lightrag_working_dir).strip() or None),
        "youtu_base_url": (str(args.youtu_base_url).strip() or None),
        "youtu_dataset": (str(args.youtu_dataset).strip() or None),
        "answer_mode": answer_mode,
    }

    if max_workers == 1:
        for task_idx, question_index, test_row, method in tasks:
            result = _run_single_task(
                task_idx,
                question_index,
                len(testset_rows),
                test_row,
                method,
                **common_kwargs,
            )
            completed_results.append(result)
            completed_tasks += 1
            if progress_every and (completed_tasks % progress_every == 0 or completed_tasks == total_tasks):
                _print_progress(
                    completed_tasks,
                    total_tasks,
                    qid=result["qid"],
                    method=result["method"],
                    question_index=result["question_index"],
                    num_questions=result["num_questions"],
                    elapsed_sec=time.perf_counter() - started_at,
                )
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    _run_single_task,
                    task_idx,
                    question_index,
                    len(testset_rows),
                    test_row,
                    method,
                    **common_kwargs,
                )
                for task_idx, question_index, test_row, method in tasks
            ]
            for future in as_completed(futures):
                result = future.result()
                completed_results.append(result)
                completed_tasks += 1
                if progress_every and (completed_tasks % progress_every == 0 or completed_tasks == total_tasks):
                    _print_progress(
                        completed_tasks,
                        total_tasks,
                        qid=result["qid"],
                        method=result["method"],
                        question_index=result["question_index"],
                        num_questions=result["num_questions"],
                        elapsed_sec=time.perf_counter() - started_at,
                    )

    completed_results.sort(key=lambda item: item["task_index"])
    merged_rows = [item["row"] for item in completed_results]
    per_method_rows: dict[str, list[dict[str, Any]]] = {method: [] for method in methods}
    for item in completed_results:
        per_method_rows[item["method"]].append(item["row"])

    write_jsonl(args.out_file, merged_rows)
    out_path = Path(args.out_file)
    out_stem = out_path.stem
    for method, rows in per_method_rows.items():
        method_file = out_path.with_name(f"{out_stem}_{method}_predictions.jsonl")
        write_jsonl(str(method_file), rows)

    print(
        json.dumps(
            {
                "testset_file": str(Path(args.testset_file).resolve()),
                "num_questions": len(testset_rows),
                "methods": methods,
                "num_rows": len(merged_rows),
                "max_workers": max_workers,
                "out_file": str(out_path.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
