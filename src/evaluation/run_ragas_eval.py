from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.ragas_converters import (
    build_langchain_clients,
    ensure_vendored_ragas_on_path,
    load_jsonl,
    write_json,
    write_jsonl,
)


def _safe_mean(values: list[Any]) -> float | None:
    numeric: list[float] = []
    for value in values:
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isnan(number):
            continue
        numeric.append(number)
    if not numeric:
        return None
    return round(sum(numeric) / len(numeric), 6)


def _coverage(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    total = len(rows)
    filled = sum(1 for row in rows if row.get(field))
    return {
        "count": filled,
        "ratio": round((filled / total), 6) if total else 0.0,
    }


def _normalize_string_ids(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        normalized.append(text)
    return normalized


def _normalize_string_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    normalized: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text:
            normalized.append(text)
    return normalized


def _flatten_trace_chunk_ids(row: dict[str, Any], field: str) -> list[str]:
    trace = row.get("retrieval_trace")
    if not isinstance(trace, dict):
        return []
    sub_questions = trace.get("sub_questions")
    if not isinstance(sub_questions, list):
        return []
    flattened: list[str] = []
    seen: set[str] = set()
    for item in sub_questions:
        if not isinstance(item, dict):
            continue
        for chunk_id in item.get(field, []) or []:
            text = str(chunk_id or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            flattened.append(text)
    return flattened


def _trace_field_present(row: dict[str, Any], field: str) -> bool:
    trace = row.get("retrieval_trace")
    if not isinstance(trace, dict):
        return False
    sub_questions = trace.get("sub_questions")
    if not isinstance(sub_questions, list):
        return False
    return any(isinstance(item, dict) and field in item for item in sub_questions)


def _overlap_metrics(candidate_ids: list[str], reference_ids: list[str], prefix: str) -> dict[str, float | None]:
    candidates = _normalize_string_ids(candidate_ids)
    references = _normalize_string_ids(reference_ids)
    if not references:
        return {
            f"{prefix}_hit": None,
            f"{prefix}_precision": None,
            f"{prefix}_recall": None,
        }
    reference_set = set(references)
    overlap = [cid for cid in candidates if cid in reference_set]
    hit = 1.0 if overlap else 0.0
    precision = (len(overlap) / len(candidates)) if candidates else 0.0
    recall = len(overlap) / len(references)
    return {
        f"{prefix}_hit": round(hit, 6),
        f"{prefix}_precision": round(precision, 6),
        f"{prefix}_recall": round(recall, 6),
    }


def _reciprocal_rank(candidate_ids: list[str], reference_ids: list[str]) -> float | None:
    candidates = _normalize_string_ids(candidate_ids)
    references = _normalize_string_ids(reference_ids)
    if not references:
        return None
    reference_set = set(references)
    for rank, candidate_id in enumerate(candidates, start=1):
        if candidate_id in reference_set:
            return round(1.0 / rank, 6)
    return 0.0


def _compute_diagnostic_metrics(row: dict[str, Any]) -> dict[str, float | None]:
    reference_chunk_ids = _normalize_string_ids(row.get("reference_context_ids"))
    reference_doc_ids = _normalize_string_ids(row.get("reference_doc_ids"))
    reference_doc_id_set = set(reference_doc_ids)
    retrieved_doc_ids = _normalize_string_list(row.get("retrieved_doc_ids"))
    support_chunk_ids = _flatten_trace_chunk_ids(row, "support_chunk_ids")
    first_stage_chunk_ids = _flatten_trace_chunk_ids(row, "first_stage_chunk_ids")
    lightweight_reranked_chunk_ids = _flatten_trace_chunk_ids(row, "lightweight_reranked_chunk_ids")
    strong_reranked_chunk_ids = _flatten_trace_chunk_ids(row, "strong_reranked_chunk_ids")
    final_selected_chunk_ids = _flatten_trace_chunk_ids(row, "final_selected_chunk_ids")
    final_chunk_ids = _normalize_string_ids(row.get("retrieved_context_ids"))

    doc_hit: float | None
    doc_purity: float | None
    if not reference_doc_id_set:
        doc_hit = None
        doc_purity = None
    else:
        matched_docs = [doc_id for doc_id in retrieved_doc_ids if doc_id in reference_doc_id_set]
        doc_hit = round(1.0 if matched_docs else 0.0, 6)
        doc_purity = round((len(matched_docs) / len(retrieved_doc_ids)) if retrieved_doc_ids else 0.0, 6)

    diagnostics: dict[str, float | None] = {
        "context_mrr": _reciprocal_rank(final_chunk_ids, reference_chunk_ids),
        "doc_mrr": _reciprocal_rank(retrieved_doc_ids, reference_doc_ids),
        "doc_hit": doc_hit,
        "doc_purity": doc_purity,
    }
    if _trace_field_present(row, "first_stage_chunk_ids"):
        diagnostics.update(_overlap_metrics(first_stage_chunk_ids, reference_chunk_ids, "first_stage_chunk"))
        diagnostics["first_stage_chunk_mrr"] = _reciprocal_rank(first_stage_chunk_ids, reference_chunk_ids)
    else:
        diagnostics.update(
            {
                "first_stage_chunk_hit": None,
                "first_stage_chunk_precision": None,
                "first_stage_chunk_recall": None,
                "first_stage_chunk_mrr": None,
            }
        )
    if _trace_field_present(row, "lightweight_reranked_chunk_ids"):
        diagnostics.update(_overlap_metrics(lightweight_reranked_chunk_ids, reference_chunk_ids, "lightweight_reranked_chunk"))
        diagnostics["lightweight_reranked_chunk_mrr"] = _reciprocal_rank(
            lightweight_reranked_chunk_ids,
            reference_chunk_ids,
        )
    else:
        diagnostics.update(
            {
                "lightweight_reranked_chunk_hit": None,
                "lightweight_reranked_chunk_precision": None,
                "lightweight_reranked_chunk_recall": None,
                "lightweight_reranked_chunk_mrr": None,
            }
        )
    if _trace_field_present(row, "strong_reranked_chunk_ids"):
        diagnostics.update(_overlap_metrics(strong_reranked_chunk_ids, reference_chunk_ids, "strong_reranked_chunk"))
        diagnostics["strong_reranked_chunk_mrr"] = _reciprocal_rank(
            strong_reranked_chunk_ids,
            reference_chunk_ids,
        )
    else:
        diagnostics.update(
            {
                "strong_reranked_chunk_hit": None,
                "strong_reranked_chunk_precision": None,
                "strong_reranked_chunk_recall": None,
                "strong_reranked_chunk_mrr": None,
            }
        )
    diagnostics.update(_overlap_metrics(support_chunk_ids, reference_chunk_ids, "support_chunk"))
    diagnostics["support_chunk_mrr"] = _reciprocal_rank(support_chunk_ids, reference_chunk_ids)
    diagnostics.update(_overlap_metrics(final_selected_chunk_ids, reference_chunk_ids, "final_selected_chunk"))
    diagnostics["final_selected_chunk_mrr"] = _reciprocal_rank(final_selected_chunk_ids, reference_chunk_ids)
    diagnostics.update(_overlap_metrics(final_chunk_ids, reference_chunk_ids, "final_chunk"))
    diagnostics["final_chunk_mrr"] = diagnostics["context_mrr"]
    return diagnostics


def _extract_response_for_eval(response: Any) -> str:
    text = str(response or "").strip()
    if not text:
        return ""

    def _normalize_heading(line: str) -> str:
        cleaned = str(line or "").strip()
        if not cleaned:
            return ""
        cleaned = cleaned.lstrip("#*- ").rstrip("*: ").strip().lower()
        cleaned = " ".join(cleaned.split())
        aliases = {
            "final answer": "final_answer",
            "inference and bridge conclusion": "inference_and_bridge_conclusion",
            "inference and conclusion": "inference_and_bridge_conclusion",
            "bridge conclusion": "inference_and_bridge_conclusion",
            "inference": "inference_and_bridge_conclusion",
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
        return sections["final_answer"]
    if sections.get("inference_and_bridge_conclusion"):
        return sections["inference_and_bridge_conclusion"]
    return text


def _build_eval_dataset(rows: list[dict[str, Any]]) -> Any:
    ensure_vendored_ragas_on_path()
    from ragas import EvaluationDataset

    return EvaluationDataset.from_list(
        [
            {
                "user_input": row.get("question"),
                "response": row.get("response_for_eval") or _extract_response_for_eval(row.get("response")),
                "reference": row.get("reference"),
                "retrieved_contexts": row.get("retrieved_contexts"),
                "reference_contexts": row.get("reference_contexts"),
                "retrieved_context_ids": row.get("retrieved_context_ids"),
                "reference_context_ids": row.get("reference_context_ids"),
                "persona_name": row.get("persona_name"),
                "query_style": row.get("query_style"),
                "query_length": row.get("query_length"),
            }
            for row in rows
        ]
    )


def _build_metrics(enable_id_based: bool, enable_factual_correctness: bool) -> list[Any]:
    ensure_vendored_ragas_on_path()
    from ragas.metrics import (
        AnswerCorrectness,
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        Faithfulness,
        IDBasedContextPrecision,
        IDBasedContextRecall,
        SemanticSimilarity,
    )

    metrics: list[Any] = [
        AnswerCorrectness(),
        AnswerRelevancy(),
        Faithfulness(),
        SemanticSimilarity(),
        ContextPrecision(),
        ContextRecall(),
    ]
    if enable_factual_correctness:
        from ragas.metrics import FactualCorrectness

        metrics.append(FactualCorrectness())
    if enable_id_based:
        metrics.extend([IDBasedContextPrecision(), IDBasedContextRecall()])
    return metrics


def _summarize_group(rows: list[dict[str, Any]], metric_names: list[str]) -> dict[str, Any]:
    summary = {
        "num_samples": len(rows),
        "with_response": _coverage(rows, "response"),
        "with_response_for_eval": _coverage(rows, "response_for_eval"),
        "with_retrieved_contexts": _coverage(rows, "retrieved_contexts"),
        "with_retrieved_context_ids": _coverage(rows, "retrieved_context_ids"),
        "with_reference_context_ids": _coverage(rows, "reference_context_ids"),
    }
    for metric_name in metric_names:
        summary[metric_name] = _safe_mean([row.get(metric_name) for row in rows])
    return summary


def _distribution(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        value = row.get(field)
        if isinstance(value, list):
            key = json.dumps(value, ensure_ascii=False)
        else:
            key = str(value or "").strip() or "<empty>"
        counts[key] += 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _refactor_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    refactor_rows = [
        row for row in rows
        if str(row.get("orchestration_mode") or "").strip() == "refactor_v1"
    ]
    legacy_rows = [
        row for row in rows
        if str(row.get("orchestration_mode") or "").strip() == "legacy"
    ]
    return {
        "orchestration_mode_distribution": _distribution(rows, "orchestration_mode"),
        "final_chunk_selection_strategy_distribution": _distribution(rows, "final_chunk_selection_strategy"),
        "answer_composition_mode_distribution": _distribution(rows, "answer_composition_mode"),
        "reasoning_plan_mode_distribution": _distribution(rows, "reasoning_plan_mode"),
        "refactor_v1_coverage": {
            "count": len(refactor_rows),
            "ratio": round((len(refactor_rows) / len(rows)), 6) if rows else 0.0,
        },
        "legacy_coverage": {
            "count": len(legacy_rows),
            "ratio": round((len(legacy_rows) / len(rows)), 6) if rows else 0.0,
        },
        "reasoning_plan_present": _coverage(rows, "reasoning_plan"),
        "semantic_alignment_present": _coverage(rows, "semantic_alignment"),
        "phase4_budget_present": _coverage(rows, "phase4_budget"),
        "evidence_matrix_nonzero": {
            "count": sum(1 for row in rows if int(row.get("evidence_matrix_size") or 0) > 0),
            "ratio": round((sum(1 for row in rows if int(row.get("evidence_matrix_size") or 0) > 0) / len(rows)), 6) if rows else 0.0,
        },
        "avg_evidence_matrix_size": _safe_mean([row.get("evidence_matrix_size") for row in rows]),
        "avg_reasoning_plan_fact_count": _safe_mean([row.get("reasoning_plan_fact_count") for row in rows]),
        "avg_sub_question_plan_count": _safe_mean([row.get("sub_question_plan_count") for row in rows]),
        "avg_evidence_requirement_count": _safe_mean([row.get("evidence_requirement_count") for row in rows]),
    }


def _write_summary_csv(path: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ragas evaluation on merged CUAD compare predictions.")
    parser.add_argument("--pred-file", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--enable-id-based", action="store_true")
    parser.add_argument(
        "--enable-factual-correctness",
        action="store_true",
        help="Enable ragas factual_correctness. Disabled by default because it is more brittle in this setup.",
    )
    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=180,
        help="Timeout in seconds for each ragas scoring job.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Maximum concurrent ragas scoring workers.",
    )
    args = parser.parse_args()

    rows = load_jsonl(args.pred_file)
    if not rows:
        raise RuntimeError(f"No prediction rows found in {args.pred_file}")

    ensure_vendored_ragas_on_path()
    from ragas.evaluation import evaluate
    from ragas.run_config import RunConfig

    llm, embeddings = build_langchain_clients(temperature=0.0)
    metrics = _build_metrics(
        enable_id_based=bool(args.enable_id_based),
        enable_factual_correctness=bool(args.enable_factual_correctness),
    )
    metric_names = [metric.name for metric in metrics]
    diagnostic_metric_names = [
        "context_mrr",
        "doc_mrr",
        "doc_hit",
        "doc_purity",
        "first_stage_chunk_hit",
        "first_stage_chunk_precision",
        "first_stage_chunk_recall",
        "first_stage_chunk_mrr",
        "lightweight_reranked_chunk_hit",
        "lightweight_reranked_chunk_precision",
        "lightweight_reranked_chunk_recall",
        "lightweight_reranked_chunk_mrr",
        "strong_reranked_chunk_hit",
        "strong_reranked_chunk_precision",
        "strong_reranked_chunk_recall",
        "strong_reranked_chunk_mrr",
        "support_chunk_hit",
        "support_chunk_precision",
        "support_chunk_recall",
        "support_chunk_mrr",
        "final_selected_chunk_hit",
        "final_selected_chunk_precision",
        "final_selected_chunk_recall",
        "final_selected_chunk_mrr",
        "final_chunk_hit",
        "final_chunk_precision",
        "final_chunk_recall",
        "final_chunk_mrr",
    ]
    run_config = RunConfig(
        timeout=max(1, int(args.timeout_sec)),
        max_workers=max(1, int(args.max_workers)),
    )

    per_sample_rows: list[dict[str, Any]] = []
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_method_synth: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

    methods = sorted({str(row.get("method", "")).strip() for row in rows if str(row.get("method", "")).strip()})
    for method in methods:
        method_rows = [row for row in rows if str(row.get("method", "")).strip() == method]
        print(
            json.dumps(
                {
                    "type": "stage",
                    "stage": "ragas_method_eval",
                    "method": method,
                    "num_samples": len(method_rows),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        dataset = _build_eval_dataset(method_rows)
        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=llm,
            embeddings=embeddings,
            run_config=run_config,
            raise_exceptions=False,
            show_progress=True,
        )
        for source_row, score_row in zip(method_rows, result.scores):
            merged = dict(source_row)
            for key, value in score_row.items():
                merged[key] = value
            merged.update(_compute_diagnostic_metrics(merged))
            per_sample_rows.append(merged)
            by_method[method].append(merged)
            by_method_synth[(method, str(merged.get("synthesizer_name", "") or "unknown"))].append(merged)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_sample_file = out_dir / "ragas_eval_per_sample.jsonl"
    summary_file = out_dir / "ragas_eval_summary.json"
    summary_csv_file = out_dir / "ragas_eval_summary.csv"

    method_summary_rows: list[dict[str, Any]] = []
    method_synth_rows: list[dict[str, Any]] = []

    for method, method_rows in sorted(by_method.items()):
        row = {"group_type": "method", "method": method, "synthesizer_name": ""}
        row.update(_summarize_group(method_rows, metric_names + diagnostic_metric_names))
        method_summary_rows.append(row)

    for (method, synthesizer_name), synth_rows in sorted(by_method_synth.items()):
        row = {
            "group_type": "method_synthesizer",
            "method": method,
            "synthesizer_name": synthesizer_name,
        }
        row.update(_summarize_group(synth_rows, metric_names + diagnostic_metric_names))
        method_synth_rows.append(row)

    summary_payload = {
        "pred_file": str(Path(args.pred_file).resolve()),
        "out_dir": str(out_dir.resolve()),
        "metrics": metric_names,
        "diagnostic_metrics": diagnostic_metric_names,
        "enable_id_based": bool(args.enable_id_based),
        "enable_factual_correctness": bool(args.enable_factual_correctness),
        "timeout_sec": int(run_config.timeout),
        "max_workers": int(run_config.max_workers),
        "num_rows": len(rows),
        "analysis": _refactor_analysis(per_sample_rows),
        "per_method": method_summary_rows,
        "per_method_synthesizer": method_synth_rows,
    }

    write_jsonl(str(per_sample_file), per_sample_rows)
    write_json(str(summary_file), summary_payload)
    _write_summary_csv(str(summary_csv_file), method_summary_rows + method_synth_rows)
    print(json.dumps(summary_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
