from __future__ import annotations

import re
from collections import Counter
from typing import Any

_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{1,}")
_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "into",
    "under",
    "shall",
    "agreement",
    "section",
    "article",
    "clause",
    "contract",
    "party",
    "parties",
    "customer",
    "customers",
    "company",
    "services",
    "service",
    "terms",
    "term",
    "provision",
    "provisions",
}


def _doc_prefix(doc_id: Any) -> str:
    return str(doc_id or "").split("#", 1)[0].strip()


def _normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _ordered_unique(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        norm = str(value or "").strip()
        if not norm or norm in seen:
            continue
        seen.add(norm)
        out.append(norm)
    return out


def _serialize_chunk_id(item: dict[str, Any]) -> str:
    raw_ids = item.get("chunk_ids") if isinstance(item.get("chunk_ids"), list) else []
    ids = _ordered_unique([str(x or "").strip() for x in raw_ids] + [str(item.get("chunk_id") or "").strip()])
    return "|".join(ids) if ids else ""


def normalize_chunk_items(evidence_chunks: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for raw in evidence_chunks or []:
        if not isinstance(raw, dict):
            continue
        item = dict(raw)
        item["chunk_id"] = str(item.get("chunk_id") or "").strip()
        item["doc_id"] = str(item.get("doc_id") or "").strip()
        item["text"] = _normalize_text(item.get("text"))
        if not item["chunk_id"] and not item["text"] and not item["doc_id"]:
            continue
        out.append(item)
    return out


def infer_answer_scope_target_doc_id(
    evidence_chunks: list[dict[str, Any]] | None,
    doc_prefix_filter: str | None = None,
) -> str:
    chunks = normalize_chunk_items(evidence_chunks)
    wanted = _doc_prefix(doc_prefix_filter)
    if wanted:
        for item in chunks:
            doc_id = str(item.get("doc_id") or "").strip()
            if _doc_prefix(doc_id) == wanted:
                return doc_id
        return wanted

    doc_ids = [str(item.get("doc_id") or "").strip() for item in chunks if str(item.get("doc_id") or "").strip()]
    if not doc_ids:
        return ""
    prefixes = [_doc_prefix(doc_id) for doc_id in doc_ids if _doc_prefix(doc_id)]
    if not prefixes:
        return ""
    counts = Counter(prefixes)
    top_prefix, top_count = counts.most_common(1)[0]
    if len(counts) == 1 or top_count > (len(prefixes) / 2.0):
        for doc_id in doc_ids:
            if _doc_prefix(doc_id) == top_prefix:
                return doc_id
    return ""


def split_primary_and_external_chunks(
    evidence_chunks: list[dict[str, Any]] | None,
    answer_scope_target_doc_id: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    chunks = normalize_chunk_items(evidence_chunks)
    target = _doc_prefix(answer_scope_target_doc_id)
    if not target:
        return chunks, []

    primary: list[dict[str, Any]] = []
    external: list[dict[str, Any]] = []
    for item in chunks:
        doc_id = str(item.get("doc_id") or "").strip()
        if doc_id and _doc_prefix(doc_id) != target:
            external.append(item)
        else:
            primary.append(item)
    return primary, external


def infer_answer_composition_mode(
    primary_chunks: list[dict[str, Any]] | None,
    external_chunks: list[dict[str, Any]] | None,
    answer_scope_target_doc_id: str | None = None,
) -> str:
    primary = normalize_chunk_items(primary_chunks)
    external = normalize_chunk_items(external_chunks)
    if answer_scope_target_doc_id:
        return "cross_document_bridge" if external else "target_contract_primary"

    doc_prefixes = {_doc_prefix(item.get("doc_id")) for item in primary if _doc_prefix(item.get("doc_id"))}
    if len(doc_prefixes) == 1:
        return "target_contract_primary"
    return "cross_document_bridge" if external else ""


def infer_semantic_alignment(
    query: str,
    primary_chunks: list[dict[str, Any]] | None,
    external_chunks: list[dict[str, Any]] | None,
    answer_composition_mode: str | None = None,
) -> dict[str, Any]:
    primary = normalize_chunk_items(primary_chunks)
    external = normalize_chunk_items(external_chunks)
    if str(answer_composition_mode or "").strip() != "cross_document_bridge" or not external:
        return {}

    def _tokenize(text: str) -> set[str]:
        tokens = {
            token.lower()
            for token in _TOKEN_RE.findall(str(text or ""))
            if len(token) >= 4 and not token.isdigit()
        }
        return {token for token in tokens if token not in _STOPWORDS}

    query_tokens = _tokenize(query)
    primary_tokens = _tokenize(" ".join(str(item.get("text") or "") for item in primary))
    external_tokens = _tokenize(" ".join(str(item.get("text") or "") for item in external))
    shared = sorted((primary_tokens & external_tokens) & (query_tokens or (primary_tokens & external_tokens)))
    if not shared:
        shared = sorted(primary_tokens & external_tokens)
    shared = shared[:8]

    alignment: dict[str, Any] = {
        "alignment_type": "heuristic_shared_concepts",
        "shared_concepts": shared,
        "safe_summary": (
            f"Primary and external evidence overlap on: {', '.join(shared[:4])}."
            if shared
            else "External related evidence was retrieved, but no stable shared concepts were inferred heuristically."
        ),
    }
    return alignment


def build_evaluation_payload(
    *,
    response: str,
    response_for_eval: str,
    response_for_eval_source: str,
    reasoning: str | None = None,
    primary_chunks: list[dict[str, Any]] | None = None,
    external_chunks: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    primary = normalize_chunk_items(primary_chunks)
    external = normalize_chunk_items(external_chunks)
    return {
        "response": str(response or "").strip(),
        "response_for_eval": str(response_for_eval or "").strip(),
        "response_for_eval_source": str(response_for_eval_source or "").strip(),
        "reasoning": str(reasoning or "").strip(),
        "retrieved_contexts": [str(item.get("text") or "") for item in primary],
        "retrieved_context_ids": [_serialize_chunk_id(item) for item in primary if _serialize_chunk_id(item)],
        "retrieved_doc_ids": [str(item.get("doc_id") or "").strip() for item in primary if str(item.get("doc_id") or "").strip()],
        "external_related_contexts": [str(item.get("text") or "") for item in external],
        "external_related_context_ids": [_serialize_chunk_id(item) for item in external if _serialize_chunk_id(item)],
        "external_related_doc_ids": [str(item.get("doc_id") or "").strip() for item in external if str(item.get("doc_id") or "").strip()],
    }


def build_minimal_trace_bundle(
    *,
    method: str,
    query: str,
    answer: str,
    evidence_chunks: list[dict[str, Any]] | None,
    doc_prefix_filter: str | None = None,
    response_for_eval: str | None = None,
    response_for_eval_source: str = "method_answer",
    reasoning: str | None = None,
    orchestration_mode: str = "",
    aggregation_strategy: str = "",
    reasoning_steps: list[dict[str, Any]] | None = None,
    final_answer_prompt_inputs: dict[str, Any] | None = None,
    retrieval_trace_extra: dict[str, Any] | None = None,
    reasoning_trace_extra: dict[str, Any] | None = None,
    answer_trace_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    chunks = normalize_chunk_items(evidence_chunks)
    target_doc_id = infer_answer_scope_target_doc_id(chunks, doc_prefix_filter=doc_prefix_filter)
    primary_chunks, external_chunks = split_primary_and_external_chunks(
        chunks,
        answer_scope_target_doc_id=target_doc_id,
    )
    answer_mode = infer_answer_composition_mode(
        primary_chunks,
        external_chunks,
        answer_scope_target_doc_id=target_doc_id,
    )
    semantic_alignment = infer_semantic_alignment(
        query,
        primary_chunks,
        external_chunks,
        answer_composition_mode=answer_mode,
    )
    eval_response = str(response_for_eval or answer or "").strip()
    eval_payload = build_evaluation_payload(
        response=str(answer or "").strip(),
        response_for_eval=eval_response,
        response_for_eval_source=response_for_eval_source,
        reasoning=reasoning,
        primary_chunks=primary_chunks,
        external_chunks=external_chunks,
    )

    retrieved_chunk_items = [dict(item) for item in primary_chunks]
    external_related_chunk_items = [dict(item) for item in external_chunks]
    prompt_inputs = {
        "answer_scope_target_doc_id": target_doc_id,
        "answer_composition_mode": answer_mode,
        "semantic_alignment": semantic_alignment,
        "backing_chunk_ids": eval_payload["retrieved_context_ids"][:8],
        "external_related_backing_chunk_ids": eval_payload["external_related_context_ids"][:4],
    }
    if isinstance(final_answer_prompt_inputs, dict):
        prompt_inputs.update(final_answer_prompt_inputs)

    retrieval_trace = {
        "query": str(query or "").strip(),
        "orchestration_mode": str(orchestration_mode or method).strip(),
        "answer_scope_target_doc_id": target_doc_id,
        "answer_composition_mode": answer_mode,
        "semantic_alignment": semantic_alignment,
        "sub_questions": [],
        "final_retrieved_chunk_ids": list(eval_payload["retrieved_context_ids"]),
        "final_retrieved_chunks": retrieved_chunk_items,
        "external_related_chunk_ids": list(eval_payload["external_related_context_ids"]),
        "external_related_chunks": external_related_chunk_items,
        "final_chunk_selection": {
            "strategy": str(aggregation_strategy or method).strip(),
            "total_limit": len(primary_chunks),
            "target_doc_id": target_doc_id,
            "selected_count": len(primary_chunks),
            "external_selected_count": len(external_chunks),
        },
    }
    if isinstance(retrieval_trace_extra, dict):
        retrieval_trace.update(retrieval_trace_extra)

    reasoning_trace = {
        "orchestration_mode": str(orchestration_mode or method).strip(),
        "answer_scope_target_doc_id": target_doc_id,
        "answer_composition_mode": answer_mode,
        "semantic_alignment": semantic_alignment,
        "sub_question_answers": [],
        "aggregation_inputs": [
            {
                "chunk_id": _serialize_chunk_id(item),
                "doc_id": str(item.get("doc_id") or "").strip(),
                "summary": str(item.get("text") or "")[:240],
            }
            for item in primary_chunks[:8]
        ],
    }
    if isinstance(reasoning_trace_extra, dict):
        reasoning_trace.update(reasoning_trace_extra)

    answer_trace = {
        "aggregation_strategy": str(aggregation_strategy or method).strip(),
        "final_answer_prompt_inputs": prompt_inputs,
        "final_answer": str(answer or "").strip(),
        "final_answer_for_eval": eval_response,
        "final_reasoning": str(reasoning or "").strip(),
        "final_answer_surface_source": str(response_for_eval_source or "method_answer").strip(),
        "final_answer_reason": "Generated from retrieved evidence with standardized compare payload alignment.",
    }
    if isinstance(answer_trace_extra, dict):
        answer_trace.update(answer_trace_extra)

    return {
        "answer_scope_target_doc_id": target_doc_id,
        "answer_composition_mode": answer_mode,
        "semantic_alignment": semantic_alignment,
        "evaluation_payload": eval_payload,
        "retrieval_trace": retrieval_trace,
        "reasoning_trace": reasoning_trace,
        "answer_trace": answer_trace,
        "reasoning_steps": list(reasoning_steps or []),
    }
