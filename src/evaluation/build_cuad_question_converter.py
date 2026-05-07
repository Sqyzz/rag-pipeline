from __future__ import annotations

import argparse
import json
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

try:
    from utils.llm_wrapper import llm_chat
except ModuleNotFoundError:
    from src.utils.llm_wrapper import llm_chat


LABEL_RE = re.compile(r'related to "([^"]+)"')
QUESTION_TYPES = ["local_retrieval", "structural_reasoning", "global_synthesis"]

TYPE_NAME_SCHEMES: dict[str, dict[str, str]] = {
    "semantic": {
        "local_retrieval": "local_retrieval",
        "structural_reasoning": "structural_reasoning",
        "global_synthesis": "global_synthesis",
    },
    "capability": {
        "local_retrieval": "local_factual",
        "structural_reasoning": "cross_clause",
        "global_synthesis": "global_summary",
    },
}

STRUCTURAL_LABELS = {
    "Most Favored Nation",
    "Competitive Restriction Exception",
    "Change Of Control",
    "Notice Period To Terminate Renewal",
    "Post-Termination Services",
    "Third Party Beneficiary",
    "Cap On Liability",
    "Uncapped Liability",
    "Anti-Assignment",
    "Termination For Convenience",
}

GLOBAL_THEMES: dict[str, list[str]] = {
    "term_and_termination": [
        "Effective Date",
        "Expiration Date",
        "Renewal Term",
        "Notice Period To Terminate Renewal",
        "Termination For Convenience",
        "Post-Termination Services",
    ],
    "assignment_and_control": [
        "Change Of Control",
        "Anti-Assignment",
        "Rofr/Rofo/Rofn",
        "Third Party Beneficiary",
    ],
    "liability_and_remedies": [
        "Cap On Liability",
        "Uncapped Liability",
        "Liquidated Damages",
        "Insurance",
        "Warranty Duration",
        "Audit Rights",
    ],
    "commercials": [
        "Revenue/Profit Sharing",
        "Price Restrictions",
        "Minimum Commitment",
        "Volume Restriction",
        "Most Favored Nation",
        "Exclusivity",
    ],
    "competition_constraints": [
        "Non-Compete",
        "No-Solicit Of Customers",
        "No-Solicit Of Employees",
        "Competitive Restriction Exception",
        "Non-Disparagement",
        "Covenant Not To Sue",
    ],
    "ip_and_license": [
        "License Grant",
        "Non-Transferable License",
        "Affiliate License-Licensor",
        "Affiliate License-Licensee",
        "Unlimited/All-You-Can-Eat-License",
        "Irrevocable Or Perpetual License",
        "Ip Ownership Assignment",
        "Joint Ip Ownership",
        "Source Code Escrow",
    ],
}

GLOBAL_THEME_INTENT: dict[str, str] = {
    "term_and_termination": "Summarize contract lifecycle and termination risk allocation.",
    "assignment_and_control": "Summarize assignment/control-transfer constraints and rights.",
    "liability_and_remedies": "Synthesize liability architecture, remedies, and enforcement.",
    "commercials": "Synthesize commercial obligations and pricing/volume/exclusivity risks.",
    "competition_constraints": "Synthesize competition restrictions, carve-outs, and post-term constraints.",
    "ip_and_license": "Synthesize IP ownership, license scope/transferability/duration, and safeguards.",
}

TYPE_ALIASES: dict[str, str] = {
    "local_factual": "local_retrieval",
    "cross_clause": "structural_reasoning",
    "global_summary": "global_synthesis",
}


@dataclass
class QAItem:
    label: str
    question: str
    answers: list[dict[str, Any]]
    is_impossible: bool


@dataclass
class DocData:
    title: str
    context: str
    labels: list[str]
    qas: list[QAItem]


def _qid_safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_") or "doc"


def _read_json(path: str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _map_type_name(canonical_type: str, scheme: str) -> str:
    mapping = TYPE_NAME_SCHEMES.get(scheme, TYPE_NAME_SCHEMES["semantic"])
    return mapping.get(canonical_type, canonical_type)


def _resolve_selected_types(raw_types: str | None) -> set[str]:
    if not str(raw_types or "").strip():
        return set(QUESTION_TYPES)
    resolved: set[str] = set()
    unknown: set[str] = set()
    for value in [x.strip() for x in str(raw_types).split(",") if x.strip()]:
        canonical = TYPE_ALIASES.get(value, value)
        if canonical not in QUESTION_TYPES:
            unknown.add(value)
            continue
        resolved.add(canonical)
    if unknown:
        allowed = QUESTION_TYPES + sorted(TYPE_ALIASES.keys())
        raise ValueError("Unknown value(s) in --types: " + ",".join(sorted(unknown)) + ". Allowed: " + ",".join(allowed))
    if not resolved:
        raise ValueError("--types must select at least one question type")
    return resolved


def _extract_docs(cuad_train_file: str) -> list[DocData]:
    payload = _read_json(cuad_train_file)
    docs: list[DocData] = []
    for item in payload.get("data", []) or []:
        title = str(item.get("title", "")).strip()
        if not title:
            continue
        context = ""
        qas: list[QAItem] = []
        labels: list[str] = []
        for para in item.get("paragraphs", []) or []:
            if not context:
                context = str(para.get("context", "") or "")
            for qa in para.get("qas", []) or []:
                q = str(qa.get("question", "") or "").strip()
                if not q:
                    continue
                m = LABEL_RE.search(q)
                label = m.group(1).strip() if m else ""
                if label:
                    labels.append(label)
                qas.append(
                    QAItem(
                        label=label,
                        question=q,
                        answers=list(qa.get("answers", []) or []),
                        is_impossible=bool(qa.get("is_impossible", False)),
                    )
                )
        docs.append(
            DocData(
                title=title,
                context=context,
                labels=sorted(set(x for x in labels if x)),
                qas=qas,
            )
        )
    return docs


def _choose_global_themes(labels: list[str], min_theme_labels: int, rng: random.Random) -> list[tuple[str, list[str]]]:
    available = set(labels)
    pool: list[tuple[str, list[str]]] = []
    for theme, theme_labels in GLOBAL_THEMES.items():
        used = [x for x in theme_labels if x in available]
        if len(used) >= int(min_theme_labels):
            pool.append((theme, used))
    rng.shuffle(pool)
    return pool


def _build_global_template(theme: str, labels: list[str], max_label_refs: int) -> tuple[str, str]:
    refs = labels[: max(1, int(max_label_refs))]
    query = f"{GLOBAL_THEME_INTENT.get(theme, 'Provide a global synthesis.')} Focus on: {', '.join(refs)}."
    answer = f"Expected synthesis should integrate: {', '.join(refs)}; explain key obligations, risks, and enforceability interactions."
    return query, answer


def _extract_answer_text(answers: list[dict[str, Any]], is_impossible: bool) -> str:
    if is_impossible:
        return "NOT_FOUND"
    vals = [str((a or {}).get("text", "") or "").strip() for a in answers]
    vals = [x for x in vals if x]
    if not vals:
        return "NOT_FOUND"
    return vals[0]


def _apply_impossible_ratio(
    pool: list[tuple[DocData, QAItem]],
    keep_ratio: float,
    rng: random.Random,
) -> tuple[list[tuple[DocData, QAItem]], int, int]:
    keep_ratio = max(0.0, min(1.0, float(keep_ratio)))
    possible = [row for row in pool if not row[1].is_impossible]
    impossible = [row for row in pool if row[1].is_impossible]
    if not impossible:
        return list(pool), 0, 0
    keep_n = min(len(impossible), max(0, int(round(len(impossible) * keep_ratio))))
    kept_impossible = rng.sample(impossible, keep_n) if keep_n > 0 else []
    merged = possible + kept_impossible
    rng.shuffle(merged)
    return merged, len(impossible), len(kept_impossible)


def _llm_generate_global_qa(
    *,
    title: str,
    context: str,
    theme: str,
    labels: list[str],
    max_label_refs: int,
    max_context_chars: int,
    temperature: float,
    max_completion_tokens: int,
) -> tuple[str, str, str | None]:
    refs = labels[: max(1, int(max_label_refs))]
    clipped = (context or "")[: max(2000, int(max_context_chars))]
    prompt = f"""
You are generating ONE global-synthesis QA pair for a legal contract.
Return strict JSON only with keys: query, answer.
Rules:
- query: one English question sentence (20-55 words), requiring whole-contract synthesis across multiple clauses.
- answer: concise evidence-grounded synthesis (2-4 sentences), based ONLY on provided contract text.
- If evidence is clearly insufficient, set answer to NOT_FOUND.

Contract title: {title}
Theme: {theme}
Focus labels: {', '.join(refs)}

Contract text (truncated):
{clipped}
""".strip()
    try:
        content, _ = llm_chat(
            [{"role": "user", "content": prompt}],
            temperature=float(temperature),
            max_tokens=int(max_completion_tokens),
            return_meta=True,
        )
        raw = str(content or "").strip()
        data = None
        try:
            data = json.loads(raw)
        except Exception:
            m = re.search(r"\{.*\}", raw, flags=re.S)
            if m:
                try:
                    data = json.loads(m.group(0))
                except Exception:
                    data = None
        if isinstance(data, dict):
            q = str(data.get("query", "") or "").strip()
            a = str(data.get("answer", "") or "").strip()
            if q and a:
                if not q.endswith("?"):
                    q = q.rstrip(". ") + "?"
                return q, a, None
        return "", "", "invalid_llm_json"
    except Exception as exc:
        return "", "", f"{type(exc).__name__}: {exc}"


def convert_cuad_questions(
    *,
    cuad_train_file: str,
    out_queries_file: str,
    out_gold_file: str,
    qas_per_type: int,
    total_per_type: int,
    mode: str,
    max_docs: int,
    min_theme_labels: int,
    max_label_refs_in_query: int,
    max_completion_tokens: int,
    temperature: float,
    seed: int,
    answer_mode: str,
    include_empty_support: bool,
    progress_every: int,
    progress_every_qa: int,
    type_name_scheme: str,
    llm_types: set[str],
    global_context_max_chars: int,
    impossible_keep_ratio: float,
    selected_types: set[str],
) -> dict[str, Any]:
    rng = random.Random(int(seed))
    docs = _extract_docs(cuad_train_file)
    if int(max_docs) > 0:
        docs = docs[: int(max_docs)]

    # Build local/cross candidate pools from original CUAD QA with gold answers.
    local_pool: list[tuple[DocData, QAItem]] = []
    cross_pool: list[tuple[DocData, QAItem]] = []
    for doc in docs:
        for qa in doc.qas:
            if not qa.label:
                continue
            if qa.label in STRUCTURAL_LABELS:
                cross_pool.append((doc, qa))
            else:
                local_pool.append((doc, qa))

    local_pool, local_impossible_total, local_impossible_kept = _apply_impossible_ratio(
        local_pool, impossible_keep_ratio, rng
    )
    cross_pool, cross_impossible_total, cross_impossible_kept = _apply_impossible_ratio(
        cross_pool, impossible_keep_ratio, rng
    )

    q_rows: list[dict[str, Any]] = []
    g_rows: list[dict[str, Any]] = []
    docs_sampled: set[str] = set()
    llm_fallbacks = 0
    skipped_docs = 0

    local_out = _map_type_name("local_retrieval", type_name_scheme)
    cross_out = _map_type_name("structural_reasoning", type_name_scheme)
    global_out = _map_type_name("global_synthesis", type_name_scheme)
    counts_by_type = {
        out_type: 0
        for canonical, out_type in (
            ("local_retrieval", local_out),
            ("structural_reasoning", cross_out),
            ("global_synthesis", global_out),
        )
        if canonical in selected_types
    }
    qid_counters: dict[tuple[str, str], int] = {}

    def _next_qid(doc_title: str, out_type: str) -> str:
        key = (_qid_safe(doc_title), out_type)
        qid_counters[key] = qid_counters.get(key, 0) + 1
        return f"{key[0]}__{out_type}__{qid_counters[key]:04d}"

    next_mark = max(int(progress_every_qa or 0), 0)

    def _append_row(
        *,
        doc: DocData,
        canonical_type: str,
        out_type: str,
        query: str,
        answer: str,
        meta_extra: dict[str, Any],
        ans_mode: str,
    ) -> None:
        nonlocal next_mark
        qid = _next_qid(doc.title, out_type)
        meta = {
            "query_doc_key": doc.title,
            "title": doc.title,
            "converter_source": "cuad_train_separate_questions",
            "converter_mode": mode,
            "converter_type": canonical_type,
            "converter_type_output": out_type,
            **meta_extra,
        }
        q_rows.append({"qid": qid, "type": out_type, "query": query, "meta": meta})
        grow = {
            "qid": qid,
            "type": out_type,
            "query": query,
            "answer": answer,
            "meta": dict(meta),
            "answer_mode": ans_mode,
        }
        if include_empty_support:
            grow["supporting_edges"] = []
            grow["supporting_chunks"] = []
            grow["supporting_communities"] = []
        g_rows.append(grow)
        counts_by_type[out_type] += 1
        docs_sampled.add(doc.title)
        if next_mark > 0 and len(q_rows) >= next_mark:
            payload = {
                "qa_generated": len(q_rows),
                "by_type": counts_by_type,
                "docs_sampled": len(docs_sampled),
                "llm_fallbacks": llm_fallbacks,
            }
            print("[qa_progress] " + json.dumps(payload, ensure_ascii=False), file=sys.stderr, flush=True)
            next_mark += max(int(progress_every_qa), 1)

    def _sample_pool(pool: list[tuple[DocData, QAItem]], target: int) -> list[tuple[DocData, QAItem]]:
        if target <= 0 or not pool:
            return []
        if target <= len(pool):
            return pool[:target]
        out = list(pool)
        while len(out) < target:
            out.append(pool[(len(out) - len(pool)) % len(pool)])
        return out[:target]

    if int(total_per_type) > 0:
        # Global total mode.
        local_target = int(total_per_type) if "local_retrieval" in selected_types else 0
        cross_target = int(total_per_type) if "structural_reasoning" in selected_types else 0
        global_target = int(total_per_type) if "global_synthesis" in selected_types else 0

        for doc, qa in _sample_pool(local_pool, local_target):
            _append_row(
                doc=doc,
                canonical_type="local_retrieval",
                out_type=local_out,
                query=qa.question,
                answer=_extract_answer_text(qa.answers, qa.is_impossible),
                meta_extra={
                    "converter_labels": [qa.label],
                    "converter_answer_source": "cuad_raw_answers",
                },
                ans_mode="extractive",
            )

        for doc, qa in _sample_pool(cross_pool, cross_target):
            _append_row(
                doc=doc,
                canonical_type="structural_reasoning",
                out_type=cross_out,
                query=qa.question,
                answer=_extract_answer_text(qa.answers, qa.is_impossible),
                meta_extra={
                    "converter_labels": [qa.label],
                    "converter_answer_source": "cuad_raw_answers",
                },
                ans_mode="extractive",
            )

        global_docs = list(docs)
        rng.shuffle(global_docs)
        if not global_docs:
            skipped_docs = 0
        for i in range(global_target):
            doc = global_docs[i % len(global_docs)]
            themes = _choose_global_themes(doc.labels, min_theme_labels=min_theme_labels, rng=rng)
            if themes:
                theme, glabels = themes[i % len(themes)]
            else:
                theme, glabels = ("fallback", doc.labels[:4] if doc.labels else ["Contract Terms"])

            use_llm = mode == "llm" and "global_synthesis" in llm_types
            err = None
            if use_llm:
                q, a, err = _llm_generate_global_qa(
                    title=doc.title,
                    context=doc.context,
                    theme=theme,
                    labels=glabels,
                    max_label_refs=max_label_refs_in_query,
                    max_context_chars=global_context_max_chars,
                    temperature=temperature,
                    max_completion_tokens=max_completion_tokens,
                )
                if not q or not a:
                    q, a = _build_global_template(theme, glabels, max_label_refs=max_label_refs_in_query)
                    if answer_mode == "placeholder":
                        a = "NOT_AVAILABLE"
                    llm_fallbacks += 1
            else:
                q, a = _build_global_template(theme, glabels, max_label_refs=max_label_refs_in_query)
                if answer_mode == "placeholder":
                    a = "NOT_AVAILABLE"
            meta_extra = {
                "converter_theme": theme,
                "converter_labels": glabels,
                "converter_answer_source": "llm_generated" if use_llm and not err else "template_generated",
            }
            if err:
                meta_extra["converter_llm_error"] = err
            _append_row(
                doc=doc,
                canonical_type="global_synthesis",
                out_type=global_out,
                query=q,
                answer=a,
                meta_extra=meta_extra,
                ans_mode="abstractive",
            )
    else:
        # Per-document mode for backward compatibility.
        for doc_idx, doc in enumerate(docs, start=1):
            raw_local_qas = [qa for qa in doc.qas if qa.label and qa.label not in STRUCTURAL_LABELS]
            raw_cross_qas = [qa for qa in doc.qas if qa.label and qa.label in STRUCTURAL_LABELS]
            local_qas = [
                qa for qa in raw_local_qas
                if (not qa.is_impossible) or rng.random() < float(impossible_keep_ratio)
            ]
            cross_qas = [
                qa for qa in raw_cross_qas
                if (not qa.is_impossible) or rng.random() < float(impossible_keep_ratio)
            ]
            if not local_qas:
                local_qas = [qa for qa in raw_local_qas if not qa.is_impossible]
            if not cross_qas:
                cross_qas = [qa for qa in raw_cross_qas if not qa.is_impossible]
            need_local = "local_retrieval" in selected_types
            need_cross = "structural_reasoning" in selected_types
            need_global = "global_synthesis" in selected_types
            if need_local and not local_qas and raw_local_qas:
                local_qas = [qa for qa in raw_local_qas if not qa.is_impossible]
            if need_cross and not cross_qas and raw_cross_qas:
                cross_qas = [qa for qa in raw_cross_qas if not qa.is_impossible]
            if (need_local and not local_qas) and (need_cross and not cross_qas) and not need_global:
                skipped_docs += 1
                continue
            rng.shuffle(local_qas)
            rng.shuffle(cross_qas)
            themes = _choose_global_themes(doc.labels, min_theme_labels=min_theme_labels, rng=rng)

            for i in range(int(qas_per_type)):
                if need_local and local_qas:
                    lqa = local_qas[i % len(local_qas)]
                    _append_row(
                        doc=doc,
                        canonical_type="local_retrieval",
                        out_type=local_out,
                        query=lqa.question,
                        answer=_extract_answer_text(lqa.answers, lqa.is_impossible),
                        meta_extra={
                            "converter_labels": [lqa.label],
                            "converter_answer_source": "cuad_raw_answers",
                        },
                        ans_mode="extractive",
                    )

                if need_cross and cross_qas:
                    cqa = cross_qas[i % len(cross_qas)]
                    _append_row(
                        doc=doc,
                        canonical_type="structural_reasoning",
                        out_type=cross_out,
                        query=cqa.question,
                        answer=_extract_answer_text(cqa.answers, cqa.is_impossible),
                        meta_extra={
                            "converter_labels": [cqa.label],
                            "converter_answer_source": "cuad_raw_answers",
                        },
                        ans_mode="extractive",
                    )

                if not need_global:
                    continue

                if themes:
                    theme, glabels = themes[i % len(themes)]
                else:
                    theme, glabels = ("fallback", doc.labels[:4] if doc.labels else ["Contract Terms"])

                use_llm = mode == "llm" and "global_synthesis" in llm_types
                err = None
                if use_llm:
                    q, a, err = _llm_generate_global_qa(
                        title=doc.title,
                        context=doc.context,
                        theme=theme,
                        labels=glabels,
                        max_label_refs=max_label_refs_in_query,
                        max_context_chars=global_context_max_chars,
                        temperature=temperature,
                        max_completion_tokens=max_completion_tokens,
                    )
                    if not q or not a:
                        q, a = _build_global_template(theme, glabels, max_label_refs=max_label_refs_in_query)
                        if answer_mode == "placeholder":
                            a = "NOT_AVAILABLE"
                        llm_fallbacks += 1
                else:
                    q, a = _build_global_template(theme, glabels, max_label_refs=max_label_refs_in_query)
                    if answer_mode == "placeholder":
                        a = "NOT_AVAILABLE"

                meta_extra = {
                    "converter_theme": theme,
                    "converter_labels": glabels,
                    "converter_answer_source": "llm_generated" if use_llm and not err else "template_generated",
                }
                if err:
                    meta_extra["converter_llm_error"] = err
                _append_row(
                    doc=doc,
                    canonical_type="global_synthesis",
                    out_type=global_out,
                    query=q,
                    answer=a,
                    meta_extra=meta_extra,
                    ans_mode="abstractive",
                )

            if int(progress_every) > 0 and (doc_idx % int(progress_every) == 0):
                payload = {
                    "sampling_mode": "per_doc",
                    "processed_docs": doc_idx,
                    "total_docs": len(docs),
                    "qa_generated": len(q_rows),
                    "by_type": counts_by_type,
                    "docs_sampled": len(docs_sampled),
                    "llm_fallbacks": llm_fallbacks,
                }
                print("[convert_cuad_questions] " + json.dumps(payload, ensure_ascii=False), file=sys.stderr, flush=True)

    _write_jsonl(out_queries_file, q_rows)
    _write_jsonl(out_gold_file, g_rows)

    return {
        "cuad_train_file": cuad_train_file,
        "out_queries_file": out_queries_file,
        "out_gold_file": out_gold_file,
        "mode": mode,
        "docs_input": len(docs),
        "docs_skipped": skipped_docs,
        "qas_per_type": int(qas_per_type),
        "total_per_type": int(total_per_type),
        "questions_generated": len(q_rows),
        "counts_by_type": counts_by_type,
        "docs_sampled": len(docs_sampled),
        "llm_fallbacks": llm_fallbacks,
        "llm_types": sorted(llm_types),
        "selected_types": sorted(selected_types),
        "answer_mode": answer_mode,
        "type_name_scheme": type_name_scheme,
        "progress_every_qa": int(progress_every_qa),
        "global_context_max_chars": int(global_context_max_chars),
        "impossible_keep_ratio": float(impossible_keep_ratio),
        "local_impossible_total": int(local_impossible_total),
        "local_impossible_kept": int(local_impossible_kept),
        "cross_impossible_total": int(cross_impossible_total),
        "cross_impossible_kept": int(cross_impossible_kept),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert CUAD QA into balanced local/cross/global sets. "
            "local/cross reuse raw CUAD query+answer; global is generated from document context."
        )
    )
    parser.add_argument("--cuad-train-file", default="data/raw/cuad/train_separate_questions.json", help="Input CUAD train_separate_questions JSON.")
    parser.add_argument("--out-queries-file", default="data/queries/cuad_converted_queries_generated.jsonl", help="Output query JSONL (qid/type/query/meta).")
    parser.add_argument("--out-gold-file", default="data/queries/cuad_converted_gold_generated.jsonl", help="Output gold JSONL (qid/type/query/answer/meta).")
    parser.add_argument("--mode", choices=["llm", "template"], default="llm", help="global generation mode: llm or template.")
    parser.add_argument(
        "--llm-types",
        default="global_synthesis",
        help=(
            "Comma-separated canonical types to generate via LLM when --mode llm. "
            "Current recommended: global_synthesis"
        ),
    )
    parser.add_argument("--qas-per-type", type=int, default=2, help="Per-document mode: number of QA items per type per document.")
    parser.add_argument("--total-per-type", type=int, default=0, help="Global sampling mode: total QA items per type across whole dataset.")
    parser.add_argument("--max-docs", type=int, default=0, help="Max documents to process; 0 means all.")
    parser.add_argument("--min-theme-labels", type=int, default=2, help="Minimum labels required for a global theme to be eligible.")
    parser.add_argument("--max-label-refs-in-query", type=int, default=4, help="Max label refs used for global generation guidance.")
    parser.add_argument("--max-completion-tokens", type=int, default=220, help="LLM max tokens per global QA generation call.")
    parser.add_argument("--temperature", type=float, default=0.2, help="LLM temperature in llm mode.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--answer-mode", choices=["placeholder", "weak_reference"], default="weak_reference", help="Affects template global answer fallback only.")
    parser.add_argument(
        "--types",
        default=",".join(QUESTION_TYPES),
        help=(
            "Comma-separated question types to emit. Supports canonical names "
            "(local_retrieval,structural_reasoning,global_synthesis) and output aliases "
            "(local_factual,cross_clause,global_summary)."
        ),
    )
    parser.add_argument(
        "--type-name-scheme",
        choices=["semantic", "capability"],
        default="semantic",
        help="Output type naming scheme.",
    )
    parser.add_argument("--no-empty-support", action="store_true", help="Do not include empty supporting arrays in gold output.")
    parser.add_argument("--progress-every", type=int, default=50, help="Per-doc/round progress interval; 0 disables.")
    parser.add_argument("--progress-every-qa", type=int, default=100, help="Progress interval by generated QA count; 0 disables.")
    parser.add_argument("--global-context-max-chars", type=int, default=12000, help="Max contract chars provided to LLM for global QA generation.")
    parser.add_argument(
        "--impossible-keep-ratio",
        type=float,
        default=0.0,
        help="For local/cross QA, retain this fraction of is_impossible=true samples from CUAD. Range [0,1].",
    )

    args = parser.parse_args()
    if int(args.total_per_type) <= 0 and int(args.qas_per_type) <= 0:
        raise ValueError("Either --total-per-type > 0 or --qas-per-type > 0 is required")

    raw_llm_types = [x.strip() for x in str(args.llm_types or "").split(",") if x.strip()]
    llm_types: set[str] = set(raw_llm_types)
    unknown = sorted(x for x in llm_types if x not in QUESTION_TYPES)
    if unknown:
        raise ValueError("Unknown value(s) in --llm-types: " + ",".join(unknown) + ". Allowed: " + ",".join(QUESTION_TYPES))
    selected_types = _resolve_selected_types(args.types)
    llm_types = llm_types & selected_types

    stats = convert_cuad_questions(
        cuad_train_file=args.cuad_train_file,
        out_queries_file=args.out_queries_file,
        out_gold_file=args.out_gold_file,
        qas_per_type=int(args.qas_per_type),
        total_per_type=int(args.total_per_type),
        mode=args.mode,
        max_docs=int(args.max_docs),
        min_theme_labels=int(args.min_theme_labels),
        max_label_refs_in_query=int(args.max_label_refs_in_query),
        max_completion_tokens=int(args.max_completion_tokens),
        temperature=float(args.temperature),
        seed=int(args.seed),
        answer_mode=str(args.answer_mode),
        include_empty_support=not args.no_empty_support,
        progress_every=int(args.progress_every),
        progress_every_qa=int(args.progress_every_qa),
        type_name_scheme=str(args.type_name_scheme),
        llm_types=llm_types,
        global_context_max_chars=int(args.global_context_max_chars),
        impossible_keep_ratio=float(args.impossible_keep_ratio),
        selected_types=selected_types,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
