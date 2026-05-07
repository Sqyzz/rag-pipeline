from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

try:
    from utils.llm_wrapper import llm_chat
except ModuleNotFoundError:
    from src.utils.llm_wrapper import llm_chat


def _doc_prefix(value: str | None) -> str:
    return str(value or "").split("#", 1)[0].strip()


def _qid_safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_") or "doc"


def _load_json(path: str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _community_chunk_map(graph: dict, communities: dict) -> dict[str, set[str]]:
    edge_by_id = {str(e.get("edge_id")): e for e in graph.get("edges", []) if e.get("edge_id")}
    out: dict[str, set[str]] = {}
    for c in communities.get("communities", []) or []:
        cid = str(c.get("community_id", "")).strip()
        if not cid:
            continue
        chunk_ids: set[str] = set()
        for eid in c.get("edges", []) or []:
            edge = edge_by_id.get(str(eid))
            if not edge:
                continue
            for m in edge.get("mentions", []) or []:
                chunk_id = str(m.get("chunk_id", "")).strip()
                if chunk_id:
                    chunk_ids.add(chunk_id)
        if chunk_ids:
            out[cid] = chunk_ids
    return out


def _mentions_by_doc(graph: dict) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for e in graph.get("edges", []) or []:
        edge_row = {
            "edge_id": str(e.get("edge_id", "")).strip(),
            "source": str(e.get("source", "")).strip(),
            "relation": str(e.get("relation", "")).strip(),
            "target": str(e.get("target", "")).strip(),
            "weight": int(e.get("weight", 1) or 1),
        }
        if not edge_row["edge_id"] or not edge_row["source"] or not edge_row["relation"] or not edge_row["target"]:
            continue
        for m in e.get("mentions", []) or []:
            doc_id = str(m.get("doc_id", "")).strip()
            chunk_id = str(m.get("chunk_id", "")).strip()
            if not doc_id:
                continue
            out.setdefault(doc_id, []).append(
                {
                    **edge_row,
                    "doc_id": doc_id,
                    "doc_key": _doc_prefix(doc_id),
                    "chunk_id": chunk_id,
                    "evidence": str(m.get("evidence", "")).strip(),
                }
            )
    return out


def _chunk_to_doc_key(mentions: dict[str, list[dict]]) -> dict[str, str]:
    out: dict[str, str] = {}
    for rows in mentions.values():
        for r in rows:
            chunk_id = str(r.get("chunk_id", "")).strip()
            doc_key = str(r.get("doc_key", "")).strip()
            if chunk_id and doc_key:
                out[chunk_id] = doc_key
    return out


def _chunk_meta_from_mentions(mentions: dict[str, list[dict]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for rows in mentions.values():
        for r in rows:
            chunk_id = str(r.get("chunk_id", "")).strip()
            if not chunk_id:
                continue
            doc_key = str(r.get("doc_key", "")).strip()
            doc_id = str(r.get("doc_id", "")).strip()
            text = str(r.get("evidence", "")).strip()
            prev = out.get(chunk_id)
            if not prev:
                out[chunk_id] = {
                    "chunk_id": chunk_id,
                    "doc_key": doc_key,
                    "doc_id": doc_id,
                    "text": text,
                }
                continue
            # Keep the longest evidence snippet as chunk proxy text.
            if len(text) > len(str(prev.get("text", "") or "")):
                prev["text"] = text
            if not str(prev.get("doc_key", "")).strip() and doc_key:
                prev["doc_key"] = doc_key
            if not str(prev.get("doc_id", "")).strip() and doc_id:
                prev["doc_id"] = doc_id
    return out


_DOC_KEY_TOKEN_STOPWORDS = {
    "agreement",
    "contract",
    "amendment",
    "restated",
    "ex",
    "inc",
    "corp",
    "corporation",
    "company",
    "ltd",
    "llc",
    "holdings",
    "services",
    "joint",
    "venture",
}


def _doc_key_anchor_tokens(doc_key: str) -> list[str]:
    toks = [x for x in re.split(r"[^a-z0-9]+", _norm_text(doc_key)) if x]
    out = [x for x in toks if len(x) >= 4 and not x.isdigit() and x not in _DOC_KEY_TOKEN_STOPWORDS]
    # Limit to keep heuristic stable and cheap.
    return out[:8]


def _chunk_quality_ok(text: str, min_chars: int = 100) -> bool:
    s = str(text or "").strip()
    if len(s) < int(min_chars):
        return False
    if _word_count(s) < 15:
        return False
    if _has_noisy_symbols(s):
        return False
    return True


def _doc_chunk_consistency_hits(doc_key: str, chunk_text: str) -> int:
    anchors = _doc_key_anchor_tokens(doc_key)
    if not anchors:
        return 0
    t = _norm_text(chunk_text)
    return sum(1 for x in anchors if x in t)


_RELATION_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is",
    "of", "on", "or", "that", "the", "this", "to", "under", "with",
}

_GENERIC_LOCAL_TARGETS = {
    "law",
    "agreement",
    "this agreement",
    "party a",
    "party b",
}


def _relation_keywords(relation: str) -> list[str]:
    tokens = [x for x in re.split(r"[^a-z0-9]+", str(relation or "").lower()) if x]
    return [x for x in tokens if x not in _RELATION_STOPWORDS and len(x) >= 3]


def _word_count(text: str) -> int:
    return len([x for x in re.split(r"\s+", str(text or "").strip()) if x])


def _has_noisy_symbols(text: str) -> bool:
    t = str(text or "")
    if not t:
        return True
    noisy = len(re.findall(r"[~`|<>_\[\]\(\){}]", t))
    return (noisy / max(len(t), 1)) > 0.06


def _row_is_grounded(row: dict, require_relation_keyword: bool = True) -> bool:
    text = str(row.get("evidence", "") or "").lower()
    source = str(row.get("source", "") or "").strip().lower()
    target = str(row.get("target", "") or "").strip().lower()
    rel = str(row.get("relation", "") or "")
    if not text or not source or not target or not rel:
        return False
    if source not in text or target not in text:
        return False
    if not require_relation_keyword:
        return True
    rel_keywords = _relation_keywords(rel)
    if not rel_keywords:
        return False
    return any(k in text for k in rel_keywords)


def _row_is_youtu_friendly_local(row: dict) -> bool:
    if not _row_is_grounded(row, require_relation_keyword=True):
        return False
    source = str(row.get("source", "")).strip()
    target = str(row.get("target", "")).strip()
    if not source or not target:
        return False
    if _word_count(source) > 6 or _word_count(target) > 6:
        return False
    if target.lower() in _GENERIC_LOCAL_TARGETS:
        return False
    if _has_noisy_symbols(source) or _has_noisy_symbols(target):
        return False
    return True


def _pair_is_youtu_friendly_cross(a: dict, b: dict) -> bool:
    if not (
        _row_is_grounded(a, require_relation_keyword=False)
        and _row_is_grounded(b, require_relation_keyword=False)
    ):
        return False
    if str(a.get("doc_id", "")).strip() != str(b.get("doc_id", "")).strip():
        return False
    if str(a.get("source", "")).strip() == str(b.get("source", "")).strip():
        return False
    if str(a.get("relation", "")).strip() == str(b.get("relation", "")).strip():
        return False
    for row in (a, b):
        source = str(row.get("source", "")).strip()
        target = str(row.get("target", "")).strip()
        if _word_count(source) > 10 or _word_count(target) > 10:
            return False
        if _has_noisy_symbols(source) or _has_noisy_symbols(target):
            return False
    return True


def _add_example(
    *,
    queries: list[dict],
    gold: list[dict],
    qid: str,
    qtype: str,
    query: str,
    answer: Any,
    doc_key: str,
    supporting_edges: list[dict] | None = None,
    supporting_chunks: list[dict] | None = None,
    supporting_communities: list[dict] | None = None,
    extra_gold: dict[str, Any] | None = None,
) -> None:
    meta = {"query_doc_key": doc_key, "title": doc_key}
    queries.append(
        {
            "qid": qid,
            "type": qtype,
            "query": query,
            "meta": meta,
        }
    )
    row = {
        "qid": qid,
        "type": qtype,
        "query": query,
        "answer": answer,
        "meta": meta,
        "supporting_edges": supporting_edges or [],
        "supporting_chunks": supporting_chunks or [],
        "supporting_communities": supporting_communities or [],
    }
    if isinstance(extra_gold, dict) and extra_gold:
        row.update(extra_gold)
    gold.append(row)


def _extract_json_obj(raw: str) -> dict | None:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if not m:
        return None
    try:
        obj = json.loads(m.group(1))
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def _norm_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def _clip(text: str, max_chars: int = 360) -> str:
    s = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(s) <= int(max_chars):
        return s
    return s[: int(max_chars) - 3] + "..."


def _llm_designed_query(
    *,
    qtype: str,
    template_query: str,
    doc_key: str,
    answer: str,
    evidence_snippets: list[str],
    model: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 96,
    debug: dict[str, Any] | None = None,
) -> str | None:
    if debug is not None:
        debug.clear()
        debug.update(
            {
                "ok": False,
                "stage": "init",
                "reason": "",
                "raw": "",
                "candidate": "",
            }
        )
    snippets = [x for x in (_clip(s, max_chars=420) for s in (evidence_snippets or [])) if x]
    evidence_block = "\n".join(f"- {s}" for s in snippets[:3]) or "- (no snippet)"
    prompt = (
        "You are designing a contract QA evaluation question.\n"
        "Return JSON only: {\"query\": \"...\"}\n"
        "Constraints:\n"
        "1) One sentence, <= 28 words, in English.\n"
        "2) Do NOT include the answer text verbatim.\n"
        "3) Keep the same capability intent as the template question.\n"
        "4) Stay faithful to provided contract context.\n\n"
        f"Question type: {qtype}\n"
        f"Contract title: {doc_key}\n"
        f"Template question: {template_query}\n"
        f"Gold answer (for grounding only, do not reveal): {answer}\n"
        f"Evidence snippets:\n{evidence_block}\n"
    )
    try:
        raw, _ = llm_chat(
            [{"role": "user", "content": prompt}],
            temperature=float(temperature),
            max_tokens=int(max_tokens),
            return_meta=True,
            model=(str(model).strip() if model is not None and str(model).strip() else None),
        )
    except Exception as exc:
        if debug is not None:
            debug.update(
                {
                    "stage": "llm_chat",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
        return None
    if debug is not None:
        debug.update({"stage": "postprocess", "raw": str(raw or "")})
    obj = _extract_json_obj(str(raw))
    cand = ""
    if isinstance(obj, dict):
        cand = str(obj.get("query", "")).strip()
    if not cand:
        cand = str(raw or "").strip().splitlines()[0].strip()
    cand = re.sub(r"^\s*(question|query)\s*:\s*", "", cand, flags=re.IGNORECASE).strip().strip('"').strip("'")
    cand = re.sub(r"\s+", " ", cand).strip()
    if not cand:
        if debug is not None:
            debug.update({"stage": "validate", "reason": "empty_candidate"})
        return None
    if len(cand) < 10 or len(cand) > 280:
        if debug is not None:
            debug.update({"stage": "validate", "reason": f"candidate_len_out_of_range:{len(cand)}", "candidate": cand})
        return None
    if "?" not in cand:
        cand = cand.rstrip(".") + "?"
    a = _norm_text(answer)
    if a and len(a) >= 4 and a in _norm_text(cand):
        if debug is not None:
            debug.update({"stage": "validate", "reason": "answer_leakage", "candidate": cand})
        return None
    if debug is not None:
        debug.update({"ok": True, "stage": "done", "reason": "", "candidate": cand})
    return cand


def _split_sentences(text: str) -> list[str]:
    s = str(text or "").strip()
    if not s:
        return []
    parts = re.split(r"(?<=[\.\!\?。！？；;])\s+", s)
    return [x.strip() for x in parts if x.strip()]


def _token_set(text: str) -> set[str]:
    toks = [x for x in re.split(r"[^a-zA-Z0-9]+", _norm_text(text)) if x]
    return set(toks)


def _contains_cjk(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", str(text or "")))


def _answer_len_ok(answer: str) -> bool:
    ans = str(answer or "").strip()
    if not ans:
        return False
    if _contains_cjk(ans):
        cjk_chars = re.findall(r"[\u4e00-\u9fff]", ans)
        n = len(cjk_chars)
        return 3 <= n <= 20
    n_tokens = len([x for x in re.split(r"\s+", ans) if x])
    return 1 <= n_tokens <= 12


def _char_ngram_overlap_ratio(a: str, b: str, n: int = 5) -> float:
    x = re.sub(r"\s+", " ", str(a or "").strip().lower())
    y = re.sub(r"\s+", " ", str(b or "").strip().lower())
    if len(x) < n or len(y) < n:
        return 0.0
    gx = {x[i : i + n] for i in range(len(x) - n + 1)}
    gy = {y[i : i + n] for i in range(len(y) - n + 1)}
    if not gx:
        return 0.0
    return len(gx & gy) / len(gx)


def _local_query_check(
    query: str,
    chunk_text: str,
    *,
    max_overlap: float = 0.72,
    max_and_or_hits: int = 1,
) -> tuple[bool, str | None, float]:
    q = re.sub(r"\s+", " ", str(query or "")).strip()
    if not q:
        return False, "empty_query", 0.0
    if len(q) < 8 or len(q) > 280:
        return False, "query_len_out_of_range", 0.0
    forbidden = [
        "why", "how", "impact", "advantage", "disadvantage", "evaluate",
        "原因", "如何", "影响", "优缺点", "评价", "分析",
    ]
    qn = _norm_text(q)
    if any(x in qn for x in forbidden):
        return False, "forbidden_explanatory", 0.0
    # One question should target one local fact; avoid conjunction-heavy asks.
    and_or_hits = sum(qn.count(x) for x in (" and ", " or ", " 以及 ", " 且 ", " 并且 "))
    if and_or_hits > int(max_and_or_hits):
        return False, "multi_constraint_like", 0.0
    # Avoid near-copying the source sentence.
    overlap = _char_ngram_overlap_ratio(q, chunk_text, n=5)
    if overlap > float(max_overlap):
        return False, "overlap_too_high", overlap
    return True, None, overlap


def _local_query_allowed(query: str, chunk_text: str) -> bool:
    ok, _, _ = _local_query_check(query, chunk_text)
    return ok


def _resolve_local_evidence_span(
    *,
    chunk_text: str,
    answer: str,
    evidence_obj: dict | None,
) -> dict[str, Any] | None:
    text = str(chunk_text or "")
    ans = str(answer or "").strip()
    if not text or not ans:
        return None
    evidence_obj = evidence_obj if isinstance(evidence_obj, dict) else {}
    quote = str(evidence_obj.get("quote", "")).strip()
    start_raw = evidence_obj.get("start_char")
    end_raw = evidence_obj.get("end_char")
    sent_idx_raw = evidence_obj.get("sentence_index")

    start = None
    end = None
    if isinstance(start_raw, int) and isinstance(end_raw, int):
        if 0 <= int(start_raw) < int(end_raw) <= len(text):
            start = int(start_raw)
            end = int(end_raw)
            if not quote:
                quote = text[start:end]
    if quote:
        pos = text.find(quote)
        if pos < 0:
            return None
        if start is None or end is None:
            start = pos
            end = pos + len(quote)
    if (start is None or end is None) and isinstance(sent_idx_raw, int):
        sents = _split_sentences(text)
        idx = int(sent_idx_raw)
        if 0 <= idx < len(sents):
            sent = sents[idx]
            pos = text.find(sent)
            if pos >= 0:
                start = pos
                end = pos + len(sent)
                quote = sent
    if start is None or end is None or not quote:
        return None
    # Answer must be extractive substring OR short locatable paraphrase tied to quote/span.
    answer_in_text = ans in text
    if not answer_in_text:
        overlap = _token_set(ans) & _token_set(quote)
        cjk_overlap = set(re.findall(r"[\u4e00-\u9fff]", ans)) & set(re.findall(r"[\u4e00-\u9fff]", quote))
        if not overlap and not cjk_overlap:
            return None
    return {
        "quote": quote,
        "start_char": int(start),
        "end_char": int(end),
        "sentence_index": int(sent_idx_raw) if isinstance(sent_idx_raw, int) else None,
        "answer_grounding": ("extractive" if answer_in_text else "abstractive_locatable"),
    }


def _llm_designed_local_qa(
    *,
    doc_key: str,
    relation: str,
    chunk_text: str,
    model: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 220,
    local_query_max_overlap: float = 0.72,
    local_query_max_and_or_hits: int = 1,
) -> tuple[dict[str, Any] | None, str | None, dict[str, Any] | None]:
    evidence = str(chunk_text or "").strip()
    if not evidence:
        return None, "empty_chunk_text", None
    prompt = (
        "You are generating ONE local_factual QA item from ONE contract chunk.\n"
        "Return JSON only with schema:\n"
        "{\"query\":\"...\",\"answer\":\"...\",\"answer_mode\":\"extractive|abstractive\","
        "\"evidence\":{\"quote\":\"...\",\"start_char\":0,\"end_char\":10,\"sentence_index\":0}}\n"
        "Hard rules:\n"
        "1) Local only: use ONLY facts from the given chunk; no outside background.\n"
        "2) Single fact only: no multi-hop / no combined constraints.\n"
        "3) Question must be factual, not explanatory. Avoid why/how/impact/pros-cons.\n"
        "4) Prefer extractive answer. If abstractive, keep it short and strictly locatable to a quote/span.\n"
        "5) Answer must be short: Chinese 3-20 chars OR English 1-12 tokens.\n"
        "6) Must provide evidence span by quote or (start_char,end_char) or sentence_index.\n"
        "7) Question should be local_factual type among entity/object/time/quantity/term/obligation/exception/liability-cap.\n"
        "8) Do not copy sentence verbatim as question.\n\n"
        f"Contract title: {doc_key}\n"
        f"Relation hint: {relation}\n"
        f"Chunk text:\n{evidence}\n"
    )
    try:
        raw, _ = llm_chat(
            [{"role": "user", "content": prompt}],
            temperature=float(temperature),
            max_tokens=int(max_tokens),
            return_meta=True,
            model=(str(model).strip() if model is not None and str(model).strip() else None),
        )
    except Exception:
        return None, "llm_call_error", None
    obj = _extract_json_obj(str(raw))
    if not isinstance(obj, dict):
        return None, "invalid_json_response", None
    query = re.sub(r"\s+", " ", str(obj.get("query", "")).strip())
    answer = re.sub(r"\s+", " ", str(obj.get("answer", "")).strip())
    answer_mode = str(obj.get("answer_mode", "")).strip().lower()
    if not query or not answer:
        return None, "missing_query_or_answer", {"query": query, "answer": answer}
    if answer_mode not in {"extractive", "abstractive"}:
        return None, "invalid_answer_mode", {"answer_mode": answer_mode}
    ok_q, q_reason, overlap = _local_query_check(
        query,
        evidence,
        max_overlap=float(local_query_max_overlap),
        max_and_or_hits=int(local_query_max_and_or_hits),
    )
    if not ok_q:
        return None, "query_constraint_failed", {
            "query_reason": q_reason,
            "rejected_query": query,
            "query_overlap": round(float(overlap), 6),
            "local_query_max_overlap": float(local_query_max_overlap),
            "local_query_max_and_or_hits": int(local_query_max_and_or_hits),
        }
    if not _answer_len_ok(answer):
        return None, "answer_length_failed", {"answer": answer}
    span = _resolve_local_evidence_span(
        chunk_text=evidence,
        answer=answer,
        evidence_obj=obj.get("evidence"),
    )
    if not span:
        return None, "span_not_locatable", {"answer": answer}
    if answer_mode == "extractive" and answer not in evidence:
        return None, "extractive_answer_not_substring", {"answer": answer}
    return {
        "query": (query if "?" in query else query.rstrip(".") + "?"),
        "answer": answer,
        "answer_mode": answer_mode,
        "evidence_span": span,
    }, None, None


def _llm_designed_cross_qa(
    *,
    doc_key: str,
    template_query: str,
    template_answer: str,
    a: dict,
    b: dict,
    model: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 160,
) -> tuple[str, str] | None:
    chain = {
        "edge_1": {
            "source": str(a.get("source", "")).strip(),
            "relation": str(a.get("relation", "")).strip(),
            "target": str(a.get("target", "")).strip(),
            "evidence": _clip(str(a.get("evidence", "")).strip(), max_chars=700),
        },
        "edge_2": {
            "source": str(b.get("source", "")).strip(),
            "relation": str(b.get("relation", "")).strip(),
            "target": str(b.get("target", "")).strip(),
            "evidence": _clip(str(b.get("evidence", "")).strip(), max_chars=700),
        },
    }
    prompt = (
        "You are designing a cross-clause QA item for contract evaluation.\n"
        "Return JSON only: {\"query\": \"...\", \"answer\": \"...\"}\n"
        "Constraints:\n"
        "1) Query: one sentence, <= 32 words, should ask for two linked clause terms.\n"
        "2) Answer: concise and fully grounded in evidence chain.\n"
        "3) Do not invent entities, relations, or facts.\n"
        "4) The answer must cover BOTH links in edge_1 and edge_2.\n\n"
        f"Contract title: {doc_key}\n"
        f"Template query: {template_query}\n"
        f"Template answer: {template_answer}\n"
        f"Evidence chain (JSON):\n{json.dumps(chain, ensure_ascii=False)}\n"
    )
    try:
        raw, _ = llm_chat(
            [{"role": "user", "content": prompt}],
            temperature=float(temperature),
            max_tokens=int(max_tokens),
            return_meta=True,
            model=(str(model).strip() if model is not None and str(model).strip() else None),
        )
    except Exception:
        return None
    obj = _extract_json_obj(str(raw))
    if not isinstance(obj, dict):
        return None
    q = re.sub(r"\s+", " ", str(obj.get("query", "")).strip())
    ans = re.sub(r"\s+", " ", str(obj.get("answer", "")).strip())
    if not q or not ans:
        return None
    if len(q) < 10 or len(q) > 300:
        return None
    if len(ans) < 8 or len(ans) > 420:
        return None
    if "?" not in q:
        q = q.rstrip(".") + "?"
    ans_norm = _norm_text(ans)
    tgt_a = _norm_text(str(a.get("target", "")).strip())
    tgt_b = _norm_text(str(b.get("target", "")).strip())
    # Keep strict grounding guard: answer should explicitly contain both gold targets.
    if tgt_a and tgt_a not in ans_norm:
        return None
    if tgt_b and tgt_b not in ans_norm:
        return None
    return q, ans


def build_cuad_capability_qa(
    graph_file: str,
    queries_out: str,
    gold_out: str,
    communities_file: str | None = None,
    per_type: int = 50,
    random_seed: int = 42,
    global_summary_chunk_cap: int = 30,
    profile: str = "default",
    summary_answer_max_chars: int = 0,
    question_style: str = "template",
    llm_question_model: str | None = None,
    llm_question_temperature: float = 0.2,
    llm_question_max_tokens: int = 96,
    llm_cross_generate_qa: bool = False,
    llm_cross_answer_max_tokens: int = 160,
    progress_every: int = 0,
    local_query_max_overlap: float = 0.72,
    local_query_max_and_or_hits: int = 1,
    global_summary_min_high_quality_chunks: int = 2,
    global_summary_target_high_quality_chunks: int = 3,
    global_summary_doc_consistency_min_hits: int = 1,
) -> dict:
    graph = _load_json(graph_file)
    communities = _load_json(communities_file) if communities_file and Path(communities_file).exists() else None
    community_chunks = (
        _community_chunk_map(graph, communities)
        if isinstance(communities, dict)
        else {}
    )
    mentions = _mentions_by_doc(graph)
    chunk_to_doc = _chunk_to_doc_key(mentions)
    chunk_meta = _chunk_meta_from_mentions(mentions)

    rng = random.Random(int(random_seed))
    queries: list[dict] = []
    gold: list[dict] = []
    seen_qid: set[str] = set()
    seen_query_text: set[str] = set()
    llm_question_attempts = 0
    llm_question_success = 0
    llm_question_fallback = 0
    llm_cross_qa_attempts = 0
    llm_cross_qa_success = 0
    llm_cross_qa_fallback = 0
    local_llm_fail_reasons: dict[str, int] = {}
    generated_by_type: dict[str, int] = {}

    progress_step = max(int(progress_every or 0), 0)
    target_total = int(per_type) * (3 if isinstance(communities, dict) else 2)
    summary_quality_stats = {
        "candidates_total": 0,
        "candidates_after_quality": 0,
        "dropped_low_quality": 0,
        "dropped_doc_consistency": 0,
        "dropped_below_min_chunks": 0,
    }

    def maybe_log_progress(stage: str) -> None:
        if progress_step <= 0:
            return
        generated = len(queries)
        if generated <= 0 or (generated % progress_step) != 0:
            return
        llm_attempts_total = int(llm_question_attempts) + int(llm_cross_qa_attempts)
        llm_success_total = int(llm_question_success) + int(llm_cross_qa_success)
        llm_fallback_total = int(llm_question_fallback) + int(llm_cross_qa_fallback)
        payload = {
            "stage": stage,
            "generated": generated,
            "target_total": target_total,
            "by_type": generated_by_type,
            "llm_question_attempts": int(llm_question_attempts),
            "llm_question_success": int(llm_question_success),
            "llm_question_fallback": int(llm_question_fallback),
            "llm_cross_qa_attempts": int(llm_cross_qa_attempts),
            "llm_cross_qa_success": int(llm_cross_qa_success),
            "llm_cross_qa_fallback": int(llm_cross_qa_fallback),
            "llm_attempts_total": llm_attempts_total,
            "llm_success_total": llm_success_total,
            "llm_fallback_total": llm_fallback_total,
            "local_llm_fail_reasons": dict(sorted(local_llm_fail_reasons.items())),
        }
        print(
            "[build_cuad_capability_qa] " + json.dumps(payload, ensure_ascii=False),
            file=sys.stderr,
            flush=True,
        )

    def add_example_tracked(stage: str, **kwargs: Any) -> None:
        _add_example(**kwargs)
        qtype = str(kwargs.get("qtype", "unknown"))
        generated_by_type[qtype] = generated_by_type.get(qtype, 0) + 1
        maybe_log_progress(stage=stage)

    def next_qid(doc_key: str, qtype: str, idx: int) -> str:
        base = f"{_qid_safe(doc_key)}__{qtype}__{idx:04d}"
        if base not in seen_qid:
            seen_qid.add(base)
            return base
        j = idx
        while True:
            j += 1
            cand = f"{_qid_safe(doc_key)}__{qtype}__{j:04d}"
            if cand not in seen_qid:
                seen_qid.add(cand)
                return cand

    def dedup_query_text(query: str, doc_key: str) -> str:
        q = str(query or "").strip()
        if not q:
            return q
        if q not in seen_query_text:
            seen_query_text.add(q)
            return q
        scoped = f'{q} [Contract: "{doc_key}"]'
        if scoped not in seen_query_text:
            seen_query_text.add(scoped)
            return scoped
        i = 2
        while True:
            cand = f"{scoped} (v{i})"
            if cand not in seen_query_text:
                seen_query_text.add(cand)
                return cand
            i += 1

    def maybe_design_query(
        *,
        qtype: str,
        template_query: str,
        doc_key: str,
        answer: str,
        evidence_snippets: list[str],
    ) -> str:
        nonlocal llm_question_attempts, llm_question_success, llm_question_fallback
        if str(question_style).strip().lower() != "llm":
            return template_query
        llm_question_attempts += 1
        designed = _llm_designed_query(
            qtype=qtype,
            template_query=template_query,
            doc_key=doc_key,
            answer=answer,
            evidence_snippets=evidence_snippets,
            model=llm_question_model,
            temperature=float(llm_question_temperature),
            max_tokens=int(llm_question_max_tokens),
        )
        if designed:
            llm_question_success += 1
            return designed
        llm_question_fallback += 1
        return template_query

    def resolve_cross_qa(a: dict, b: dict, doc_key: str) -> tuple[str, str]:
        nonlocal llm_cross_qa_attempts, llm_cross_qa_success, llm_cross_qa_fallback
        template_q = (
            "Provide the exact terms for two clause links: "
            f'`{a["source"]}` `{a["relation"]}` ?, and `{b["source"]}` `{b["relation"]}` ?'
        )
        template_ans = f'{a["relation"]}: {a["target"]}; {b["relation"]}: {b["target"]}'

        q = template_q
        ans = template_ans
        if str(question_style).strip().lower() == "llm" and bool(llm_cross_generate_qa):
            llm_cross_qa_attempts += 1
            designed = _llm_designed_cross_qa(
                doc_key=doc_key,
                template_query=template_q,
                template_answer=template_ans,
                a=a,
                b=b,
                model=llm_question_model,
                temperature=float(llm_question_temperature),
                max_tokens=int(llm_cross_answer_max_tokens),
            )
            if designed:
                q, ans = designed
                llm_cross_qa_success += 1
            else:
                llm_cross_qa_fallback += 1
                q = maybe_design_query(
                    qtype="cross_clause",
                    template_query=template_q,
                    doc_key=doc_key,
                    answer=template_ans,
                    evidence_snippets=[
                        str(a.get("evidence", "")).strip(),
                        str(b.get("evidence", "")).strip(),
                    ],
                )
                ans = template_ans
        else:
            q = maybe_design_query(
                qtype="cross_clause",
                template_query=template_q,
                doc_key=doc_key,
                answer=template_ans,
                evidence_snippets=[
                    str(a.get("evidence", "")).strip(),
                    str(b.get("evidence", "")).strip(),
                ],
            )
            ans = template_ans
        return dedup_query_text(q, doc_key), ans

    # local_factual
    # Build a richer candidate pool and split into:
    # 1) single-target rows where (doc, source, relation) maps to a unique target;
    # 2) multi-target groups rendered as list questions.
    local_candidates: list[dict] = []
    for _, rows in mentions.items():
        if not rows:
            continue
        local_candidates.extend(sorted(rows, key=lambda x: x["weight"], reverse=True))
    rng.shuffle(local_candidates)
    grouped_local: dict[tuple[str, str, str], list[dict]] = {}
    for row in local_candidates:
        key = (
            str(row.get("doc_key", "")).strip(),
            str(row.get("source", "")).strip().lower(),
            str(row.get("relation", "")).strip().lower(),
        )
        grouped_local.setdefault(key, []).append(row)
    single_target_rows: list[dict] = []
    multi_target_groups: list[dict] = []
    for (_, _, _), rows in grouped_local.items():
        if not rows:
            continue
        by_target: dict[str, list[dict]] = {}
        for row in rows:
            tgt_key = str(row.get("target", "")).strip().lower()
            if not tgt_key:
                continue
            by_target.setdefault(tgt_key, []).append(row)
        unique_targets = sorted(by_target.keys())
        if len(unique_targets) == 1:
            # Keep only unique (source, relation)->target rows for local factual singles.
            best_row = max(rows, key=lambda x: int(x.get("weight", 1) or 1))
            single_target_rows.append(best_row)
            continue
        dedup_targets: list[str] = []
        supporting_rows: list[dict] = []
        for tgt_key in unique_targets:
            candidates = by_target.get(tgt_key, [])
            if not candidates:
                continue
            best_row = max(candidates, key=lambda x: int(x.get("weight", 1) or 1))
            supporting_rows.append(best_row)
            dedup_targets.append(str(best_row.get("target", "")).strip())
        if not supporting_rows or not dedup_targets:
            continue
        seed = max(supporting_rows, key=lambda x: int(x.get("weight", 1) or 1))
        multi_target_groups.append(
            {
                "doc_key": str(seed.get("doc_key", "")).strip(),
                "doc_id": str(seed.get("doc_id", "")).strip(),
                "source": str(seed.get("source", "")).strip(),
                "relation": str(seed.get("relation", "")).strip(),
                "targets": dedup_targets,
                "rows": supporting_rows,
                "weight": sum(int(x.get("weight", 1) or 1) for x in supporting_rows),
            }
        )
    single_target_rows = sorted(single_target_rows, key=lambda x: int(x.get("weight", 1) or 1), reverse=True)
    multi_target_groups = sorted(multi_target_groups, key=lambda x: int(x.get("weight", 1) or 1), reverse=True)
    rng.shuffle(single_target_rows)
    rng.shuffle(multi_target_groups)
    local_idx = 0
    local_used_edges: set[str] = set()
    local_used_docs: set[str] = set()
    local_used_shapes: set[str] = set()

    def local_row_ok(row: dict, strict_profile: bool = True) -> bool:
        if profile == "youtu_friendly":
            if strict_profile:
                return _row_is_youtu_friendly_local(row)
            return _row_is_grounded(row, require_relation_keyword=False)
        return _row_is_grounded(row, require_relation_keyword=True)

    def add_local_row(row: dict) -> None:
        nonlocal local_idx, llm_question_attempts, llm_question_success, llm_question_fallback
        local_idx += 1
        doc_key = row["doc_key"]
        template_q = (
            f'What is the `{row["relation"]}` target for '
            f'"{row["source"]}"? Return the exact phrase.'
        )
        q = template_q
        ans: Any = row["target"]
        extra_gold: dict[str, Any] | None = None
        if str(question_style).strip().lower() == "llm":
            llm_question_attempts += 1
            designed, fail_reason, fail_detail = _llm_designed_local_qa(
                doc_key=doc_key,
                relation=str(row.get("relation", "")).strip(),
                chunk_text=str(row.get("evidence", "")).strip(),
                model=llm_question_model,
                temperature=float(llm_question_temperature),
                max_tokens=max(int(llm_question_max_tokens), 180),
                local_query_max_overlap=float(local_query_max_overlap),
                local_query_max_and_or_hits=int(local_query_max_and_or_hits),
            )
            if designed:
                llm_question_success += 1
                q = str(designed.get("query", "")).strip() or template_q
                ans = str(designed.get("answer", "")).strip() or row["target"]
                extra_gold = {
                    "answer_mode": str(designed.get("answer_mode", "")).strip(),
                    "evidence_span": designed.get("evidence_span"),
                    "chunk_text": str(row.get("evidence", "")).strip(),
                }
            else:
                llm_question_fallback += 1
                reason_key = str(fail_reason or "unknown_local_failure").strip()
                local_llm_fail_reasons[reason_key] = local_llm_fail_reasons.get(reason_key, 0) + 1
                print(
                    "[build_cuad_capability_qa] "
                    + json.dumps(
                        {
                            "stage": "local_factual",
                            "event": "llm_fallback",
                            "reason": reason_key,
                            "reason_detail": (fail_detail if isinstance(fail_detail, dict) else {}),
                            "doc_key": doc_key,
                            "edge_id": str(row.get("edge_id", "")).strip(),
                            "relation": str(row.get("relation", "")).strip(),
                        },
                        ensure_ascii=False,
                    ),
                    file=sys.stderr,
                    flush=True,
                )
                q = template_q
                ans = row["target"]
        else:
            q = maybe_design_query(
                qtype="local_factual",
                template_query=template_q,
                doc_key=doc_key,
                answer=str(row.get("target", "")).strip(),
                evidence_snippets=[str(row.get("evidence", "")).strip()],
            )
        q = dedup_query_text(q, doc_key)
        add_example_tracked(
            stage="local_factual",
            queries=queries,
            gold=gold,
            qid=next_qid(doc_key, "local_factual", local_idx),
            qtype="local_factual",
            query=q,
            answer=ans,
            doc_key=doc_key,
            supporting_edges=[{"edge_id": row["edge_id"]}],
            supporting_chunks=[{"chunk_id": row["chunk_id"], "doc_id": row["doc_id"]}],
            extra_gold=extra_gold,
        )
        local_used_edges.add(str(row.get("edge_id", "")))
        local_used_docs.add(doc_key)
        shape = f'{str(row.get("relation", "")).strip().lower()}|||{str(row.get("source", "")).strip().lower()}'
        if shape.strip("|"):
            local_used_shapes.add(shape)

    # Pass 1: prioritize question-shape diversity on unique-target samples.
    for row in single_target_rows:
        if local_idx >= int(per_type):
            break
        doc_key = row["doc_key"]
        if doc_key in local_used_docs:
            continue
        if not local_row_ok(row, strict_profile=True):
            continue
        shape = f'{str(row.get("relation", "")).strip().lower()}|||{str(row.get("source", "")).strip().lower()}'
        if shape in local_used_shapes:
            continue
        add_local_row(row)

    # Pass 2: fill remaining quota on unique-target samples without shape diversity constraint.
    for row in single_target_rows:
        if local_idx >= int(per_type):
            break
        doc_key = row["doc_key"]
        if doc_key in local_used_docs:
            continue
        if str(row.get("edge_id", "")) in local_used_edges:
            continue
        if not local_row_ok(row, strict_profile=(profile != "youtu_friendly")):
            continue
        add_local_row(row)

    # Pass 3: for multi-target (source, relation) groups, switch to list-style question.
    for grp in multi_target_groups:
        if local_idx >= int(per_type):
            break
        doc_key = str(grp.get("doc_key", "")).strip()
        if not doc_key or doc_key in local_used_docs:
            continue
        rows = [r for r in (grp.get("rows") or []) if isinstance(r, dict)]
        if not rows:
            continue
        if not all(local_row_ok(r, strict_profile=(profile != "youtu_friendly")) for r in rows):
            continue
        source = str(grp.get("source", "")).strip()
        relation = str(grp.get("relation", "")).strip()
        targets = [str(x).strip() for x in (grp.get("targets") or []) if str(x).strip()]
        if not source or not relation or len(targets) < 2:
            continue
        local_idx += 1
        q = f'List all `{relation}` targets for "{source}". Return a JSON array of exact phrases.'
        # Keep multi-target list questions deterministic to preserve
        # one-fact-point local_factual constraint in LLM mode.
        if str(question_style).strip().lower() != "llm":
            q = maybe_design_query(
                qtype="local_factual",
                template_query=q,
                doc_key=doc_key,
                answer="; ".join(targets),
                evidence_snippets=[str(r.get("evidence", "")).strip() for r in rows[:3]],
            )
        q = dedup_query_text(q, doc_key)
        supporting_edges = []
        supporting_chunks = []
        seen_edge_ids: set[str] = set()
        seen_chunk_ids: set[str] = set()
        for r in rows:
            edge_id = str(r.get("edge_id", "")).strip()
            chunk_id = str(r.get("chunk_id", "")).strip()
            doc_id = str(r.get("doc_id", "")).strip()
            if edge_id and edge_id not in seen_edge_ids:
                supporting_edges.append({"edge_id": edge_id})
                seen_edge_ids.add(edge_id)
                local_used_edges.add(edge_id)
            if chunk_id and chunk_id not in seen_chunk_ids:
                chunk_ref = {"chunk_id": chunk_id}
                if doc_id:
                    chunk_ref["doc_id"] = doc_id
                supporting_chunks.append(chunk_ref)
                seen_chunk_ids.add(chunk_id)
        add_example_tracked(
            stage="local_factual",
            queries=queries,
            gold=gold,
            qid=next_qid(doc_key, "local_factual", local_idx),
            qtype="local_factual",
            query=q,
            answer=targets,
            doc_key=doc_key,
            supporting_edges=supporting_edges,
            supporting_chunks=supporting_chunks,
        )
        local_used_docs.add(doc_key)
        shape = f"{relation.strip().lower()}|||{source.strip().lower()}"
        if shape.strip("|"):
            local_used_shapes.add(shape)

    # cross_clause
    cross_candidates: list[tuple[dict, dict]] = []
    for _, rows in mentions.items():
        by_rel: dict[str, list[dict]] = {}
        for r in rows:
            by_rel.setdefault(r["relation"], []).append(r)
        rels = sorted(by_rel.keys())
        if len(rels) < 2:
            continue
        a = max(by_rel[rels[0]], key=lambda x: x["weight"])
        b = max(by_rel[rels[1]], key=lambda x: x["weight"])
        cross_candidates.append((a, b))
    rng.shuffle(cross_candidates)
    cross_idx = 0
    cross_used_pairs: set[str] = set()
    for (a, b) in cross_candidates:
        if profile == "youtu_friendly":
            if not _pair_is_youtu_friendly_cross(a, b):
                continue
        elif not (
            _row_is_grounded(a, require_relation_keyword=False)
            and _row_is_grounded(b, require_relation_keyword=False)
        ):
            continue
        cross_idx += 1
        if cross_idx > int(per_type):
            break
        doc_key = a["doc_key"]
        q, ans = resolve_cross_qa(a, b, doc_key)
        add_example_tracked(
            stage="cross_clause",
            queries=queries,
            gold=gold,
            qid=next_qid(doc_key, "cross_clause", cross_idx),
            qtype="cross_clause",
            query=q,
            answer=ans,
            doc_key=doc_key,
            supporting_edges=[{"edge_id": a["edge_id"]}, {"edge_id": b["edge_id"]}],
            supporting_chunks=[
                {"chunk_id": a["chunk_id"], "doc_id": a["doc_id"]},
                {"chunk_id": b["chunk_id"], "doc_id": b["doc_id"]},
            ],
        )
        cross_used_pairs.add(f"{a.get('edge_id')}|{b.get('edge_id')}")
    if profile == "youtu_friendly" and cross_idx < int(per_type):
        for (a, b) in cross_candidates:
            if cross_idx >= int(per_type):
                break
            key = f"{a.get('edge_id')}|{b.get('edge_id')}"
            if key in cross_used_pairs:
                continue
            if not (
                _row_is_grounded(a, require_relation_keyword=False)
                and _row_is_grounded(b, require_relation_keyword=False)
            ):
                continue
            cross_idx += 1
            doc_key = a["doc_key"]
            q, ans = resolve_cross_qa(a, b, doc_key)
            add_example_tracked(
                stage="cross_clause",
                queries=queries,
                gold=gold,
                qid=next_qid(doc_key, "cross_clause", cross_idx),
                qtype="cross_clause",
                query=q,
                answer=ans,
                doc_key=doc_key,
                supporting_edges=[{"edge_id": a["edge_id"]}, {"edge_id": b["edge_id"]}],
                supporting_chunks=[
                    {"chunk_id": a["chunk_id"], "doc_id": a["doc_id"]},
                    {"chunk_id": b["chunk_id"], "doc_id": b["doc_id"]},
                ],
            )
            cross_used_pairs.add(key)

    # global_summary
    summary_candidates: list[dict] = []
    if isinstance(communities, dict):
        for c in communities.get("communities", []) or []:
            cid = str(c.get("community_id", "")).strip()
            summary = str(c.get("summary", "")).strip()
            if not cid or not summary:
                continue
            doc_keys = set()
            for chunk_id in community_chunks.get(cid, set()):
                doc_key = chunk_to_doc.get(chunk_id)
                if doc_key:
                    doc_keys.add(doc_key)
            for doc_key in sorted(doc_keys):
                doc_chunk_ids = sorted(
                    [x for x in community_chunks.get(cid, set()) if chunk_to_doc.get(x) == doc_key]
                )
                if not doc_chunk_ids:
                    continue
                summary_quality_stats["candidates_total"] += 1
                qualified_chunk_ids: list[str] = []
                for chunk_id in doc_chunk_ids:
                    meta_row = chunk_meta.get(chunk_id, {})
                    text = str(meta_row.get("text", "")).strip()
                    if not _chunk_quality_ok(text, min_chars=100):
                        summary_quality_stats["dropped_low_quality"] += 1
                        continue
                    hits = _doc_chunk_consistency_hits(doc_key, text)
                    if int(global_summary_doc_consistency_min_hits) > 0 and hits < int(global_summary_doc_consistency_min_hits):
                        summary_quality_stats["dropped_doc_consistency"] += 1
                        continue
                    qualified_chunk_ids.append(chunk_id)
                if len(qualified_chunk_ids) < int(global_summary_min_high_quality_chunks):
                    summary_quality_stats["dropped_below_min_chunks"] += 1
                    continue
                summary_quality_stats["candidates_after_quality"] += 1
                # Keep at most target high-quality chunks per candidate by default.
                if int(global_summary_target_high_quality_chunks) > 0 and len(qualified_chunk_ids) > int(global_summary_target_high_quality_chunks):
                    qualified_chunk_ids = sorted(
                        rng.sample(qualified_chunk_ids, int(global_summary_target_high_quality_chunks))
                    )
                summary_candidates.append(
                    {
                        "doc_key": doc_key,
                        "community_id": cid,
                        "summary": summary,
                        "chunk_ids": qualified_chunk_ids,
                    }
                )
    rng.shuffle(summary_candidates)
    used_doc_for_summary: set[str] = set()
    picked_summary: list[dict] = []
    for row in summary_candidates:
        if row["doc_key"] in used_doc_for_summary:
            continue
        if profile == "youtu_friendly":
            n_chunks = len(row.get("chunk_ids", []) or [])
            if n_chunks < 3 or n_chunks > 15:
                continue
        used_doc_for_summary.add(row["doc_key"])
        picked_summary.append(row)
        if len(picked_summary) >= int(per_type):
            break
    if profile == "youtu_friendly" and len(picked_summary) < int(per_type):
        for row in summary_candidates:
            if len(picked_summary) >= int(per_type):
                break
            if row["doc_key"] in used_doc_for_summary:
                continue
            used_doc_for_summary.add(row["doc_key"])
            picked_summary.append(row)
    for i, row in enumerate(picked_summary, start=1):
        doc_key = row["doc_key"]
        q = f'For contract "{doc_key}", provide a concise global risk/theme summary from graph communities.'
        q = maybe_design_query(
            qtype="global_summary",
            template_query=q,
            doc_key=doc_key,
            answer=str(row.get("summary", "")).strip(),
            evidence_snippets=[str(row.get("summary", "")).strip()],
        )
        q = dedup_query_text(q, doc_key)
        candidate_chunk_ids = list(row["chunk_ids"])
        if int(global_summary_chunk_cap) > 0 and len(candidate_chunk_ids) > int(global_summary_chunk_cap):
            # Keep deterministic but diverse evidence coverage to avoid overfitting
            # chunk recall to a tiny fixed subset.
            candidate_chunk_ids = sorted(
                rng.sample(candidate_chunk_ids, int(global_summary_chunk_cap))
            )
        sup_chunks = []
        for c in candidate_chunk_ids:
            chunk_ref = {"chunk_id": c}
            doc_id = str(chunk_to_doc.get(c, "")).strip()
            if doc_id:
                chunk_ref["doc_id"] = doc_id
            sup_chunks.append(chunk_ref)
        summary_answer = str(row["summary"])
        if int(summary_answer_max_chars) > 0 and len(summary_answer) > int(summary_answer_max_chars):
            summary_answer = summary_answer[: int(summary_answer_max_chars)]
        add_example_tracked(
            stage="global_summary",
            queries=queries,
            gold=gold,
            qid=next_qid(doc_key, "global_summary", i),
            qtype="global_summary",
            query=q,
            answer=summary_answer,
            doc_key=doc_key,
            supporting_communities=[{"community_id": row["community_id"]}],
            supporting_chunks=sup_chunks,
        )

    Path(queries_out).parent.mkdir(parents=True, exist_ok=True)
    Path(gold_out).parent.mkdir(parents=True, exist_ok=True)
    with Path(queries_out).open("w", encoding="utf-8") as f:
        for r in queries:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with Path(gold_out).open("w", encoding="utf-8") as f:
        for r in gold:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    by_type: dict[str, int] = {}
    for r in queries:
        t = str(r.get("type", "unknown"))
        by_type[t] = by_type.get(t, 0) + 1
    return {
        "graph_file": graph_file,
        "communities_file": communities_file,
        "queries_out": queries_out,
        "gold_out": gold_out,
        "num_queries": len(queries),
        "num_gold": len(gold),
        "per_type_target": int(per_type),
        "global_summary_chunk_cap": int(global_summary_chunk_cap),
        "profile": str(profile),
        "summary_answer_max_chars": int(summary_answer_max_chars),
        "question_style": str(question_style),
        "llm_question_model": (None if llm_question_model is None else (str(llm_question_model).strip() or None)),
        "llm_question_temperature": float(llm_question_temperature),
        "llm_question_max_tokens": int(llm_question_max_tokens),
        "llm_cross_generate_qa": bool(llm_cross_generate_qa),
        "llm_cross_answer_max_tokens": int(llm_cross_answer_max_tokens),
        "llm_question_attempts": int(llm_question_attempts),
        "llm_question_success": int(llm_question_success),
        "llm_question_fallback": int(llm_question_fallback),
        "llm_cross_qa_attempts": int(llm_cross_qa_attempts),
        "llm_cross_qa_success": int(llm_cross_qa_success),
        "llm_cross_qa_fallback": int(llm_cross_qa_fallback),
        "local_llm_fail_reasons": dict(sorted(local_llm_fail_reasons.items())),
        "local_query_max_overlap": float(local_query_max_overlap),
        "local_query_max_and_or_hits": int(local_query_max_and_or_hits),
        "global_summary_min_high_quality_chunks": int(global_summary_min_high_quality_chunks),
        "global_summary_target_high_quality_chunks": int(global_summary_target_high_quality_chunks),
        "global_summary_doc_consistency_min_hits": int(global_summary_doc_consistency_min_hits),
        "summary_quality_stats": summary_quality_stats,
        "progress_every": int(progress_step),
        "by_type": by_type,
        "random_seed": int(random_seed),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CUAD capability-oriented QA set for RAG comparison.")
    parser.add_argument("--graph-file", default="outputs/graph/cuad_graph_test.json")
    parser.add_argument("--communities-file", default="outputs/graph/cuad_communities_test.json")
    parser.add_argument("--queries-out", default="data/queries/cuad_capability_queries.jsonl")
    parser.add_argument("--gold-out", default="data/queries/cuad_capability_gold.jsonl")
    parser.add_argument("--per-type", type=int, default=20)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--global-summary-chunk-cap", type=int, default=30)
    parser.add_argument("--profile", choices=["default", "youtu_friendly"], default="default")
    parser.add_argument(
        "--question-style",
        choices=["template", "llm"],
        default="template",
        help="template: deterministic templates; llm: use LLM to design natural questions with template fallback.",
    )
    parser.add_argument(
        "--llm-question-model",
        default=None,
        help="Optional model override for LLM question design.",
    )
    parser.add_argument(
        "--llm-question-temperature",
        type=float,
        default=0.2,
        help="Temperature for LLM question design.",
    )
    parser.add_argument(
        "--llm-question-max-tokens",
        type=int,
        default=96,
        help="Max completion tokens for each LLM-designed question.",
    )
    parser.add_argument(
        "--llm-cross-generate-qa",
        action="store_true",
        help="When question-style=llm, let LLM generate BOTH cross_clause question and answer with fallback to rule-based answer.",
    )
    parser.add_argument(
        "--llm-cross-answer-max-tokens",
        type=int,
        default=160,
        help="Max completion tokens for LLM cross_clause QA generation.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=0,
        help="Print progress JSON to stderr every N generated QA rows (0 disables progress logs).",
    )
    parser.add_argument(
        "--summary-answer-max-chars",
        type=int,
        default=0,
        help="Optional max chars for global_summary gold answer (0 means no truncation).",
    )
    parser.add_argument(
        "--global-summary-min-high-quality-chunks",
        type=int,
        default=2,
        help="Minimum number of high-quality same-doc chunks required for each global_summary QA.",
    )
    parser.add_argument(
        "--global-summary-target-high-quality-chunks",
        type=int,
        default=3,
        help="Preferred number of high-quality same-doc chunks retained per global_summary QA candidate.",
    )
    parser.add_argument(
        "--global-summary-doc-consistency-min-hits",
        type=int,
        default=1,
        help="Minimum anchor-token hits between doc_key and chunk text for consistency screening (0 disables).",
    )
    parser.add_argument(
        "--local-query-max-overlap",
        type=float,
        default=0.72,
        help="Max allowed char n-gram overlap ratio between local question and chunk text.",
    )
    parser.add_argument(
        "--local-query-max-and-or-hits",
        type=int,
        default=1,
        help="Max allowed conjunction hits (and/or/以及/且/并且) in local question.",
    )
    args = parser.parse_args()
    stats = build_cuad_capability_qa(
        graph_file=args.graph_file,
        queries_out=args.queries_out,
        gold_out=args.gold_out,
        communities_file=args.communities_file,
        per_type=args.per_type,
        random_seed=args.random_seed,
        global_summary_chunk_cap=args.global_summary_chunk_cap,
        profile=args.profile,
        summary_answer_max_chars=args.summary_answer_max_chars,
        question_style=args.question_style,
        llm_question_model=args.llm_question_model,
        llm_question_temperature=args.llm_question_temperature,
        llm_question_max_tokens=args.llm_question_max_tokens,
        llm_cross_generate_qa=args.llm_cross_generate_qa,
        llm_cross_answer_max_tokens=args.llm_cross_answer_max_tokens,
        progress_every=args.progress_every,
        local_query_max_overlap=args.local_query_max_overlap,
        local_query_max_and_or_hits=args.local_query_max_and_or_hits,
        global_summary_min_high_quality_chunks=args.global_summary_min_high_quality_chunks,
        global_summary_target_high_quality_chunks=args.global_summary_target_high_quality_chunks,
        global_summary_doc_consistency_min_hits=args.global_summary_doc_consistency_min_hits,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
