from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

try:
    from evaluation.metrics import similarity
    from evaluation.judges import semantic_equivalent_yes_no
    from utils.embedder import embed_texts
except ModuleNotFoundError:
    from src.evaluation.metrics import similarity
    from src.evaluation.judges import semantic_equivalent_yes_no
    from src.utils.embedder import embed_texts

METHODS = ("vector_rag", "kg_rag", "graph_rag")
PLOT_STYLE = "whitegrid"


def _read_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_json(path: str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _norm_text(text: Any) -> str:
    text = "" if text is None else str(text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _binary(value: bool) -> int:
    return 1 if value else 0


def _answer_items(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        out: list[str] = []
        for x in value:
            s = str(x).strip()
            if s:
                out.append(s)
        return out
    s = str(value).strip()
    if not s:
        return []
    if s.startswith("[") and s.endswith("]"):
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                out = []
                for x in parsed:
                    t = str(x).strip()
                    if t:
                        out.append(t)
                if out:
                    return out
        except Exception:
            pass
    if "\n" in s:
        lines = [re.sub(r"^\s*[-*\d\.\)]\s*", "", x).strip() for x in s.splitlines()]
        lines = [x for x in lines if x]
        if len(lines) > 1:
            return lines
    if ";" in s:
        parts = [x.strip() for x in s.split(";")]
        parts = [x for x in parts if x]
        if len(parts) > 1:
            return parts
    return [s]


def _answer_set_norm(value: Any) -> set[str]:
    return {_norm_text(x) for x in _answer_items(value) if _norm_text(x)}


def _answer_set_relaxed(value: Any) -> set[str]:
    out: set[str] = set()
    for x in _answer_items(value):
        nx = _norm_text(x)
        if not nx:
            continue
        rx = re.sub(r"[^a-z0-9]+", "", nx)
        if rx:
            out.add(rx)
    return out


def _answer_to_text(value: Any) -> str:
    items = _answer_items(value)
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return "; ".join(items)


def _set_f1(pred_set: set[str], gold_set: set[str]) -> float:
    if not pred_set or not gold_set:
        return 0.0
    hit = len(pred_set & gold_set)
    if hit <= 0:
        return 0.0
    prec = hit / len(pred_set)
    rec = hit / len(gold_set)
    return (2 * prec * rec) / (prec + rec)


def _normalize_token(tok: str) -> str:
    t = re.sub(r"[^a-z0-9]+", "", str(tok or "").lower())
    if len(t) > 3 and t.endswith("ies"):
        return t[:-3] + "y"
    if len(t) > 3 and t.endswith("es"):
        return t[:-2]
    if len(t) > 3 and t.endswith("s"):
        return t[:-1]
    return t


def _tokenize_norm(text: str) -> list[str]:
    toks = [_normalize_token(x) for x in re.split(r"[^a-zA-Z0-9]+", _norm_text(text)) if x]
    return [x for x in toks if x]


def _fuzzy_token_overlap_f1(pred_text: str, gold_text: str, sim_threshold: float = 0.82) -> float:
    p_toks = _tokenize_norm(pred_text)
    g_toks = _tokenize_norm(gold_text)
    if not p_toks or not g_toks:
        return 0.0
    matched_g = 0
    used_pred: set[int] = set()
    for g in g_toks:
        best_idx = -1
        best_sim = 0.0
        for i, p in enumerate(p_toks):
            if i in used_pred:
                continue
            sim = similarity(p, g)
            if sim > best_sim:
                best_sim = sim
                best_idx = i
        if best_idx >= 0 and best_sim >= float(sim_threshold):
            used_pred.add(best_idx)
            matched_g += 1
    if matched_g <= 0:
        return 0.0
    prec = matched_g / len(p_toks)
    rec = matched_g / len(g_toks)
    return (2 * prec * rec) / (prec + rec)


def _fuzzy_token_recall(pred_text: str, gold_text: str, sim_threshold: float = 0.82) -> float:
    p_toks = _tokenize_norm(pred_text)
    g_toks = _tokenize_norm(gold_text)
    if not p_toks or not g_toks:
        return 0.0
    matched_g = 0
    used_pred: set[int] = set()
    for g in g_toks:
        best_idx = -1
        best_sim = 0.0
        for i, p in enumerate(p_toks):
            if i in used_pred:
                continue
            sim = similarity(p, g)
            if sim > best_sim:
                best_sim = sim
                best_idx = i
        if best_idx >= 0 and best_sim >= float(sim_threshold):
            used_pred.add(best_idx)
            matched_g += 1
    return matched_g / len(g_toks)


def _match_scores(pred: Any, gold: Any) -> dict:
    pred_text = _answer_to_text(pred)
    gold_text = _answer_to_text(gold)
    pred_items = _answer_items(pred)
    gold_items = _answer_items(gold)
    p_set = _answer_set_norm(pred)
    g_set = _answer_set_norm(gold)
    is_multi = len(pred_items) > 1 or len(gold_items) > 1

    if is_multi:
        exact = bool(g_set) and p_set == g_set
        contains = bool(g_set) and (g_set.issubset(p_set) or p_set.issubset(g_set))
        # Bullet-heavy free-form answers are often split into many lines, which
        # makes strict set-level matching too harsh. Keep the set metric, but
        # backstop it with whole-answer token overlap so long-form partial hits
        # do not collapse to zero.
        token_f1 = max(_set_f1(p_set, g_set), _fuzzy_token_overlap_f1(pred_text, gold_text))
        p_relaxed = _answer_set_relaxed(pred)
        g_relaxed = _answer_set_relaxed(gold)
        token_f1_relaxed = _set_f1(p_relaxed, g_relaxed)
        contains_relaxed = bool(g_relaxed) and (
            g_relaxed.issubset(p_relaxed)
            or p_relaxed.issubset(g_relaxed)
            or token_f1_relaxed >= 0.5
        )
        exact_relaxed = bool(g_relaxed) and p_relaxed == g_relaxed
    else:
        p = _norm_text(pred_text)
        g = _norm_text(gold_text)
        exact = bool(g) and p == g
        contains = bool(g) and (g in p or p in g)
        p_alnum = re.sub(r"[^a-z0-9]+", "", p)
        g_alnum = re.sub(r"[^a-z0-9]+", "", g)
        exact_relaxed = bool(g_alnum) and p_alnum == g_alnum
        token_f1 = _fuzzy_token_overlap_f1(pred_text, gold_text)
        token_recall = _fuzzy_token_recall(pred_text, gold_text)
        contains_relaxed = bool(g_alnum) and (
            g_alnum in p_alnum
            or p_alnum in g_alnum
            or token_f1 >= 0.5
            or token_recall >= 0.67
        )
    # For multi-answer tasks, text-order-sensitive SequenceMatcher can be misleading.
    # Use set-level F1 as answer_similarity to align with multi-answer evaluation intent.
    if is_multi:
        answer_similarity = token_f1
    else:
        # Single-answer: be robust to short gold phrase vs longer faithful sentence.
        token_recall = _fuzzy_token_recall(pred_text, gold_text)
        answer_similarity = max(float(similarity(pred_text, gold_text)), float(token_f1), float(token_recall))
    return {
        "answer_similarity": round(float(answer_similarity), 6),
        "answer_exact": _binary(exact),
        "answer_contains": _binary(contains),
        "answer_exact_relaxed": _binary(exact_relaxed),
        "answer_contains_relaxed": _binary(contains_relaxed),
        "answer_token_f1": round(float(token_f1), 6),
    }


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _clip_for_semantic(text: str, max_chars: int) -> str:
    s = str(text or "").strip()
    if not s:
        return ""
    return s[: int(max_chars)]


def _attach_semantic_similarity(
    rows: list[dict],
    enabled: bool = True,
    target_types: set[str] | None = None,
    max_chars: int = 3000,
) -> dict:
    if not enabled:
        for row in rows:
            row["summary_semantic_similarity"] = None
            row["answer_score_primary"] = row.get("answer_similarity")
            row["answer_score_primary_name"] = "answer_similarity"
        return {"enabled": False, "reason": "disabled"}

    tset = {str(x).strip().lower() for x in (target_types or {"global_summary"}) if str(x).strip()}
    pairs: list[tuple[int, str, str]] = []
    uniq_texts: list[str] = []
    text_to_idx: dict[str, int] = {}
    for i, row in enumerate(rows):
        qtype = str(row.get("type", "")).strip().lower()
        if qtype not in tset:
            row["summary_semantic_similarity"] = None
            row["answer_score_primary"] = row.get("answer_similarity")
            row["answer_score_primary_name"] = "answer_similarity"
            continue
        pred = _clip_for_semantic(str(row.get("pred_answer", "")), max_chars=max_chars)
        gold = _clip_for_semantic(str(row.get("gold_answer", "")), max_chars=max_chars)
        if (not pred) or (not gold):
            row["summary_semantic_similarity"] = None
            row["answer_score_primary"] = row.get("answer_similarity")
            row["answer_score_primary_name"] = "answer_similarity"
            continue
        if pred not in text_to_idx:
            text_to_idx[pred] = len(uniq_texts)
            uniq_texts.append(pred)
        if gold not in text_to_idx:
            text_to_idx[gold] = len(uniq_texts)
            uniq_texts.append(gold)
        pairs.append((i, pred, gold))

    if not pairs:
        return {"enabled": True, "pairs": 0, "unique_texts": len(uniq_texts), "target_types": sorted(tset)}

    try:
        vecs, meta = embed_texts(uniq_texts, return_meta=True)
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"
        for row in rows:
            if str(row.get("type", "")).strip().lower() in tset:
                row["summary_semantic_similarity"] = None
                row["answer_score_primary"] = row.get("answer_similarity")
                row["answer_score_primary_name"] = "answer_similarity"
        return {"enabled": True, "error": err, "pairs": len(pairs), "target_types": sorted(tset)}

    for i, pred, gold in pairs:
        a = vecs[text_to_idx[pred]]
        b = vecs[text_to_idx[gold]]
        sim = round(_cosine_similarity(a, b), 6)
        rows[i]["summary_semantic_similarity"] = sim
        rows[i]["answer_score_primary"] = sim
        rows[i]["answer_score_primary_name"] = "summary_semantic_similarity"

    return {
        "enabled": True,
        "pairs": len(pairs),
        "unique_texts": len(uniq_texts),
        "target_types": sorted(tset),
        "embedding_backend": meta.get("backend"),
        "embedding_model": meta.get("model"),
        "embedding_latency_ms": int(meta.get("latency_ms", 0) or 0),
    }


def _to_float_or_none(v: Any) -> float | None:
    try:
        if v is None:
            return None
        if isinstance(v, str) and not v.strip():
            return None
        return float(v)
    except Exception:
        return None


def _to_int_binary_or_none(v: Any) -> int | None:
    if v is None:
        return None
    if isinstance(v, bool):
        return 1 if v else 0
    try:
        x = int(v)
    except Exception:
        return None
    return 1 if x > 0 else 0


def _apply_llm_yesno_judge(
    rows: list[dict],
    judge_mode: str = "off",
    judge_model: str = "qwen-flash",
) -> dict:
    mode = str(judge_mode or "off").strip().lower()
    if mode != "llm_yesno":
        for row in rows:
            row["answer_semantic_yesno"] = None
        return {"enabled": False, "judge_mode": mode}

    cache: dict[tuple[str, str], int] = {}
    calls = 0
    errors = 0
    for row in rows:
        pred = str(row.get("pred_answer", "") or "").strip()
        gold = str(row.get("gold_answer", "") or "").strip()
        if (not pred) or (not gold):
            row["answer_semantic_yesno"] = 0
            continue
        key = (pred, gold)
        if key not in cache:
            try:
                cache[key] = int(
                    semantic_equivalent_yes_no(
                        pred=pred,
                        gold=gold,
                        model=str(judge_model or "qwen-flash"),
                    )
                )
                calls += 1
            except Exception:
                cache[key] = 0
                errors += 1
        row["answer_semantic_yesno"] = int(cache[key])
    return {
        "enabled": True,
        "judge_mode": mode,
        "judge_model": str(judge_model or "qwen-flash"),
        "llm_calls": int(calls),
        "cache_size": len(cache),
        "errors": int(errors),
    }


def _apply_topk_accuracy_fields(rows: list[dict]) -> dict:
    support_ones = 0
    correct_ones = 0
    accuracy_ones = 0
    used_yesno = 0
    used_exact = 0
    for row in rows:
        support = 1 if float(row.get("evidence_recall_primary", 0.0) or 0.0) > 0.0 else 0
        yesno = _to_int_binary_or_none(row.get("answer_semantic_yesno"))
        if yesno is None:
            correct = 1 if int(row.get("answer_exact_relaxed", 0) or 0) > 0 else 0
            used_exact += 1
        else:
            correct = yesno
            used_yesno += 1
        acc = int(support * correct)
        row["topk_support"] = support
        row["topk_correct"] = correct
        row["topk_accuracy"] = acc
        support_ones += support
        correct_ones += correct
        accuracy_ones += acc
    return {
        "rows": len(rows),
        "support_ones": support_ones,
        "correct_ones": correct_ones,
        "accuracy_ones": accuracy_ones,
        "answer_correct_source": {
            "answer_semantic_yesno": used_yesno,
            "answer_exact_relaxed": used_exact,
        },
    }


def _apply_primary_answer_policy(
    rows: list[dict],
    global_summary_mode: str = "semantic",
    global_summary_alpha_semantic: float = 1.0,
) -> dict:
    mode = str(global_summary_mode or "semantic").strip().lower()
    alpha = float(global_summary_alpha_semantic)
    alpha = min(max(alpha, 0.0), 1.0)
    touched = 0
    for row in rows:
        qtype = str(row.get("type", "")).strip().lower()
        if qtype != "global_summary":
            if row.get("answer_score_primary") is None:
                row["answer_score_primary"] = row.get("answer_similarity")
                row["answer_score_primary_name"] = "answer_similarity"
            continue
        sem = _to_float_or_none(row.get("summary_semantic_similarity"))
        rec = _to_float_or_none(row.get("evidence_recall_primary"))
        if mode == "composite":
            if sem is not None and rec is not None:
                score = alpha * sem + (1.0 - alpha) * rec
                row["answer_score_primary"] = round(float(score), 6)
                row["answer_score_primary_name"] = (
                    f"global_summary_composite(alpha={alpha:.2f})"
                )
            elif sem is not None:
                row["answer_score_primary"] = sem
                row["answer_score_primary_name"] = "summary_semantic_similarity"
            else:
                row["answer_score_primary"] = row.get("answer_similarity")
                row["answer_score_primary_name"] = "answer_similarity_fallback"
        else:
            if sem is not None:
                row["answer_score_primary"] = sem
                row["answer_score_primary_name"] = "summary_semantic_similarity"
            else:
                row["answer_score_primary"] = row.get("answer_similarity")
                row["answer_score_primary_name"] = "answer_similarity_fallback"
        touched += 1
    return {"global_summary_mode": mode, "global_summary_alpha_semantic": alpha, "rows_touched": touched}


def _precision_recall_f1(pred_set: set[str], gold_set: set[str]) -> tuple[float, float, float]:
    if not gold_set:
        return 0.0, 0.0, 0.0
    if not pred_set:
        return 0.0, 0.0, 0.0
    hit = len(pred_set & gold_set)
    if hit == 0:
        return 0.0, 0.0, 0.0
    p = hit / len(pred_set)
    r = hit / len(gold_set)
    f1 = 2 * p * r / (p + r)
    return p, r, f1


def _valid_doc_id(value: Any) -> str:
    if value is None:
        return ""
    s = _doc_prefix(str(value)).strip()
    if not s:
        return ""
    if s.lower() in {"none", "null", "nan"}:
        return ""
    return s


def _valid_chunk_id(value: Any) -> str:
    if value is None:
        return ""
    s = str(value).strip()
    if not s:
        return ""
    if s.lower() in {"none", "null", "nan"}:
        return ""
    return s


def _expand_chunk_ids(value: Any) -> list[str]:
    raw = _valid_chunk_id(value)
    if not raw:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for part in str(raw).split("|"):
        cid = _valid_chunk_id(part)
        if cid and cid not in seen:
            seen.add(cid)
            out.append(cid)
    return out


_INLINE_CHUNK_ID_RE = re.compile(r"\[chunk_id=([^\]\n]+)\]")
_INLINE_DOC_ID_RE = re.compile(r"\[doc_id:\s*([^\],\n]+)")


def _extract_inline_chunk_ids(text: Any) -> list[str]:
    if text is None:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for match in _INLINE_CHUNK_ID_RE.findall(str(text)):
        cid = _valid_chunk_id(match)
        if cid and cid not in seen:
            seen.add(cid)
            out.append(cid)
    return out


def _extract_inline_doc_ids(text: Any) -> list[str]:
    if text is None:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for match in _INLINE_DOC_ID_RE.findall(str(text)):
        did = _valid_doc_id(match)
        if did and did not in seen:
            seen.add(did)
            out.append(did)
    return out


def _collect_pred_ids(method_payload: dict, community_chunk_map: dict[str, set[str]] | None = None) -> dict[str, set[str]]:
    doc_ids = set()
    chunk_ids = set()
    edge_ids = set()
    community_ids = set()

    for e in method_payload.get("evidence", []) or []:
        if isinstance(e, dict):
            did = _valid_doc_id(e.get("doc_id"))
            if did:
                doc_ids.add(did)
            for cid in _expand_chunk_ids(e.get("chunk_id")):
                chunk_ids.add(cid)
            comm = str(e.get("community_id", "")).strip()
            if comm:
                community_ids.add(comm)

    for e in method_payload.get("subgraph_edges", []) or []:
        if isinstance(e, dict):
            eid = str(e.get("edge_id", "")).strip()
            if eid:
                edge_ids.add(eid)
            for field in ("source", "target", "text"):
                for did in _extract_inline_doc_ids(e.get(field)):
                    doc_ids.add(did)
                for cid in _extract_inline_chunk_ids(e.get(field)):
                    chunk_ids.add(cid)

    for c in method_payload.get("communities", []) or []:
        cid = str(c).strip()
        if cid:
            community_ids.add(cid)

    for e in method_payload.get("evidence_chunks", []) or []:
        if isinstance(e, dict):
            did = _valid_doc_id(e.get("doc_id"))
            if did:
                doc_ids.add(did)
            for cid in _expand_chunk_ids(e.get("chunk_id")):
                chunk_ids.add(cid)
            for did in _extract_inline_doc_ids(e.get("text")):
                doc_ids.add(did)
            for inline_cid in _extract_inline_chunk_ids(e.get("text")):
                chunk_ids.add(inline_cid)

    if community_chunk_map:
        for cid in community_ids:
            chunk_ids.update(community_chunk_map.get(cid, set()))

    # Backward-compat fallback when payload has no chunk ids.
    if not chunk_ids and doc_ids:
        chunk_ids.update(doc_ids)

    # Backward-compat fallback when payload has no doc ids.
    if not doc_ids:
        for e in method_payload.get("evidence", []) or []:
            if isinstance(e, dict):
                cid = _valid_chunk_id(e.get("chunk_id"))
                if cid:
                    doc_ids.add(cid)
        for e in method_payload.get("evidence_chunks", []) or []:
            if isinstance(e, dict):
                cid = _valid_chunk_id(e.get("chunk_id"))
                if cid:
                    doc_ids.add(cid)
        if community_chunk_map:
            for cid in community_ids:
                doc_ids.update(community_chunk_map.get(cid, set()))

    return {
        "chunks": chunk_ids,
        "docs": doc_ids,
        "edges": edge_ids,
        "communities": community_ids,
    }


def _collect_pred_ranked_ids(
    method_payload: dict,
    community_chunk_map: dict[str, set[str]] | None = None,
) -> dict[str, list[str]]:
    ranked_docs: list[str] = []
    ranked_chunks: list[str] = []
    seen_docs: set[str] = set()
    seen_chunks: set[str] = set()

    def _append_doc(value: Any) -> None:
        did = _valid_doc_id(value)
        if did and did not in seen_docs:
            seen_docs.add(did)
            ranked_docs.append(did)

    def _append_chunk(value: Any) -> None:
        for cid in _expand_chunk_ids(value):
            if cid and cid not in seen_chunks:
                seen_chunks.add(cid)
                ranked_chunks.append(cid)

    for e in method_payload.get("evidence", []) or []:
        if not isinstance(e, dict):
            continue
        _append_doc(e.get("doc_id"))
        _append_chunk(e.get("chunk_id"))
        for did in _extract_inline_doc_ids(e.get("text")):
            _append_doc(did)
        for cid in _extract_inline_chunk_ids(e.get("text")):
            _append_chunk(cid)

    for e in method_payload.get("evidence_chunks", []) or []:
        if not isinstance(e, dict):
            continue
        _append_doc(e.get("doc_id"))
        _append_chunk(e.get("chunk_id"))
        for did in _extract_inline_doc_ids(e.get("text")):
            _append_doc(did)
        for cid in _extract_inline_chunk_ids(e.get("text")):
            _append_chunk(cid)

    for e in method_payload.get("subgraph_edges", []) or []:
        if not isinstance(e, dict):
            continue
        for field in ("source", "target", "text"):
            for did in _extract_inline_doc_ids(e.get(field)):
                _append_doc(did)
            for cid in _extract_inline_chunk_ids(e.get(field)):
                _append_chunk(cid)

    for cid in method_payload.get("communities", []) or []:
        community_id = str(cid).strip()
        if not community_id or not community_chunk_map:
            continue
        for chunk_id in sorted(community_chunk_map.get(community_id, set())):
            _append_chunk(chunk_id)

    if not ranked_chunks and ranked_docs:
        for did in ranked_docs:
            _append_chunk(did)
    if not ranked_docs and ranked_chunks:
        for cid in ranked_chunks:
            _append_doc(cid)

    return {"docs": ranked_docs, "chunks": ranked_chunks}


def _hit_and_recall_at_k(ranked_ids: list[str], gold_ids: set[str], k: int) -> tuple[float, float]:
    if not gold_ids:
        return 0.0, 0.0
    topk = []
    seen: set[str] = set()
    for item in ranked_ids:
        s = str(item or "").strip()
        if not s or s in seen:
            continue
        seen.add(s)
        topk.append(s)
        if len(topk) >= int(k):
            break
    hits = len(set(topk) & set(gold_ids))
    return (1.0 if hits > 0 else 0.0, hits / len(gold_ids))


def _doc_prefix(value: str) -> str:
    return str(value or "").split("#", 1)[0].strip()


def _is_global_type(value: Any) -> bool:
    qtype = str(value or "").strip().lower()
    return qtype.startswith("global")


def _norm_doc_key(value: str) -> str:
    s = _doc_prefix(value).lower()
    return re.sub(r"[^a-z0-9]+", "", s)


def _extract_query_doc_key(row: dict, gold: dict) -> str:
    candidates = [
        str((row.get("meta") or {}).get("query_doc_key", "")).strip(),
        str((gold.get("meta") or {}).get("query_doc_key", "")).strip(),
        str((gold.get("meta") or {}).get("title", "")).strip(),
    ]
    qid = str(row.get("qid", "")).strip()
    if "__" in qid:
        candidates.append(qid.split("__", 1)[0].strip())
    for c in candidates:
        if c:
            return c
    return ""


def _extract_pred_doc_ids(method_payload: dict) -> list[str]:
    out: list[str] = []
    for e in method_payload.get("evidence", []) or []:
        if isinstance(e, dict):
            did = _valid_doc_id(e.get("doc_id"))
            if did:
                out.append(did)
    for e in method_payload.get("evidence_chunks", []) or []:
        if isinstance(e, dict):
            did = _valid_doc_id(e.get("doc_id"))
            if did:
                out.append(did)
    return out


def _retrieval_doc_diagnostics(pred_doc_ids: list[str], query_doc_key: str) -> dict:
    if not pred_doc_ids:
        return {
            "retrieved_unique_docs": 0,
            "retrieved_top1_doc_share": 0.0,
            "retrieved_doc_entropy": 0.0,
            "evidence_doc_self_hit_rate": None if not query_doc_key else 0.0,
        }
    counts: dict[str, int] = {}
    for d in pred_doc_ids:
        counts[d] = counts.get(d, 0) + 1
    total = len(pred_doc_ids)
    top1 = max(counts.values()) / total
    if len(counts) <= 1:
        entropy = 0.0
    else:
        h = 0.0
        for c in counts.values():
            p = c / total
            h -= p * math.log2(p)
        entropy = h / math.log2(len(counts))
    self_hit_rate = None
    if query_doc_key:
        qn = _norm_doc_key(query_doc_key)
        if qn:
            self_hits = 0
            for d in pred_doc_ids:
                dn = _norm_doc_key(d)
                if dn and (dn == qn or qn in dn or dn in qn):
                    self_hits += 1
            self_hit_rate = self_hits / total
    return {
        "retrieved_unique_docs": len(counts),
        "retrieved_top1_doc_share": round(float(top1), 6),
        "retrieved_doc_entropy": round(float(entropy), 6),
        "evidence_doc_self_hit_rate": (round(float(self_hit_rate), 6) if self_hit_rate is not None else None),
    }


def _method_specific_diagnostics(method: str, payload: dict) -> dict:
    out: dict[str, float] = {}
    if method == "kg_rag":
        diag = payload.get("diagnostics", {}) if isinstance(payload.get("diagnostics"), dict) else {}
        out["kg_startnode_hub_ratio"] = round(float(diag.get("startnode_hub_ratio", 0.0) or 0.0), 6)
        out["kg_traversed_doc_span"] = int(diag.get("retrieved_doc_span", 0) or 0)
    if method == "graph_rag":
        communities = payload.get("communities", []) or []
        n = len(communities)
        uniq = len(set(str(x) for x in communities))
        reuse_rate = 0.0 if n <= 0 else max((n - uniq) / max(n, 1), 0.0)
        out["graph_selected_community_reuse_rate"] = round(float(reuse_rate), 6)
        out["graph_summary_coverage_at_query_level"] = round(
            float(payload.get("selected_precomputed_summary_coverage", 0.0) or 0.0),
            6,
        )
    return out


def _build_community_chunk_map(graph_file: str | None, communities_file: str | None) -> dict[str, set[str]]:
    if not graph_file or not communities_file:
        return {}
    gp = Path(graph_file)
    cp = Path(communities_file)
    if not gp.exists() or not cp.exists():
        return {}

    graph = _load_json(str(gp))
    communities_payload = _load_json(str(cp))
    communities = communities_payload if isinstance(communities_payload, list) else communities_payload.get("communities", [])
    edge_by_id = {str(e.get("edge_id")): e for e in graph.get("edges", []) if e.get("edge_id")}

    out: dict[str, set[str]] = {}
    for c in communities:
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


def _build_community_hierarchy_map(communities_file: str | None) -> dict[str, set[str]]:
    if not communities_file:
        return {}
    cp = Path(communities_file)
    if not cp.exists():
        return {}
    payload = _load_json(str(cp))
    communities = payload if isinstance(payload, list) else payload.get("communities", [])
    if not isinstance(communities, list):
        return {}
    graph: dict[str, set[str]] = {}
    for c in communities:
        if not isinstance(c, dict):
            continue
        cid = str(c.get("community_id", "")).strip()
        if not cid:
            continue
        graph.setdefault(cid, set())
        parent_id = c.get("parent_id")
        if parent_id:
            pid = str(parent_id).strip()
            if pid:
                graph.setdefault(pid, set()).add(cid)
                graph[cid].add(pid)
        for ch in c.get("children_ids", []) or []:
            chid = str(ch).strip()
            if not chid:
                continue
            graph.setdefault(chid, set()).add(cid)
            graph.setdefault(cid, set()).add(chid)
    return graph


def _expand_community_family(ids: set[str], hierarchy_map: dict[str, set[str]]) -> set[str]:
    if not ids or not hierarchy_map:
        return set(ids)
    seen = set(ids)
    stack = list(ids)
    while stack:
        cur = stack.pop()
        for nxt in hierarchy_map.get(cur, set()):
            if nxt in seen:
                continue
            seen.add(nxt)
            stack.append(nxt)
    return seen


def _collect_gold_ids(gold_row: dict) -> dict[str, set[str]]:
    doc_ids = {
        _doc_prefix(str(x.get("doc_id", "") or ""))
        for x in gold_row.get("supporting_chunks", []) or []
        if _doc_prefix(str(x.get("doc_id", "") or "")).strip()
    }
    chunk_ids = {
        cid
        for x in gold_row.get("supporting_chunks", []) or []
        for cid in _expand_chunk_ids(x.get("chunk_id"))
    }
    # Backward-compat fallback when gold chunk_id is not provided.
    if not chunk_ids and doc_ids:
        chunk_ids = set(doc_ids)
    # Backward-compat fallback when gold doc_id is not provided.
    if not doc_ids and chunk_ids:
        doc_ids = set(chunk_ids)
    if not doc_ids:
        query_doc_key = str((gold_row.get("meta") or {}).get("query_doc_key", "") or "").strip()
        title = str((gold_row.get("meta") or {}).get("title", "") or "").strip()
        fallback_doc = _doc_prefix(query_doc_key or title)
        if fallback_doc:
            doc_ids = {fallback_doc}
    edge_ids: set[str] = set()
    community_ids: set[str] = set()
    return {
        "chunks": chunk_ids,
        "docs": doc_ids,
        "edges": edge_ids,
        "communities": community_ids,
    }


def _has_compare_layout(row: dict) -> bool:
    regimes = row.get("regimes")
    return isinstance(regimes, dict) and bool(regimes)


def _detect_compare_methods(pred_rows: list[dict]) -> tuple[str, ...]:
    found: set[str] = set()
    for row in pred_rows:
        if not isinstance(row, dict):
            continue
        regimes = row.get("regimes")
        if not isinstance(regimes, dict):
            continue
        for by_method in regimes.values():
            if not isinstance(by_method, dict):
                continue
            for method in by_method.keys():
                m = str(method).strip()
                if m:
                    found.add(m)
    if found:
        return tuple(sorted(found))
    return METHODS


def _is_youtu_method(method: str) -> bool:
    return "youtu" in str(method).strip().lower()


def _apply_method_mode(methods: tuple[str, ...], method_mode: str) -> tuple[str, ...]:
    mode = str(method_mode or "all").strip().lower()
    if mode == "exclude_youtu":
        return tuple(m for m in methods if not _is_youtu_method(m))
    if mode == "only_youtu":
        return tuple(m for m in methods if _is_youtu_method(m))
    return methods


def _extract_method_telemetry(method: str, payload: dict | None) -> dict:
    if not isinstance(payload, dict):
        return {}
    if method == "vector_rag":
        telemetry = payload.get("telemetry")
        if isinstance(telemetry, dict):
            aggregate = telemetry.get("aggregate")
            if isinstance(aggregate, dict):
                return aggregate
        return {}
    telemetry = payload.get("telemetry")
    return telemetry if isinstance(telemetry, dict) else {}


def _eval_compare(
    pred_rows: list[dict],
    gold_map: dict[str, dict],
    include_evidence: bool,
    methods: tuple[str, ...],
    community_chunk_map: dict[str, set[str]] | None = None,
    community_hierarchy_map: dict[str, set[str]] | None = None,
) -> list[dict]:
    res = []
    for row in pred_rows:
        qid = str(row.get("qid"))
        gold = gold_map.get(qid)
        if not gold:
            continue

        regimes = row.get("regimes", {}) or {}
        for regime, by_method in regimes.items():
            for method in methods:
                payload = (by_method or {}).get(method, {}) or {}
                pred_answer_raw = payload.get("answer", "")
                gold_answer_raw = gold.get("answer", "")
                pred_answer = _answer_to_text(pred_answer_raw)
                gold_answer = _answer_to_text(gold_answer_raw)
                base = {
                    "qid": qid,
                    "type": str(row.get("type", gold.get("type", "unknown"))),
                    "query": str(row.get("query", gold.get("query", ""))),
                    "regime": str(regime),
                    "method": method,
                    "mode": str(row.get("mode", "reject")),
                    "top_k": int(row.get("top_k", 0) or 0),
                    "gold_answer": gold_answer,
                    "pred_answer": pred_answer,
                }
                base.update(_match_scores(pred_answer_raw, gold_answer_raw))

                budget = payload.get("budget_check") or {}
                base["budget_within"] = budget.get("within_budget")
                base["budget_error"] = budget.get("error")
                telemetry = _extract_method_telemetry(method, payload)
                base["llm_calls"] = int(telemetry.get("llm_calls", 0) or 0)
                base["embedding_calls"] = int(telemetry.get("embedding_calls", 0) or 0)
                base["prompt_tokens"] = int(telemetry.get("prompt_tokens", 0) or 0)
                base["completion_tokens"] = int(telemetry.get("completion_tokens", 0) or 0)
                base["total_tokens"] = int(telemetry.get("total_tokens", 0) or 0)
                base["llm_latency_ms"] = int(telemetry.get("llm_latency_ms", 0) or 0)
                base["embedding_latency_ms"] = int(telemetry.get("embedding_latency_ms", 0) or 0)
                base["latency_ms_total"] = base["llm_latency_ms"] + base["embedding_latency_ms"]
                query_doc_key = _extract_query_doc_key(row=row, gold=gold)
                pred_doc_ids = _extract_pred_doc_ids(payload)
                base.update(_retrieval_doc_diagnostics(pred_doc_ids=pred_doc_ids, query_doc_key=query_doc_key))
                base.update(_method_specific_diagnostics(method=method, payload=payload))
                raw_chunks = payload.get("evidence_chunks", []) if isinstance(payload.get("evidence_chunks"), list) else []
                raw_edges = payload.get("subgraph_edges", []) if isinstance(payload.get("subgraph_edges"), list) else []
                base["raw_evidence_chunks"] = len(raw_chunks)
                base["raw_evidence_chunks_empty_chunk_id"] = sum(
                    1
                    for x in raw_chunks
                    if not (isinstance(x, dict) and str(x.get("chunk_id", "")).strip())
                )
                base["raw_subgraph_edges"] = len(raw_edges)
                base["raw_subgraph_edges_empty_edge_id"] = sum(
                    1
                    for x in raw_edges
                    if not (isinstance(x, dict) and str(x.get("edge_id", "")).strip())
                )

                if include_evidence:
                    pred_ids = _collect_pred_ids(payload, community_chunk_map=community_chunk_map)
                    pred_ranked_ids = _collect_pred_ranked_ids(payload, community_chunk_map=community_chunk_map)
                    gold_ids = _collect_gold_ids(gold)
                    p_chunk, r_chunk, f_chunk = _precision_recall_f1(pred_ids["chunks"], gold_ids["chunks"])
                    p_doc, r_doc, f_doc = _precision_recall_f1(pred_ids["docs"], gold_ids["docs"])
                    k_eval = int(row.get("top_k", 0) or 0)
                    chunk_hit_at_k, chunk_recall_at_k = _hit_and_recall_at_k(
                        pred_ranked_ids["chunks"],
                        gold_ids["chunks"],
                        k_eval,
                    )
                    doc_hit_at_k, doc_recall_at_k = _hit_and_recall_at_k(
                        pred_ranked_ids["docs"],
                        gold_ids["docs"],
                        k_eval,
                    )
                    # Primary retrieval policy is type-aware: global* uses doc recall,
                    # other types keep chunk recall.
                    p_e, r_e, f_e = 0.0, 0.0, 0.0
                    p_m, r_m, f_m = 0.0, 0.0, 0.0
                    p_m_relaxed, r_m_relaxed, f_m_relaxed = 0.0, 0.0, 0.0
                    p_a, r_a, f_a = p_chunk, r_chunk, f_chunk
                    qtype = str(base.get("type", ""))
                    primary_recall = r_doc if _is_global_type(qtype) else r_chunk
                    base.update(
                        {
                            "evidence_precision_docs": round(p_doc, 6),
                            "evidence_recall_docs": round(r_doc, 6),
                            "evidence_f1_docs": round(f_doc, 6),
                            "evidence_precision_chunks": round(p_chunk, 6),
                            "evidence_recall_chunks": round(r_chunk, 6),
                            "evidence_f1_chunks": round(f_chunk, 6),
                            "evidence_precision_edges": round(p_e, 6),
                            "evidence_recall_edges": round(r_e, 6),
                            "evidence_f1_edges": round(f_e, 6),
                            "evidence_precision_communities": round(p_m, 6),
                            "evidence_recall_communities": round(r_m, 6),
                            "evidence_f1_communities": round(f_m, 6),
                            "evidence_precision_communities_relaxed": round(p_m_relaxed, 6),
                            "evidence_recall_communities_relaxed": round(r_m_relaxed, 6),
                            "evidence_f1_communities_relaxed": round(f_m_relaxed, 6),
                            "evidence_precision_all": round(p_a, 6),
                            "evidence_recall_all": round(r_a, 6),
                            "evidence_f1_all": round(f_a, 6),
                            "evidence_recall_primary": round(primary_recall, 6),
                            "chunk_hit_rate_at_k": round(chunk_hit_at_k, 6),
                            "chunk_recall_at_k": round(chunk_recall_at_k, 6),
                            "doc_hit_rate_at_k": round(doc_hit_at_k, 6),
                            "doc_recall_at_k": round(doc_recall_at_k, 6),
                            "pred_chunks": len(pred_ids["chunks"]),
                            "pred_docs": len(pred_ids["docs"]),
                            "pred_edges": len(pred_ids["edges"]),
                            "pred_communities": len(pred_ids["communities"]),
                            "gold_chunks": len(gold_ids["chunks"]),
                            "gold_docs": len(gold_ids["docs"]),
                            "gold_edges": len(gold_ids["edges"]),
                            "gold_communities": len(gold_ids["communities"]),
                        }
                    )
                res.append(base)
    return res


def _eval_legacy(pred_rows: list[dict], gold_map: dict[str, dict]) -> list[dict]:
    res = []
    for row in pred_rows:
        qid = str(row.get("qid"))
        gold = gold_map.get(qid)
        if not gold:
            continue
        pred_answer_raw = row.get("answer", "")
        gold_answer_raw = gold.get("answer", "")
        pred_answer = _answer_to_text(pred_answer_raw)
        gold_answer = _answer_to_text(gold_answer_raw)
        item = {
            "qid": qid,
            "type": str(gold.get("type", "unknown")),
            "query": str(gold.get("query", "")),
            "regime": "legacy",
            "method": "legacy",
            "gold_answer": gold_answer,
            "pred_answer": pred_answer,
        }
        item.update(_match_scores(pred_answer_raw, gold_answer_raw))
        res.append(item)
    return res


def _build_summary(rows: list[dict]) -> dict:
    df = pd.DataFrame(rows)
    if df.empty:
        return {"num_rows": 0, "overall": [], "by_type": []}

    group_dims = [x for x in ("regime", "method", "mode", "top_k") if x in df.columns]
    by_type_dims = group_dims + (["type"] if "type" in df.columns else [])

    metric_cols = [
        c
        for c in (
            "answer_similarity",
            "answer_semantic_yesno",
            "summary_semantic_similarity",
            "answer_score_primary",
            "answer_exact",
            "answer_contains",
            "answer_exact_relaxed",
            "answer_contains_relaxed",
            "answer_token_f1",
            "topk_support",
            "topk_correct",
            "topk_accuracy",
            "evidence_precision_all",
            "evidence_recall_all",
            "evidence_f1_all",
            "evidence_recall_primary",
            "chunk_hit_rate_at_k",
            "chunk_recall_at_k",
            "doc_hit_rate_at_k",
            "doc_recall_at_k",
            "evidence_precision_docs",
            "evidence_recall_docs",
            "evidence_f1_docs",
            "evidence_precision_chunks",
            "evidence_recall_chunks",
            "evidence_f1_chunks",
            "evidence_recall_edges",
            "evidence_recall_communities",
            "evidence_recall_communities_relaxed",
            "llm_calls",
            "embedding_calls",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "llm_latency_ms",
            "embedding_latency_ms",
            "latency_ms_total",
            "retrieved_unique_docs",
            "retrieved_top1_doc_share",
            "retrieved_doc_entropy",
            "evidence_doc_self_hit_rate",
            "kg_startnode_hub_ratio",
            "kg_traversed_doc_span",
            "graph_selected_community_reuse_rate",
            "graph_summary_coverage_at_query_level",
        )
        if c in df.columns
    ]

    agg = {m: "mean" for m in metric_cols}
    agg["qid"] = "count"
    overall = (
        df.groupby(group_dims, dropna=False)
        .agg(agg)
        .rename(columns={"qid": "num_queries"})
        .reset_index()
    )

    by_type = (
        df.groupby(by_type_dims, dropna=False)
        .agg(agg)
        .rename(columns={"qid": "num_queries"})
        .reset_index()
    )

    if "budget_within" in df.columns:
        budget_numeric = pd.to_numeric(df["budget_within"], errors="coerce")
        bw_cols = {d: df[d] for d in group_dims}
        bw_cols["budget_within"] = budget_numeric
        bw = pd.DataFrame(bw_cols).groupby(group_dims, dropna=False)["budget_within"].mean().reset_index()
        bw = bw.rename(columns={"budget_within": "budget_within_rate"})
        overall = overall.merge(bw, on=group_dims, how="left")

    for d in (overall, by_type):
        for c in d.columns:
            if c not in ("regime", "method", "type", "mode", "top_k", "num_queries"):
                d[c] = d[c].astype(float).round(6)

    return {
        "num_rows": int(len(df)),
        "overall": overall.to_dict(orient="records"),
        "by_type": by_type.to_dict(orient="records"),
    }


def _build_retrieval_diagnostics(rows: list[dict]) -> dict:
    df = pd.DataFrame(rows)
    if df.empty:
        return {"overall": [], "by_type": []}
    group_dims = [x for x in ("regime", "method", "mode", "top_k") if x in df.columns]
    by_type_dims = group_dims + (["type"] if "type" in df.columns else [])
    metric_cols = [
        c
        for c in (
            "retrieved_unique_docs",
            "retrieved_top1_doc_share",
            "retrieved_doc_entropy",
            "evidence_doc_self_hit_rate",
            "kg_startnode_hub_ratio",
            "kg_traversed_doc_span",
            "graph_selected_community_reuse_rate",
            "graph_summary_coverage_at_query_level",
        )
        if c in df.columns
    ]
    if not metric_cols:
        return {"overall": [], "by_type": []}
    overall = (
        df.groupby(group_dims, dropna=False)[metric_cols]
        .mean(numeric_only=True)
        .reset_index()
    )
    by_type = (
        df.groupby(by_type_dims, dropna=False)[metric_cols]
        .mean(numeric_only=True)
        .reset_index()
    )
    for d in (overall, by_type):
        for c in metric_cols:
            if c in d.columns:
                d[c] = d[c].astype(float).round(6)
    return {
        "overall": overall.to_dict(orient="records"),
        "by_type": by_type.to_dict(orient="records"),
    }


def _build_alignment_preflight(rows: list[dict]) -> dict:
    df = pd.DataFrame(rows)
    if df.empty or "method" not in df.columns:
        return {"overall": [], "alerts": []}
    if "raw_evidence_chunks" not in df.columns or "raw_subgraph_edges" not in df.columns:
        return {"overall": [], "alerts": []}

    methods = sorted(set(str(x) for x in df["method"].dropna().tolist()))
    out_rows: list[dict] = []
    alerts: list[str] = []
    for method in methods:
        mdf = df[df["method"] == method]
        chunks_total = float(mdf.get("raw_evidence_chunks", pd.Series(dtype=float)).sum())
        chunks_empty = float(mdf.get("raw_evidence_chunks_empty_chunk_id", pd.Series(dtype=float)).sum())
        edges_total = float(mdf.get("raw_subgraph_edges", pd.Series(dtype=float)).sum())
        edges_empty = float(mdf.get("raw_subgraph_edges_empty_edge_id", pd.Series(dtype=float)).sum())
        chunk_empty_rate = (chunks_empty / chunks_total) if chunks_total > 0 else None
        edge_empty_rate = (edges_empty / edges_total) if edges_total > 0 else None
        row = {
            "method": method,
            "raw_evidence_chunks_total": int(chunks_total),
            "raw_evidence_chunks_empty_chunk_id_total": int(chunks_empty),
            "raw_subgraph_edges_total": int(edges_total),
            "raw_subgraph_edges_empty_edge_id_total": int(edges_empty),
            "chunk_id_empty_rate": (round(float(chunk_empty_rate), 6) if chunk_empty_rate is not None else None),
            "edge_id_empty_rate": (round(float(edge_empty_rate), 6) if edge_empty_rate is not None else None),
        }
        out_rows.append(row)
        if "youtu" in method.lower():
            if chunk_empty_rate is not None and chunk_empty_rate >= 0.95:
                alerts.append(f"{method}: chunk_id_empty_rate={chunk_empty_rate:.3f} (>=0.95)")
            if edge_empty_rate is not None and edge_empty_rate >= 0.95:
                alerts.append(f"{method}: edge_id_empty_rate={edge_empty_rate:.3f} (>=0.95)")
    return {"overall": out_rows, "alerts": alerts}


def _dedup_compare_eval_rows(rows: list[dict]) -> list[dict]:
    """Keep the latest row for each qid+regime+method+mode+top_k to avoid incremental duplicates."""
    dedup: dict[tuple[str, str, str, str, str], dict] = {}
    ordered_keys: list[tuple[str, str, str, str, str]] = []
    for row in rows:
        key = (
            str(row.get("qid", "")),
            str(row.get("regime", "")),
            str(row.get("method", "")),
            str(row.get("mode", "")),
            str(row.get("top_k", "")),
        )
        if key not in dedup:
            ordered_keys.append(key)
        dedup[key] = row
    return [dedup[k] for k in ordered_keys]


def _plot_metric_bar(
    df: pd.DataFrame,
    metric: str,
    title: str,
    out_file: Path,
    hue: str = "regime",
) -> None:
    if metric not in df.columns:
        return
    plt.figure(figsize=(8, 4.8))
    ax = sns.barplot(data=df, x="method", y=metric, hue=hue)
    ax.set_title(title)
    ax.set_xlabel("Method")
    ax.set_ylabel(metric)
    ax.tick_params(axis="x", rotation=0)
    plt.tight_layout()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=160)
    plt.close()


def _generate_plots(summary: dict, plots_dir: str) -> list[str]:
    out_paths: list[str] = []
    payload = summary.get("summary", {})
    overall = pd.DataFrame(payload.get("overall", []))
    by_type = pd.DataFrame(payload.get("by_type", []))
    out_dir = Path(plots_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not overall.empty:
        metrics = [
            ("answer_similarity", "Answer Similarity by Method/Regime", "answer_similarity.png"),
            ("answer_semantic_yesno", "Answer Semantic Yes/No by Method/Regime", "answer_semantic_yesno.png"),
            ("answer_score_primary", "Primary Answer Score by Method/Regime", "answer_score_primary.png"),
            (
                "summary_semantic_similarity",
                "Summary Semantic Similarity by Method/Regime",
                "summary_semantic_similarity.png",
            ),
            ("answer_contains", "Answer Contains Rate by Method/Regime", "answer_contains.png"),
            ("answer_exact", "Answer Exact Match Rate by Method/Regime", "answer_exact.png"),
            ("answer_contains_relaxed", "Answer Contains Relaxed by Method/Regime", "answer_contains_relaxed.png"),
            ("answer_exact_relaxed", "Answer Exact Relaxed by Method/Regime", "answer_exact_relaxed.png"),
            ("answer_token_f1", "Answer Token F1 by Method/Regime", "answer_token_f1.png"),
            ("evidence_recall_docs", "Evidence Doc Recall by Method/Regime", "evidence_recall_docs.png"),
            ("evidence_recall_chunks", "Evidence Chunk Recall by Method/Regime", "evidence_recall_chunks.png"),
            ("evidence_recall_edges", "Evidence Edge Recall by Method/Regime", "evidence_recall_edges.png"),
            ("evidence_recall_all", "Evidence Recall (All IDs) by Method/Regime", "evidence_recall_all.png"),
            ("budget_within_rate", "Budget Within Rate by Method/Regime", "budget_within_rate.png"),
            ("retrieved_unique_docs", "Retrieved Unique Docs by Method/Regime", "retrieved_unique_docs.png"),
            ("retrieved_top1_doc_share", "Retrieved Top1 Doc Share by Method/Regime", "retrieved_top1_doc_share.png"),
            ("retrieved_doc_entropy", "Retrieved Doc Entropy by Method/Regime", "retrieved_doc_entropy.png"),
            ("kg_startnode_hub_ratio", "KG Start Node Hub Ratio by Method/Regime", "kg_startnode_hub_ratio.png"),
            (
                "graph_summary_coverage_at_query_level",
                "Graph Summary Coverage at Query Level",
                "graph_summary_coverage_at_query_level.png",
            ),
        ]
        for metric, title, filename in metrics:
            out_file = out_dir / filename
            _plot_metric_bar(overall, metric=metric, title=title, out_file=out_file)
            if out_file.exists():
                out_paths.append(str(out_file))

    if not by_type.empty and "answer_similarity" in by_type.columns:
        g = sns.catplot(
            data=by_type,
            x="method",
            y="answer_similarity",
            hue="regime",
            col="type",
            kind="bar",
            height=4,
            aspect=1,
            col_wrap=2,
            sharey=True,
        )
        g.fig.subplots_adjust(top=0.9)
        g.fig.suptitle("Answer Similarity by Type / Method / Regime")
        out_file = out_dir / "answer_similarity_by_type.png"
        g.savefig(out_file, dpi=160)
        plt.close(g.fig)
        if out_file.exists():
            out_paths.append(str(out_file))

    if not by_type.empty and "answer_score_primary" in by_type.columns:
        g = sns.catplot(
            data=by_type,
            x="method",
            y="answer_score_primary",
            hue="regime",
            col="type",
            kind="bar",
            height=4,
            aspect=1,
            col_wrap=2,
            sharey=True,
        )
        g.fig.subplots_adjust(top=0.9)
        g.fig.suptitle("Primary Answer Score by Type / Method / Regime")
        out_file = out_dir / "answer_score_primary_by_type.png"
        g.savefig(out_file, dpi=160)
        plt.close(g.fig)
        if out_file.exists():
            out_paths.append(str(out_file))

    if not by_type.empty and "answer_semantic_yesno" in by_type.columns:
        g = sns.catplot(
            data=by_type,
            x="method",
            y="answer_semantic_yesno",
            hue="regime",
            col="type",
            kind="bar",
            height=4,
            aspect=1,
            col_wrap=2,
            sharey=True,
        )
        g.fig.subplots_adjust(top=0.9)
        g.fig.suptitle("Answer Semantic Yes/No by Type / Method / Regime")
        out_file = out_dir / "answer_semantic_yesno_by_type.png"
        g.savefig(out_file, dpi=160)
        plt.close(g.fig)
        if out_file.exists():
            out_paths.append(str(out_file))

    if not by_type.empty and "evidence_recall_chunks" in by_type.columns:
        g = sns.catplot(
            data=by_type,
            x="method",
            y="evidence_recall_chunks",
            hue="regime",
            col="type",
            kind="bar",
            height=4,
            aspect=1,
            col_wrap=2,
            sharey=True,
        )
        g.fig.subplots_adjust(top=0.9)
        g.fig.suptitle("Evidence Chunk Recall by Type / Method / Regime")
        out_file = out_dir / "evidence_recall_chunks_by_type.png"
        g.savefig(out_file, dpi=160)
        plt.close(g.fig)
        if out_file.exists():
            out_paths.append(str(out_file))

    if not by_type.empty:
        by_type_metrics = [
            ("total_tokens", "Token Usage by Type / Method / Regime", "total_tokens_by_type.png"),
            ("latency_ms_total", "Latency by Type / Method / Regime", "latency_ms_total_by_type.png"),
            ("prompt_tokens", "Prompt Tokens by Type / Method / Regime", "prompt_tokens_by_type.png"),
            ("completion_tokens", "Completion Tokens by Type / Method / Regime", "completion_tokens_by_type.png"),
            ("retrieved_unique_docs", "Retrieved Unique Docs by Type / Method / Regime", "retrieved_unique_docs_by_type.png"),
            (
                "graph_summary_coverage_at_query_level",
                "Graph Summary Coverage by Type / Method / Regime",
                "graph_summary_coverage_by_type.png",
            ),
        ]
        for metric, title, filename in by_type_metrics:
            if metric not in by_type.columns:
                continue
            g = sns.catplot(
                data=by_type,
                x="method",
                y=metric,
                hue="regime",
                col="type",
                kind="bar",
                height=4,
                aspect=1,
                col_wrap=2,
                sharey=True,
            )
            g.fig.subplots_adjust(top=0.9)
            g.fig.suptitle(title)
            out_file = out_dir / filename
            g.savefig(out_file, dpi=160)
            plt.close(g.fig)
            if out_file.exists():
                out_paths.append(str(out_file))

    return out_paths


def run_eval(
    pred_file: str,
    gold_file: str,
    out_csv: str,
    out_summary: str,
    graph_file: str | None = None,
    communities_file: str | None = None,
    expand_community_chunks: bool = True,
    make_plots: bool = True,
    plots_dir: str = "outputs/results/eval_plots",
    method_mode: str = "all",
    semantic_similarity: bool = True,
    semantic_target_types: str = "global_summary",
    semantic_max_chars: int = 3000,
    global_summary_primary_mode: str = "semantic",
    global_summary_alpha_semantic: float = 1.0,
    judge_mode: str = "off",
    judge_model: str = "qwen-flash",
) -> dict:
    sns.set_theme(style=PLOT_STYLE)
    pred_rows = _read_jsonl(pred_file)
    gold_rows = _read_jsonl(gold_file)
    gold_map = {str(x.get("qid")): x for x in gold_rows if x.get("qid")}
    include_evidence = any("supporting_chunks" in x or "supporting_edges" in x for x in gold_rows)
    community_chunk_map = (
        _build_community_chunk_map(graph_file=graph_file, communities_file=communities_file)
        if expand_community_chunks
        else {}
    )
    community_hierarchy_map = _build_community_hierarchy_map(communities_file=communities_file)

    has_compare_layout = any(isinstance(x, dict) and _has_compare_layout(x) for x in pred_rows)
    if has_compare_layout:
        methods = _apply_method_mode(_detect_compare_methods(pred_rows), method_mode=method_mode)
        rows = _eval_compare(
            pred_rows,
            gold_map,
            include_evidence=include_evidence,
            methods=methods,
            community_chunk_map=community_chunk_map,
            community_hierarchy_map=community_hierarchy_map,
        )
        rows = _dedup_compare_eval_rows(rows)
    else:
        methods = ("legacy",)
        rows = _eval_legacy(pred_rows, gold_map)

    semantic_types = {
        x.strip().lower()
        for x in str(semantic_target_types or "").split(",")
        if x.strip()
    } or {"global_summary"}
    semantic_meta = _attach_semantic_similarity(
        rows=rows,
        enabled=bool(semantic_similarity),
        target_types=semantic_types,
        max_chars=int(semantic_max_chars),
    )
    judge_meta = _apply_llm_yesno_judge(
        rows=rows,
        judge_mode=judge_mode,
        judge_model=judge_model,
    )
    topk_meta = _apply_topk_accuracy_fields(rows=rows)
    primary_policy_meta = _apply_primary_answer_policy(
        rows=rows,
        global_summary_mode=global_summary_primary_mode,
        global_summary_alpha_semantic=float(global_summary_alpha_semantic),
    )

    out_csv_path = Path(out_csv)
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    summary = {
        "pred_file": pred_file,
        "gold_file": gold_file,
        "out_csv": out_csv,
        "num_pred_rows": len(pred_rows),
        "num_gold_rows": len(gold_rows),
        "mode": "compare" if has_compare_layout else "legacy",
        "method_mode": method_mode,
        "methods": list(methods),
        "expand_community_chunks": bool(expand_community_chunks),
        "community_chunk_map_size": len(community_chunk_map),
        "community_hierarchy_nodes": len(community_hierarchy_map),
        "evidence_metric_policy": {
            "default_primary_recall": "evidence_recall_chunks",
            "global_primary_recall": "evidence_recall_docs",
            "primary_recall_by_type_rule": "type startswith('global') -> evidence_recall_docs; else -> evidence_recall_chunks",
            "matching_granularity": "chunk_id",
            "ranked_retrieval_metrics": {
                "hit_rate_at_k": "I[|TopK ∩ Gold| > 0]",
                "recall_at_k": "|TopK ∩ Gold| / |Gold|",
                "ranking_source": "returned evidence order",
            },
        },
        "answer_metric_policy": {
            "default_primary_answer_score": "answer_similarity",
            "global_summary_primary_answer_score": (
                "composite(summary_semantic_similarity, evidence_recall_primary)"
                if str(global_summary_primary_mode).strip().lower() == "composite"
                else "summary_semantic_similarity"
            ),
            "global_summary_primary_mode": str(global_summary_primary_mode),
            "global_summary_alpha_semantic": float(global_summary_alpha_semantic),
            "semantic_similarity_enabled": bool(semantic_similarity),
            "semantic_target_types": sorted(semantic_types),
            "semantic_max_chars": int(semantic_max_chars),
            "judge_mode": str(judge_mode),
            "judge_model": str(judge_model),
            "topk_accuracy_definition": "I[answer_correct=1 AND support_at_k=1]",
            "support_at_k_source": "evidence_recall_primary > 0",
            "answer_correct_priority": ["answer_semantic_yesno", "answer_exact_relaxed"],
        },
        "semantic_similarity": semantic_meta,
        "judge": judge_meta,
        "topk_accuracy": topk_meta,
        "primary_answer_policy": primary_policy_meta,
        "summary": _build_summary(rows),
        "retrieval_diagnostics": _build_retrieval_diagnostics(rows),
        "alignment_preflight": _build_alignment_preflight(rows),
    }
    if make_plots:
        summary["plots"] = _generate_plots(summary, plots_dir=plots_dir)
        summary["plots_dir"] = plots_dir
    out_summary_path = Path(out_summary)
    out_summary_path.parent.mkdir(parents=True, exist_ok=True)
    out_summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate comparison answers against gold references.")
    parser.add_argument("--pred-file", default="outputs/results/compare_answers.jsonl")
    parser.add_argument("--gold-file", default="data/queries/gold_qa.jsonl")
    parser.add_argument("--out-csv", default="outputs/results/eval_compare.csv")
    parser.add_argument("--out-summary", default="outputs/results/eval_compare_summary.json")
    parser.add_argument("--graph-file", default="outputs/graph/graph.json")
    parser.add_argument("--communities-file", default="outputs/graph/communities.json")
    parser.add_argument("--disable-community-chunk-expand", action="store_true")
    parser.add_argument("--disable-plots", action="store_true")
    parser.add_argument("--plots-dir", default="outputs/results/eval_plots")
    parser.add_argument(
        "--method-mode",
        choices=["all", "exclude_youtu", "only_youtu"],
        default="all",
        help="Filter compared methods: all / exclude_youtu / only_youtu",
    )
    parser.add_argument("--disable-semantic-similarity", action="store_true")
    parser.add_argument(
        "--semantic-target-types",
        default="global_summary",
        help="Comma-separated query types using embedding-based semantic similarity.",
    )
    parser.add_argument("--semantic-max-chars", type=int, default=3000)
    parser.add_argument(
        "--global-summary-primary-mode",
        choices=["semantic", "composite"],
        default="semantic",
        help="Primary score mode for global_summary rows.",
    )
    parser.add_argument(
        "--global-summary-alpha-semantic",
        type=float,
        default=1.0,
        help="Composite mode weight: alpha*semantic + (1-alpha)*community_recall_relaxed.",
    )
    parser.add_argument(
        "--judge-mode",
        choices=["off", "llm_yesno"],
        default="off",
        help="Answer judge mode: off or llm_yesno.",
    )
    parser.add_argument(
        "--judge-model",
        default="qwen-flash",
        help="Judge model name used when --judge-mode llm_yesno.",
    )
    args = parser.parse_args()

    summary = run_eval(
        pred_file=args.pred_file,
        gold_file=args.gold_file,
        out_csv=args.out_csv,
        out_summary=args.out_summary,
        graph_file=args.graph_file,
        communities_file=args.communities_file,
        expand_community_chunks=not args.disable_community_chunk_expand,
        make_plots=not args.disable_plots,
        plots_dir=args.plots_dir,
        method_mode=args.method_mode,
        semantic_similarity=not args.disable_semantic_similarity,
        semantic_target_types=args.semantic_target_types,
        semantic_max_chars=args.semantic_max_chars,
        global_summary_primary_mode=args.global_summary_primary_mode,
        global_summary_alpha_semantic=args.global_summary_alpha_semantic,
        judge_mode=args.judge_mode,
        judge_model=args.judge_model,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
