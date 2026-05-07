import json
import time
from pathlib import Path

import faiss
import numpy as np

from utils.embedder import embed_texts
from utils.llm_wrapper import llm


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def _doc_prefix(value):
    return str(value or "").split("#", 1)[0].strip()


def _read_chunk_meta(chunks_file):
    texts = []
    meta = []
    with open(chunks_file, encoding="utf-8") as r:
        for line in r:
            obj = json.loads(line)
            texts.append(obj["text"])
            meta.append(obj)
    return texts, meta


def build_index(chunks_file, idx_file, store_file):
    t0 = time.perf_counter()
    texts, meta = _read_chunk_meta(chunks_file)

    t_embed_start = time.perf_counter()
    embs = embed_texts(texts)
    embs = _normalize_rows(np.asarray(embs, dtype="float32"))
    t_embed_end = time.perf_counter()

    t_index_start = time.perf_counter()
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embs)
    faiss.write_index(index, idx_file)
    t_index_end = time.perf_counter()

    with open(store_file, "w", encoding="utf-8") as w:
        json.dump(meta, w, ensure_ascii=False)

    idx_path = Path(idx_file)
    store_path = Path(store_file)
    metrics = {
        "chunks_file": chunks_file,
        "idx_file": idx_file,
        "store_file": store_file,
        "num_chunks": len(meta),
        "embedding_dim": int(dim),
        "embedding_time_sec": round(t_embed_end - t_embed_start, 4),
        "index_build_time_sec": round(t_index_end - t_index_start, 4),
        "total_build_time_sec": round(time.perf_counter() - t0, 4),
        "index_size_bytes": idx_path.stat().st_size if idx_path.exists() else 0,
        "store_size_bytes": store_path.stat().st_size if store_path.exists() else 0,
    }
    return metrics


def build_index_from_embeddings(chunks_file, embeddings_file, idx_file, store_file):
    t0 = time.perf_counter()
    _, meta = _read_chunk_meta(chunks_file)

    embs = np.load(embeddings_file).astype("float32")
    if embs.shape[0] != len(meta):
        raise ValueError(
            f"Embedding rows ({embs.shape[0]}) != chunks ({len(meta)}). "
            "Ensure the same chunks file was used."
        )
    embs = _normalize_rows(embs)

    t_index_start = time.perf_counter()
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embs)
    faiss.write_index(index, idx_file)
    t_index_end = time.perf_counter()

    with open(store_file, "w", encoding="utf-8") as w:
        json.dump(meta, w, ensure_ascii=False)

    idx_path = Path(idx_file)
    store_path = Path(store_file)
    metrics = {
        "chunks_file": chunks_file,
        "embeddings_file": embeddings_file,
        "idx_file": idx_file,
        "store_file": store_file,
        "num_chunks": len(meta),
        "embedding_dim": int(dim),
        "embedding_time_sec": None,
        "index_build_time_sec": round(t_index_end - t_index_start, 4),
        "total_build_time_sec": round(time.perf_counter() - t0, 4),
        "index_size_bytes": idx_path.stat().st_size if idx_path.exists() else 0,
        "store_size_bytes": store_path.stat().st_size if store_path.exists() else 0,
    }
    return metrics


def retrieve_and_answer(query, idx_file, store_file, top_k):
    evidence, _ = retrieve_with_evidence(
        query=query,
        idx_file=idx_file,
        store_file=store_file,
        top_k=top_k,
        return_meta=True,
    )
    contexts = [e["text"] for e in evidence]
    return contexts


def retrieve_with_evidence(
    query,
    idx_file,
    store_file,
    top_k,
    return_meta=False,
    doc_prefix_filter=None,
    scan_multiplier=50,
):
    index = faiss.read_index(idx_file)
    with open(store_file, encoding="utf-8") as r:
        store = json.load(r)

    q_emb_out = embed_texts([query], return_meta=True)
    q_emb = np.asarray(q_emb_out[0], dtype="float32")
    emb_meta = q_emb_out[1]
    q_emb = _normalize_rows(q_emb)

    if doc_prefix_filter:
        search_k = min(len(store), max(int(top_k), int(top_k) * int(scan_multiplier)))
    else:
        search_k = int(top_k)
    scores, indices = index.search(q_emb, search_k)
    evidence = []
    wanted_prefix = _doc_prefix(doc_prefix_filter) if doc_prefix_filter else ""
    for rank, idx in enumerate(indices[0], start=1):
        if not (0 <= idx < len(store)):
            continue
        chunk = store[idx]
        if wanted_prefix and _doc_prefix(chunk.get("doc_id")) != wanted_prefix:
            continue
        evidence.append(
            {
                "rank": len(evidence) + 1,
                "score": float(scores[0][rank - 1]),
                "chunk_id": chunk.get("chunk_id"),
                "doc_id": chunk.get("doc_id"),
                "source": chunk.get("source"),
                "meta": chunk.get("meta", {}),
                "text": chunk.get("text", ""),
            }
        )
        if len(evidence) >= int(top_k):
            break
    if return_meta:
        return evidence, {"embedding": emb_meta}
    return evidence


def answer_with_context(
    query,
    contexts,
    max_completion_tokens=None,
    return_meta=False,
    query_type: str | None = None,
    answer_mode: str = "reject",
):
    mode = str(answer_mode or "reject").strip().lower()
    reject_line = "If evidence is insufficient, return exactly: NOT_FOUND"
    open_line = (
        "If evidence is insufficient, you may answer using general knowledge. "
        "Prefix the answer with: OUTSIDE_EVIDENCE:"
    )
    insufficient_policy = reject_line if mode == "reject" else open_line

    if str(query_type or "").strip().lower() == "global_summary":
        prompt = f"""
Summarize the key themes and risks from the provided contexts.
Use only the provided contexts.
Return 3 concise bullet points (no preface, no legal advice).
{insufficient_policy}

{contexts}

Question:
{query}
"""
    else:
        prompt = f"""
Answer the question using only the following contexts.
This is an extractive QA task.
Return the shortest exact span copied from the context that answers the question.
Do not add explanation, bullet points, or legal commentary.
{insufficient_policy}

{contexts}

Question:
{query}
"""
    return llm.chat(prompt, max_tokens=max_completion_tokens, return_meta=return_meta)
