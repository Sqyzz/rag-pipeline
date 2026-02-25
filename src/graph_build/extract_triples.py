from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.llm_wrapper import llm_chat
from utils.telemetry import Telemetry

JSON_BLOCK_RE = re.compile(r"```json\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _progress(message: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[extract_triples {ts}] {message}", flush=True)


@dataclass
class Triple:
    chunk_id: str
    subject: str
    subject_type: str
    relation: str
    object: str
    object_type: str
    evidence: str

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "subject": self.subject,
            "subject_type": self.subject_type,
            "relation": self.relation,
            "object": self.object,
            "object_type": self.object_type,
            "evidence": self.evidence,
        }


def _safe_json_loads(content: str):
    content = content.strip()
    if not content:
        return []
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        m = JSON_BLOCK_RE.search(content)
        if not m:
            start = content.find("[")
            end = content.rfind("]")
            if start >= 0 and end > start:
                return json.loads(content[start : end + 1])
            raise
        return json.loads(m.group(1))


def _normalize_entity_text(text: str) -> str:
    """Light normalization to reduce trivial surface-form drift."""
    t = re.sub(r"\s+", " ", str(text or "")).strip()
    t = t.strip("`\"' ")
    t = re.sub(r"^[,;:.()\[\]{}]+|[,;:.()\[\]{}]+$", "", t).strip()
    return t


def _normalize_relation_text(text: str) -> str:
    t = re.sub(r"\s+", " ", str(text or "")).strip()
    t = t.strip("`\"' ")
    t = re.sub(r"[_/]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"^[,;:.()\[\]{}]+|[,;:.()\[\]{}]+$", "", t).strip()
    return t.lower()


def _load_schema(schema_file: str | None) -> dict[str, Any] | None:
    if not schema_file:
        return None
    p = Path(schema_file)
    if not p.exists():
        return None
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return None
    return data


def _compile_schema(schema: dict[str, Any] | None) -> dict[str, Any]:
    if not schema:
        return {"enabled": False}
    entity_types = {str(x).strip() for x in schema.get("entity_types", []) if str(x).strip()}
    entity_type_alias_to_name: dict[str, str] = {t.lower(): t for t in entity_types}
    rel_rows = schema.get("relations", [])
    relation_alias_to_name: dict[str, str] = {}
    relation_constraints: dict[str, dict[str, set[str]]] = {}
    for row in rel_rows:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name", "")).strip()
        if not name:
            continue
        canon = name.lower()
        relation_alias_to_name[canon] = name
        for alias in row.get("aliases", []):
            a = str(alias).strip().lower()
            if a:
                relation_alias_to_name[a] = name
        relation_constraints[name] = {
            "subject_types": {str(x).strip() for x in row.get("subject_types", []) if str(x).strip()},
            "object_types": {str(x).strip() for x in row.get("object_types", []) if str(x).strip()},
        }
    return {
        "enabled": True,
        "entity_types": entity_types,
        "entity_type_alias_to_name": entity_type_alias_to_name,
        "relation_alias_to_name": relation_alias_to_name,
        "relation_constraints": relation_constraints,
    }


def _schema_prompt_block(schema: dict[str, Any] | None) -> str:
    if not schema:
        return ""
    return f"""
You must follow this extraction schema:
{json.dumps(schema, ensure_ascii=False)}

Output MUST include:
- subject_type
- object_type

relation must be one of schema.relations.name (or its aliases mapped to canonical name).
subject_type/object_type must be one of schema.entity_types.
"""


def _validate_row_to_triple(
    row: dict[str, Any],
    chunk_id: str,
    compiled_schema: dict[str, Any],
) -> Triple | None:
    s = _normalize_entity_text(row.get("subject", ""))
    r_raw = _normalize_relation_text(row.get("relation", ""))
    o = _normalize_entity_text(row.get("object", ""))
    e = str(row.get("evidence", "")).strip()
    st_raw = str(row.get("subject_type", "")).strip()
    ot_raw = str(row.get("object_type", "")).strip()
    if not (s and r_raw and o):
        return None
    if not compiled_schema.get("enabled"):
        return Triple(
            chunk_id=chunk_id,
            subject=s,
            subject_type=st_raw or "unknown",
            relation=r_raw,
            object=o,
            object_type=ot_raw or "unknown",
            evidence=e,
        )

    alias_map = compiled_schema["relation_alias_to_name"]
    relation = alias_map.get(r_raw.lower())
    if not relation:
        return None
    type_alias_map = compiled_schema["entity_type_alias_to_name"]
    st = type_alias_map.get(st_raw.lower(), st_raw)
    ot = type_alias_map.get(ot_raw.lower(), ot_raw)
    entity_types = compiled_schema["entity_types"]
    if st not in entity_types or ot not in entity_types:
        return None
    cons = compiled_schema["relation_constraints"].get(relation, {})
    allow_s = cons.get("subject_types", set())
    allow_o = cons.get("object_types", set())
    if allow_s and st not in allow_s:
        return None
    if allow_o and ot not in allow_o:
        return None
    return Triple(
        chunk_id=chunk_id,
        subject=s,
        subject_type=st,
        relation=relation,
        object=o,
        object_type=ot,
        evidence=e,
    )


def _extract_with_llm(
    text: str,
    chunk_id: str,
    max_text_chars: int,
    schema: dict[str, Any] | None,
    compiled_schema: dict[str, Any],
) -> tuple[list[Triple], dict, int]:
    prompt = f"""
Extract knowledge triples from the text below.
Return STRICT JSON array only. No extra words.
{_schema_prompt_block(schema)}

Schema for each item:
{{
  "subject": "entity A",
  "subject_type": "type of subject entity",
  "relation": "verb phrase",
  "object": "entity B",
  "object_type": "type of object entity",
  "evidence": "short quote from source text"
}}

Rules:
1) Keep relation concise (1-5 words).
2) Use explicit entities, avoid pronouns.
3) Extract at most 8 triples.
4) If no confident triple, return [].
5) Use canonical entity names consistently (e.g., avoid alternating abbreviations/full names in the same chunk).

Text:
\"\"\"
{text[:max_text_chars]}
\"\"\"
"""
    content, meta = llm_chat(
        [{"role": "user", "content": prompt}],
        temperature=0,
        return_meta=True,
    )
    rows = _safe_json_loads(content)
    triples: list[Triple] = []
    dropped = 0
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict):
                continue
            t = _validate_row_to_triple(row, chunk_id=chunk_id, compiled_schema=compiled_schema)
            if t is None:
                dropped += 1
                continue
            triples.append(t)
    return triples, meta, dropped


def _extract_batch_with_llm(
    chunks: list[dict],
    max_text_chars: int,
    schema: dict[str, Any] | None,
    compiled_schema: dict[str, Any],
) -> tuple[list[Triple], dict, int]:
    chunk_payload = [{"chunk_id": c["chunk_id"], "text": c["text"][:max_text_chars]} for c in chunks]
    prompt = f"""
Extract knowledge triples from multiple chunks.
Return STRICT JSON array only. No extra words.
{_schema_prompt_block(schema)}

Input:
{chunk_payload}

Schema for each output item:
{{
  "chunk_id": "source chunk id from input",
  "subject": "entity A",
  "subject_type": "type of subject entity",
  "relation": "verb phrase",
  "object": "entity B",
  "object_type": "type of object entity",
  "evidence": "short quote from source text"
}}

Rules:
1) Keep relation concise (1-5 words).
2) Use explicit entities, avoid pronouns.
3) At most 8 triples per chunk.
4) If no confident triple for a chunk, emit none for that chunk.
5) chunk_id must come from input.
6) Use canonical entity names consistently within and across chunks when possible.
"""
    content, meta = llm_chat(
        [{"role": "user", "content": prompt}],
        temperature=0,
        return_meta=True,
    )
    rows = _safe_json_loads(content)
    triples: list[Triple] = []
    dropped = 0
    valid_chunk_ids = {c["chunk_id"] for c in chunks}
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict):
                continue
            cid = str(row.get("chunk_id", "")).strip()
            if cid not in valid_chunk_ids:
                continue
            t = _validate_row_to_triple(row, chunk_id=cid, compiled_schema=compiled_schema)
            if t is None:
                dropped += 1
                continue
            triples.append(t)
    return triples, meta, dropped


def extract_triples(
    chunks_file: str,
    out_file: str,
    metrics_file: str = "outputs/results/triple_extract_metrics.json",
    progress_every: int = 50,
    mode: str = "per_chunk",
    batch_size: int = 4,
    min_chars: int = 60,
    max_chunks: int | None = None,
    max_text_chars: int = 5000,
    concurrency: int = 1,
    schema_file: str | None = "config_triple_schema.json",
) -> dict:
    telemetry = Telemetry()
    total_chunks = 0
    eligible_chunks = 0
    total_triples = 0
    parse_failures = 0
    filtered_by_schema = 0
    llm_calls = 0
    schema = _load_schema(schema_file)
    compiled_schema = _compile_schema(schema)

    out_path = Path(out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    _progress(
        f"start mode={mode} chunks_file={chunks_file} schema_file={schema_file} schema_enabled={compiled_schema.get('enabled')}"
    )
    def _iter_eligible_chunks() -> list[dict]:
        rows: list[dict] = []
        with open(chunks_file, encoding="utf-8") as f:
            for line in f:
                if max_chunks is not None and total_chunks + len(rows) >= int(max_chunks):
                    break
                chunk = json.loads(line)
                chunk_id = str(chunk.get("chunk_id", "")).strip()
                doc_id = str(chunk.get("doc_id", "")).strip() or None
                text = str(chunk.get("text", "")).strip()
                if not chunk_id:
                    continue
                rows.append({"chunk_id": chunk_id, "doc_id": doc_id, "text": text})
        return rows

    def _chunk_batches(rows: list[dict], size: int) -> list[list[dict]]:
        if size <= 0:
            size = 1
        return [rows[i : i + size] for i in range(0, len(rows), size)]

    def _worker_per_chunk(chunk_row: dict) -> dict:
        text = chunk_row["text"]
        if not text or len(text) < int(min_chars):
            return {"ok": True, "meta": None, "triples": [], "chunks": 1}
        try:
            triples, meta, dropped = _extract_with_llm(
                text,
                chunk_id=chunk_row["chunk_id"],
                max_text_chars=max_text_chars,
                schema=schema,
                compiled_schema=compiled_schema,
            )
            return {"ok": True, "meta": meta, "triples": triples, "chunks": 1, "dropped": dropped}
        except Exception:
            return {"ok": False, "meta": None, "triples": [], "chunks": 1}

    def _worker_batched(batch_rows: list[dict]) -> dict:
        eligible = [x for x in batch_rows if x["text"] and len(x["text"]) >= int(min_chars)]
        if not eligible:
            return {"ok": True, "meta": None, "triples": [], "chunks": len(batch_rows)}
        try:
            triples, meta, dropped = _extract_batch_with_llm(
                eligible,
                max_text_chars=max_text_chars,
                schema=schema,
                compiled_schema=compiled_schema,
            )
            return {
                "ok": True,
                "meta": meta,
                "triples": triples,
                "chunks": len(batch_rows),
                "chunk_doc": {x["chunk_id"]: x["doc_id"] for x in eligible},
                "dropped": dropped,
            }
        except Exception:
            return {"ok": False, "meta": None, "triples": [], "chunks": len(batch_rows)}

    rows = _iter_eligible_chunks()
    total_chunks = len(rows)
    eligible_chunks = sum(1 for x in rows if x["text"] and len(x["text"]) >= int(min_chars))
    _progress(
        f"prepared chunks total={total_chunks}, eligible={eligible_chunks}, mode={mode}, concurrency={max(int(concurrency),1)}"
    )

    with out_path.open("w", encoding="utf-8") as out:
        processed_chunks = 0
        if mode == "batched":
            tasks = _chunk_batches(rows, int(batch_size))
            with ThreadPoolExecutor(max_workers=max(int(concurrency), 1)) as ex:
                for result in ex.map(_worker_batched, tasks):
                    processed_chunks += int(result.get("chunks", 0))
                    if not result.get("ok", True):
                        parse_failures += 1
                        continue
                    if result.get("meta"):
                        telemetry.add_llm(result["meta"])
                        llm_calls += 1
                    filtered_by_schema += int(result.get("dropped", 0))
                    triples = result.get("triples", [])
                    chunk_doc = result.get("chunk_doc") or {}
                    for t in triples:
                        record = {
                            "chunk_id": t.chunk_id,
                            "doc_id": chunk_doc.get(t.chunk_id),
                            **t.to_dict(),
                        }
                        out.write(json.dumps(record, ensure_ascii=False) + "\n")
                        total_triples += 1
                    if progress_every > 0 and processed_chunks % int(progress_every) == 0:
                        _progress(
                            f"processed_chunks={processed_chunks}, eligible_chunks={eligible_chunks}, llm_calls={llm_calls}, total_triples={total_triples}, parse_failures={parse_failures}"
                        )
        else:
            chunk_doc = {x["chunk_id"]: x["doc_id"] for x in rows}
            with ThreadPoolExecutor(max_workers=max(int(concurrency), 1)) as ex:
                for result in ex.map(_worker_per_chunk, rows):
                    processed_chunks += int(result.get("chunks", 0))
                    if not result.get("ok", True):
                        parse_failures += 1
                        continue
                    if result.get("meta"):
                        telemetry.add_llm(result["meta"])
                        llm_calls += 1
                    filtered_by_schema += int(result.get("dropped", 0))
                    triples = result.get("triples", [])
                    for t in triples:
                        record = {
                            "chunk_id": t.chunk_id,
                            "doc_id": chunk_doc.get(t.chunk_id),
                            **t.to_dict(),
                        }
                        out.write(json.dumps(record, ensure_ascii=False) + "\n")
                        total_triples += 1
                    if progress_every > 0 and processed_chunks % int(progress_every) == 0:
                        _progress(
                            f"processed_chunks={processed_chunks}, eligible_chunks={eligible_chunks}, llm_calls={llm_calls}, total_triples={total_triples}, parse_failures={parse_failures}"
                        )

    metrics = {
        "chunks_file": chunks_file,
        "triples_file": out_file,
        "mode": mode,
        "batch_size": int(batch_size) if mode == "batched" else 1,
        "min_chars": int(min_chars),
        "max_chunks": int(max_chunks) if max_chunks is not None else None,
        "max_text_chars": int(max_text_chars),
        "concurrency": int(concurrency),
        "schema_file": schema_file,
        "schema_enabled": bool(compiled_schema.get("enabled")),
        "total_chunks": total_chunks,
        "eligible_chunks": eligible_chunks,
        "llm_calls": llm_calls,
        "total_triples": total_triples,
        "parse_failures": parse_failures,
        "filtered_by_schema": filtered_by_schema,
        "avg_triples_per_chunk": round(total_triples / max(total_chunks, 1), 4),
        "telemetry": telemetry.to_dict(),
    }
    metrics_path = Path(metrics_file)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    _progress(
        f"done mode={mode}, total_chunks={total_chunks}, eligible_chunks={eligible_chunks}, llm_calls={llm_calls}, total_triples={total_triples}, parse_failures={parse_failures}, filtered_by_schema={filtered_by_schema}"
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-based triple extraction from chunks")
    parser.add_argument("--chunks-file", default="data/processed/chunks_sampled.jsonl")
    parser.add_argument("--out-file", default="outputs/graph/triples.jsonl")
    parser.add_argument("--metrics-file", default="outputs/results/triple_extract_metrics.json")
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--mode", choices=["per_chunk", "batched"], default="per_chunk")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--min-chars", type=int, default=60)
    parser.add_argument("--max-chunks", type=int, default=None)
    parser.add_argument("--max-text-chars", type=int, default=5000)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--schema-file", default="config_triple_schema.json")
    args = parser.parse_args()
    result = extract_triples(
        args.chunks_file,
        args.out_file,
        args.metrics_file,
        progress_every=args.progress_every,
        mode=args.mode,
        batch_size=args.batch_size,
        min_chars=args.min_chars,
        max_chunks=args.max_chunks,
        max_text_chars=args.max_text_chars,
        concurrency=args.concurrency,
        schema_file=args.schema_file,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
