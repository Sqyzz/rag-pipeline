from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.config import cfg

_READY_DATASETS: dict[str, str] = {}
_CHUNK_INDEX_CACHE: dict[str, tuple[str, dict[str, dict[str, str | None]], dict[str, str]]] = {}
_EDGE_MAP_CACHE: dict[str, tuple[str, dict[tuple[str, str, str], dict[str, Any]]]] = {}
_SYNC_STATE_CACHE: dict[str, dict[str, str]] | None = None


@dataclass
class YoutuSettings:
    base_url: str
    dataset: str
    timeout_sec: int
    search_timeout_sec: int
    search_max_retries: int
    search_retry_backoff_sec: int
    construct_poll_sec: int
    construct_timeout_sec: int
    require_id_alignment: bool
    chunks_file: str
    graph_dir: str
    sync_state_file: str


def _cfg_get(path: str, default: Any) -> Any:
    cur: Any = cfg
    for part in path.split("."):
        cur = getattr(cur, part, None)
        if cur is None:
            return default
    return cur


def _to_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _load_settings() -> YoutuSettings:
    env_chunks_file = os.getenv("GRAPH_RAG_CHUNKS_FILE", "").strip()
    chunks_file = env_chunks_file or str(_cfg_get("graph.youtu.chunks_file", "data/processed/chunks_sampled.jsonl"))
    return YoutuSettings(
        base_url=str(os.getenv("YOUTU_BASE_URL") or _cfg_get("graph.youtu.base_url", "http://127.0.0.1:8000")).rstrip(
            "/"
        ),
        dataset=str(os.getenv("YOUTU_DATASET") or _cfg_get("graph.youtu.dataset", "enterprise")).strip(),
        timeout_sec=int(os.getenv("YOUTU_TIMEOUT_SEC") or _cfg_get("graph.youtu.timeout_sec", 120)),
        search_timeout_sec=int(
            os.getenv("YOUTU_SEARCH_TIMEOUT_SEC") or _cfg_get("graph.youtu.search_timeout_sec", 900)
        ),
        search_max_retries=int(
            os.getenv("YOUTU_SEARCH_MAX_RETRIES") or _cfg_get("graph.youtu.search_max_retries", 2)
        ),
        search_retry_backoff_sec=int(
            os.getenv("YOUTU_SEARCH_RETRY_BACKOFF_SEC") or _cfg_get("graph.youtu.search_retry_backoff_sec", 3)
        ),
        construct_poll_sec=int(
            os.getenv("YOUTU_CONSTRUCT_POLL_SEC") or _cfg_get("graph.youtu.construct_poll_sec", 2)
        ),
        construct_timeout_sec=int(
            os.getenv("YOUTU_CONSTRUCT_TIMEOUT_SEC") or _cfg_get("graph.youtu.construct_timeout_sec", 1800)
        ),
        require_id_alignment=_to_bool(
            os.getenv("YOUTU_REQUIRE_ID_ALIGNMENT"),
            _to_bool(_cfg_get("graph.youtu.require_id_alignment", True), True),
        ),
        chunks_file=chunks_file,
        graph_dir=str(os.getenv("YOUTU_GRAPH_DIR") or _cfg_get("graph.youtu.graph_dir", "youtu-graphrag/output/graphs")),
        sync_state_file=str(
            os.getenv("YOUTU_SYNC_STATE_FILE") or _cfg_get("graph.youtu.sync_state_file", "outputs/cache/youtu_sync_state.json")
        ),
    )


def _normalize_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def _normalize_term(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def _triple_key(s: Any, r: Any, o: Any) -> tuple[str, str, str]:
    return (_normalize_term(s), _normalize_term(r), _normalize_term(o))


def _request_json(method: str, url: str, timeout_sec: int | tuple[int, int], **kwargs) -> dict:
    try:
        resp = requests.request(method=method, url=url, timeout=timeout_sec, **kwargs)
    except requests.Timeout as exc:
        raise TimeoutError(f"youtu request timeout: {method} {url} (timeout={timeout_sec})") from exc
    except requests.RequestException as exc:
        raise RuntimeError(f"youtu request failed: {method} {url}: {exc}") from exc
    if resp.status_code >= 400:
        snippet = (resp.text or "").strip().replace("\n", " ")[:400]
        raise RuntimeError(f"youtu http {resp.status_code} for {method} {url}: {snippet}")
    try:
        return resp.json()
    except ValueError as exc:
        snippet = (resp.text or "").strip().replace("\n", " ")[:400]
        raise RuntimeError(f"youtu response is not valid json for {method} {url}: {snippet}") from exc


def _file_fingerprint(path: Path) -> str:
    if not path.exists():
        return f"missing:{path}"
    stat = path.stat()
    return f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}"


def _load_sync_state(path: str) -> dict[str, dict[str, str]]:
    global _SYNC_STATE_CACHE
    if _SYNC_STATE_CACHE is not None:
        return _SYNC_STATE_CACHE

    p = Path(path)
    if not p.exists():
        _SYNC_STATE_CACHE = {}
        return _SYNC_STATE_CACHE
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        _SYNC_STATE_CACHE = {}
        return _SYNC_STATE_CACHE
    if not isinstance(payload, dict):
        _SYNC_STATE_CACHE = {}
        return _SYNC_STATE_CACHE
    out: dict[str, dict[str, str]] = {}
    for k, v in payload.items():
        if not isinstance(k, str):
            continue
        if isinstance(v, str):
            # Backward compatibility for legacy format: {key: "fingerprint"}.
            out[k] = {"fingerprint": v}
            continue
        if isinstance(v, dict):
            fp = v.get("fingerprint")
            if not isinstance(fp, str):
                continue
            row: dict[str, str] = {"fingerprint": fp}
            task_id = v.get("task_id")
            if isinstance(task_id, str) and task_id.strip():
                row["task_id"] = task_id.strip()
            out[k] = row
    _SYNC_STATE_CACHE = out
    return _SYNC_STATE_CACHE


def _save_sync_state(path: str, state: dict[str, dict[str, str]]) -> None:
    global _SYNC_STATE_CACHE
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    _SYNC_STATE_CACHE = dict(state)


def _dataset_ready(settings: YoutuSettings) -> bool:
    try:
        payload = _request_json(
            "GET",
            f"{settings.base_url}/api/datasets",
            timeout_sec=settings.timeout_sec,
        )
    except Exception:
        return False
    rows = payload.get("datasets")
    if not isinstance(rows, list):
        return False
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name", "")).strip()
        status = str(row.get("status", "")).strip().lower()
        if name == settings.dataset and status == "ready":
            return True
    return False


def _search_with_retry(settings: YoutuSettings, payload: dict[str, Any]) -> dict:
    attempts = max(1, int(settings.search_max_retries) + 1)
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return _request_json(
                "POST",
                f"{settings.base_url}/api/v1/datasets/{settings.dataset}/search",
                # Keep connect timeout short, allow long read timeout for first-query warm-up.
                timeout_sec=(10, max(30, int(settings.search_timeout_sec))),
                json=payload,
            )
        except TimeoutError as exc:
            last_exc = exc
            if attempt >= attempts:
                break
            time.sleep(max(1, int(settings.search_retry_backoff_sec)) * attempt)
    raise RuntimeError(
        f"youtu search timed out after {attempts} attempts "
        f"(read_timeout={settings.search_timeout_sec}s, dataset={settings.dataset})"
    ) from last_exc


def _load_rows_from_chunks_file(chunks_file: str) -> list[dict]:
    p = Path(chunks_file)
    if not p.exists():
        raise FileNotFoundError(f"chunks file not found: {chunks_file}")

    if p.suffix.lower() == ".jsonl":
        rows = []
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
    else:
        payload = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"chunks file must be jsonl or json list: {chunks_file}")
        rows = payload

    seen: set[str] = set()
    out: list[dict] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        chunk_id = str(row.get("chunk_id", "")).strip()
        text = str(row.get("text", "") or "")
        if not chunk_id or not text.strip():
            continue
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        out.append(
            {
                "chunk_id": chunk_id,
                "doc_id": str(row.get("doc_id", "") or "").strip() or None,
                "text": text,
            }
        )
    if not out:
        raise ValueError(f"no valid rows loaded from chunks file: {chunks_file}")
    return out


def _load_chunk_index(chunks_file: str) -> tuple[dict[str, dict[str, str | None]], dict[str, str]]:
    fp = _file_fingerprint(Path(chunks_file))
    cached = _CHUNK_INDEX_CACHE.get(chunks_file)
    if cached and cached[0] == fp:
        return cached[1], cached[2]

    rows = _load_rows_from_chunks_file(chunks_file)
    chunk_index: dict[str, dict[str, str | None]] = {}
    text_to_chunk: dict[str, str] = {}
    for row in rows:
        cid = row["chunk_id"]
        chunk_index[cid] = {"doc_id": row.get("doc_id"), "text": row.get("text")}
        norm = _normalize_text(row.get("text"))
        if norm and norm not in text_to_chunk:
            text_to_chunk[norm] = cid

    _CHUNK_INDEX_CACHE[chunks_file] = (fp, chunk_index, text_to_chunk)
    return chunk_index, text_to_chunk


def _sync_dataset(settings: YoutuSettings, chunk_rows: list[dict]) -> None:
    corpus_rows = [
        {
            "title": row.get("doc_id") or row["chunk_id"],
            "text": row["text"],
            "chunk_id": row["chunk_id"],
            "doc_id": row.get("doc_id"),
        }
        for row in chunk_rows
    ]

    tmp_path = None
    try:
        try:
            _request_json(
                "DELETE",
                f"{settings.base_url}/api/datasets/{settings.dataset}",
                timeout_sec=settings.timeout_sec,
            )
        except Exception:
            pass

        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".json", delete=False) as tf:
            tmp_path = tf.name
            json.dump(corpus_rows, tf, ensure_ascii=False)

        with open(tmp_path, "rb") as fh:
            payload = _request_json(
                "POST",
                f"{settings.base_url}/api/upload",
                timeout_sec=settings.timeout_sec,
                data={"dataset_name": settings.dataset},
                files=[("files", ("chunks.json", fh, "application/json"))],
            )

        if not bool(payload.get("success", False)):
            raise RuntimeError(f"youtu upload failed: {payload}")
        actual_dataset = str(payload.get("dataset_name", "") or "").strip()
        if actual_dataset and actual_dataset != settings.dataset:
            raise RuntimeError(
                f"uploaded dataset name mismatch: expected={settings.dataset}, got={actual_dataset}. "
                "Please ensure target dataset is deletable and name is valid."
            )
    finally:
        if tmp_path and Path(tmp_path).exists():
            try:
                Path(tmp_path).unlink()
            except OSError:
                pass


def _task_status(settings: YoutuSettings, task_id: str) -> tuple[str, dict[str, Any]]:
    url = f"{settings.base_url}/api/construct-graph/{task_id}"
    try:
        resp = requests.get(url, timeout=settings.timeout_sec)
    except requests.RequestException as exc:
        raise RuntimeError(f"youtu request failed: GET {url}: {exc}") from exc
    if resp.status_code == 404:
        return "not_found", {}
    if resp.status_code >= 400:
        snippet = (resp.text or "").strip().replace("\n", " ")[:400]
        raise RuntimeError(f"youtu http {resp.status_code} for GET {url}: {snippet}")
    try:
        payload = resp.json()
    except ValueError as exc:
        snippet = (resp.text or "").strip().replace("\n", " ")[:400]
        raise RuntimeError(f"youtu response is not valid json for GET {url}: {snippet}") from exc
    status = str(payload.get("status", "")).strip().lower()
    return status, payload


def _wait_construct_task(settings: YoutuSettings, task_id: str) -> None:
    deadline = time.time() + int(settings.construct_timeout_sec) if int(settings.construct_timeout_sec) > 0 else None
    while True:
        status, status_payload = _task_status(settings, task_id)
        if status == "succeeded":
            return
        if status == "not_found":
            raise RuntimeError(f"construct-graph task not found: task_id={task_id}")
        if status == "failed":
            raise RuntimeError(
                f"construct-graph failed: {status_payload.get('error') or status_payload.get('message') or status_payload}"
            )
        if deadline is not None and time.time() >= deadline:
            # Soft timeout: keep polling instead of hard-failing.
            # Youtu graph construction can be long on large corpora.
            deadline = time.time() + int(settings.construct_timeout_sec)
        time.sleep(max(1, int(settings.construct_poll_sec)))


def _construct_dataset(settings: YoutuSettings, existing_task_id: str | None = None) -> str:
    task_id = str(existing_task_id or "").strip()
    if not task_id:
        accepted = _request_json(
            "POST",
            f"{settings.base_url}/api/construct-graph",
            timeout_sec=settings.timeout_sec,
            json={"dataset_name": settings.dataset, "force_rebuild": True},
        )
        task_id = str(accepted.get("task_id", "")).strip()
        if not task_id:
            raise RuntimeError(f"construct-graph missing task_id: {accepted}")
    return task_id


def _ensure_dataset_ready(settings: YoutuSettings) -> None:
    file_fp = _file_fingerprint(Path(settings.chunks_file))
    chunks_path = str(Path(settings.chunks_file).resolve())
    key = f"{settings.base_url}|{settings.dataset}|{chunks_path}"
    legacy_key = f"{settings.base_url}|{settings.dataset}|{settings.chunks_file}"
    state = _load_sync_state(settings.sync_state_file)
    if key not in state and legacy_key in state:
        state[key] = state[legacy_key]
        _save_sync_state(settings.sync_state_file, state)
    entry = state.get(key) or {}
    ready = _dataset_ready(settings)

    if _READY_DATASETS.get(key) == file_fp and ready:
        return
    if entry.get("fingerprint") == file_fp and ready:
        _READY_DATASETS[key] = file_fp
        return
    if entry.get("fingerprint") == file_fp and not ready and entry.get("task_id"):
        task_id = str(entry.get("task_id", "")).strip()
        if task_id:
            try:
                _wait_construct_task(settings, task_id)
                if _dataset_ready(settings):
                    _READY_DATASETS[key] = file_fp
                    return
            except Exception:
                # Task may have been evicted or failed; fall back to fresh sync + construct.
                pass
    # Bootstrap local state when remote dataset is already ready.
    # This prevents unnecessary rebuilds after introducing sync_state,
    # while still forcing rebuild when local fingerprint later changes.
    if ready and key not in state:
        state[key] = {"fingerprint": file_fp}
        _READY_DATASETS[key] = file_fp
        _save_sync_state(settings.sync_state_file, state)
        return

    chunk_rows = _load_rows_from_chunks_file(settings.chunks_file)
    _sync_dataset(settings, chunk_rows)
    task_id = _construct_dataset(settings)
    state[key] = {"fingerprint": file_fp, "task_id": task_id}
    _save_sync_state(settings.sync_state_file, state)
    _wait_construct_task(settings, task_id)
    if not _dataset_ready(settings):
        raise RuntimeError(f"construct-graph finished but dataset still not ready: {settings.dataset}")
    state[key] = {"fingerprint": file_fp, "task_id": task_id}
    _save_sync_state(settings.sync_state_file, state)
    _READY_DATASETS[key] = file_fp


def _load_edge_map(graph_file: str) -> dict[tuple[str, str, str], dict[str, Any]]:
    fp = _file_fingerprint(Path(graph_file))
    cached = _EDGE_MAP_CACHE.get(graph_file)
    if cached and cached[0] == fp:
        return cached[1]

    graph = json.loads(Path(graph_file).read_text(encoding="utf-8"))
    out: dict[tuple[str, str, str], dict[str, Any]] = {}
    for edge in graph.get("edges", []) if isinstance(graph, dict) else []:
        if not isinstance(edge, dict):
            continue
        source = edge.get("source")
        relation = edge.get("relation")
        target = edge.get("target")
        edge_id = str(edge.get("edge_id", "")).strip()
        if not (source and relation and target and edge_id):
            continue
        key = _triple_key(source, relation, target)
        if key not in out:
            out[key] = {
                "edge_id": edge_id,
                "source": source,
                "relation": relation,
                "target": target,
                "weight": edge.get("weight"),
            }

    _EDGE_MAP_CACHE[graph_file] = (fp, out)
    return out


def _load_current_community_chunk_sets(graph_file: str, communities_file: str) -> list[dict]:
    if not Path(graph_file).exists() or not Path(communities_file).exists():
        return []
    graph = json.loads(Path(graph_file).read_text(encoding="utf-8"))
    community_payload = json.loads(Path(communities_file).read_text(encoding="utf-8"))
    communities = community_payload if isinstance(community_payload, list) else community_payload.get("communities", [])
    edge_by_id = {str(e.get("edge_id")): e for e in graph.get("edges", []) if isinstance(e, dict) and e.get("edge_id")}

    out = []
    for c in communities:
        if not isinstance(c, dict):
            continue
        cid = str(c.get("community_id", "")).strip()
        if not cid:
            continue
        chunk_ids: set[str] = set()
        for eid in c.get("edges", []) or []:
            edge = edge_by_id.get(str(eid))
            if not edge:
                continue
            for m in edge.get("mentions", []) or []:
                if not isinstance(m, dict):
                    continue
                chunk_id = str(m.get("chunk_id", "")).strip()
                if chunk_id:
                    chunk_ids.add(chunk_id)
        out.append(
            {
                "community_id": cid,
                "chunk_ids": chunk_ids,
                "summary": str(c.get("summary", "") or "").strip(),
            }
        )
    return out


def _load_youtu_communities(settings: YoutuSettings) -> list[dict]:
    graph_path = Path(settings.graph_dir) / f"{settings.dataset}_new.json"
    if not graph_path.exists():
        return []
    payload = json.loads(graph_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return []

    communities: dict[str, dict[str, Any]] = {}

    def _ensure(name: str, description: str) -> dict[str, Any]:
        row = communities.get(name)
        if row is None:
            row = {"name": name, "description": description, "chunk_ids": set(), "members": set()}
            communities[name] = row
        elif description and not row.get("description"):
            row["description"] = description
        return row

    def _node_info(node: Any) -> tuple[str, dict]:
        if not isinstance(node, dict):
            return "", {}
        label = str(node.get("label", "")).strip().lower()
        props = node.get("properties")
        if not isinstance(props, dict):
            props = {}
        return label, props

    for rel in payload:
        if not isinstance(rel, dict):
            continue
        s_label, s_props = _node_info(rel.get("start_node"))
        e_label, e_props = _node_info(rel.get("end_node"))

        if s_label == "community":
            name = str(s_props.get("name", "")).strip()
            if name:
                row = _ensure(name, str(s_props.get("description", "")).strip())
                for m in s_props.get("members", []) if isinstance(s_props.get("members"), list) else []:
                    m_name = str(m).strip()
                    if m_name:
                        row["members"].add(m_name)
                chunk_id = str(e_props.get("chunk id") or e_props.get("chunk_id") or "").strip()
                if chunk_id:
                    row["chunk_ids"].add(chunk_id)

        if e_label == "community":
            name = str(e_props.get("name", "")).strip()
            if name:
                row = _ensure(name, str(e_props.get("description", "")).strip())
                for m in e_props.get("members", []) if isinstance(e_props.get("members"), list) else []:
                    m_name = str(m).strip()
                    if m_name:
                        row["members"].add(m_name)
                chunk_id = str(s_props.get("chunk id") or s_props.get("chunk_id") or "").strip()
                if chunk_id:
                    row["chunk_ids"].add(chunk_id)

    return [
        {
            "name": name,
            "description": row.get("description", ""),
            "chunk_ids": set(row.get("chunk_ids", set())),
            "members": set(row.get("members", set())),
        }
        for name, row in communities.items()
    ]


def _map_communities(
    retrieved_chunk_ids: list[str],
    graph_file: str,
    communities_file: str,
    settings: YoutuSettings,
    top_communities: int,
) -> tuple[list[str], list[str], int]:
    top_k = max(1, int(top_communities))
    youtu_communities = _load_youtu_communities(settings)
    current_communities = _load_current_community_chunk_sets(graph_file, communities_file)
    if not youtu_communities or not current_communities:
        return [], [], 0

    query_chunks = set(str(x).strip() for x in retrieved_chunk_ids if str(x).strip())
    scored = []
    for row in youtu_communities:
        overlap = len(row["chunk_ids"] & query_chunks) if query_chunks else 0
        scored.append((overlap, len(row["chunk_ids"]), row))
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)

    selected = [row for overlap, _, row in scored if overlap > 0][:top_k]
    if not selected:
        selected = [row for _, _, row in scored[:top_k]]

    mapped_ids: list[str] = []
    mapped_summaries: list[str] = []
    unmapped_count = 0

    for yc in selected:
        best_id = ""
        best_score = 0.0
        for cc in current_communities:
            intersection = len(yc["chunk_ids"] & cc["chunk_ids"])
            if intersection <= 0:
                continue
            union = len(yc["chunk_ids"] | cc["chunk_ids"]) or 1
            score = intersection / union
            if score > best_score:
                best_score = score
                best_id = cc["community_id"]

        if best_id:
            if best_id not in mapped_ids:
                mapped_ids.append(best_id)
                mapped_summaries.append(str(yc.get("description", "") or "").strip())
        else:
            unmapped_count += 1

    return mapped_ids, mapped_summaries, unmapped_count


def _map_edges(
    triples_struct: list[dict],
    graph_file: str,
    max_evidence: int,
) -> tuple[list[dict], int]:
    edge_map = _load_edge_map(graph_file)
    out: list[dict] = []
    unmapped = 0
    for row in triples_struct:
        if len(out) >= max(1, int(max_evidence)):
            break
        if not isinstance(row, dict):
            continue
        s = str(row.get("subject", "")).strip()
        r = str(row.get("relation", "")).strip()
        o = str(row.get("object", "")).strip()
        if not (s and r and o):
            continue
        score = float(row.get("score", 0.0) or 0.0)
        mapped = edge_map.get(_triple_key(s, r, o))
        if mapped:
            out.append(
                {
                    "edge_id": mapped["edge_id"],
                    "source": mapped["source"],
                    "relation": mapped["relation"],
                    "target": mapped["target"],
                    "weight": mapped.get("weight"),
                    "score": score,
                }
            )
        else:
            unmapped += 1
            out.append(
                {
                    "edge_id": "",
                    "source": s,
                    "relation": r,
                    "target": o,
                    "score": score,
                    "unmapped": True,
                }
            )
    return out, unmapped


def _build_evidence_chunks(
    retrieved_chunk_ids: list[Any],
    retrieved_chunks: list[Any],
    chunk_index: dict[str, dict[str, str | None]],
    text_to_chunk: dict[str, str],
    max_evidence: int,
) -> tuple[list[dict], int, bool]:
    ids = [str(x).strip() for x in (retrieved_chunk_ids or []) if str(x).strip()]
    texts = [str(x) for x in (retrieved_chunks or [])]
    length_mismatch = False

    if ids and len(ids) != len(texts):
        length_mismatch = True
        n = min(len(ids), len(texts))
        ids = ids[:n]
        texts = texts[:n]
    elif (not ids) and texts:
        for text in texts:
            ids.append(text_to_chunk.get(_normalize_text(text), ""))

    out: list[dict] = []
    unmapped = 0
    for chunk_id, text in zip(ids, texts):
        if len(out) >= max(1, int(max_evidence)):
            break
        meta = chunk_index.get(chunk_id)
        if not chunk_id or meta is None:
            unmapped += 1
            out.append({"chunk_id": chunk_id, "doc_id": None, "text": text})
            continue
        out.append({"chunk_id": chunk_id, "doc_id": meta.get("doc_id"), "text": text})
    return out, unmapped, length_mismatch


def _build_map_partial_answers(search_data: dict, max_evidence: int) -> list[dict]:
    out = []
    for sq in search_data.get("sub_questions", []) or []:
        if len(out) >= max(1, int(max_evidence)):
            break
        if not isinstance(sq, dict):
            continue
        out.append(
            {
                "community_id": None,
                "level": None,
                "summary": "",
                "partial_answer": str(sq.get("sub-question", "")).strip(),
            }
        )
    return out


def answer_with_graphrag(
    query: str,
    graph_file: str,
    communities_file: str,
    top_communities: int = 3,
    max_evidence: int = 12,
    query_level: int = 0,
    use_hierarchy: bool = True,
    use_community_summaries: bool = True,
    shuffle_communities: bool = True,
    use_map_reduce: bool = True,
    max_summary_chars: int = 1800,
    map_keypoints_limit: int = 5,
    max_completion_tokens: int | None = None,
    max_llm_calls: int | None = None,
    max_total_tokens: int | None = None,
    evidence_token_limit: int | None = None,
    strict_budget: bool = False,
    generate_summary_on_demand: bool = True,
    use_embedding_cache: bool = True,
    embedding_cache_dir: str = "outputs/cache",
    embedding_cache_text_chars: int = 4000,
    include_chunk_evidence: bool = True,
) -> dict:
    settings = _load_settings()
    _ensure_dataset_ready(settings)

    chunk_index, text_to_chunk = _load_chunk_index(settings.chunks_file)
    search_request: dict[str, Any] = {"question": query}
    if max_completion_tokens is not None and int(max_completion_tokens) > 0:
        search_request["max_completion_tokens"] = int(max_completion_tokens)
    if max_llm_calls is not None and int(max_llm_calls) > 0:
        search_request["max_llm_calls"] = int(max_llm_calls)
    if max_total_tokens is not None and int(max_total_tokens) > 0:
        search_request["max_total_tokens"] = int(max_total_tokens)
    if strict_budget:
        search_request["top_k_filter"] = max(1, int(max_evidence))
        search_request["evidence_max_triples"] = max(1, int(max_evidence))
        search_request["evidence_max_chunks"] = max(1, int(max_evidence))
        if evidence_token_limit is not None and int(evidence_token_limit) > 0:
            # Approximate token-to-char budget to bound prompt context size.
            search_request["context_max_chars"] = int(evidence_token_limit) * 4
        if not bool(use_map_reduce):
            search_request["disable_iterative_reasoning"] = True
    search_payload = _search_with_retry(settings, search_request)
    if not bool(search_payload.get("success", False)):
        raise RuntimeError(f"youtu search failed: {search_payload}")

    data = search_payload.get("data", {}) or {}
    meta = search_payload.get("meta", {}) or {}

    retrieved_chunks = [str(x) for x in (data.get("retrieved_chunks") or [])]
    retrieved_chunk_ids = [str(x).strip() for x in (data.get("retrieved_chunk_ids") or []) if str(x).strip()]
    retrieved_triples_struct = [
        x for x in (data.get("retrieved_triples_struct") or []) if isinstance(x, dict)
    ]

    evidence_chunks, unmapped_chunk_count, chunk_length_mismatch = _build_evidence_chunks(
        retrieved_chunk_ids=retrieved_chunk_ids,
        retrieved_chunks=retrieved_chunks,
        chunk_index=chunk_index,
        text_to_chunk=text_to_chunk,
        max_evidence=max_evidence,
    )

    if settings.require_id_alignment:
        if retrieved_chunks and not retrieved_chunk_ids:
            raise RuntimeError("youtu returned retrieved_chunks but no retrieved_chunk_ids")
        if chunk_length_mismatch:
            raise RuntimeError("youtu returned chunk ids/chunks with mismatched lengths")
        if unmapped_chunk_count > 0:
            raise RuntimeError(f"chunk_id alignment failed: unmapped_chunk_count={unmapped_chunk_count}")

    subgraph_edges, unmapped_edge_count = _map_edges(
        triples_struct=retrieved_triples_struct,
        graph_file=graph_file,
        max_evidence=max_evidence,
    )
    communities, community_summaries, unmapped_community_count = _map_communities(
        retrieved_chunk_ids=retrieved_chunk_ids,
        graph_file=graph_file,
        communities_file=communities_file,
        settings=settings,
        top_communities=top_communities,
    )
    map_partial_answers = _build_map_partial_answers(data, max_evidence=max_evidence)
    evidence = [
        {"community_id": cid, "level": None, "summary": summary}
        for cid, summary in zip(communities, community_summaries)
    ][: max(1, int(max_evidence))]

    llm_calls = meta.get("llm_calls") or []
    llm_latency_ms = 0
    for call in llm_calls:
        if isinstance(call, dict):
            llm_latency_ms += int(call.get("latency_ms", 0) or 0)
    usage = meta.get("usage") or {}
    telemetry = {
        "llm_calls": len(llm_calls) if isinstance(llm_calls, list) else 0,
        "embedding_calls": 0,
        "llm_latency_ms": llm_latency_ms,
        "embedding_latency_ms": 0,
        "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
        "total_tokens": int(usage.get("total_tokens", 0) or 0),
        "extra": {
            "trace_id": meta.get("trace_id"),
            "dataset_name": meta.get("dataset_name", settings.dataset),
            "timings_ms": meta.get("timings_ms") or {},
            "budget": meta.get("budget") or {},
            "unmapped_chunk_count": int(unmapped_chunk_count),
            "unmapped_edge_count": int(unmapped_edge_count),
            "unmapped_community_count": int(unmapped_community_count),
            "chunk_length_mismatch": bool(chunk_length_mismatch),
            "id_alignment_required": bool(settings.require_id_alignment),
        },
    }

    return {
        "answer": str(data.get("answer", "") or ""),
        "communities": communities,
        "community_summaries": community_summaries,
        "query_level": query_level,
        "use_hierarchy": use_hierarchy,
        "use_community_summaries": use_community_summaries,
        "shuffle_communities": shuffle_communities,
        "use_map_reduce": use_map_reduce,
        "map_keypoints_limit": map_keypoints_limit,
        "generate_summary_on_demand": generate_summary_on_demand,
        "embedding_cache": {"enabled": False, "backend": "youtu_http"},
        "on_demand_summaries_generated": 0,
        "map_partial_answers": map_partial_answers,
        "subgraph_edges": subgraph_edges,
        "evidence": evidence,
        "evidence_chunks": evidence_chunks if include_chunk_evidence else [],
        "telemetry": telemetry,
    }
