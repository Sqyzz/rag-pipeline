from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any


def _stable_dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def compute_graph_fingerprint(
    chunks_file: str,
    dataset: str,
    build_params: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    chunks_sha = _sha256_file(chunks_file)
    material = {
        "dataset": str(dataset),
        "chunks_sha256": chunks_sha,
        "build_params": build_params or {},
    }
    digest = hashlib.sha256(_stable_dumps(material).encode("utf-8")).hexdigest()
    return digest, {
        "fingerprint_method": "full_sha256",
        "chunks_sha256": chunks_sha,
    }


def load_graph_state(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def save_graph_state(path: str, payload: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def decide_graph_reuse(
    graph_state_file: str,
    chunks_file: str,
    dataset: str,
    build_params: dict[str, Any] | None,
    reuse_graph: bool,
    force_rebuild: bool,
    require_local_assets: bool = True,
    graph_file: str | None = None,
    communities_file: str | None = None,
) -> dict[str, Any]:
    fingerprint, fingerprint_meta = compute_graph_fingerprint(
        chunks_file=chunks_file,
        dataset=dataset,
        build_params=build_params,
    )
    state = load_graph_state(graph_state_file)

    if force_rebuild or not reuse_graph:
        return {
            "used_cached_graph": False,
            "reason": "forced_rebuild",
            "fingerprint": fingerprint,
            **fingerprint_meta,
            "state": state,
        }

    matches = state.get("fingerprint") == fingerprint
    local_ok = True
    if require_local_assets:
        if not graph_file or not communities_file:
            local_ok = False
        else:
            local_ok = Path(graph_file).exists() and Path(communities_file).exists()

    if matches and local_ok:
        return {
            "used_cached_graph": True,
            "reason": "fingerprint_match",
            "fingerprint": fingerprint,
            **fingerprint_meta,
            "state": state,
        }

    return {
        "used_cached_graph": False,
        "reason": "fingerprint_changed",
        "fingerprint": fingerprint,
        **fingerprint_meta,
        "state": state,
    }


def build_state_payload(
    dataset: str,
    fingerprint: str,
    build_params: dict[str, Any],
    graph_task_id: str | None,
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "fingerprint": fingerprint,
        "graph_task_id": graph_task_id,
        "built_at": int(time.time()),
        "build_params": build_params,
    }
