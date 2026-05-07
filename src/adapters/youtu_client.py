from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import requests


class YoutuClient:
    def __init__(self, base_url: str, timeout_sec: int = 120) -> None:
        self.base_url = str(base_url).rstrip("/")
        self.timeout_sec = int(timeout_sec)
        self._session = requests.Session()

    def _url(self, path: str) -> str:
        if path.startswith("http://") or path.startswith("https://"):
            return path
        if not path.startswith("/"):
            path = "/" + path
        return self.base_url + path

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        url = self._url(path)
        timeout = kwargs.pop("timeout", self.timeout_sec)
        try:
            resp = self._session.request(method=method.upper(), url=url, timeout=timeout, **kwargs)
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(f"Youtu request failed: {method.upper()} {url}: {exc}") from exc

        ctype = (resp.headers.get("content-type") or "").lower()
        if "application/json" in ctype:
            return resp.json()
        text = resp.text.strip()
        if not text:
            return {}
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"raw": text}

    def _extract(self, payload: Any, keys: list[str]) -> Any:
        if not isinstance(payload, dict):
            return None
        cur: Any = payload
        for key in keys:
            if not isinstance(cur, dict):
                return None
            cur = cur.get(key)
        return cur

    def _first_non_empty(self, payload: Any, candidates: list[list[str]]) -> Any:
        for keys in candidates:
            v = self._extract(payload, keys)
            if v is not None and v != "":
                return v
        return None

    def health_check(self) -> dict[str, Any]:
        paths = ["/health", "/api/health"]
        errors: list[str] = []
        for path in paths:
            try:
                body = self._request("GET", path)
                return {"ok": True, "path": path, "body": body}
            except Exception as exc:  # noqa: BLE001
                errors.append(str(exc))
        return {"ok": False, "errors": errors}

    def list_datasets(self) -> list[dict[str, Any]]:
        body = self._request("GET", "/api/datasets")
        if isinstance(body, dict) and isinstance(body.get("datasets"), list):
            return [x for x in body.get("datasets", []) if isinstance(x, dict)]
        if isinstance(body, list):
            return [x for x in body if isinstance(x, dict)]
        return []

    def construct_graph(self, dataset_name: str, **kwargs: Any) -> str:
        payload = {"dataset_name": dataset_name}
        payload.update(kwargs)
        body = self._request("POST", "/api/construct-graph", json=payload)
        task_id = self._first_non_empty(
            body,
            [
                ["task_id"],
                ["id"],
                ["data", "task_id"],
                ["data", "id"],
            ],
        )
        if not task_id:
            raise RuntimeError(f"construct-graph response missing task id: {body}")
        return str(task_id)

    def get_construct_status(self, task_id: str) -> dict[str, Any]:
        body = self._request("GET", f"/api/construct-graph/{task_id}")
        return body if isinstance(body, dict) else {"data": body}

    def poll_construct(self, task_id: str, timeout_sec: int, poll_sec: int = 2) -> dict[str, Any]:
        deadline = time.time() + int(timeout_sec)
        poll = max(1, int(poll_sec))
        last: dict[str, Any] = {}
        while time.time() < deadline:
            last = self.get_construct_status(task_id)
            status = str(
                self._first_non_empty(
                    last,
                    [
                        ["status"],
                        ["data", "status"],
                        ["data", "state"],
                        ["state"],
                    ],
                )
                or ""
            ).lower()

            if status in {"done", "completed", "success", "succeeded", "finished"}:
                return last
            if status in {"failed", "error", "cancelled", "canceled"}:
                raise RuntimeError(f"construct-graph failed: task_id={task_id}, payload={last}")
            time.sleep(poll)

        raise TimeoutError(f"construct-graph timeout: task_id={task_id}, last={last}")

    def _has_search_answer(self, payload: Any) -> bool:
        if not isinstance(payload, dict):
            return bool(payload)
        answer_keys = {
            "answer",
            "retrieved_triples",
            "retrieved_chunks",
            "retrieved_chunk_ids",
            "retrieved_chunk_items",
            "retrieved_triples_struct",
            "sub_questions",
            "reasoning_steps",
            "visualization_data",
        }
        if any(k in payload for k in answer_keys):
            return True
        data = payload.get("data")
        if isinstance(data, dict) and any(k in data for k in answer_keys):
            return True
        return False

    def _normalize_search_response(
        self,
        body: Any,
        *,
        endpoint: str,
        task_id: str | None = None,
        task_status: str | None = None,
    ) -> dict[str, Any]:
        if not isinstance(body, dict):
            meta: dict[str, Any] = {"endpoint": endpoint}
            if task_id:
                meta["task_id"] = task_id
            if task_status:
                meta["task_status"] = task_status
            return {"data": {"answer": str(body or "")}, "meta": meta}

        task_meta: dict[str, Any] = {}
        for key in ("task_id", "dataset_name", "status", "progress", "message", "error", "created_at", "updated_at"):
            if key in body:
                task_meta[key] = body.get(key)

        nested_data = body.get("data") if isinstance(body.get("data"), dict) else None
        if isinstance(nested_data, dict):
            # Spec: GET /api/ask-question/{task_id} returns a task envelope whose
            # top-level `data` field is the QuestionResponse payload.
            data = dict(nested_data)
            base_meta = dict(nested_data.get("meta") or {}) if isinstance(nested_data.get("meta"), dict) else {}
        else:
            data = {
                "answer": body.get("answer"),
                "retrieved_triples": body.get("retrieved_triples") or [],
                "retrieved_chunks": body.get("retrieved_chunks") or [],
                "retrieved_chunk_ids": body.get("retrieved_chunk_ids") or [],
                "retrieved_chunk_items": body.get("retrieved_chunk_items") or [],
                "retrieved_triples_struct": body.get("retrieved_triples_struct") or [],
                "sub_questions": body.get("sub_questions") or [],
                "reasoning_steps": body.get("reasoning_steps") or [],
                "visualization_data": body.get("visualization_data") or {},
            }
            base_meta = {}

        meta = dict(task_meta)
        for k, v in base_meta.items():
            if k not in meta:
                meta[k] = v
        if isinstance(data.get("meta"), dict):
            for k, v in (data.get("meta") or {}).items():
                if k not in meta:
                    meta[k] = v
        top_meta = body.get("meta")
        if isinstance(top_meta, dict):
            for k, v in top_meta.items():
                if k not in meta:
                    meta[k] = v

        # Compatibility: accept telemetry-like fields either at top-level or under meta.
        if "usage" in body and "usage" not in meta:
            meta["usage"] = body.get("usage")
        if "llm_calls" in body and "llm_calls" not in meta:
            meta["llm_calls"] = body.get("llm_calls")
        if "embedding_calls" in body and "embedding_calls" not in meta:
            meta["embedding_calls"] = body.get("embedding_calls")
        # Dynamic routing metadata may be returned at top-level or data-level by backend.
        for route_key in ("route_type", "route_confidence", "route_reason", "route_fallback"):
            if route_key in body and route_key not in meta:
                meta[route_key] = body.get(route_key)
            if route_key in data and route_key not in meta:
                meta[route_key] = data.get(route_key)
        routing = None
        if isinstance(meta.get("routing"), dict):
            routing = meta.get("routing")
        elif isinstance((data.get("meta") if isinstance(data.get("meta"), dict) else {}).get("routing"), dict):
            routing = (data.get("meta") if isinstance(data.get("meta"), dict) else {}).get("routing")
        if isinstance(routing, dict):
            for route_key in ("route_type", "route_confidence", "route_reason", "route_fallback", "routing_mode", "decomposition_debug"):
                if route_key in routing and route_key not in meta:
                    meta[route_key] = routing.get(route_key)
        # Async task metadata is useful for diagnostics.
        for task_key in ("task_id", "status", "progress", "message", "error"):
            if task_key in body and task_key not in meta:
                meta[task_key] = body.get(task_key)
        meta["endpoint"] = endpoint
        if task_id and "task_id" not in meta:
            meta["task_id"] = task_id
        if task_status and "task_status" not in meta:
            meta["task_status"] = task_status
        return {"data": data, "meta": meta}

    def get_question_status(self, task_id: str) -> dict[str, Any]:
        body = self._request("GET", f"/api/ask-question/{task_id}")
        return body if isinstance(body, dict) else {"data": body}

    def poll_question(self, task_id: str, timeout_sec: int, poll_sec: float = 1.0) -> dict[str, Any]:
        deadline = time.time() + int(timeout_sec)
        poll = max(0.2, float(poll_sec))
        last: dict[str, Any] = {}
        while time.time() < deadline:
            last = self.get_question_status(task_id)
            status = str(
                self._first_non_empty(
                    last,
                    [
                        ["status"],
                        ["data", "status"],
                        ["data", "state"],
                        ["state"],
                    ],
                )
                or ""
            ).lower()
            if status in {"done", "completed", "success", "succeeded", "finished"}:
                return last
            if status in {"failed", "error", "cancelled", "canceled"}:
                raise RuntimeError(f"ask-question failed: task_id={task_id}, payload={last}")
            time.sleep(poll)
        raise TimeoutError(f"ask-question timeout: task_id={task_id}, last={last}")

    def search(self, dataset_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        # Local youtu backend endpoint.
        q = str(payload.get("question", "") or payload.get("query", "") or "").strip()
        ask_payload = {
            "dataset_name": dataset_name,
            "question": q,
        }
        client_id = str(payload.get("client_id", "") or "").strip()
        ask_params: dict[str, Any] = {}
        if client_id:
            ask_params["client_id"] = client_id
        route_type = str(payload.get("route_type", "") or "").strip()
        router_type = str(payload.get("router_type", "") or "").strip()
        if route_type:
            ask_payload["route_type"] = route_type
        elif router_type:
            # Backward compatibility with older backend naming.
            ask_payload["router_type"] = router_type
        try:
            body = self._request("POST", "/api/ask-question", json=ask_payload, params=ask_params)
            if not isinstance(body, dict):
                return self._normalize_search_response(body, endpoint="/api/ask-question")

            post_status = str(
                self._first_non_empty(
                    body,
                    [
                        ["status"],
                        ["data", "status"],
                        ["data", "state"],
                        ["state"],
                    ],
                )
                or ""
            ).lower()
            task_id = self._first_non_empty(
                body,
                [
                    ["task_id"],
                    ["id"],
                    ["data", "task_id"],
                    ["data", "id"],
                ],
            )
            task_id_str = str(task_id).strip() if task_id is not None else ""

            # Async API: POST returns a task id; poll status endpoint until completed.
            if task_id_str and not self._has_search_answer(body):
                final = self.poll_question(task_id=task_id_str, timeout_sec=self.timeout_sec, poll_sec=1.0)
                final_status = str(
                    self._first_non_empty(
                        final,
                        [
                            ["status"],
                            ["data", "status"],
                            ["data", "state"],
                            ["state"],
                        ],
                    )
                    or ""
                ).lower()
                return self._normalize_search_response(
                    final,
                    endpoint=f"/api/ask-question/{task_id_str}",
                    task_id=task_id_str,
                    task_status=final_status,
                )

            if post_status in {"failed", "error", "cancelled", "canceled"}:
                raise RuntimeError(f"ask-question failed: payload={body}")

            return self._normalize_search_response(
                body,
                endpoint="/api/ask-question",
                task_id=task_id_str or None,
                task_status=post_status or None,
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Youtu search failed on /api/ask-question: {exc}") from exc

    def _download_file(self, url: str, dst_path: str) -> None:
        p = Path(dst_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        resp = self._session.get(self._url(url), timeout=self.timeout_sec)
        resp.raise_for_status()
        p.write_bytes(resp.content)

    def _discover_artifact_urls(self, dataset_name: str, task_id: str | None = None) -> dict[str, str]:
        candidates: list[str] = []
        if task_id:
            candidates.append(f"/api/construct-graph/{task_id}")
            candidates.append(f"/api/construct-graph/{task_id}/artifacts")
        candidates.append(f"/api/v1/datasets/{dataset_name}/artifacts")

        for path in candidates:
            try:
                body = self._request("GET", path)
            except Exception:  # noqa: BLE001
                continue
            graph_url = self._first_non_empty(
                body,
                [
                    ["graph_url"],
                    ["data", "graph_url"],
                    ["artifacts", "graph_url"],
                    ["data", "artifacts", "graph_url"],
                ],
            )
            communities_url = self._first_non_empty(
                body,
                [
                    ["communities_url"],
                    ["data", "communities_url"],
                    ["artifacts", "communities_url"],
                    ["data", "artifacts", "communities_url"],
                ],
            )
            if graph_url and communities_url:
                return {
                    "graph_url": str(graph_url),
                    "communities_url": str(communities_url),
                    "source": path,
                }

        return {
            "graph_url": f"/api/v1/datasets/{dataset_name}/graph.json",
            "communities_url": f"/api/v1/datasets/{dataset_name}/communities.json",
            "source": "default_dataset_artifacts",
        }

    def export_graph_artifacts(
        self,
        dataset_name: str,
        graph_file: str,
        communities_file: str,
        task_id: str | None = None,
    ) -> dict[str, Any]:
        urls = self._discover_artifact_urls(dataset_name=dataset_name, task_id=task_id)
        self._download_file(urls["graph_url"], graph_file)
        self._download_file(urls["communities_url"], communities_file)
        return {
            "downloaded": True,
            "graph_url": urls["graph_url"],
            "communities_url": urls["communities_url"],
            "source": urls.get("source"),
        }
