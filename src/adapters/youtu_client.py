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

    def search(self, dataset_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = self._request("POST", f"/api/v1/datasets/{dataset_name}/search", json=payload)
        if not isinstance(body, dict):
            return {"data": body, "meta": {}}
        data = body.get("data") if "data" in body else body
        meta = body.get("meta") if isinstance(body.get("meta"), dict) else {}
        return {"data": data or {}, "meta": meta}

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
