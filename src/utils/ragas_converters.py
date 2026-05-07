from __future__ import annotations

import json
import os
import re
import sys
import asyncio
from collections import defaultdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from utils.config import cfg
from utils.embedder import embed_texts
from utils.llm_wrapper import llm_chat

_HOP_PREFIX_RE = re.compile(r"^<\d+-hop>\s*")

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(dotenv_path=_PROJECT_ROOT / ".env")


def ensure_vendored_ragas_on_path() -> Path:
    ragas_src = _PROJECT_ROOT / "ragas" / "src"
    value = str(ragas_src)
    if value not in sys.path:
        sys.path.insert(0, value)
    return ragas_src


def load_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str, payload: dict[str, Any]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _resolve_cfg_api_key(env_name: str | None, fallback: str | None = "DASHSCOPE_API_KEY") -> str | None:
    for name in (env_name, fallback):
        if not name:
            continue
        value = str(os.getenv(name, "")).strip()
        if value:
            return value
    return None


def _ragas_llm_cfg() -> dict[str, Any]:
    ragas_cfg = getattr(cfg, "ragas", None)
    llm_cfg = getattr(ragas_cfg, "llm", None)
    if llm_cfg is None:
        return {}

    backend = str(getattr(llm_cfg, "backend", getattr(cfg.llm, "backend", "api"))).strip().lower()
    if backend == "local":
        local_cfg = getattr(llm_cfg, "local", None)
        global_local = getattr(cfg.llm, "local", None)
        return {
            "backend": "local",
            "base_url": str(getattr(local_cfg, "base_url", getattr(global_local, "base_url", ""))).strip(),
            "model": str(getattr(local_cfg, "model", getattr(global_local, "model", ""))).strip(),
            "api_key": str(getattr(local_cfg, "api_key", getattr(global_local, "api_key", ""))).strip(),
        }

    api_cfg = getattr(llm_cfg, "api", None)
    global_api = getattr(cfg.llm, "api", None)
    return {
        "backend": "api",
        "base_url": str(getattr(api_cfg, "base_url", getattr(global_api, "base_url", ""))).strip(),
        "model": str(getattr(api_cfg, "model", getattr(global_api, "model", ""))).strip(),
        "api_key_env": str(getattr(api_cfg, "api_key_env", getattr(global_api, "api_key_env", ""))).strip(),
    }


def build_langchain_clients(temperature: float = 0.0) -> tuple[Any, Any]:
    ensure_vendored_ragas_on_path()
    try:
        from langchain_core.outputs import Generation, LLMResult
        from ragas.embeddings.base import BaseRagasEmbeddings
        from ragas.llms.base import BaseRagasLLM
        from ragas.run_config import RunConfig
    except ImportError as exc:  # pragma: no cover - exercised in runtime only
        raise ImportError(
            "Missing ragas runtime deps. Install `langchain-core`, `datasets`, and vendored ragas dependencies."
        ) from exc

    ragas_llm_cfg = _ragas_llm_cfg()
    llm_backend = str(ragas_llm_cfg.get("backend") or getattr(cfg.llm, "backend", "api")).strip().lower()
    if llm_backend != "local":
        llm_api_key_env = str(ragas_llm_cfg.get("api_key_env") or getattr(cfg.llm.api, "api_key_env", "")).strip() or None
        llm_api_key = _resolve_cfg_api_key(llm_api_key_env)
        if not llm_api_key:
            raise RuntimeError(
                f"Missing LLM API key. Set {llm_api_key_env or cfg.llm.api.api_key_env} or DASHSCOPE_API_KEY."
            )

    emb_backend = str(getattr(cfg.embedding, "backend", "api")).strip().lower()
    if emb_backend != "local":
        emb_api_key = _resolve_cfg_api_key(getattr(cfg.embedding.api, "api_key_env", None))
        if not emb_api_key:
            raise RuntimeError(
                f"Missing embedding API key. Set {cfg.embedding.api.api_key_env} or DASHSCOPE_API_KEY."
            )

    class RepoRagasLLM(BaseRagasLLM):
        def __init__(self) -> None:
            super().__init__(cache=None)
            self.run_config = RunConfig()
            self._temperature = float(temperature)

        def _prompt_to_text(self, prompt: Any) -> str:
            if hasattr(prompt, "to_string"):
                try:
                    return str(prompt.to_string())
                except Exception:
                    pass
            return str(prompt)

        def _generate_once(self, prompt_text: str, temperature_value: float | None) -> str:
            model = str(ragas_llm_cfg.get("model") or "").strip() or None
            backend = str(ragas_llm_cfg.get("backend") or "").strip() or None
            base_url = str(ragas_llm_cfg.get("base_url") or "").strip() or None
            api_key = str(ragas_llm_cfg.get("api_key") or "").strip() or None
            api_key_env = str(ragas_llm_cfg.get("api_key_env") or "").strip() or None
            return str(
                llm_chat(
                    [{"role": "user", "content": prompt_text}],
                    temperature=float(self._temperature if temperature_value is None else temperature_value),
                    model=model,
                    backend=backend,
                    base_url=base_url,
                    api_key=api_key,
                    api_key_env=api_key_env,
                )
                or ""
            ).strip()

        def generate_text(
            self,
            prompt: Any,
            n: int = 1,
            temperature: float | None = 0.01,
            stop: list[str] | None = None,
            callbacks: Any = None,
        ) -> Any:
            del stop, callbacks
            prompt_text = self._prompt_to_text(prompt)
            generations = [
                Generation(text=self._generate_once(prompt_text, temperature), generation_info={"finish_reason": "stop"})
                for _ in range(max(1, int(n)))
            ]
            return LLMResult(generations=[generations])

        def generate_prompt(
            self,
            prompts: list[Any],
            n: int = 1,
            stop: list[str] | None = None,
            callbacks: Any = None,
        ) -> Any:
            del stop, callbacks
            all_generations = []
            for prompt in prompts:
                prompt_text = self._prompt_to_text(prompt)
                generations = [
                    Generation(
                        text=self._generate_once(prompt_text, None),
                        generation_info={"finish_reason": "stop"},
                    )
                    for _ in range(max(1, int(n)))
                ]
                all_generations.append(generations)
            return LLMResult(generations=all_generations)

        async def agenerate_text(
            self,
            prompt: Any,
            n: int = 1,
            temperature: float | None = 0.01,
            stop: list[str] | None = None,
            callbacks: Any = None,
        ) -> Any:
            del stop, callbacks
            prompt_text = self._prompt_to_text(prompt)
            tasks = [
                asyncio.to_thread(self._generate_once, prompt_text, temperature)
                for _ in range(max(1, int(n)))
            ]
            outputs = await asyncio.gather(*tasks)
            generations = [
                Generation(text=str(text or "").strip(), generation_info={"finish_reason": "stop"})
                for text in outputs
            ]
            return LLMResult(generations=[generations])

        async def agenerate_prompt(
            self,
            prompts: list[Any],
            stop: list[str] | None = None,
            callbacks: Any = None,
        ) -> Any:
            del stop, callbacks
            all_generations = []
            for prompt in prompts:
                prompt_text = self._prompt_to_text(prompt)
                text = await asyncio.to_thread(self._generate_once, prompt_text, None)
                all_generations.append(
                    [Generation(text=str(text or "").strip(), generation_info={"finish_reason": "stop"})]
                )
            return LLMResult(generations=all_generations)

        def is_finished(self, response: Any) -> bool:
            del response
            return True

    class RepoRagasEmbeddings(BaseRagasEmbeddings):
        def __init__(self) -> None:
            super().__init__(cache=None)
            self.set_run_config(RunConfig())

        def embed_query(self, text: str) -> list[float]:
            arr = embed_texts([str(text or "")], return_meta=False)
            return [float(x) for x in arr[0].tolist()]

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            arr = embed_texts([str(text or "") for text in texts], return_meta=False)
            return [[float(v) for v in row.tolist()] for row in arr]

        async def aembed_query(self, text: str) -> list[float]:
            return await asyncio.to_thread(self.embed_query, text)

        async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
            return await asyncio.to_thread(self.embed_documents, texts)

    return RepoRagasLLM(), RepoRagasEmbeddings()


def _norm_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _norm_key(value: Any) -> str:
    return _norm_text(value).lower()


def strip_hop_prefix(text: Any) -> str:
    return _HOP_PREFIX_RE.sub("", str(text or "").strip(), count=1)


def build_chunk_lookup(chunks: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    lookup: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in chunks:
        text = _norm_key(row.get("text"))
        if text:
            lookup[text].append(row)
    return dict(lookup)


def match_reference_contexts(
    reference_contexts: list[str] | None,
    chunk_lookup: dict[str, list[dict[str, Any]]],
) -> tuple[list[str], list[str]]:
    ref_chunk_ids: list[str] = []
    ref_doc_ids: list[str] = []
    seen_chunk_ids: set[str] = set()
    seen_doc_ids: set[str] = set()

    for raw_context in reference_contexts or []:
        key = _norm_key(strip_hop_prefix(raw_context))
        matched_rows = chunk_lookup.get(key, [])
        if not matched_rows:
            continue
        row = matched_rows[0]
        chunk_id = str(row.get("chunk_id", "") or "").strip()
        doc_id = str(row.get("doc_id", "") or "").strip()
        if chunk_id and chunk_id not in seen_chunk_ids:
            seen_chunk_ids.add(chunk_id)
            ref_chunk_ids.append(chunk_id)
        if doc_id and doc_id not in seen_doc_ids:
            seen_doc_ids.add(doc_id)
            ref_doc_ids.append(doc_id)

    return ref_chunk_ids, ref_doc_ids


def _split_compound_chunk_id(value: Any) -> list[str]:
    raw = str(value or "").strip()
    if not raw:
        return []
    if "|" in raw:
        return [part.strip() for part in raw.split("|") if part.strip()]
    return [raw]


def normalize_evidence_chunks(evidence_chunks: list[dict[str, Any]] | None) -> dict[str, list[str]]:
    retrieved_contexts: list[str] = []
    retrieved_context_ids: list[str] = []
    retrieved_doc_ids: list[str] = []
    seen_chunk_ids: set[str] = set()
    seen_doc_ids: set[str] = set()

    for item in evidence_chunks or []:
        text = _norm_text(item.get("text"))
        if text:
            retrieved_contexts.append(text)

        chunk_ids = item.get("chunk_ids")
        if isinstance(chunk_ids, list) and chunk_ids:
            candidates = [str(x or "").strip() for x in chunk_ids if str(x or "").strip()]
        else:
            candidates = _split_compound_chunk_id(item.get("chunk_id"))
        for chunk_id in candidates:
            if chunk_id and chunk_id not in seen_chunk_ids:
                seen_chunk_ids.add(chunk_id)
                retrieved_context_ids.append(chunk_id)

        doc_id = str(item.get("doc_id", "") or "").strip()
        if doc_id and doc_id not in seen_doc_ids:
            seen_doc_ids.add(doc_id)
            retrieved_doc_ids.append(doc_id)

    return {
        "retrieved_contexts": retrieved_contexts,
        "retrieved_context_ids": retrieved_context_ids,
        "retrieved_doc_ids": retrieved_doc_ids,
    }


def infer_doc_prefix(reference_doc_ids: list[str] | None) -> str:
    prefixes: list[str] = []
    seen: set[str] = set()
    for doc_id in reference_doc_ids or []:
        value = str(doc_id or "").strip()
        if not value:
            continue
        prefix = value.split("#", 1)[0].strip()
        if not prefix or prefix in seen:
            continue
        seen.add(prefix)
        prefixes.append(prefix)
    if len(prefixes) == 1:
        return prefixes[0]
    return ""


def scope_cuad_question(question: str, doc_prefix: str | None) -> str:
    doc_prefix = str(doc_prefix or "").strip()
    if not doc_prefix:
        return str(question or "")
    return f'Contract title: "{doc_prefix}"\nQuestion: {str(question or "").strip()}'


def testset_summary(rows: list[dict[str, Any]], chunk_count: int | None = None) -> dict[str, Any]:
    by_synthesizer: dict[str, int] = {}
    empty_reference = 0
    empty_reference_contexts = 0
    missing_reference_context_ids = 0
    missing_reference_doc_ids = 0

    for row in rows:
        synth = str(row.get("synthesizer_name", "") or "unknown").strip() or "unknown"
        by_synthesizer[synth] = by_synthesizer.get(synth, 0) + 1
        if not _norm_text(row.get("reference")):
            empty_reference += 1
        if not (row.get("reference_contexts") or []):
            empty_reference_contexts += 1
        if not (row.get("reference_context_ids") or []):
            missing_reference_context_ids += 1
        if not (row.get("reference_doc_ids") or []):
            missing_reference_doc_ids += 1

    total = len(rows)
    denom = total or 1
    summary = {
        "num_samples": total,
        "chunk_count": chunk_count,
        "by_synthesizer": by_synthesizer,
        "empty_reference_ratio": round(empty_reference / denom, 6),
        "empty_reference_contexts_ratio": round(empty_reference_contexts / denom, 6),
        "missing_reference_context_ids_ratio": round(missing_reference_context_ids / denom, 6),
        "missing_reference_doc_ids_ratio": round(missing_reference_doc_ids / denom, 6),
    }
    return summary
