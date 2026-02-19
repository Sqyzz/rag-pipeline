import numpy as np
import time
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from .config import cfg
from .telemetry import usage_from_body

_local_model = None


def _resolve_api_key() -> str | None:
    import os

    configured = os.getenv(cfg.embedding.api.api_key_env)
    if configured:
        return configured
    # DashScope users often export DASHSCOPE_API_KEY directly.
    return os.getenv("DASHSCOPE_API_KEY")


def _prepare_api_text(text: str, max_chars: int = 6000) -> str:
    text = str(text).strip()
    if not text:
        text = "."
    return text[:max_chars]


def embed_texts(texts, return_meta: bool = False):
    if cfg.embedding.backend == "local":
        global _local_model
        t0 = time.perf_counter()
        if _local_model is None:
            _local_model = SentenceTransformer(cfg.embedding.local.model)
        # Use single-item encoding for stability across constrained/macOS environments.
        vectors = [
            np.asarray(_local_model.encode([text])[0], dtype="float32")
            for text in tqdm(texts, desc="Embedding (local)", unit="chunk")
        ]
        arr = np.vstack(vectors)
        if not return_meta:
            return arr
        return arr, {
            "backend": "local",
            "model": cfg.embedding.local.model,
            "latency_ms": int((time.perf_counter() - t0) * 1000),
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    import requests

    base_url = cfg.embedding.api.base_url.rstrip("/")
    url = base_url if base_url.endswith("/embeddings") else f"{base_url}/embeddings"
    api_key = _resolve_api_key()
    if not api_key:
        raise RuntimeError(
            f"Missing embedding API key. Set {cfg.embedding.api.api_key_env} "
            "or DASHSCOPE_API_KEY."
        )
    headers = {"Authorization": f"Bearer {api_key}"}
    cleaned_texts = [_prepare_api_text(t) for t in texts]
    batch_size = 16
    all_vectors = []
    prompt_tokens = 0
    total_tokens = 0
    t0 = time.perf_counter()

    for i in tqdm(
        range(0, len(cleaned_texts), batch_size),
        desc="Embedding (api)",
        unit="batch",
    ):
        batch = cleaned_texts[i : i + batch_size]
        payload = {"model": cfg.embedding.api.model, "input": batch}
        r = requests.post(url, json=payload, headers=headers, timeout=120)
        if not r.ok:
            # Fallback to single-item requests to isolate problematic entries.
            vectors = []
            for text in batch:
                payload = {"model": cfg.embedding.api.model, "input": [text]}
                rr = requests.post(url, json=payload, headers=headers, timeout=120)
                rr.raise_for_status()
                b = rr.json()
                usage = usage_from_body(b)
                prompt_tokens += usage["prompt_tokens"]
                total_tokens += usage["total_tokens"]
                if "data" in b:
                    vectors.append(b["data"][0]["embedding"])
                elif "output" in b and "embeddings" in b["output"]:
                    vectors.append(b["output"]["embeddings"][0]["embedding"])
                else:
                    raise ValueError(f"Unexpected embedding response format: {b}")
            all_vectors.extend(vectors)
            continue

        body = r.json()
        usage = usage_from_body(body)
        prompt_tokens += usage["prompt_tokens"]
        total_tokens += usage["total_tokens"]

        if "data" in body:
            vectors = [e["embedding"] for e in body["data"]]
        elif "output" in body and "embeddings" in body["output"]:
            vectors = [e["embedding"] for e in body["output"]["embeddings"]]
        else:
            raise ValueError(f"Unexpected embedding response format: {body}")
        all_vectors.extend(vectors)

    arr = np.array(all_vectors, dtype="float32")
    if not return_meta:
        return arr
    return arr, {
        "backend": "api",
        "model": cfg.embedding.api.model,
        "latency_ms": int((time.perf_counter() - t0) * 1000),
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 0, "total_tokens": total_tokens},
    }
