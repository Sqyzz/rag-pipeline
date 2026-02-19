from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from openai import OpenAI

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from baselines.vector_rag import build_index_from_embeddings
from utils.config import cfg


def _resolve_api_key() -> str:
    key = os.getenv(cfg.embedding.api.api_key_env) or os.getenv("DASHSCOPE_API_KEY")
    if not key:
        raise RuntimeError(
            f"Missing API key. Set {cfg.embedding.api.api_key_env} or DASHSCOPE_API_KEY."
        )
    return key


def _base_url() -> str:
    return cfg.embedding.api.base_url.rstrip("/")


def _safe_text(text: str, max_chars: int = 6000) -> str:
    t = str(text).strip()
    if not t:
        t = "."
    return t[:max_chars]


def prepare_batch_input(chunks_file: str, out_jsonl: str, model: str, dimensions: int | None) -> int:
    out = Path(out_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with open(chunks_file, encoding="utf-8") as src, out.open("w", encoding="utf-8") as dst:
        for i, line in enumerate(src):
            obj = json.loads(line)
            text = _safe_text(obj.get("text", ""))
            body = {
                "model": model,
                "input": text,
                "encoding_format": "float",
            }
            if dimensions is not None:
                body["dimensions"] = dimensions

            req = {
                "custom_id": f"chunk-{i:08d}",
                "method": "POST",
                "url": "/v1/embeddings",
                "body": body,
            }
            dst.write(json.dumps(req, ensure_ascii=False) + "\n")
            total += 1
    return total


def _poll_batch(client: OpenAI, batch_id: str, interval_sec: int, timeout_sec: int):
    t0 = time.time()
    last_status = None
    while True:
        batch = client.batches.retrieve(batch_id)
        status = batch.status
        counts = getattr(batch, "request_counts", None)
        if status != last_status:
            print(f"batch status: {status}")
            last_status = status
        if counts:
            print(
                f"progress total={counts.total} completed={counts.completed} failed={counts.failed}"
            )

        if status in {"completed", "failed", "expired", "cancelled"}:
            return batch

        if time.time() - t0 > timeout_sec:
            raise TimeoutError(f"Batch {batch_id} timed out after {timeout_sec}s")

        time.sleep(interval_sec)


def _write_file_content(client: OpenAI, file_id: str, out_path: str) -> None:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    content = client.files.content(file_id)
    if hasattr(content, "write_to_file"):
        content.write_to_file(str(p))
        return
    data = content.read() if hasattr(content, "read") else bytes(content)
    p.write_bytes(data)


def _parse_embedding_from_line(line_obj: dict) -> list[float] | None:
    response = line_obj.get("response") or {}
    body = response.get("body") or {}

    if "data" in body:
        return body["data"][0]["embedding"]
    if "output" in body and "embeddings" in body["output"]:
        return body["output"]["embeddings"][0]["embedding"]
    return None


def parse_output_embeddings(output_jsonl: str, out_npy: str, expected_rows: int) -> dict:
    vectors = [None] * expected_rows
    completed = 0
    failed = 0

    with open(output_jsonl, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            cid = obj.get("custom_id", "")
            try:
                row = int(cid.split("-")[-1])
            except Exception:
                failed += 1
                continue

            vec = _parse_embedding_from_line(obj)
            if vec is None:
                failed += 1
                continue

            vectors[row] = vec
            completed += 1

    missing = sum(1 for v in vectors if v is None)
    if missing > 0:
        raise ValueError(f"Missing {missing} embeddings in batch output. Check error file.")

    arr = np.asarray(vectors, dtype="float32")
    np.save(out_npy, arr)

    return {
        "completed": completed,
        "failed": failed,
        "rows": int(arr.shape[0]),
        "dim": int(arr.shape[1]),
        "npy_file": out_npy,
    }


def run(args):
    key = _resolve_api_key()
    client = OpenAI(api_key=key, base_url=_base_url())

    model = args.model or cfg.embedding.api.model
    request_count = prepare_batch_input(
        chunks_file=args.chunks_file,
        out_jsonl=args.batch_input_jsonl,
        model=model,
        dimensions=args.dimensions,
    )
    print(f"prepared batch input: {args.batch_input_jsonl} rows={request_count}")

    with open(args.batch_input_jsonl, "rb") as f:
        uploaded = client.files.create(file=f, purpose="batch")
    print(f"uploaded file_id: {uploaded.id}")

    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/embeddings",
        completion_window=args.completion_window,
        metadata={"ds_name": "enterprise-graphrag-embedding-batch"},
    )
    print(f"created batch_id: {batch.id}")

    if args.no_wait:
        return

    batch = _poll_batch(
        client,
        batch_id=batch.id,
        interval_sec=args.poll_interval_sec,
        timeout_sec=args.poll_timeout_sec,
    )

    if batch.status != "completed":
        raise RuntimeError(f"Batch finished with status={batch.status}")

    if not batch.output_file_id:
        raise RuntimeError("Batch completed but output_file_id is empty")

    _write_file_content(client, batch.output_file_id, args.batch_output_jsonl)
    print(f"downloaded output: {args.batch_output_jsonl}")

    if getattr(batch, "error_file_id", None):
        _write_file_content(client, batch.error_file_id, args.batch_error_jsonl)
        print(f"downloaded error file: {args.batch_error_jsonl}")

    parse_stats = parse_output_embeddings(
        output_jsonl=args.batch_output_jsonl,
        out_npy=args.embeddings_npy,
        expected_rows=request_count,
    )
    print("embedding parse stats:", parse_stats)

    if args.build_index:
        idx_metrics = build_index_from_embeddings(
            chunks_file=args.chunks_file,
            embeddings_file=args.embeddings_npy,
            idx_file=args.index_file,
            store_file=args.store_file,
        )
        print("index metrics:", idx_metrics)

        Path(args.metrics_json).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "batch_id": batch.id,
            "input_file_id": uploaded.id,
            "output_file_id": batch.output_file_id,
            "request_count": request_count,
            "parse_stats": parse_stats,
            "index_metrics": idx_metrics,
        }
        Path(args.metrics_json).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"saved metrics: {args.metrics_json}")


def main():
    parser = argparse.ArgumentParser(description="DashScope/OpenAI-compatible batch embeddings")
    parser.add_argument("--chunks-file", default="data/processed/chunks_sampled.jsonl")
    parser.add_argument("--model", default=None)
    parser.add_argument("--dimensions", type=int, default=None)

    parser.add_argument("--batch-input-jsonl", default="outputs/batch/embedding_requests.jsonl")
    parser.add_argument("--batch-output-jsonl", default="outputs/batch/embedding_output.jsonl")
    parser.add_argument("--batch-error-jsonl", default="outputs/batch/embedding_error.jsonl")
    parser.add_argument("--embeddings-npy", default="outputs/indexes/embeddings_sampled.npy")

    parser.add_argument("--completion-window", default="24h")
    parser.add_argument("--poll-interval-sec", type=int, default=20)
    parser.add_argument("--poll-timeout-sec", type=int, default=172800)
    parser.add_argument("--no-wait", action="store_true")

    parser.add_argument("--build-index", action="store_true")
    parser.add_argument("--index-file", default="outputs/indexes/faiss_sampled_batch.idx")
    parser.add_argument("--store-file", default="outputs/indexes/chunk_store_sampled_batch.json")
    parser.add_argument("--metrics-json", default="outputs/results/index_build_metrics_batch.json")

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
