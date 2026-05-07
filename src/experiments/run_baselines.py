from pathlib import Path
import sys
import json
import argparse
import time
import subprocess

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from ingestion.chunking import chunk_texts
from ingestion.load_docs import load_enron, load_cuad
from baselines.text_mapreduce import mapreduce_answer
from baselines.vector_rag import answer_with_context, build_index, retrieve_and_answer
from utils.config import load_config


def _append_jsonl(path: str, obj: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _progress(step: int, total: int, message: str) -> None:
    print(f"[{step}/{total}] {message}")


def run(
    mode: str = "all",
    embedding_mode: str = "realtime",
    batch_completion_window: str = "24h",
    dataset: str = "enron",
    cuad_raw_file: str = "data/raw/cuad/CUADv1.json",
    cuad_split_name: str | None = None,
):
    if dataset == "cuad":
        docs_file = "data/processed/cuad_docs.jsonl"
        chunks_file = "data/processed/cuad_chunks.jsonl"
        idx_file = "outputs/indexes/faiss_cuad.idx"
        store_file = "outputs/indexes/chunk_store_cuad.json"
        if not Path(docs_file).exists():
            _progress(1, 5, f"docs not found, loading raw CUAD data: {cuad_raw_file}")
            load_cuad(cuad_raw_file, docs_file, split_name=cuad_split_name)
    else:
        docs_file = "data/processed/enron_docs_sampled.jsonl"
        chunks_file = "data/processed/chunks_sampled.jsonl"
        idx_file = "outputs/indexes/faiss_sampled.idx"
        store_file = "outputs/indexes/chunk_store_sampled.json"

        if not Path(docs_file).exists():
            _progress(1, 5, "sampled docs not found, loading raw enron data")
            load_enron("data/raw/enron", "data/processed/enron_docs.jsonl")
            docs_file = "data/processed/enron_docs.jsonl"
            chunks_file = "data/processed/chunks.jsonl"
            idx_file = "outputs/indexes/faiss.idx"
            store_file = "outputs/indexes/chunk_store.json"

    if mode in {"all", "build_only"}:
        t_build = time.perf_counter()
        _progress(2, 5, f"chunking docs -> {chunks_file}")
        cfg = load_config("/Users/sqy/Desktop/sensei/毕业论文/imp/enterprise-graphrag/config.yaml")
        chunk_size = cfg.chunking.chunk_size
        overlap = cfg.chunking.overlap
        chunk_texts(docs_file, chunks_file, chunk_size, overlap)
        _progress(3, 5, f"building vector index ({embedding_mode}) -> {idx_file}")

        if embedding_mode == "batch":
            batch_metrics_path = "outputs/results/index_build_metrics_batch.json"
            cmd = [
                sys.executable,
                "src/ingestion/batch_embed.py",
                "--chunks-file",
                chunks_file,
                "--build-index",
                "--index-file",
                idx_file,
                "--store-file",
                store_file,
                "--metrics-json",
                batch_metrics_path,
                "--completion-window",
                batch_completion_window,
                "--embeddings-npy",
                "outputs/indexes/embeddings_sampled.npy",
                "--batch-input-jsonl",
                "outputs/batch/embedding_requests.jsonl",
                "--batch-output-jsonl",
                "outputs/batch/embedding_output.jsonl",
                "--batch-error-jsonl",
                "outputs/batch/embedding_error.jsonl",
            ]
            subprocess.run(cmd, check=True)
            if not Path(batch_metrics_path).exists():
                raise FileNotFoundError(
                    f"Batch metrics not found: {batch_metrics_path}"
                )
            metrics = json.loads(
                Path(batch_metrics_path).read_text(encoding="utf-8")
            ).get("index_metrics", {})
            metrics["embedding_mode"] = "batch"
        else:
            metrics = build_index(chunks_file, idx_file, store_file)
            metrics["embedding_mode"] = "realtime"
            Path("outputs/results").mkdir(parents=True, exist_ok=True)
            Path("outputs/results/index_build_metrics.json").write_text(
                json.dumps(metrics, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        metrics["build_pipeline_time_sec"] = round(time.perf_counter() - t_build, 4)
        Path("outputs/results").mkdir(parents=True, exist_ok=True)
        Path("outputs/results/index_build_metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        _progress(4, 5, "index metrics saved: outputs/results/index_build_metrics.json")
        print("index metrics:", metrics)

    if mode in {"all", "qa_only"}:
        if not Path(idx_file).exists() or not Path(store_file).exists():
            raise FileNotFoundError(
                "Index files not found. Run with --mode build_only first."
            )
        _progress(5, 5, "running retrieval + generation")
        query = "What are the main issues discussed?"
        ctx = retrieve_and_answer(query, idx_file, store_file, 8)
        print("contexts:", ctx)
        vector_ans = answer_with_context(query, ctx)
        print("vector-rag ans:", vector_ans)
        _append_jsonl(
            "outputs/results/vector_answers.jsonl",
            {"qid": "demo-001", "query": query, "contexts": ctx, "answer": vector_ans},
        )

        ans = mapreduce_answer(query, chunks_file)
        print("map-reduce ans:", ans)
        _append_jsonl(
            "outputs/results/mapreduce_answers.jsonl",
            {"qid": "demo-001", "query": query, "answer": ans},
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["enron", "cuad"],
        default="enron",
        help="Select dataset for docs/chunks/index paths.",
    )
    parser.add_argument(
        "--mode",
        choices=["all", "build_only", "qa_only"],
        default="all",
        help="all: build index + QA; build_only: only chunk and build index; qa_only: only retrieval/generation",
    )
    parser.add_argument(
        "--embedding-mode",
        choices=["realtime", "batch"],
        default="batch",
        help="realtime: normal embedding API; batch: DashScope/OpenAI-compatible Batch File API",
    )
    parser.add_argument(
        "--batch-completion-window",
        default="24h",
        help="batch completion window, e.g. 24h or 7d",
    )
    parser.add_argument(
        "--cuad-raw-file",
        default="data/raw/cuad/CUADv1.json",
        help="CUAD raw SQuAD-format json file path.",
    )
    parser.add_argument(
        "--cuad-split-name",
        default=None,
        help="Optional split tag written into CUAD doc meta.",
    )
    args = parser.parse_args()
    run(
        mode=args.mode,
        embedding_mode=args.embedding_mode,
        batch_completion_window=args.batch_completion_window,
        dataset=args.dataset,
        cuad_raw_file=args.cuad_raw_file,
        cuad_split_name=args.cuad_split_name,
    )
