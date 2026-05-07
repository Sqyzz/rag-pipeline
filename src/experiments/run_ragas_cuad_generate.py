from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.ragas_converters import (
    build_chunk_lookup,
    build_langchain_clients,
    ensure_vendored_ragas_on_path,
    load_jsonl,
    match_reference_contexts,
    testset_summary,
    write_json,
    write_jsonl,
)


def _build_documents(chunks: list[dict[str, Any]]) -> list[Any]:
    try:
        from langchain_core.documents import Document
    except ImportError as exc:
        raise ImportError(
            "Missing ragas runtime deps. Install `langchain-core`, `langchain-openai`, and `datasets`."
        ) from exc

    docs: list[Any] = []
    for row in chunks:
        text = str(row.get("text", "") or "").strip()
        if not text:
            continue
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "chunk_id": str(row.get("chunk_id", "") or "").strip(),
                    "doc_id": str(row.get("doc_id", "") or "").strip(),
                    "source": str(row.get("source", "") or "").strip(),
                    "meta": row.get("meta") if isinstance(row.get("meta"), dict) else {},
                },
            )
        )
    return docs


def _parse_query_distribution(raw: str | None, generator: Any) -> Any:
    ensure_vendored_ragas_on_path()
    from ragas.testset.synthesizers import default_query_distribution

    canonical_default = (
        "single_hop_specific=0.333333,"
        "multi_hop_specific=0.333333,"
        "multi_hop_abstract=0.333333"
    )
    distribution = default_query_distribution(generator.llm)
    alias_map = {
        "single_hop_specific": "single_hop_specific_query_synthesizer",
        "multi_hop_specific": "multi_hop_specific_query_synthesizer",
        "multi_hop_abstract": "multi_hop_abstract_query_synthesizer",
    }
    if not raw or str(raw).strip() == canonical_default:
        # Let ragas resolve the actually available synthesizers after transforms
        # are applied to the knowledge graph. This is important for smoke tests
        # with very few chunks where multi-hop synthesizers are unavailable.
        return None

    requested: dict[str, float] = {}
    for part in str(raw).split(","):
        name, _, value = part.partition("=")
        key = str(name or "").strip()
        if not key:
            continue
        requested[alias_map.get(key, key)] = float(value or 0.0)

    total = sum(requested.values())
    if total <= 0:
        raise ValueError("query distribution weights must sum to a positive value")

    normalized = []
    for synthesizer, _ in distribution:
        synth_name = getattr(synthesizer, "name", "")
        weight = requested.get(synth_name)
        if weight is None:
            weight = requested.get(type(synthesizer).__name__)
        if weight is None:
            continue
        normalized.append((synthesizer, float(weight / total)))
    if not normalized:
        available = [
            getattr(synthesizer, "name", type(synthesizer).__name__)
            for synthesizer, _ in distribution
        ]
        raise ValueError(
            "query distribution did not match any available synthesizer. "
            f"Available names: {available}"
        )
    return normalized


def build_testset_rows(testset: Any, chunk_lookup: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, sample in enumerate(testset.samples, start=1):
        eval_sample = sample.eval_sample
        reference_contexts = list(eval_sample.reference_contexts or [])
        reference_context_ids, reference_doc_ids = match_reference_contexts(reference_contexts, chunk_lookup)
        rows.append(
            {
                "qid": f"ragas-cuad-{idx:04d}",
                "question": str(eval_sample.user_input or "").strip(),
                "reference": str(eval_sample.reference or "").strip(),
                "reference_contexts": reference_contexts,
                "reference_context_ids": reference_context_ids,
                "reference_doc_ids": reference_doc_ids,
                "synthesizer_name": str(sample.synthesizer_name or "").strip(),
                "persona_name": getattr(eval_sample, "persona_name", None),
                "query_style": getattr(eval_sample, "query_style", None),
                "query_length": getattr(eval_sample, "query_length", None),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate CUAD ragas testset from pre-chunked assets.")
    parser.add_argument("--chunks-file", default="", help="Chunk jsonl file used for ragas generate_with_chunks.")
    parser.add_argument("--docs-file", default="", help="Deprecated alias for --chunks-file.")
    parser.add_argument("--out-testset-file", required=True)
    parser.add_argument("--out-summary-file", required=True)
    parser.add_argument("--testset-size", type=int, default=20)
    parser.add_argument("--max-chunks", type=int, default=None, help="Only use the first N chunks for smoke testing.")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--query-distribution",
        default="single_hop_specific=0.333333,multi_hop_specific=0.333333,multi_hop_abstract=0.333333",
        help="Comma-separated synthesizer weights, e.g. single_hop_specific=0.4,multi_hop_specific=0.4,multi_hop_abstract=0.2",
    )
    args = parser.parse_args()

    chunks_file = str(args.chunks_file or args.docs_file).strip()
    if not chunks_file:
        raise SystemExit("Either --chunks-file or --docs-file must be provided.")

    ensure_vendored_ragas_on_path()
    from ragas.run_config import RunConfig
    from ragas.testset.synthesizers.generate import TestsetGenerator

    chunks = load_jsonl(chunks_file)
    if args.max_chunks is not None:
        chunks = chunks[: max(0, int(args.max_chunks))]
    documents = _build_documents(chunks)
    if not documents:
        raise RuntimeError(f"No non-empty chunks found in {chunks_file}")

    random.seed(int(args.random_seed))
    llm, embeddings = build_langchain_clients(temperature=0.2)
    generator = TestsetGenerator.from_langchain(llm=llm, embedding_model=embeddings)
    query_distribution = _parse_query_distribution(args.query_distribution, generator)
    testset = generator.generate_with_chunks(
        chunks=documents,
        testset_size=int(args.testset_size),
        query_distribution=query_distribution,
        run_config=RunConfig(timeout=180),
        raise_exceptions=True,
    )

    chunk_lookup = build_chunk_lookup(chunks)
    rows = build_testset_rows(testset, chunk_lookup)
    summary = testset_summary(rows, chunk_count=len(documents))
    summary.update(
        {
            "chunks_file": str(Path(chunks_file).resolve()),
            "out_testset_file": str(Path(args.out_testset_file).resolve()),
            "random_seed": int(args.random_seed),
            "requested_testset_size": int(args.testset_size),
            "actual_testset_size": len(rows),
            "max_chunks": int(args.max_chunks) if args.max_chunks is not None else None,
            "query_distribution": (
                {
                    getattr(synthesizer, "name", type(synthesizer).__name__): round(float(weight), 6)
                    for synthesizer, weight in query_distribution
                }
                if query_distribution is not None
                else {"mode": "auto_available_synthesizers"}
            ),
        }
    )

    write_jsonl(args.out_testset_file, rows)
    write_json(args.out_summary_file, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
