from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any


def normalize_text(text: str) -> str:
    text = text.strip().lower()
    return re.sub(r"\s+", " ", text)


def text_hash(text: str) -> str:
    return hashlib.sha1(normalize_text(text).encode("utf-8")).hexdigest()


def run(
    input_path: str,
    output_path: str,
    sample_size: int | None,
    sample_ratio: float | None,
    seed: int,
    min_chars: int,
) -> dict[str, Any]:
    if sample_size is None and sample_ratio is None:
        raise ValueError("Provide either --sample-size or --sample-ratio")
    if sample_ratio is not None and not (0 < sample_ratio <= 1):
        raise ValueError("--sample-ratio must be in (0, 1]")
    if sample_size is not None and sample_size <= 0:
        raise ValueError("--sample-size must be > 0")

    rng = random.Random(seed)
    seen_hashes: set[str] = set()

    total_count = 0
    valid_count = 0
    dup_count = 0
    unique_count = 0

    sampled: list[dict[str, Any]] = []

    with Path(input_path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            total_count += 1
            obj = json.loads(line)
            text = str(obj.get("text", ""))

            if len(text) < min_chars:
                continue

            valid_count += 1
            h = text_hash(text)
            if h in seen_hashes:
                dup_count += 1
                continue

            seen_hashes.add(h)
            unique_count += 1

            if sample_ratio is not None:
                if rng.random() <= sample_ratio:
                    sampled.append(obj)
                continue

            # Reservoir sampling for exact-size sample over de-duplicated stream.
            assert sample_size is not None
            if len(sampled) < sample_size:
                sampled.append(obj)
            else:
                j = rng.randint(1, unique_count)
                if j <= sample_size:
                    sampled[j - 1] = obj

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as out:
        for item in sampled:
            out.write(json.dumps(item, ensure_ascii=False) + "\n")

    kept_count = len(sampled)
    unique_keep_ratio = (kept_count / unique_count) if unique_count else 0.0

    return {
        "input": input_path,
        "output": output_path,
        "total_docs": total_count,
        "valid_docs": valid_count,
        "duplicate_docs": dup_count,
        "unique_docs": unique_count,
        "sampled_docs": kept_count,
        "sample_size": sample_size,
        "sample_ratio": sample_ratio,
        "seed": seed,
        "min_chars": min_chars,
        "unique_keep_ratio": round(unique_keep_ratio, 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Deduplicate and sample large jsonl corpora")
    parser.add_argument("--input", default="data/processed/enron_docs.jsonl", help="input jsonl")
    parser.add_argument(
        "--output",
        default="data/processed/enron_docs_sampled.jsonl",
        help="sampled output jsonl",
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--sample-size", type=int, help="fixed sample size after de-dup")
    mode.add_argument("--sample-ratio", type=float, help="sample ratio in (0, 1] after de-dup")

    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--min-chars", type=int, default=20, help="minimum text length")
    parser.add_argument(
        "--stats-output",
        default="outputs/results/sample_dedup_stats.json",
        help="stats output json",
    )

    args = parser.parse_args()

    stats = run(
        input_path=args.input,
        output_path=args.output,
        sample_size=args.sample_size,
        sample_ratio=args.sample_ratio,
        seed=args.seed,
        min_chars=args.min_chars,
    )

    stats_path = Path(args.stats_output)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
