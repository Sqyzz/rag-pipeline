from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import yaml


def load_chunk_size(config_path: str) -> int:
    with Path(config_path).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return int(cfg.get("chunking", {}).get("chunk_size", 1000))


def percentile_90(lengths: list[int]) -> int:
    if not lengths:
        return 0
    return int(np.percentile(np.asarray(lengths, dtype=np.int32), 90))


def estimate_chunks(length: float, chunk_size: int, overlap: int) -> float:
    if length <= 0:
        return 0.0
    if chunk_size <= overlap:
        return 0.0
    if length <= chunk_size:
        return 1.0
    step = chunk_size - overlap
    return 1.0 + math.ceil((length - chunk_size) / step)


def recommend_chunk_size(current: int, avg_len: float, p90_len: int) -> tuple[int, str]:
    ratio = p90_len / current if current > 0 else 0

    if ratio < 0.8:
        rec = max(400, int(round((p90_len / 0.8) / 100.0) * 100))
        reason = "P90 明显小于当前 chunk_size，建议下调以减少空白上下文。"
    elif ratio > 3.0:
        rec = min(2000, int(round((current * 1.5) / 100.0) * 100))
        reason = "P90 远大于当前 chunk_size，建议上调以减少跨块语义断裂。"
    else:
        rec = current
        reason = "当前 chunk_size 与文档长度分布匹配，保持不变更稳妥。"

    return rec, reason


def analyze(input_path: str, config_path: str, overlap: int = 200) -> dict:
    lengths: list[int] = []
    count = 0
    total_len = 0

    with Path(input_path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            text = obj.get("text", "")
            length = len(text)
            lengths.append(length)
            total_len += length
            count += 1

    avg_len = (total_len / count) if count else 0.0
    p90_len = percentile_90(lengths)

    current_chunk_size = load_chunk_size(config_path)
    rec_chunk_size, reason = recommend_chunk_size(current_chunk_size, avg_len, p90_len)

    avg_chunks_per_doc = estimate_chunks(avg_len, current_chunk_size, overlap)
    p90_chunks_per_doc = estimate_chunks(p90_len, current_chunk_size, overlap)

    return {
        "doc_count": count,
        "avg_length": round(avg_len, 2),
        "p90_length": p90_len,
        "current_chunk_size": current_chunk_size,
        "overlap": overlap,
        "estimated_avg_chunks_per_doc": avg_chunks_per_doc,
        "estimated_p90_chunks_per_doc": p90_chunks_per_doc,
        "recommended_chunk_size": rec_chunk_size,
        "recommendation": reason,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Document length stats for chunk_size decision")
    parser.add_argument("--input", default="data/processed/enron_docs.jsonl", help="input docs jsonl")
    parser.add_argument("--config", default="config.yaml", help="config path")
    parser.add_argument("--output", default="outputs/results/doc_stats.json", help="output json summary")
    parser.add_argument("--overlap", type=int, default=200, help="chunk overlap for chunk estimate")
    args = parser.parse_args()

    result = analyze(args.input, args.config, overlap=args.overlap)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
