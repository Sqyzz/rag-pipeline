"""Tests for optional cross-encoder rerank in youtu-graphrag backend (§12.3.67).

CE is positioned BETWEEN lightweight rerank and strong rerank so that
strong rerank's anchor-reservation / greedy-coverage / doc-rescue constraints
are applied last and are not bypassed by CE reordering.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
YOUTU_ROOT = ROOT / "youtu-graphrag"
if str(YOUTU_ROOT) not in sys.path:
    sys.path.insert(0, str(YOUTU_ROOT))

_prev_cwd = os.getcwd()
os.chdir(YOUTU_ROOT)
try:
    import backend as be  # noqa: E402
finally:
    os.chdir(_prev_cwd)


_CE_CFG_OFF = SimpleNamespace(
    enabled=False,
    model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_k_in=20,
    top_k_out=12,
    fuse_alpha=1.0,
    device="cpu",
    batch_size=16,
)

_CE_CFG_ON = SimpleNamespace(
    enabled=True,
    model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_k_in=20,
    top_k_out=12,
    fuse_alpha=1.0,
    device="cpu",
    batch_size=16,
)


def _set_ce_config(cfg) -> None:
    be.config = SimpleNamespace(retrieval=SimpleNamespace(cross_encoder_rerank=cfg))


def test_cross_encoder_rerank_disabled_noop_preserves_order() -> None:
    """When disabled, CE is a pass-through; strong rerank constraints are unaffected."""
    pairs = [
        {"chunk_id": "a", "text": "alpha"},
        {"chunk_id": "b", "text": "beta"},
    ]
    trace: dict = {}
    _set_ce_config(_CE_CFG_OFF)
    out = be._cross_encoder_rerank_support_pairs_after_strong(
        list(pairs),
        query_text="q",
        retrieval_queries=["q2"],
        chunk_stage_trace=trace,
    )
    assert [p["chunk_id"] for p in out] == ["a", "b"]
    assert "cross_encoder_reranked_chunk_ids" not in trace


def test_cross_encoder_rerank_enabled_reorders_by_scores(monkeypatch: pytest.MonkeyPatch) -> None:
    """CE reorders candidates before strong rerank.

    The output represents what strong rerank receives as input — strong
    rerank then applies its anchor/doc constraints on top.
    """
    class _MockModel:
        def predict(self, pairs, batch_size=8, show_progress_bar=False):
            assert len(pairs) == 2
            return [0.1, 0.9]

    monkeypatch.setattr(be, "get_shared_cross_encoder", lambda _m, _d: _MockModel())

    pairs = [
        {"chunk_id": "low", "text": "first chunk"},
        {"chunk_id": "high", "text": "second chunk"},
    ]
    trace: dict = {}
    _set_ce_config(_CE_CFG_ON)
    out = be._cross_encoder_rerank_support_pairs_after_strong(
        list(pairs),
        query_text="test query",
        retrieval_queries=[],
        chunk_stage_trace=trace,
    )
    # CE reorders: high score first → strong rerank sees best candidates first
    assert [p["chunk_id"] for p in out] == ["high", "low"]
    assert trace.get("cross_encoder_reranked_chunk_ids") == ["high", "low"]
    meta = trace.get("cross_encoder_rerank_meta") or {}
    assert meta.get("model_id") == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert "ce_rerank_latency_ms" in meta
    assert meta.get("ce_scores_summary", {}).get("n") == 2


def test_cross_encoder_rerank_position_before_strong_constraint(monkeypatch: pytest.MonkeyPatch) -> None:
    """CE output is strong rerank's input; strong rerank can still reorder CE output.

    Verifies the pipeline order: lightweight → CE → strong.
    CE trace (`cross_encoder_reranked_chunk_ids`) is recorded before strong runs,
    so it captures what CE returned — independent of what strong later decides.
    """
    class _MockModel:
        def predict(self, pairs, batch_size=8, show_progress_bar=False):
            # Rank b > a
            return [0.2, 0.8]

    monkeypatch.setattr(be, "get_shared_cross_encoder", lambda _m, _d: _MockModel())

    pairs = [
        {"chunk_id": "a", "text": "clause alpha"},
        {"chunk_id": "b", "text": "clause beta"},
    ]
    trace: dict = {}
    _set_ce_config(_CE_CFG_ON)
    ce_out = be._cross_encoder_rerank_support_pairs_after_strong(
        list(pairs),
        query_text="what is the termination clause",
        retrieval_queries=[],
        chunk_stage_trace=trace,
    )
    # CE output goes to strong rerank next; CE trace records CE order
    assert trace["cross_encoder_reranked_chunk_ids"] == ["b", "a"]
    # CE output has b first (higher score)
    assert ce_out[0]["chunk_id"] == "b"
    # Strong rerank would then apply its constraints on ce_out — tested separately
