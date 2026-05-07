from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for candidate in (ROOT, SRC):
    value = str(candidate)
    if value not in sys.path:
        sys.path.insert(0, value)

from src.utils.ragas_converters import (
    _ragas_llm_cfg,
    infer_doc_prefix,
    match_reference_contexts,
    normalize_evidence_chunks,
    scope_cuad_question,
)
from src.utils import ragas_converters


def test_match_reference_contexts_strips_multi_hop_prefix() -> None:
    chunk_lookup = {
        "payment terms apply.": [
            {"chunk_id": "chunk-1", "doc_id": "ContractA#p0", "text": "Payment terms apply."}
        ],
        "termination clause text.": [
            {"chunk_id": "chunk-2", "doc_id": "ContractA#p1", "text": "Termination clause text."}
        ],
    }

    chunk_ids, doc_ids = match_reference_contexts(
        ["<1-hop>\n\nPayment terms apply.", "<2-hop>\n\nTermination clause text."],
        chunk_lookup,
    )

    assert chunk_ids == ["chunk-1", "chunk-2"]
    assert doc_ids == ["ContractA#p0", "ContractA#p1"]


def test_normalize_evidence_chunks_expands_merged_chunk_ids() -> None:
    payload = normalize_evidence_chunks(
        [
            {
                "chunk_id": "chunk-a",
                "chunk_ids": ["chunk-a", "chunk-b"],
                "doc_id": "ContractA#p0",
                "text": "Merged context",
            },
            {
                "chunk_id": "chunk-c|chunk-d",
                "doc_id": "ContractA#p1",
                "text": "Second context",
            },
        ]
    )

    assert payload["retrieved_contexts"] == ["Merged context", "Second context"]
    assert payload["retrieved_context_ids"] == ["chunk-a", "chunk-b", "chunk-c", "chunk-d"]
    assert payload["retrieved_doc_ids"] == ["ContractA#p0", "ContractA#p1"]


def test_scope_and_doc_prefix_follow_cuad_contract_title_format() -> None:
    assert infer_doc_prefix(["ContractA#p3"]) == "ContractA"
    assert scope_cuad_question("What is the term?", "ContractA") == (
        'Contract title: "ContractA"\nQuestion: What is the term?'
    )


def test_infer_doc_prefix_disables_single_contract_title_for_multi_doc_questions() -> None:
    assert infer_doc_prefix(["ContractA#p3", "ContractB#p1"]) == ""
    assert scope_cuad_question("How do the clauses relate?", infer_doc_prefix(["ContractA#p3", "ContractB#p1"])) == (
        "How do the clauses relate?"
    )


def test_ragas_llm_cfg_prefers_ragas_specific_settings() -> None:
    original_cfg = ragas_converters.cfg
    ragas_converters.cfg = SimpleNamespace(
        llm=SimpleNamespace(
            backend="api",
            api=SimpleNamespace(base_url="https://global.example/v1", model="global-model", api_key_env="GLOBAL_KEY"),
            local=SimpleNamespace(base_url="http://global-local/v1", model="global-local-model", api_key="EMPTY"),
        ),
        ragas=SimpleNamespace(
            llm=SimpleNamespace(
                backend="api",
                api=SimpleNamespace(
                    base_url="https://ragas.example/v1",
                    model="ragas-model",
                    api_key_env="RAGAS_KEY",
                ),
            )
        ),
    )
    try:
        assert _ragas_llm_cfg() == {
            "backend": "api",
            "base_url": "https://ragas.example/v1",
            "model": "ragas-model",
            "api_key_env": "RAGAS_KEY",
        }
    finally:
        ragas_converters.cfg = original_cfg
