from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for candidate in (ROOT, SRC):
    value = str(candidate)
    if value not in sys.path:
        sys.path.insert(0, value)

from src.baselines import graph_rag
from src.baselines.graph_rag import _collect_chunk_evidence_from_communities
from src.utils.ragas_converters import normalize_evidence_chunks


class TestGraphRagEvidenceAlignment(unittest.TestCase):
    def test_collect_chunk_evidence_includes_text_for_ragas_contexts(self) -> None:
        selected = [{"community_id": "c1", "edges": ["e1"]}]
        edge_by_id = {
            "e1": {
                "edge_id": "e1",
                "mentions": [
                    {
                        "chunk_id": "chunk-1",
                        "doc_id": "doc-1#p0",
                        "evidence": "Payment obligations survive termination.",
                    }
                ],
            }
        }

        evidence_chunks = _collect_chunk_evidence_from_communities(
            selected=selected,
            edge_by_id=edge_by_id,
            max_chunks=5,
        )
        normalized = normalize_evidence_chunks(evidence_chunks)

        self.assertEqual(len(evidence_chunks), 1)
        self.assertEqual(evidence_chunks[0]["text"], "Payment obligations survive termination.")
        self.assertEqual(normalized["retrieved_contexts"], ["Payment obligations survive termination."])
        self.assertEqual(normalized["retrieved_context_ids"], ["chunk-1"])

    def test_answer_with_graphrag_emits_alignment_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            graph_file = root / "graph.json"
            communities_file = root / "communities.json"
            graph_file.write_text(
                json.dumps(
                    {
                        "edges": [
                            {
                                "edge_id": "e1",
                                "mentions": [
                                    {
                                        "chunk_id": "chunk-1",
                                        "doc_id": "doc-1#p0",
                                        "text": "Primary clause text about payment obligations.",
                                    },
                                    {
                                        "chunk_id": "chunk-2",
                                        "doc_id": "doc-2#p0",
                                        "text": "External clause text about payment obligations and third party costs.",
                                    },
                                ],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            communities_file.write_text(
                json.dumps(
                    {
                        "communities": [
                            {
                                "community_id": "community-1",
                                "level": 0,
                                "summary": "Summary for payment obligations.",
                                "edges": ["e1"],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            with patch.object(
                graph_rag,
                "_rank_communities_by_embedding",
                return_value=[
                    {
                        "community_id": "community-1",
                        "level": 0,
                        "summary": "Summary for payment obligations.",
                        "edges": ["e1"],
                    }
                ],
            ):
                with patch.object(
                    graph_rag,
                    "_map_summary_answer",
                    return_value=(
                        {
                            "community_id": "community-1",
                            "level": 0,
                            "summary": "Summary for payment obligations.",
                            "partial_answer": "partial answer",
                        },
                        {},
                    ),
                ):
                    with patch.object(graph_rag, "_reduce_answers", return_value=("final answer", {})):
                        out = graph_rag.answer_with_graphrag(
                            query="What are the payment obligations?",
                            graph_file=str(graph_file),
                            communities_file=str(communities_file),
                            doc_prefix_filter="doc-1",
                            strict_doc_scope=False,
                        )

        self.assertEqual(out["evaluation_payload"]["retrieved_context_ids"], ["chunk-1"])
        self.assertEqual(out["evaluation_payload"]["external_related_context_ids"], ["chunk-2"])
        self.assertEqual(out["answer_scope_target_doc_id"], "doc-1#p0")
        self.assertEqual(out["answer_composition_mode"], "cross_document_bridge")
        self.assertEqual(out["retrieval_trace"]["answer_scope_target_doc_id"], "doc-1#p0")
        self.assertEqual(out["answer_trace"]["final_chunk_selection_strategy"], "graph_selected_community_mentions")
        self.assertEqual(out["reasoning_steps"][0]["type"], "community_map")
        self.assertEqual(out["reasoning_steps"][-1]["type"], "community_reduce")


if __name__ == "__main__":
    unittest.main()
