from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from baselines.graph_rag import _filter_communities_by_doc_prefix
from baselines.kg_rag import _filter_chunks_by_doc_prefix, _filter_edges_by_doc_prefix
from graph_build.build_graph import build_graph


class TestDocScopeStrictMode(unittest.TestCase):
    def test_kg_chunk_filter_strict(self) -> None:
        chunks = [{"doc_id": "A#p0", "chunk_id": "c1"}, {"doc_id": "B#p0", "chunk_id": "c2"}]
        out = _filter_chunks_by_doc_prefix(chunks, "C", strict_doc_scope=True)
        self.assertEqual(out, [])

    def test_kg_edge_filter_strict(self) -> None:
        edges = [
            {
                "edge_id": "e1",
                "mentions": [{"doc_id": "A#p0", "chunk_id": "c1"}],
            }
        ]
        out = _filter_edges_by_doc_prefix(edges, "C", strict_doc_scope=True)
        self.assertEqual(out, [])

    def test_graph_community_filter_strict(self) -> None:
        communities = [{"community_id": "cm1", "edges": ["e1"]}]
        edge_by_id = {
            "e1": {"mentions": [{"doc_id": "A#p0", "chunk_id": "c1"}]},
        }
        out = _filter_communities_by_doc_prefix(
            communities,
            edge_by_id=edge_by_id,
            doc_prefix_filter="C",
            strict_doc_scope=True,
        )
        self.assertEqual(out, [])


class TestGraphEdgeMergeMode(unittest.TestCase):
    def test_doc_scoped_merge_separates_edges(self) -> None:
        triples = [
            {
                "chunk_id": "c1",
                "doc_id": "A#p0",
                "subject": "Agreement",
                "relation": "governed_by",
                "object": "LawX",
                "evidence": "x",
            },
            {
                "chunk_id": "c2",
                "doc_id": "B#p0",
                "subject": "Agreement",
                "relation": "governed_by",
                "object": "LawX",
                "evidence": "y",
            },
        ]
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            triples_file = root / "triples.jsonl"
            graph_global = root / "graph_global.json"
            graph_scoped = root / "graph_scoped.json"
            with triples_file.open("w", encoding="utf-8") as f:
                for row in triples:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

            mg = build_graph(str(triples_file), str(graph_global), edge_merge_mode="global")
            ms = build_graph(str(triples_file), str(graph_scoped), edge_merge_mode="doc_scoped")

            self.assertEqual(mg["num_edges"], 1)
            self.assertEqual(ms["num_edges"], 2)


if __name__ == "__main__":
    unittest.main()
