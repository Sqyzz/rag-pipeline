from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from graph_build.build_communities import _attach_hierarchy
from graph_build.build_graph import build_graph
from graph_build.extract_triples import _compile_schema, _validate_row_to_triple


class TestTripleSchemaCanonicalization(unittest.TestCase):
    def test_prompt_only_canonicalizes_relation_and_types(self) -> None:
        schema = {
            "entity_types": ["party", "agreement"],
            "relations": [
                {
                    "name": "party_to",
                    "subject_types": ["party"],
                    "object_types": ["agreement"],
                    "aliases": ["party to", "is a party to"],
                }
            ],
        }
        compiled = _compile_schema(schema)
        row = {
            "subject": "Acme",
            "subject_type": "Party",
            "relation": "party to",
            "object": "Master Agreement",
            "object_type": "agreement",
            "evidence": "Acme is a party to the Master Agreement.",
        }
        t = _validate_row_to_triple(
            row=row,
            chunk_id="c1",
            compiled_schema=compiled,
            schema_apply_mode="prompt_only",
        )
        self.assertIsNotNone(t)
        assert t is not None
        self.assertEqual(t.relation, "party_to")
        self.assertEqual(t.subject_type, "party")
        self.assertEqual(t.object_type, "agreement")

    def test_strict_rejects_unknown_relation(self) -> None:
        schema = {
            "entity_types": ["party", "agreement"],
            "relations": [
                {
                    "name": "party_to",
                    "subject_types": ["party"],
                    "object_types": ["agreement"],
                    "aliases": ["party to"],
                }
            ],
        }
        compiled = _compile_schema(schema)
        row = {
            "subject": "Acme",
            "subject_type": "party",
            "relation": "includes",
            "object": "Master Agreement",
            "object_type": "agreement",
            "evidence": "x",
        }
        t = _validate_row_to_triple(
            row=row,
            chunk_id="c1",
            compiled_schema=compiled,
            schema_apply_mode="strict",
        )
        self.assertIsNone(t)


class TestGraphNodeSemanticMerge(unittest.TestCase):
    def test_node_merge_casefold_and_type_scope(self) -> None:
        triples = [
            {
                "chunk_id": "c1",
                "doc_id": "D#p0",
                "subject": "Acme Corp",
                "subject_type": "organization",
                "relation": "owns",
                "object": "Project X",
                "object_type": "project",
                "evidence": "x",
            },
            {
                "chunk_id": "c2",
                "doc_id": "D#p1",
                "subject": "acme corp ",
                "subject_type": "organization",
                "relation": "owns",
                "object": "Project X",
                "object_type": "project",
                "evidence": "y",
            },
        ]
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            triples_file = root / "triples.jsonl"
            graph_file = root / "graph.json"
            with triples_file.open("w", encoding="utf-8") as f:
                for row in triples:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

            metrics = build_graph(
                str(triples_file),
                str(graph_file),
                edge_merge_mode="global",
                node_merge_mode="casefold",
                node_scope_with_type=True,
            )
            graph = json.loads(graph_file.read_text(encoding="utf-8"))

        self.assertEqual(metrics["num_nodes"], 2)
        self.assertEqual(metrics["num_edges"], 1)
        self.assertEqual(graph["edges"][0]["weight"], 2)
        self.assertIn("organization", graph["nodes"][0]["types"] + graph["nodes"][1]["types"])


class TestHierarchyOverlapGuard(unittest.TestCase):
    def test_attach_hierarchy_respects_min_overlap(self) -> None:
        levels = [
            {
                "level": 0,
                "communities": [
                    {"community_id": "l0_c1", "nodes": ["a", "b", "c"]},
                    {"community_id": "l0_c2", "nodes": ["d", "e"]},
                ],
            },
            {
                "level": 1,
                "communities": [
                    {"community_id": "l1_c1", "nodes": ["a"]},
                    {"community_id": "l1_c2", "nodes": ["x", "y"]},
                ],
            },
        ]

        meta = _attach_hierarchy(levels, min_parent_overlap=0.8)
        by_id = {
            c["community_id"]: c
            for lv in levels
            for c in lv["communities"]
        }

        self.assertEqual(by_id["l1_c1"]["parent_id"], "l0_c1")
        self.assertIsNone(by_id["l1_c2"]["parent_id"])
        self.assertEqual(meta["linked_children"], 1)
        self.assertEqual(meta["unlinked_children"], 1)


if __name__ == "__main__":
    unittest.main()
