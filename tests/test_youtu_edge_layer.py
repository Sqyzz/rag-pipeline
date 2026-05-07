from __future__ import annotations

import importlib.util
import json
import ast
import types
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
YOUTU_ROOT = ROOT / "youtu-graphrag"
for candidate in (YOUTU_ROOT, ROOT):
    value = str(candidate)
    if value not in sys.path:
        sys.path.insert(0, value)


logger_stub = types.SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None, error=lambda *a, **k: None)
sys.modules.setdefault("utils.logger", types.SimpleNamespace(logger=logger_stub))
sys.modules.setdefault("json_repair", types.SimpleNamespace(loads=json.loads))
sys.modules.setdefault(
    "utils.call_llm_api",
    types.SimpleNamespace(LLMCompletionCall=lambda *a, **k: None),
)


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _extract_class_method(path: Path, class_name: str, method_name: str):
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    target = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == method_name:
                    target = child
                    break
    assert target is not None, f"{class_name}.{method_name} not found"

    module = ast.Module(body=[target], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "Dict": dict,
        "List": list,
        "logger": logger_stub,
        "os": __import__("os"),
        "json": json,
        "torch": types.SimpleNamespace(load=lambda *a, **k: {}),
    }
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[method_name]


graph_processor = _load_module("youtu_graph_processor_test", YOUTU_ROOT / "utils" / "graph_processor.py")
config_loader = _load_module("youtu_config_loader_test", YOUTU_ROOT / "config" / "config_loader.py")
agentic_decomposer = _load_module("youtu_agentic_decomposer_test", YOUTU_ROOT / "models" / "retriever" / "agentic_decomposer.py")


class TestYoutuEdgeLayer(unittest.TestCase):
    def test_graph_loader_marks_auxiliary_edges(self) -> None:
        schema = {
            "relation_types": ["governed_by"],
            "relations": [{"name": "governed_by", "relation": "governed_by"}],
        }
        relationships = [
            {
                "start_node": {"label": "entity", "properties": {"name": "A"}},
                "relation": "has_attribute",
                "end_node": {"label": "attribute", "properties": {"name": "attr"}},
            },
            {
                "start_node": {"label": "entity", "properties": {"name": "A"}},
                "relation": "governed_by",
                "end_node": {"label": "entity", "properties": {"name": "B"}},
            },
        ]

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as handle:
            json.dump(relationships, handle, ensure_ascii=False)
            graph_path = handle.name
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as handle:
            json.dump(schema, handle, ensure_ascii=False)
            schema_path = handle.name

        graph = graph_processor.load_graph_from_json(graph_path, use_edge_layer=True, schema_path=schema_path)
        layers = {(data["relation"], data.get("edge_layer")) for _, _, data in graph.edges(data=True)}

        self.assertIn(("has_attribute", "auxiliary"), layers)
        self.assertIn(("governed_by", "semantic"), layers)

    def test_config_loader_reads_edge_layer_switch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            corpus = tmp_path / "corpus.json"
            schema = tmp_path / "schema.json"
            corpus.write_text("[]", encoding="utf-8")
            schema.write_text("{}", encoding="utf-8")
            graph_output = tmp_path / "graph.json"
            config_path = tmp_path / "config.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "datasets:",
                        "  demo:",
                        f"    corpus_path: {corpus}",
                        "    qa_path: qa.json",
                        f"    schema_path: {schema}",
                        f"    graph_output: {graph_output}",
                        "triggers:",
                        "  mode: noagent",
                        "construction: {}",
                        "retrieval:",
                        "  edge_layer:",
                        "    enabled: true",
                        "embeddings: {}",
                        "nlp: {}",
                        "prompts: {}",
                        "output: {}",
                        "performance: {}",
                        "evaluation: {}",
                    ]
                ),
                encoding="utf-8",
            )

            manager = config_loader.ConfigManager(str(config_path))

        self.assertTrue(manager.retrieval.edge_layer.enabled)

    def test_config_loader_reads_doc_consistency_switch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            corpus = tmp_path / "corpus.json"
            schema = tmp_path / "schema.json"
            corpus.write_text("[]", encoding="utf-8")
            schema.write_text("{}", encoding="utf-8")
            graph_output = tmp_path / "graph.json"
            config_path = tmp_path / "config.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "datasets:",
                        "  demo:",
                        f"    corpus_path: {corpus}",
                        "    qa_path: qa.json",
                        f"    schema_path: {schema}",
                        f"    graph_output: {graph_output}",
                        "triggers:",
                        "  mode: noagent",
                        "construction: {}",
                        "retrieval:",
                        "  doc_consistency:",
                        "    enabled: true",
                        "    rerank_same_doc_bonus: 0.2",
                        "embeddings: {}",
                        "nlp: {}",
                        "prompts: {}",
                        "output: {}",
                        "performance: {}",
                        "evaluation: {}",
                    ]
                ),
                encoding="utf-8",
            )

            manager = config_loader.ConfigManager(str(config_path))

        self.assertTrue(manager.retrieval.doc_consistency.enabled)
        self.assertEqual(manager.retrieval.doc_consistency.rerank_same_doc_bonus, 0.2)

    def test_config_loader_reads_decomposition_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            corpus = tmp_path / "corpus.json"
            schema = tmp_path / "schema.json"
            corpus.write_text("[]", encoding="utf-8")
            schema.write_text("{}", encoding="utf-8")
            graph_output = tmp_path / "graph.json"
            config_path = tmp_path / "config.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "datasets:",
                        "  demo:",
                        f"    corpus_path: {corpus}",
                        "    qa_path: qa.json",
                        f"    schema_path: {schema}",
                        f"    graph_output: {graph_output}",
                        "triggers:",
                        "  mode: noagent",
                        "construction: {}",
                        "retrieval:",
                        "  decomposition:",
                        "    mode: retrieval_requirements",
                        "    enable_query_compilation: true",
                        "    max_query_variants: 3",
                        "embeddings: {}",
                        "nlp: {}",
                        "prompts: {}",
                        "output: {}",
                        "performance: {}",
                        "evaluation: {}",
                    ]
                ),
                encoding="utf-8",
            )

            manager = config_loader.ConfigManager(str(config_path))

        self.assertEqual(manager.retrieval.decomposition.mode, "retrieval_requirements")
        self.assertTrue(manager.retrieval.decomposition.enable_query_compilation)
        self.assertEqual(manager.retrieval.decomposition.max_query_variants, 3)

    def test_decomposer_accepts_list_top_level_requirements(self) -> None:
        config = types.SimpleNamespace(
            retrieval=types.SimpleNamespace(
                decomposition=types.SimpleNamespace(mode="retrieval_requirements")
            )
        )
        decomposer = agentic_decomposer.GraphQ("cuad_v3", config=config)
        decomposer.read_schema = lambda _: "{}"
        decomposer.llm_client = types.SimpleNamespace(
            call_api=lambda _prompt: json.dumps(
                [
                    {
                        "route_type": "structural",
                        "intent": "bridge_lookup",
                        "route_reason": "bridge",
                        "entities": ["customer"],
                        "terms": ["breach"],
                        "anchors": ["Section 8"],
                        "query_keywords": ["breach consequences"],
                        "target_patterns": ["which clause states"],
                        "left_endpoint": "Section 8 breach consequences",
                        "right_endpoint": "Section 10.3 indemnification procedures",
                        "bridge_relation": "explicit legal connection",
                        "scope": "",
                        "involved_types": {
                            "nodes": ["clause"],
                            "relations": ["breach_of"],
                            "attributes": ["section_number"],
                        },
                    }
                ]
            )
        )

        result = decomposer.decompose("q", "schema.json")

        self.assertEqual(result["retrieval_requirements"][0]["route_type"], "structural")
        self.assertEqual(result["retrieval_requirements"][0]["left_endpoint"], "Section 8 breach consequences")
        self.assertEqual(result["involved_types"]["nodes"], ["clause"])
        self.assertEqual(result["decomposition_debug"]["mode"], "retrieval_requirements")

    def test_decomposer_normalizes_stringified_requirement_fields(self) -> None:
        config = types.SimpleNamespace(
            retrieval=types.SimpleNamespace(
                decomposition=types.SimpleNamespace(mode="retrieval_requirements")
            )
        )
        decomposer = agentic_decomposer.GraphQ("cuad_v3", config=config)
        decomposer.read_schema = lambda _: "{}"
        decomposer.llm_client = types.SimpleNamespace(
            call_api=lambda _prompt: json.dumps(
                {
                    "retrieval_requirements": [
                        {
                            "route_type": "structural",
                            "intent": "To identify the relationship between two clauses",
                            "route_reason": "bridge",
                            "entities": "['clause', 'party', 'Section 8']",
                            "terms": "['breach', 'indemnification']",
                            "anchors": "['Section 8', 'Section 10.3']",
                            "query_keywords": "['breach consequences', 'indemnification procedures']",
                            "target_patterns": "['(clause)-[terminates_on_event]->(event)']",
                            "left_endpoint": "clause (Section 8)",
                            "right_endpoint": "clause (Section 10.3)",
                            "bridge_relation": "terminates_on_event",
                            "scope": "",
                        }
                    ],
                    "involved_types": {"nodes": [], "relations": [], "attributes": []},
                }
            )
        )

        result = decomposer.decompose("q", "schema.json")
        req = result["retrieval_requirements"][0]

        self.assertEqual(req["intent"], "bridge_lookup")
        self.assertEqual(req["entities"], ["Section 8"])
        self.assertEqual(req["anchors"], ["Section 8", "Section 10.3"])
        self.assertEqual(req["left_endpoint"], "Section 8")
        self.assertEqual(req["right_endpoint"], "Section 10.3")
        self.assertIn("what clause states termination trigger", req["target_patterns"])

    def test_graph_loader_falls_back_without_schema(self) -> None:
        relationships = [
            {
                "start_node": {"label": "entity", "properties": {"name": "A"}},
                "relation": "kw_filter_by",
                "end_node": {"label": "keyword", "properties": {"name": "term"}},
            },
            {
                "start_node": {"label": "entity", "properties": {"name": "A"}},
                "relation": "custom_relation",
                "end_node": {"label": "entity", "properties": {"name": "B"}},
            },
        ]

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as handle:
            json.dump(relationships, handle, ensure_ascii=False)
            graph_path = handle.name

        graph = graph_processor.load_graph_from_json(graph_path, use_edge_layer=True)
        layers = {(data["relation"], data.get("edge_layer")) for _, _, data in graph.edges(data=True)}

        self.assertIn(("kw_filter_by", "auxiliary"), layers)
        self.assertIn(("custom_relation", "semantic"), layers)

    def test_faiss_filter_skips_community_path_when_index_missing(self) -> None:
        method = _extract_class_method(
            YOUTU_ROOT / "models" / "retriever" / "faiss_filter.py",
            "DualFAISSRetriever",
            "retrieve_via_communities",
        )

        retriever = types.SimpleNamespace(comm_index=None)
        result = method(retriever, query_embed=object(), top_k=3)

        self.assertEqual(result, [])

    def test_faiss_filter_build_indices_treats_missing_comm_cache_as_optional(self) -> None:
        method = _extract_class_method(
            YOUTU_ROOT / "models" / "retriever" / "faiss_filter.py",
            "DualFAISSRetriever",
            "build_indices",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = Path(tmpdir) / "demo"
            dataset_dir.mkdir(parents=True, exist_ok=True)
            for filename in [
                "node.index",
                "relation.index",
                "triple.index",
                "node_embeddings.pt",
                "relation_embeddings.pt",
                "node_map.json",
            ]:
                (dataset_dir / filename).write_text("{}", encoding="utf-8")

            loaded = {"value": False}
            retriever = types.SimpleNamespace(
                cache_dir=tmpdir,
                dataset="demo",
                graph=types.SimpleNamespace(nodes=lambda: []),
                model_dim=384,
                _load_indices=lambda: loaded.__setitem__("value", True),
                load_embedding_cache=lambda: True,
                _preload_faiss_indices=lambda: None,
            )

            method(retriever)

            self.assertTrue(loaded["value"])


if __name__ == "__main__":
    unittest.main()
