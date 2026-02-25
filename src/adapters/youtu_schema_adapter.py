from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def schema_sha256(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def adapt_schema_for_youtu(schema: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    # Pass-through if schema already looks like youtu seed schema.
    if "entities" in schema and "relations" in schema and ("entity_types" in schema or "triples_schema" in schema):
        meta = {
            "adapter_mode": "passthrough",
            "entity_count": len(schema.get("entity_types") or schema.get("entities") or []),
            "relation_count": len(schema.get("relations") or []),
        }
        return schema, meta

    entity_types = [str(x).strip() for x in (schema.get("entity_types") or []) if str(x).strip()]
    relation_rows = schema.get("relations") or []
    if not isinstance(relation_rows, list):
        raise ValueError("Invalid schema: `relations` must be a list")

    entities = [
        {
            "name": et,
            "entity_type": et,
        }
        for et in entity_types
    ]

    adapted_relations: list[dict[str, Any]] = []
    triples_schema: list[dict[str, Any]] = []
    relation_constraints: dict[str, dict[str, list[str]]] = {}
    relation_types: list[str] = []

    for row in relation_rows:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name", "")).strip()
        if not name:
            continue
        subject_types = [str(x).strip() for x in (row.get("subject_types") or []) if str(x).strip()]
        object_types = [str(x).strip() for x in (row.get("object_types") or []) if str(x).strip()]
        aliases = [str(x).strip() for x in (row.get("aliases") or []) if str(x).strip()]
        relation_types.append(name)
        relation_constraints[name] = {
            "subject_types": subject_types,
            "object_types": object_types,
        }
        adapted_relations.append(
            {
                "name": name,
                "relation": name,
                "aliases": aliases,
                "subject_types": subject_types,
                "object_types": object_types,
                "head_types": subject_types,
                "tail_types": object_types,
            }
        )
        for st in subject_types:
            for ot in object_types:
                triples_schema.append(
                    {
                        "subject_type": st,
                        "relation": name,
                        "predicate": name,
                        "object_type": ot,
                    }
                )

    adapted = {
        "schema_version": "enterprise_to_youtu_v1",
        "entity_types": entity_types,
        "entities": entities,
        "relation_types": relation_types,
        "relations": adapted_relations,
        "relation_constraints": relation_constraints,
        "triples_schema": triples_schema,
    }
    meta = {
        "adapter_mode": "enterprise_to_youtu_v1",
        "entity_count": len(entity_types),
        "relation_count": len(adapted_relations),
        "triple_pattern_count": len(triples_schema),
    }
    return adapted, meta


def load_and_adapt_schema(schema_file: str | None) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    raw = str(schema_file or "").strip()
    if not raw or raw.lower() in {"none", "null", "off", "false"}:
        return None, {"enabled": False, "reason": "disabled"}

    p = Path(raw)
    if not p.exists():
        raise FileNotFoundError(f"Youtu schema file not found: {p}")
    loaded = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Youtu schema must be a JSON object: {p}")

    adapted, adapt_meta = adapt_schema_for_youtu(loaded)
    return adapted, {
        "enabled": True,
        "schema_file": str(p),
        "schema_sha256": schema_sha256(adapted),
        **adapt_meta,
    }
