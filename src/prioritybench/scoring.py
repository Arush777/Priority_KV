"""Deterministic scorers for PriorityBench-A (no LLM-as-judge).

Each scorer returns a float in [0, 1]. Near-miss adversarial cases belong in
unit tests (plan §3.2: ≥3 scorer tests per category).
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Optional

from prioritybench.schema import Category, PriorityExample


def _as_obj(text: str) -> Any:
    """Parse model output as JSON if possible; otherwise return raw string."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Tolerate a trailing fence-less JSON object embedded in prose.
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass
        return text


def score_tool_schema(output: str, scoring: Mapping[str, Any]) -> float:
    """JSON / tool-call validity against an expected JSON Schema fragment.

    Expected ``scoring`` keys:
      - expected_schema: dict (JSON Schema for the tool arguments or full call)
      - path: optional dotted path under the parsed object to validate
        (default: whole object). Example: ``arguments``.
      - required_fields: optional list of field names that must be present
        at ``path`` (lightweight check when full draft-schema isn't needed).
      - allowed_tool_names: optional list; if present, output.tool or
        output.name must be in this set.
    """
    expected = scoring.get("expected_schema")
    required = list(scoring.get("required_fields") or [])
    allowed = scoring.get("allowed_tool_names")
    path = scoring.get("path")

    obj = _as_obj(output)
    if not isinstance(obj, dict):
        return 0.0

    if allowed is not None:
        tool_name = obj.get("name") or obj.get("tool") or obj.get("tool_name")
        if tool_name not in set(allowed):
            return 0.0

    target: Any = obj
    if path:
        for part in str(path).split("."):
            if not isinstance(target, dict) or part not in target:
                return 0.0
            target = target[part]

    if required:
        if not isinstance(target, dict):
            return 0.0
        if any(f not in target for f in required):
            return 0.0

    if expected is not None:
        # Minimal subset check: type + required + enum (no external jsonschema dep).
        ok = _validate_schema_subset(target, expected)
        if not ok:
            return 0.0

    return 1.0


def _validate_schema_subset(value: Any, schema: Mapping[str, Any]) -> bool:
    """Tiny JSON-Schema subset: type, required, properties, enum, const."""
    if "const" in schema and value != schema["const"]:
        return False
    if "enum" in schema and value not in schema["enum"]:
        return False

    schema_type = schema.get("type")
    if schema_type == "object":
        if not isinstance(value, dict):
            return False
        for key in schema.get("required") or []:
            if key not in value:
                return False
        props = schema.get("properties") or {}
        for key, sub in props.items():
            if key in value and not _validate_schema_subset(value[key], sub):
                return False
        return True
    if schema_type == "array":
        if not isinstance(value, list):
            return False
        item_schema = schema.get("items")
        if item_schema is not None:
            return all(_validate_schema_subset(v, item_schema) for v in value)
        return True
    if schema_type == "string":
        return isinstance(value, str)
    if schema_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if schema_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if schema_type == "boolean":
        return isinstance(value, bool)
    if schema_type == "null":
        return value is None
    # No type declared: accept.
    return True


def score_instruction_supersession(output: str, scoring: Mapping[str, Any]) -> float:
    """Latest-constraint check via regex (deterministic; no LLM judge).

    Expected ``scoring`` keys:
      - constraint_pattern: regex that *must* match the final output
      - forbidden_pattern: optional regex that must *not* match (old constraint)
      - flags: optional list of re flag names, e.g. [\"IGNORECASE\", \"DOTALL\"]
    """
    pattern = scoring.get("constraint_pattern")
    if not pattern:
        raise ValueError("constraint_pattern required for instruction_supersession")
    forbidden = scoring.get("forbidden_pattern")
    flags = 0
    for name in scoring.get("flags") or []:
        flags |= getattr(re, str(name))
    if not re.search(str(pattern), output, flags):
        return 0.0
    if forbidden and re.search(str(forbidden), output, flags):
        return 0.0
    return 1.0


def score_multi_turn_state(output: str, scoring: Mapping[str, Any]) -> float:
    """Exact-match slot extraction: every required slot string must appear.

    Expected ``scoring`` keys:
      - required_slots: mapping slot_name -> exact string value
      - mode: \"all\" (default, fraction of slots found) or \"strict\" (0/1)
    """
    slots = scoring.get("required_slots") or {}
    if not isinstance(slots, Mapping) or not slots:
        raise ValueError("required_slots required for multi_turn_state")
    hits = sum(1 for v in slots.values() if str(v) in output)
    mode = scoring.get("mode", "all")
    if mode == "strict":
        return 1.0 if hits == len(slots) else 0.0
    return hits / len(slots)


def score_example(example: PriorityExample, model_output: str) -> float:
    """Dispatch to the category scorer."""
    if example.category == Category.TOOL_SCHEMA:
        return score_tool_schema(model_output, example.scoring)
    if example.category == Category.INSTRUCTION_SUPERSESSION:
        return score_instruction_supersession(model_output, example.scoring)
    if example.category == Category.MULTI_TURN_STATE:
        return score_multi_turn_state(model_output, example.scoring)
    raise ValueError(f"unknown category: {example.category}")


def explain_score(example: PriorityExample, model_output: str) -> dict:
    """Return score plus a tiny debug payload for audits."""
    score = score_example(example, model_output)
    return {
        "example_id": example.example_id,
        "category": example.category.value,
        "score": score,
        "output_preview": model_output[:200],
    }
