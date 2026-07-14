"""Unit tests for PriorityBench-A scorers (≥3 per category, incl. near-misses)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from prioritybench.schema import (  # noqa: E402
    Category,
    PriorityExample,
    Split,
    validate_example_shape,
)
from prioritybench.scoring import score_example  # noqa: E402


def _ex(category: Category, scoring: dict, **kwargs) -> PriorityExample:
    base = dict(
        example_id="ex-test",
        category=category,
        split=Split.CALIBRATION,
        context_length=8_000,
        template_id="t0",
        seed=0,
        messages=[],
        scoring=scoring,
    )
    base.update(kwargs)
    return PriorityExample(**base)


# --- tool_schema ---


def test_tool_schema_valid_call():
    ex = _ex(
        Category.TOOL_SCHEMA,
        {
            "allowed_tool_names": ["search"],
            "path": "arguments",
            "expected_schema": {
                "type": "object",
                "required": ["query"],
                "properties": {"query": {"type": "string"}},
            },
        },
    )
    out = json.dumps({"name": "search", "arguments": {"query": "kv cache"}})
    assert score_example(ex, out) == 1.0


def test_tool_schema_wrong_tool_name():
    ex = _ex(
        Category.TOOL_SCHEMA,
        {"allowed_tool_names": ["search"], "required_fields": ["query"], "path": "arguments"},
    )
    out = json.dumps({"name": "browse", "arguments": {"query": "x"}})
    assert score_example(ex, out) == 0.0


def test_tool_schema_near_miss_missing_required_field():
    ex = _ex(
        Category.TOOL_SCHEMA,
        {
            "allowed_tool_names": ["search"],
            "path": "arguments",
            "expected_schema": {
                "type": "object",
                "required": ["query", "limit"],
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer"},
                },
            },
        },
    )
    # Near-miss: looks plausible but omits required `limit`.
    out = json.dumps({"name": "search", "arguments": {"query": "x"}})
    assert score_example(ex, out) == 0.0


def test_tool_schema_enum_violation():
    ex = _ex(
        Category.TOOL_SCHEMA,
        {
            "path": "arguments",
            "expected_schema": {
                "type": "object",
                "required": ["mode"],
                "properties": {"mode": {"type": "string", "enum": ["fast", "thorough"]}},
            },
        },
    )
    out = json.dumps({"arguments": {"mode": "speedy"}})
    assert score_example(ex, out) == 0.0


# --- instruction_supersession ---


def test_supersession_follows_latest():
    ex = _ex(
        Category.INSTRUCTION_SUPERSESSION,
        {
            "constraint_pattern": r"ANSWER:\s*blue",
            "forbidden_pattern": r"ANSWER:\s*red",
        },
    )
    assert score_example(ex, "Thinking...\nANSWER: blue") == 1.0


def test_supersession_stale_constraint():
    ex = _ex(
        Category.INSTRUCTION_SUPERSESSION,
        {
            "constraint_pattern": r"ANSWER:\s*blue",
            "forbidden_pattern": r"ANSWER:\s*red",
        },
    )
    # Near-miss: obeys the revoked (red) rule.
    assert score_example(ex, "ANSWER: red") == 0.0


def test_supersession_both_present_fails():
    ex = _ex(
        Category.INSTRUCTION_SUPERSESSION,
        {
            "constraint_pattern": r"use-metric=ndcg",
            "forbidden_pattern": r"use-metric=recall",
        },
    )
    assert score_example(ex, "use-metric=ndcg and also use-metric=recall") == 0.0


# --- multi_turn_state ---


def test_multi_turn_all_slots():
    ex = _ex(
        Category.MULTI_TURN_STATE,
        {"required_slots": {"order_id": "ORD-42", "path": "/tmp/a.txt"}, "mode": "all"},
    )
    out = "Resume ORD-42 from /tmp/a.txt please"
    assert score_example(ex, out) == 1.0


def test_multi_turn_partial_credit():
    ex = _ex(
        Category.MULTI_TURN_STATE,
        {"required_slots": {"order_id": "ORD-42", "path": "/tmp/a.txt"}, "mode": "all"},
    )
    # Near-miss: remembers order id, drops path.
    assert score_example(ex, "Continue with ORD-42") == 0.5


def test_multi_turn_strict_requires_all():
    ex = _ex(
        Category.MULTI_TURN_STATE,
        {"required_slots": {"a": "AA", "b": "BB"}, "mode": "strict"},
    )
    assert score_example(ex, "AA only") == 0.0
    assert score_example(ex, "AA and BB") == 1.0


def test_validate_example_shape_rejects_bad_context():
    ex = _ex(Category.TOOL_SCHEMA, {}, context_length=4_000)
    assert validate_example_shape(ex) is not None


if __name__ == "__main__":
    test_tool_schema_valid_call()
    test_tool_schema_wrong_tool_name()
    test_tool_schema_near_miss_missing_required_field()
    test_tool_schema_enum_violation()
    test_supersession_follows_latest()
    test_supersession_stale_constraint()
    test_supersession_both_present_fails()
    test_multi_turn_all_slots()
    test_multi_turn_partial_credit()
    test_multi_turn_strict_requires_all()
    test_validate_example_shape_rejects_bad_context()
    print("ok")
