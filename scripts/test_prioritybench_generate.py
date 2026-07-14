"""Tests for PriorityBench-A W1 generator + tool_schema templates."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from prioritybench.generate import (  # noqa: E402
    W1_MASTER_SEED,
    assign_split,
    generate_one,
    generate_tool_schema_pilot,
    gold_tool_call,
    load_jsonl,
)
from prioritybench.schema import Category, CONTEXT_LENGTHS, validate_example_shape
from prioritybench.scoring import score_example
from prioritybench.templates import TOOL_SCHEMA_TEMPLATES


def test_assign_split_stable():
    a = assign_split("tool_schema.search_docs.v1__c8000__s1")
    b = assign_split("tool_schema.search_docs.v1__c8000__s1")
    assert a == b


def test_pilot_count_and_categories():
    examples = generate_tool_schema_pilot(40, master_seed=W1_MASTER_SEED)
    assert len(examples) == 40
    assert all(ex.category == Category.TOOL_SCHEMA for ex in examples)
    assert {ex.context_length for ex in examples} == set(CONTEXT_LENGTHS)
    assert len({ex.template_id for ex in examples}) == len(TOOL_SCHEMA_TEMPLATES)


def test_pilot_deterministic():
    a = generate_tool_schema_pilot(12, master_seed=W1_MASTER_SEED)
    b = generate_tool_schema_pilot(12, master_seed=W1_MASTER_SEED)
    assert [ex.to_dict() for ex in a] == [ex.to_dict() for ex in b]


def test_gold_calls_score_one():
    examples = generate_tool_schema_pilot(8, master_seed=W1_MASTER_SEED)
    for ex in examples:
        err = validate_example_shape(ex)
        assert err is None, err
        gold = gold_tool_call(ex)
        assert score_example(ex, gold) == 1.0


def test_each_template_builds():
    for t in TOOL_SCHEMA_TEMPLATES:
        ex = generate_one(t, seed=99, context_length=8_000)
        assert ex.template_id == t.template_id
        assert len(ex.messages) >= 4
        assert score_example(ex, gold_tool_call(ex)) == 1.0


def test_fixture_if_present():
    path = ROOT / "data" / "prioritybench" / "fixtures" / "tool_schema_smoke.jsonl"
    if not path.exists():
        return
    examples = load_jsonl(path)
    assert len(examples) >= 1
    for ex in examples:
        assert score_example(ex, gold_tool_call(ex)) == 1.0
        # Round-trip JSONL shape.
        assert Category(ex.category.value)


def test_near_miss_wrong_tool_still_zero():
    ex = generate_tool_schema_pilot(1, master_seed=W1_MASTER_SEED)[0]
    bad = json.loads(gold_tool_call(ex))
    bad["name"] = "not_a_real_tool"
    assert score_example(ex, json.dumps(bad)) == 0.0


if __name__ == "__main__":
    test_assign_split_stable()
    test_pilot_count_and_categories()
    test_pilot_deterministic()
    test_gold_calls_score_one()
    test_each_template_builds()
    test_fixture_if_present()
    test_near_miss_wrong_tool_still_zero()
    print("ok")
