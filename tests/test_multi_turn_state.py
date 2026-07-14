"""Tests for multi_turn_state templates."""

from __future__ import annotations

from prioritybench.generate import generate_one
from prioritybench.scoring import score_example
from prioritybench.templates import MULTI_TURN_STATE_TEMPLATES


def test_multi_turn_gold_scores_one():
    for tmpl in MULTI_TURN_STATE_TEMPLATES:
        ex = generate_one(tmpl, seed=3, context_length=8000)
        line = ex.scoring["required_slots"]["line"]
        assert score_example(ex, line) == 1.0
        assert score_example(ex, "wrong") == 0.0
