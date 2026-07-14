"""Tests for fixed supersession format tags."""

from __future__ import annotations

from prioritybench.generate import generate_one
from prioritybench.scoring import score_example
from prioritybench.templates import INSTRUCTION_SUPERSESSION_TEMPLATES


def test_format_flip_gold_tag_scores_one():
    tmpl = next(
        t
        for t in INSTRUCTION_SUPERSESSION_TEMPLATES
        if "format_flip" in t.template_id
    )
    ex = generate_one(tmpl, seed=7, context_length=8000)
    latest = ex.scoring["latest_constraint"]
    revoked = ex.scoring["revoked_constraint"]
    good = f"[[FMT:{latest}]] Status is green."
    bad_old = f"[[FMT:{revoked}]] Status is green."
    assert score_example(ex, good) == 1.0
    assert score_example(ex, bad_old) == 0.0
    assert score_example(ex, "Status is green.") == 0.0
