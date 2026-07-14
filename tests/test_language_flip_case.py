"""Case-insensitive language_flip scoring."""

from __future__ import annotations

from prioritybench.generate import generate_one
from prioritybench.scoring import score_example
from prioritybench.templates import INSTRUCTION_SUPERSESSION_TEMPLATES


def test_language_flip_accepts_capitalized_codename():
    tmpl = next(
        t for t in INSTRUCTION_SUPERSESSION_TEMPLATES if "language_flip" in t.template_id
    )
    ex = generate_one(tmpl, seed=11, context_length=8000)
    # Extract the expected token from the pattern (escaped word).
    import re

    raw = ex.scoring["constraint_pattern"]
    word = re.sub(r"\\(.)", r"\1", raw)
    assert score_example(ex, f"Short reply with {word}.") == 1.0
    assert score_example(ex, f"Short reply with {word.capitalize()}.") == 1.0
