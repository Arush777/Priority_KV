"""instruction_supersession templates: follow the *latest* constraint."""

from __future__ import annotations

import random
from typing import Any, Dict, List, Mapping, Tuple

from prioritybench.schema import Category
from prioritybench.templates.base import TemplateSpec, pad_with_filler_turns

FORMATS: tuple[str, ...] = ("ALL_CAPS", "snake_case", "quoted", "bullet")


def _build_format_flip(
    rng: random.Random, target_tokens: int
) -> Tuple[List[Dict[str, str]], Mapping[str, Any]]:
    first = rng.choice(FORMATS)
    rest = [f for f in FORMATS if f != first]
    second = rng.choice(rest)
    topic = rng.choice(["status report", "deploy notes", "cache summary"])

    messages: List[Dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "You answer user questions. Obey formatting constraints exactly. "
                "If a constraint is updated, follow the latest one only."
            ),
        },
        {
            "role": "user",
            "content": f"Constraint v1: Write your final answer in {first} about {topic}.",
        },
        {
            "role": "assistant",
            "content": f"Understood. I will use {first}.",
        },
        {
            "role": "user",
            "content": (
                f"Constraint v2 (REPLACES v1): Ignore the previous format. "
                f"Write the final answer in {second} about {topic}."
            ),
        },
        {
            "role": "assistant",
            "content": f"Updated. Final answers will use {second} only.",
        },
        {
            "role": "user",
            "content": f"FINAL: Give a one-sentence {topic} using the latest constraint.",
        },
    ]
    messages = pad_with_filler_turns(messages, rng, target_tokens)

    # Deterministic checkers: must mention the new format token; must not praise old.
    scoring = {
        "constraint_pattern": second.replace("_", "[_ ]?"),
        "forbidden_pattern": first.replace("_", "[_ ]?"),
        "latest_constraint": second,
        "revoked_constraint": first,
    }
    return messages, scoring


def _build_language_flip(
    rng: random.Random, target_tokens: int
) -> Tuple[List[Dict[str, str]], Mapping[str, Any]]:
    langs = [("English", "en"), ("Spanish", "es"), ("French", "fr")]
    a, b = rng.sample(langs, 2)
    word = rng.choice(["alpha", "bravo", "charlie"])
    messages: List[Dict[str, str]] = [
        {
            "role": "system",
            "content": "Follow the latest language constraint for the final answer.",
        },
        {
            "role": "user",
            "content": f"Constraint v1: Answer only in {a[0]}. Codename is {word}.",
        },
        {"role": "assistant", "content": f"Language set to {a[0]}."},
        {
            "role": "user",
            "content": (
                f"Constraint v2: Revoke prior language. Answer only in {b[0]}. "
                f"Still include the codename {word}."
            ),
        },
        {"role": "assistant", "content": f"Language updated to {b[0]}."},
        {
            "role": "user",
            "content": "FINAL: Reply with one short sentence under the latest constraint.",
        },
    ]
    messages = pad_with_filler_turns(messages, rng, target_tokens)
    scoring = {
        "constraint_pattern": word,
        "forbidden_pattern": None,
        "latest_language": b[0],
        "required_slots": [word],
    }
    return messages, scoring


INSTRUCTION_SUPERSESSION_TEMPLATES: tuple[TemplateSpec, ...] = (
    TemplateSpec(
        "instruction_supersession.format_flip.v1",
        Category.INSTRUCTION_SUPERSESSION,
        _build_format_flip,
    ),
    TemplateSpec(
        "instruction_supersession.language_flip.v1",
        Category.INSTRUCTION_SUPERSESSION,
        _build_language_flip,
    ),
)
