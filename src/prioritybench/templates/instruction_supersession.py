"""instruction_supersession templates: follow the *latest* constraint.

v1: FINAL restated the latest tag/language (weak stress).
v2: FINAL does not restate — model must recall the latest constraint.
"""

from __future__ import annotations

import random
import re
from typing import Any, Dict, List, Mapping, Tuple

from prioritybench.schema import Category
from prioritybench.templates.base import TemplateSpec, pad_with_filler_turns

FORMATS: tuple[str, ...] = ("ALL_CAPS", "snake_case", "quoted", "bullet")


def _fmt_tag(name: str) -> str:
    return f"[[FMT:{name}]]"


def _build_format_flip_v1(
    rng: random.Random, target_tokens: int
) -> Tuple[List[Dict[str, str]], Mapping[str, Any]]:
    first = rng.choice(FORMATS)
    rest = [f for f in FORMATS if f != first]
    second = rng.choice(rest)
    topic = rng.choice(["status report", "deploy notes", "cache summary"])
    tag_new = _fmt_tag(second)
    tag_old = _fmt_tag(first)

    messages: List[Dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "You answer user questions. Obey formatting constraints exactly. "
                "If a constraint is updated, follow the latest one only. "
                "When asked for a FINAL answer, the first token sequence must be "
                "the format tag [[FMT:...]] then a single sentence."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Constraint v1: Start FINAL answers with {tag_old} then one "
                f"sentence about {topic}."
            ),
        },
        {"role": "assistant", "content": f"Understood. I will start with {tag_old}."},
        {
            "role": "user",
            "content": (
                f"Constraint v2 (REPLACES v1): Ignore {tag_old}. "
                f"Start FINAL answers with {tag_new} then one sentence about {topic}."
            ),
        },
        {
            "role": "assistant",
            "content": f"Updated. FINAL answers will start with {tag_new} only.",
        },
        {
            "role": "user",
            "content": (
                f"FINAL: Emit {tag_new} then one sentence on {topic}. "
                f"Do not emit {tag_old}."
            ),
        },
    ]
    messages = pad_with_filler_turns(messages, rng, target_tokens)
    scoring = {
        "constraint_pattern": re.escape(tag_new),
        "forbidden_pattern": re.escape(tag_old),
        "latest_constraint": second,
        "revoked_constraint": first,
        "flags": [],
    }
    return messages, scoring


def _build_language_flip_v1(
    rng: random.Random, target_tokens: int
) -> Tuple[List[Dict[str, str]], Mapping[str, Any]]:
    langs = [("English", "en"), ("Spanish", "es"), ("French", "fr")]
    a, b = rng.sample(langs, 2)
    word = rng.choice(["alpha", "bravo", "charlie"])
    messages: List[Dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "Follow the latest language constraint for the final answer. "
                "Always include the codename token exactly once."
            ),
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
                f"Still include the codename {word} exactly."
            ),
        },
        {"role": "assistant", "content": f"Language updated to {b[0]}."},
        {
            "role": "user",
            "content": f"FINAL: One short sentence in {b[0]} that includes {word}.",
        },
    ]
    messages = pad_with_filler_turns(messages, rng, target_tokens)
    scoring = {
        "constraint_pattern": re.escape(word),
        "forbidden_pattern": None,
        "latest_language": b[0],
        "flags": ["IGNORECASE"],
    }
    return messages, scoring


def _build_format_flip_v2(
    rng: random.Random, target_tokens: int
) -> Tuple[List[Dict[str, str]], Mapping[str, Any]]:
    first = rng.choice(FORMATS)
    rest = [f for f in FORMATS if f != first]
    second = rng.choice(rest)
    topic = rng.choice(["status report", "deploy notes", "cache summary"])
    tag_new = _fmt_tag(second)
    tag_old = _fmt_tag(first)

    messages: List[Dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "You answer user questions. Obey formatting constraints exactly. "
                "If a constraint is updated, follow the latest one only. "
                "When asked for a FINAL answer, the first token sequence must be "
                "the format tag [[FMT:...]] then a single sentence."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Constraint v1: Start FINAL answers with {tag_old} then one "
                f"sentence about {topic}."
            ),
        },
        {"role": "assistant", "content": f"Understood. I will start with {tag_old}."},
        {
            "role": "user",
            "content": (
                f"Constraint v2 (REPLACES v1): Ignore {tag_old}. "
                f"Start FINAL answers with {tag_new} then one sentence about {topic}."
            ),
        },
        {
            "role": "assistant",
            "content": f"Updated. FINAL answers will start with {tag_new} only.",
        },
        {
            "role": "user",
            "content": (
                "FINAL: Answer now under the *latest* formatting constraint "
                "only (emit the correct [[FMT:...]] tag first, then one "
                f"sentence about {topic}). Do not use revoked tags."
            ),
        },
    ]
    messages = pad_with_filler_turns(messages, rng, target_tokens)
    scoring = {
        "constraint_pattern": re.escape(tag_new),
        "forbidden_pattern": re.escape(tag_old),
        "latest_constraint": second,
        "revoked_constraint": first,
        "flags": [],
    }
    return messages, scoring


def _build_language_flip_v2(
    rng: random.Random, target_tokens: int
) -> Tuple[List[Dict[str, str]], Mapping[str, Any]]:
    langs = [("English", "en"), ("Spanish", "es"), ("French", "fr")]
    a, b = rng.sample(langs, 2)
    word = rng.choice(["alpha", "bravo", "charlie"])
    messages: List[Dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "Follow the latest language constraint for the final answer. "
                "Always include the codename token exactly once."
            ),
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
                f"Still include the codename {word} exactly."
            ),
        },
        {"role": "assistant", "content": f"Language updated to {b[0]}."},
        {
            "role": "user",
            "content": (
                "FINAL: One short sentence in the currently required language, "
                "including the stored codename exactly once."
            ),
        },
    ]
    messages = pad_with_filler_turns(messages, rng, target_tokens)
    scoring = {
        "constraint_pattern": re.escape(word),
        "forbidden_pattern": None,
        "latest_language": b[0],
        "flags": ["IGNORECASE"],
    }
    return messages, scoring


INSTRUCTION_SUPERSESSION_TEMPLATES_V1: tuple[TemplateSpec, ...] = (
    TemplateSpec(
        "instruction_supersession.format_flip.v1",
        Category.INSTRUCTION_SUPERSESSION,
        _build_format_flip_v1,
    ),
    TemplateSpec(
        "instruction_supersession.language_flip.v1",
        Category.INSTRUCTION_SUPERSESSION,
        _build_language_flip_v1,
    ),
)

INSTRUCTION_SUPERSESSION_TEMPLATES_V2: tuple[TemplateSpec, ...] = (
    TemplateSpec(
        "instruction_supersession.format_flip.v2",
        Category.INSTRUCTION_SUPERSESSION,
        _build_format_flip_v2,
    ),
    TemplateSpec(
        "instruction_supersession.language_flip.v2",
        Category.INSTRUCTION_SUPERSESSION,
        _build_language_flip_v2,
    ),
)

# New pilots use v2; v1 stays registered for old manifests.
INSTRUCTION_SUPERSESSION_TEMPLATES: tuple[TemplateSpec, ...] = (
    INSTRUCTION_SUPERSESSION_TEMPLATES_V2
)
