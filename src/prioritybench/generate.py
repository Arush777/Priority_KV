"""Deterministic PriorityBench-A example generator (seeds + templates)."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from prioritybench.schema import (
    CONTEXT_LENGTHS,
    Category,
    PriorityExample,
    Split,
    validate_example_shape,
)
from prioritybench.templates import (
    INSTRUCTION_SUPERSESSION_TEMPLATES_V1,
    INSTRUCTION_SUPERSESSION_TEMPLATES_V2,
    MULTI_TURN_STATE_TEMPLATES_V1,
    MULTI_TURN_STATE_TEMPLATES_V2,
    TEMPLATES_BY_ID,
    TOOL_SCHEMA_TEMPLATES,
)
from prioritybench.templates.base import TemplateSpec, messages_approx_tokens

# Stable master seed for the W1 40-example pilot (tool_schema first).
W1_MASTER_SEED = 20260714
W2_MASTER_SEED = 20260721
W2B_MASTER_SEED = 20260728
W2D_MASTER_SEED = 20260804

# Split fractions from plan §3.2 (calibration 40% / validation 20% / test 40%).
_SPLIT_THRESHOLDS = (
    (0.40, Split.CALIBRATION),
    (0.60, Split.VALIDATION),
    (1.00, Split.TEST),
)


def assign_split(example_id: str) -> Split:
    """Deterministic split assignment from example_id (not from content)."""
    # Stable across processes (do not use built-in hash(); PYTHONHASHSEED varies).
    digest = hashlib.md5(f"split::{example_id}".encode()).hexdigest()
    h = int(digest[:8], 16) % 10_000
    u = h / 10_000.0
    for thresh, split in _SPLIT_THRESHOLDS:
        if u < thresh:
            return split
    return Split.TEST


def _example_id(template_id: str, context_length: int, seed: int) -> str:
    return f"{template_id}__c{context_length}__s{seed}"


def generate_one(
    template: TemplateSpec,
    *,
    seed: int,
    context_length: int,
) -> PriorityExample:
    rng = random.Random(seed)
    messages, scoring = template.build(rng, context_length)
    eid = _example_id(template.template_id, context_length, seed)
    ex = PriorityExample(
        example_id=eid,
        category=template.category,
        split=assign_split(eid),
        context_length=context_length,
        template_id=template.template_id,
        seed=seed,
        messages=messages,
        scoring=dict(scoring),
        meta={
            "approx_tokens": messages_approx_tokens(messages),
            "generator": "prioritybench.generate",
            "master_seed_ref": W1_MASTER_SEED,
        },
    )
    err = validate_example_shape(ex)
    if err:
        raise ValueError(f"invalid example {eid}: {err}")
    return ex


def generate_tool_schema_pilot(
    n: int = 40,
    *,
    master_seed: int = W1_MASTER_SEED,
    context_lengths: Sequence[int] = CONTEXT_LENGTHS,
    templates: Sequence[TemplateSpec] = TOOL_SCHEMA_TEMPLATES,
) -> List[PriorityExample]:
    """W1 pilot: ``n`` tool_schema examples round-robin across templates × strata."""
    return _generate_round_robin(
        n,
        master_seed=master_seed,
        context_lengths=context_lengths,
        templates=templates,
    )


def generate_w2_mixed_pilot(
    n_tool: int = 80,
    n_supersession: int = 40,
    *,
    master_seed: int = W2_MASTER_SEED,
    context_lengths: Sequence[int] = CONTEXT_LENGTHS,
) -> List[PriorityExample]:
    """W2 growth toward ~145: more tool_schema + first supersession set."""
    tools = _generate_round_robin(
        n_tool,
        master_seed=master_seed,
        context_lengths=context_lengths,
        templates=TOOL_SCHEMA_TEMPLATES,
    )
    supers = _generate_round_robin(
        n_supersession,
        master_seed=master_seed + 10_000,
        context_lengths=context_lengths,
        templates=INSTRUCTION_SUPERSESSION_TEMPLATES_V1,
    )
    return tools + supers


def generate_w2b_pilot(
    n_tool: int = 80,
    n_supersession: int = 40,
    n_multi_turn: int = 25,
    *,
    master_seed: int = W2B_MASTER_SEED,
    context_lengths: Sequence[int] = CONTEXT_LENGTHS,
) -> List[PriorityExample]:
    """W2b: ~145 examples across all three categories (80+40+25). Uses v1 templates."""
    base = generate_w2_mixed_pilot(
        n_tool=n_tool,
        n_supersession=n_supersession,
        master_seed=master_seed,
        context_lengths=context_lengths,
    )
    multi = _generate_round_robin(
        n_multi_turn,
        master_seed=master_seed + 20_000,
        context_lengths=context_lengths,
        templates=MULTI_TURN_STATE_TEMPLATES_V1,
    )
    return base + multi


def generate_w2d_pilot(
    n_tool: int = 80,
    n_supersession: int = 40,
    n_multi_turn: int = 25,
    *,
    master_seed: int = W2D_MASTER_SEED,
    context_lengths: Sequence[int] = CONTEXT_LENGTHS,
) -> List[PriorityExample]:
    """W2d: same shape as w2b but non-leaking v2 supersession + multi_turn."""
    tools = _generate_round_robin(
        n_tool,
        master_seed=master_seed,
        context_lengths=context_lengths,
        templates=TOOL_SCHEMA_TEMPLATES,
    )
    supers = _generate_round_robin(
        n_supersession,
        master_seed=master_seed + 10_000,
        context_lengths=context_lengths,
        templates=INSTRUCTION_SUPERSESSION_TEMPLATES_V2,
    )
    multi = _generate_round_robin(
        n_multi_turn,
        master_seed=master_seed + 20_000,
        context_lengths=context_lengths,
        templates=MULTI_TURN_STATE_TEMPLATES_V2,
    )
    return tools + supers + multi


def _generate_round_robin(
    n: int,
    *,
    master_seed: int,
    context_lengths: Sequence[int],
    templates: Sequence[TemplateSpec],
) -> List[PriorityExample]:
    if n <= 0:
        return []
    if not templates:
        raise ValueError("no templates provided")
    lengths = list(context_lengths) or list(CONTEXT_LENGTHS)
    out: List[PriorityExample] = []
    for i in range(n):
        template = templates[i % len(templates)]
        ctx = lengths[i % len(lengths)]
        seed = master_seed + i * 17 + (i % 7)
        out.append(generate_one(template, seed=seed, context_length=ctx))
    return out


def write_jsonl(path: Path, examples: Iterable[PriorityExample]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex.to_dict(), ensure_ascii=False) + "\n")
            n += 1
    return n


def write_split_dirs(
    root: Path,
    examples: Sequence[PriorityExample],
) -> dict[str, int]:
    """Write examples into calibration/validation/test JSONL under ``root``."""
    buckets: dict[Split, List[PriorityExample]] = {
        Split.CALIBRATION: [],
        Split.VALIDATION: [],
        Split.TEST: [],
    }
    for ex in examples:
        buckets[ex.split].append(ex)
    counts: dict[str, int] = {}
    for split, items in buckets.items():
        counts[split.value] = write_jsonl(root / split.value / "examples.jsonl", items)
    return counts


def load_jsonl(path: Path) -> List[PriorityExample]:
    examples: List[PriorityExample] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            examples.append(PriorityExample.from_dict(json.loads(line)))
    return examples


def template_by_id(template_id: str) -> Optional[TemplateSpec]:
    return TEMPLATES_BY_ID.get(template_id)


def gold_tool_call(example: PriorityExample) -> str:
    """Build a perfect tool-call string from scoring consts (for smoke tests)."""
    scoring = example.scoring
    names = scoring.get("allowed_tool_names") or ["tool"]
    name = names[0]
    schema = scoring.get("expected_schema") or {}
    props = schema.get("properties") or {}
    args = {}
    for key, sub in props.items():
        if "const" in sub:
            args[key] = sub["const"]
        elif "enum" in sub:
            args[key] = sub["enum"][0]
        else:
            # Fallback placeholders for optional non-const fields.
            t = sub.get("type")
            args[key] = "" if t == "string" else 0
    return json.dumps({"name": name, "arguments": args})
