"""Example schema for PriorityBench-A (240 examples, 3 categories).

Locked construction rules live in docs/PRIORITYKV_IMPLEMENTATION_PLAN.md §3.2.
This module only defines the on-disk / in-memory contract for generators and scorers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional


class Category(str, Enum):
    """§3.1 categories (80 examples each)."""

    TOOL_SCHEMA = "tool_schema"
    INSTRUCTION_SUPERSESSION = "instruction_supersession"
    MULTI_TURN_STATE = "multi_turn_state"


CATEGORIES: tuple[Category, ...] = (
    Category.TOOL_SCHEMA,
    Category.INSTRUCTION_SUPERSESSION,
    Category.MULTI_TURN_STATE,
)


class Split(str, Enum):
    """§3.2 splits: 40% calibration / 20% validation / 40% locked test."""

    CALIBRATION = "calibration"
    VALIDATION = "validation"
    TEST = "test"


# Context-length strata from the plan (tokens).
CONTEXT_LENGTHS: tuple[int, ...] = (8_000, 16_000, 32_000)


@dataclass(frozen=True)
class PriorityExample:
    """One programmatically scored PriorityBench-A item.

    ``messages`` is a chat-style trace (role/content dicts). Scoring fields are
    category-specific and ignored by other scorers:

    - tool_schema: ``expected_schema`` (JSON Schema for the tool call),
      ``allowed_tool_names`` (optional)
    - instruction_supersession: ``constraint_pattern`` (regex that must match
      the *final* assistant output), ``forbidden_pattern`` (optional)
    - multi_turn_state: ``required_slots`` (exact string values that must
      appear in the final output)
    """

    example_id: str
    category: Category
    split: Split
    context_length: int
    template_id: str
    seed: int
    messages: List[Dict[str, str]]
    # Category-specific scoring payload (kept loosely typed for template variety).
    scoring: Mapping[str, Any] = field(default_factory=dict)
    meta: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["category"] = self.category.value
        d["split"] = self.split.value
        return d

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PriorityExample":
        return cls(
            example_id=str(data["example_id"]),
            category=Category(data["category"]),
            split=Split(data["split"]),
            context_length=int(data["context_length"]),
            template_id=str(data["template_id"]),
            seed=int(data["seed"]),
            messages=list(data.get("messages") or []),
            scoring=dict(data.get("scoring") or {}),
            meta=dict(data.get("meta") or {}),
        )


def validate_example_shape(ex: PriorityExample) -> Optional[str]:
    """Return an error string if the example violates the contract, else None."""
    if ex.context_length not in CONTEXT_LENGTHS:
        return f"context_length {ex.context_length} not in {CONTEXT_LENGTHS}"
    if ex.category not in CATEGORIES:
        return f"unknown category {ex.category}"
    if not ex.example_id:
        return "example_id empty"
    if not ex.template_id:
        return "template_id empty"
    return None
