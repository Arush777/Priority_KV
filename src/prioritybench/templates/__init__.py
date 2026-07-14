"""PriorityBench-A template registry (W1+)."""

from __future__ import annotations

from prioritybench.templates.base import TemplateSpec
from prioritybench.templates.instruction_supersession import (
    INSTRUCTION_SUPERSESSION_TEMPLATES,
    INSTRUCTION_SUPERSESSION_TEMPLATES_V1,
    INSTRUCTION_SUPERSESSION_TEMPLATES_V2,
)
from prioritybench.templates.multi_turn_state import (
    MULTI_TURN_STATE_TEMPLATES,
    MULTI_TURN_STATE_TEMPLATES_V1,
    MULTI_TURN_STATE_TEMPLATES_V2,
)
from prioritybench.templates.tool_schema import TOOL_SCHEMA_TEMPLATES

ALL_TEMPLATES: tuple[TemplateSpec, ...] = (
    TOOL_SCHEMA_TEMPLATES
    + INSTRUCTION_SUPERSESSION_TEMPLATES_V1
    + INSTRUCTION_SUPERSESSION_TEMPLATES_V2
    + MULTI_TURN_STATE_TEMPLATES_V1
    + MULTI_TURN_STATE_TEMPLATES_V2
)

TEMPLATES_BY_ID: dict[str, TemplateSpec] = {t.template_id: t for t in ALL_TEMPLATES}

__all__ = [
    "ALL_TEMPLATES",
    "TEMPLATES_BY_ID",
    "TOOL_SCHEMA_TEMPLATES",
    "INSTRUCTION_SUPERSESSION_TEMPLATES",
    "INSTRUCTION_SUPERSESSION_TEMPLATES_V1",
    "INSTRUCTION_SUPERSESSION_TEMPLATES_V2",
    "MULTI_TURN_STATE_TEMPLATES",
    "MULTI_TURN_STATE_TEMPLATES_V1",
    "MULTI_TURN_STATE_TEMPLATES_V2",
    "TemplateSpec",
]
