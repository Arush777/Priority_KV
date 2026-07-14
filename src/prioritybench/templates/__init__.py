"""PriorityBench-A template registry (W1+)."""

from __future__ import annotations

from prioritybench.templates.base import TemplateSpec
from prioritybench.templates.tool_schema import TOOL_SCHEMA_TEMPLATES

TEMPLATES_BY_ID: dict[str, TemplateSpec] = {
    t.template_id: t for t in TOOL_SCHEMA_TEMPLATES
}

__all__ = ["TEMPLATES_BY_ID", "TOOL_SCHEMA_TEMPLATES", "TemplateSpec"]
