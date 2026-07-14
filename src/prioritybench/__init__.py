"""PriorityBench-A: agent-reliability eval scaffolding (Workstream A / S1).

Canon: docs/PRIORITYKV_IMPLEMENTATION_PLAN.md §3.
"""

from prioritybench.schema import (
    Category,
    PriorityExample,
    Split,
    CATEGORIES,
)
from prioritybench.scoring import score_example

__all__ = [
    "CATEGORIES",
    "Category",
    "PriorityExample",
    "Split",
    "score_example",
]
