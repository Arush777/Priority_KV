"""PriorityBench-A: agent-reliability eval scaffolding (Workstream A / S1).

The frozen dataset contract is documented in docs/DATASET.md.
"""

from prioritybench.generate import generate_tool_schema_pilot
from prioritybench.pins import (
    QWEN3_8B_MODEL_ID,
    QWEN3_8B_REVISION,
    QWEN3_ENABLE_THINKING,
)
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
    "QWEN3_8B_MODEL_ID",
    "QWEN3_8B_REVISION",
    "QWEN3_ENABLE_THINKING",
    "Split",
    "generate_tool_schema_pilot",
    "score_example",
]
