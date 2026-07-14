"""SnapKV baseline hook (W2 start).

Reproduction target: matched-byte eviction baseline Q3 in the plan.
This module is a scaffold — full KVPress/SnapKV wiring comes next; do not
claim SnapKV numbers until the reproduction harness is green.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SnapKVConfig:
    """Matched-byte SnapKV settings (placeholders until KVPress path lands)."""

    # Fraction of FullKV bytes to match (same budgets as PriorityKV).
    budget_frac: float = 0.50
    # Window / observation sizes follow common SnapKV defaults; lock after repro.
    window_size: int = 32
    max_capacity_prompt: int = 256
    kernel_size: int = 5
    pooling: str = "avgpool"


def status() -> dict:
    return {
        "baseline_id": "Q3",
        "name": "SnapKV",
        "implemented": False,
        "config": SnapKVConfig().__dict__,
        "next": "wire KVPress SnapKV on Qwen3-8B; reproduce pub numbers ≤4 days",
    }
