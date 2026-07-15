"""SnapKV baseline hook (Q3).

Reproduction target: matched-byte eviction baseline in the plan.
Prefers NVIDIA kvpress.SnapKVPress when installed; otherwise remains a stub.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class SnapKVConfig:
    """Matched-byte SnapKV settings (lock after first green repro).

    kvpress ``compression_ratio`` = fraction of KV pairs *removed*.
    For keep_frac=0.25 use compression_ratio=0.75.
    """

    budget_frac: float = 0.25
    compression_ratio: float = 0.75
    window_size: int = 64
    max_capacity_prompt: int = 256
    kernel_size: int = 5
    pooling: str = "avgpool"


def _try_import_press():
    try:
        from kvpress import SnapKVPress  # type: ignore

        return SnapKVPress
    except Exception:
        return None


def make_press(cfg: Optional[SnapKVConfig] = None):
    """Return a kvpress SnapKVPress instance, or None if unavailable."""
    cfg = cfg or SnapKVConfig()
    cls = _try_import_press()
    if cls is None:
        return None
    # Prefer compression_ratio kw; fall back to defaults if signature differs.
    try:
        return cls(compression_ratio=cfg.compression_ratio)
    except TypeError:
        return cls()


def status() -> dict[str, Any]:
    press = make_press()
    return {
        "baseline_id": "Q3",
        "name": "SnapKV",
        "implemented": press is not None,
        "config": asdict(SnapKVConfig()),
        "kvpress_snapkv": press is not None,
        "next": (
            "scripts/run_snapkv_quality.py --config configs/w4_snapkv_matched.yaml"
            if press is not None
            else "uv sync --extra kvpress (H200); or LOCK_Q_DROPKEEP via run_snapkv_attempt.py"
        ),
    }
