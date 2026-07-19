"""kvpress factory helpers: SnapKV, H2O-style ObservedAttention, PyramidKV."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class AttnPressConfig:
    """Matched-byte attention-eviction settings (compression_ratio = fraction removed)."""

    keep_frac: float = 0.25
    compression_ratio: float = 0.75
    window_size: int = 64
    kernel_size: int = 5
    # ObservedAttentionPress (H2O-related) needs eager attention.
    h2o_attn_implementation: str = "eager"


def compression_ratio_for_keep(keep_frac: float) -> float:
    return float(max(0.0, min(0.999, 1.0 - keep_frac)))


def _try_import(name: str):
    try:
        import kvpress  # noqa: F401
    except Exception:
        return None
    return getattr(__import__("kvpress", fromlist=[name]), name, None)


def make_snapkv_press(cfg: Optional[AttnPressConfig] = None):
    cfg = cfg or AttnPressConfig()
    cls = _try_import("SnapKVPress")
    if cls is None:
        return None
    try:
        return cls(
            compression_ratio=cfg.compression_ratio,
            window_size=cfg.window_size,
            kernel_size=cfg.kernel_size,
        )
    except TypeError:
        try:
            return cls(compression_ratio=cfg.compression_ratio)
        except TypeError:
            return cls()


def make_h2o_press(cfg: Optional[AttnPressConfig] = None):
    """H2O-related baseline via kvpress ObservedAttentionPress (cumulative attn)."""
    cfg = cfg or AttnPressConfig()
    cls = _try_import("ObservedAttentionPress")
    if cls is None:
        return None
    try:
        return cls(compression_ratio=cfg.compression_ratio)
    except TypeError:
        return cls()


def make_pyramid_press(cfg: Optional[AttnPressConfig] = None):
    cfg = cfg or AttnPressConfig()
    cls = _try_import("PyramidKVPress")
    if cls is None:
        return None
    try:
        return cls(
            compression_ratio=cfg.compression_ratio,
            window_size=cfg.window_size,
            kernel_size=cfg.kernel_size,
        )
    except TypeError:
        try:
            return cls(compression_ratio=cfg.compression_ratio)
        except TypeError:
            return cls()


def press_status() -> dict[str, Any]:
    return {
        "snapkv": make_snapkv_press() is not None,
        "h2o_observed_attention": make_h2o_press() is not None,
        "pyramidkv": make_pyramid_press() is not None,
        "config": asdict(AttnPressConfig()),
    }
