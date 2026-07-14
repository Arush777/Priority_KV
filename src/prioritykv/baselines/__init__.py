"""Baselines package (SnapKV, later FixedHot / Random / ProtectedRole)."""

from prioritykv.baselines.snapkv import SnapKVConfig, status as snapkv_status

__all__ = ["SnapKVConfig", "snapkv_status"]
