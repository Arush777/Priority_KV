"""Uniform INT4 KV reference (Q2): group-wise asymmetric quant + HF QuantizedCache.

KIVI-style along the token axis for keys (group_size=32 default). Used as:
- CPU unit-testable round-trip helper
- QuantizedCache generate path on H200 (quanto backend when installed)
- Fallback: post-prefill fake-quant of past_key_values when quanto is missing
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional

import numpy as np


@dataclass(frozen=True)
class Int4KvConfig:
    """Uniform INT4 storage settings (plan §1.2 / Q2)."""

    nbits: int = 4
    group_size: int = 32  # token axis for K; channel for V in full KIVI — we use token for both in ref
    backend: str = "quanto"  # quanto | fake
    # Transformers QuantizedCache axis defaults (quanto).
    axis_key: int = 0
    axis_value: int = 0


def pack_range(nbits: int) -> tuple[int, int]:
    """Inclusive integer range for signed asymmetric codes mapped via zp."""
    qmax = (1 << nbits) - 1
    return 0, qmax


def quantize_groupwise(
    x: np.ndarray,
    *,
    group_size: int = 32,
    nbits: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Asymmetric per-group INT quant along axis=-1.

    Returns (q, scale, zero_point) with q uint8 in [0, 2^nbits-1].
    """
    if x.dtype not in (np.float16, np.float32, np.float64):
        x = x.astype(np.float32)
    else:
        x = x.astype(np.float32, copy=False)

    *lead, n = x.shape
    pad = (-n) % group_size
    if pad:
        x_pad = np.pad(x, [(0, 0)] * len(lead) + [(0, pad)], mode="constant")
    else:
        x_pad = x
    g = x_pad.shape[-1] // group_size
    grouped = x_pad.reshape(*lead, g, group_size)
    mn = grouped.min(axis=-1, keepdims=True)
    mx = grouped.max(axis=-1, keepdims=True)
    qmin, qmax = pack_range(nbits)
    scale = (mx - mn) / max(qmax - qmin, 1)
    scale = np.maximum(scale, 1e-8)
    zp = np.round(qmin - mn / scale).astype(np.float32)
    zp = np.clip(zp, qmin, qmax)
    q = np.round(grouped / scale + zp)
    q = np.clip(q, qmin, qmax).astype(np.uint8)
    # Drop padding on q only for the trailing dim when returning flat view.
    q_flat = q.reshape(*lead, g * group_size)
    if pad:
        q_flat = q_flat[..., :n]
        # scales/zp stay per-group including pad group (caller uses group_size)
    return q_flat, scale.squeeze(-1), zp.squeeze(-1)


def dequantize_groupwise(
    q: np.ndarray,
    scale: np.ndarray,
    zero_point: np.ndarray,
    *,
    group_size: int = 32,
) -> np.ndarray:
    """Inverse of quantize_groupwise."""
    *lead, n = q.shape
    pad = (-n) % group_size
    if pad:
        q_pad = np.pad(q, [(0, 0)] * len(lead) + [(0, pad)], mode="constant")
    else:
        q_pad = q
    g = q_pad.shape[-1] // group_size
    grouped = q_pad.reshape(*lead, g, group_size).astype(np.float32)
    # scale/zp may be (...g) already
    sc = scale[..., :g]
    zp = zero_point[..., :g]
    while sc.ndim < grouped.ndim:
        sc = np.expand_dims(sc, -1)
        zp = np.expand_dims(zp, -1)
    x = (grouped - zp) * sc
    x = x.reshape(*lead, g * group_size)
    if pad:
        x = x[..., :n]
    return x


def fake_quant_roundtrip(x: np.ndarray, cfg: Int4KvConfig | None = None) -> np.ndarray:
    """Quantize then dequantize — CPU reference for uniform INT4 error."""
    cfg = cfg or Int4KvConfig()
    q, scale, zp = quantize_groupwise(x, group_size=cfg.group_size, nbits=cfg.nbits)
    return dequantize_groupwise(q, scale, zp, group_size=cfg.group_size)


def status() -> dict[str, Any]:
    cfg = Int4KvConfig()
    quanto_ok = False
    try:
        import optimum.quanto  # noqa: F401

        quanto_ok = True
    except Exception:
        try:
            import quanto  # noqa: F401

            quanto_ok = True
        except Exception:
            quanto_ok = False
    cache_cls = None
    try:
        from transformers import QuantoQuantizedCache  # type: ignore

        cache_cls = "QuantoQuantizedCache"
    except Exception:
        try:
            from transformers.cache_utils import QuantizedCache  # type: ignore

            cache_cls = "QuantizedCache"
        except Exception:
            cache_cls = None
    return {
        "baseline_id": "Q2",
        "name": "Uniform INT4",
        "implemented": True,
        "config": asdict(cfg),
        "quanto_available": quanto_ok,
        "transformers_cache": cache_cls,
        "path": "HF QuantizedCache (quanto) preferred; fake groupwise fallback otherwise",
    }


def make_quantized_cache(
    *,
    max_batch_size: int = 1,
    max_cache_len: int = 32768,
    cfg: Optional[Int4KvConfig] = None,
):
    """Build a Transformers quantized KV cache, or None if unavailable."""
    cfg = cfg or Int4KvConfig()
    try:
        from transformers import QuantizedCacheConfig, QuantoQuantizedCache

        qcfg = QuantizedCacheConfig(
            backend=cfg.backend if cfg.backend != "fake" else "quanto",
            nbits=cfg.nbits,
            axis_key=cfg.axis_key,
            axis_value=cfg.axis_value,
            q_group_size=cfg.group_size,
            compute_dtype="bfloat16",
        )
        return QuantoQuantizedCache(cache_config=qcfg)
    except Exception:
        pass
    try:
        from transformers import QuantizedCacheConfig
        from transformers.cache_utils import QuantizedCache

        qcfg = QuantizedCacheConfig(
            backend="quanto",
            nbits=cfg.nbits,
            q_group_size=cfg.group_size,
        )
        return QuantizedCache(cache_config=qcfg)
    except Exception:
        return None
