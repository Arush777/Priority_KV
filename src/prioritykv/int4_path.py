"""INT4 append/decode reference path (W3 start).

CPU/torch-free numpy reference for quantize-on-write. CUDA kernels come later.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from prioritykv.int4_kv import Int4KvConfig, dequantize_groupwise, quantize_groupwise


@dataclass
class PackedInt4Page:
    """One physical page of INT4 KV for a single layer/head slice (reference)."""

    q: np.ndarray  # uint8 codes, shape (..., n_tokens) or (..., head_dim) depending on axis
    scale: np.ndarray
    zero_point: np.ndarray
    n_tokens: int
    group_size: int
    nbits: int = 4

    def dequant(self) -> np.ndarray:
        return dequantize_groupwise(
            self.q, self.scale, self.zero_point, group_size=self.group_size
        )


def append_quantize(
    bf16_slice: np.ndarray,
    *,
    cfg: Int4KvConfig | None = None,
) -> PackedInt4Page:
    """Quantize a BF16 (or float) KV slice on write → packed INT4 page."""
    cfg = cfg or Int4KvConfig()
    q, scale, zp = quantize_groupwise(
        bf16_slice, group_size=cfg.group_size, nbits=cfg.nbits
    )
    n_tokens = int(bf16_slice.shape[-1]) if bf16_slice.ndim >= 1 else 0
    return PackedInt4Page(
        q=q,
        scale=scale,
        zero_point=zp,
        n_tokens=n_tokens,
        group_size=cfg.group_size,
        nbits=cfg.nbits,
    )


def decode_gather_reference(pages: list[PackedInt4Page]) -> np.ndarray:
    """Concatenate dequantized pages along the token axis (homogenous INT4 ref)."""
    if not pages:
        raise ValueError("no pages")
    parts = [p.dequant() for p in pages]
    return np.concatenate(parts, axis=-1)
