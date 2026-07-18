"""INT4 append/decode path (W3 + M2 GPU torch).

CPU numpy reference for quantize-on-write; torch groupwise ops keep pack/dequant
on device (Fable M2 — no custom CUDA kernels).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from prioritykv.int4_kv import Int4KvConfig, dequantize_groupwise, pack_range, quantize_groupwise


@dataclass
class PackedInt4Page:
    """One physical page of INT4 KV (numpy host or torch device tensors)."""

    q: Any  # uint8 codes
    scale: Any
    zero_point: Any
    n_tokens: int
    group_size: int
    nbits: int = 4

    def is_torch(self) -> bool:
        try:
            import torch

            return torch.is_tensor(self.q)
        except Exception:
            return False

    def dequant(self) -> Any:
        if self.is_torch():
            return dequantize_groupwise_torch(
                self.q, self.scale, self.zero_point, group_size=self.group_size
            )
        return dequantize_groupwise(
            self.q, self.scale, self.zero_point, group_size=self.group_size
        )

    def payload_bytes(self) -> int:
        """Packed codes + scale + zero_point (excludes BF16 source tensor)."""
        if self.is_torch():
            return int(
                self.q.numel() * self.q.element_size()
                + self.scale.numel() * self.scale.element_size()
                + self.zero_point.numel() * self.zero_point.element_size()
            )
        return int(self.q.nbytes + self.scale.nbytes + self.zero_point.nbytes)

    def to_numpy(self) -> "PackedInt4Page":
        """Host copy for parity / CPU consumers."""
        if not self.is_torch():
            return self
        return PackedInt4Page(
            q=self.q.detach().cpu().numpy(),
            scale=self.scale.detach().float().cpu().numpy(),
            zero_point=self.zero_point.detach().float().cpu().numpy(),
            n_tokens=self.n_tokens,
            group_size=self.group_size,
            nbits=self.nbits,
        )


def quantize_groupwise_torch(
    x: Any,
    *,
    group_size: int = 32,
    nbits: int = 4,
) -> tuple[Any, Any, Any]:
    """Asymmetric per-group INT quant along axis=-1 (torch, any device)."""
    import torch

    if not torch.is_tensor(x):
        raise TypeError(f"expected torch tensor, got {type(x)}")
    x = x.to(dtype=torch.float32)
    *lead, n = x.shape
    pad = (-n) % group_size
    if pad:
        x_pad = torch.nn.functional.pad(x, (0, pad))
    else:
        x_pad = x
    g = x_pad.shape[-1] // group_size
    grouped = x_pad.reshape(*lead, g, group_size)
    mn = grouped.amin(dim=-1, keepdim=True)
    mx = grouped.amax(dim=-1, keepdim=True)
    qmin, qmax = pack_range(nbits)
    scale = (mx - mn) / max(qmax - qmin, 1)
    scale = torch.clamp(scale, min=1e-8)
    zp = torch.round(qmin - mn / scale)
    zp = torch.clamp(zp, qmin, qmax)
    q = torch.round(grouped / scale + zp)
    q = torch.clamp(q, qmin, qmax).to(torch.uint8)
    q_flat = q.reshape(*lead, g * group_size)
    if pad:
        q_flat = q_flat[..., :n]
    return q_flat, scale.squeeze(-1), zp.squeeze(-1)


def dequantize_groupwise_torch(
    q: Any,
    scale: Any,
    zero_point: Any,
    *,
    group_size: int = 32,
) -> Any:
    """Inverse of quantize_groupwise_torch."""
    import torch

    *lead, n = q.shape
    pad = (-n) % group_size
    if pad:
        q_pad = torch.nn.functional.pad(q, (0, pad))
    else:
        q_pad = q
    g = q_pad.shape[-1] // group_size
    grouped = q_pad.reshape(*lead, g, group_size).to(torch.float32)
    sc = scale[..., :g]
    zp = zero_point[..., :g]
    while sc.ndim < grouped.ndim:
        sc = sc.unsqueeze(-1)
        zp = zp.unsqueeze(-1)
    x = (grouped - zp.to(torch.float32)) * sc.to(torch.float32)
    x = x.reshape(*lead, g * group_size)
    if pad:
        x = x[..., :n]
    return x


def append_quantize(
    bf16_slice: np.ndarray,
    *,
    cfg: Int4KvConfig | None = None,
) -> PackedInt4Page:
    """Quantize a BF16 (or float) KV slice on write → packed INT4 page (numpy)."""
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


def append_quantize_torch(
    bf16_slice: Any,
    *,
    cfg: Int4KvConfig | None = None,
) -> PackedInt4Page:
    """GPU/CPU torch quantize-on-write → PackedInt4Page with device tensors."""
    import torch

    cfg = cfg or Int4KvConfig()
    if not torch.is_tensor(bf16_slice):
        raise TypeError(f"expected torch tensor, got {type(bf16_slice)}")
    q, scale, zp = quantize_groupwise_torch(
        bf16_slice, group_size=cfg.group_size, nbits=cfg.nbits
    )
    n_tokens = int(bf16_slice.shape[-1]) if bf16_slice.ndim >= 1 else 0
    return PackedInt4Page(
        q=q.contiguous(),
        scale=scale.contiguous(),
        zero_point=zp.contiguous(),
        n_tokens=n_tokens,
        group_size=cfg.group_size,
        nbits=cfg.nbits,
    )


def decode_gather_reference(pages: list[PackedInt4Page]) -> np.ndarray:
    """Concatenate dequantized pages along the token axis (homogenous INT4 ref)."""
    if not pages:
        raise ValueError("no pages")
    parts = []
    for p in pages:
        y = p.dequant()
        if hasattr(y, "detach"):
            y = y.detach().float().cpu().numpy()
        parts.append(y)
    return np.concatenate(parts, axis=-1)
