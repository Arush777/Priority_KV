"""Mixed BF16/INT4 cache reference (W3): dequantize-then-attend on numpy.

W4 kernels (FlashInfer multi-call + LSE merge) must match this within tolerance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from prioritykv.int4_path import PackedInt4Page, append_quantize
from prioritykv.page_roles import StorageDtype


@dataclass
class RefPage:
    """One physical page of KV for a single (layer, head) slice."""

    dtype: StorageDtype
    # BF16 path: store float tensor (n_tokens, head_dim)
    # INT4 path: packed
    tokens_bf16: Optional[np.ndarray] = None
    packed: Optional[PackedInt4Page] = None

    @property
    def n_tokens(self) -> int:
        if self.packed is not None:
            return self.packed.n_tokens
        assert self.tokens_bf16 is not None
        return int(self.tokens_bf16.shape[0])

    def materialize(self) -> np.ndarray:
        """Return float32 array shaped (n_tokens, dim)."""
        if self.dtype == StorageDtype.BF16:
            assert self.tokens_bf16 is not None
            return self.tokens_bf16.astype(np.float32)
        assert self.packed is not None
        x = self.packed.dequant()
        # We store INT4 of shape (dim, n_tokens) from append_quantize(chunk.T)
        if x.ndim == 2 and x.shape[1] == self.packed.n_tokens:
            return x.T.astype(np.float32)
        if x.ndim == 2 and x.shape[0] == self.packed.n_tokens:
            return x.astype(np.float32)
        raise ValueError(f"unexpected INT4 dequant shape {x.shape}")


def pages_from_sequence(
    kv: np.ndarray,
    dtypes: Sequence[StorageDtype],
    *,
    page_tokens: int = 16,
) -> List[RefPage]:
    """Split (seq, dim) KV into pages with assigned dtypes (one dtype per page)."""
    seq, _dim = kv.shape
    pages: List[RefPage] = []
    for i, start in enumerate(range(0, seq, page_tokens)):
        chunk = kv[start : start + page_tokens]
        dt = dtypes[min(i, len(dtypes) - 1)]
        if dt == StorageDtype.BF16:
            pages.append(RefPage(dtype=dt, tokens_bf16=chunk.copy()))
        else:
            # quantize along token axis → shape (dim, n_tok) then store as (n_tok, dim) view
            packed = append_quantize(chunk.T.astype(np.float32))  # (dim, n_tok)
            pages.append(RefPage(dtype=dt, packed=packed))
    return pages


def gather_kv(pages: Sequence[RefPage]) -> np.ndarray:
    """Materialize and concat along the sequence axis → (seq, dim)."""
    parts = [p.materialize() for p in pages]
    return np.concatenate(parts, axis=0)


def attention_reference(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Single-head attention: q (tq,d), k/v (tk,d) → (tq,d)."""
    scale = 1.0 / np.sqrt(max(q.shape[-1], 1))
    logits = (q @ k.T) * scale
    logits = logits - logits.max(axis=-1, keepdims=True)
    w = np.exp(logits)
    w = w / (w.sum(axis=-1, keepdims=True) + 1e-9)
    return w @ v


def mixed_attend(
    q: np.ndarray,
    pages: Sequence[RefPage],
) -> np.ndarray:
    """Dequantize-then-attend reference for mixed BF16/INT4 pages."""
    k = gather_kv(pages)
    # For V use same pages (caller passes V pages) — here pages are K; V handled outside.
    return attention_reference(q, k, k)  # placeholder misuse — see mixed_attend_kv


def mixed_attend_kv(
    q: np.ndarray,
    k_pages: Sequence[RefPage],
    v_pages: Sequence[RefPage],
) -> np.ndarray:
    k = gather_kv(k_pages)
    v = gather_kv(v_pages)
    assert k.shape[0] == v.shape[0]
    return attention_reference(q, k, v)


def attention_with_lse(
    q: np.ndarray, k: np.ndarray, v: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Single-head attention returning (out, lse) with lse = logsumexp(logits).

    ``lse`` shape (tq,). Used by multi-call merge (FlashInfer / W4).
    """
    scale = 1.0 / np.sqrt(max(q.shape[-1], 1))
    logits = (q @ k.T) * scale
    m = logits.max(axis=-1, keepdims=True)
    exp = np.exp(logits - m)
    se = exp.sum(axis=-1, keepdims=True) + 1e-9
    out = (exp / se) @ v
    lse = (m + np.log(se)).reshape(-1)
    return out, lse


def lse_merge_pair(
    out_a: np.ndarray,
    lse_a: np.ndarray,
    out_b: np.ndarray,
    lse_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact two-way merge of attention partials via LSE (FlashInfer target)."""
    assert out_a.shape == out_b.shape
    assert lse_a.shape == lse_b.shape == (out_a.shape[0],)
    m = np.maximum(lse_a, lse_b)
    wa = np.exp(lse_a - m)[:, None]
    wb = np.exp(lse_b - m)[:, None]
    denom = wa + wb + 1e-9
    out = (wa * out_a + wb * out_b) / denom
    lse = m + np.log(np.exp(lse_a - m) + np.exp(lse_b - m) + 1e-9)
    return out, lse


def mixed_attend_kv_multicall(
    q: np.ndarray,
    k_pages: Sequence[RefPage],
    v_pages: Sequence[RefPage],
) -> np.ndarray:
    """Multi-call attention: attend each page separately, LSE-merge.

    Must match ``mixed_attend_kv`` within tight tolerance (W4 FlashInfer gate).
    """
    assert len(k_pages) == len(v_pages)
    if not k_pages:
        raise ValueError("empty pages")
    out, lse = attention_with_lse(q, k_pages[0].materialize(), v_pages[0].materialize())
    for kp, vp in zip(k_pages[1:], v_pages[1:]):
        ob, lb = attention_with_lse(q, kp.materialize(), vp.materialize())
        out, lse = lse_merge_pair(out, lse, ob, lb)
    return out
