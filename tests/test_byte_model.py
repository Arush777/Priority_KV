"""Unit tests for PriorityKV byte accounting (CPU-only; runs here and on H200)."""

from __future__ import annotations

from prioritykv.byte_model import (
    PHYSICAL_PAGE_TOKENS,
    KvShape,
    bf16_kv_bytes,
    int4_kv_bytes,
    realized_bytes,
)


def test_bf16_full_page():
    shape = KvShape(num_kv_heads=8, head_dim=128, num_tokens=PHYSICAL_PAGE_TOKENS)
    # 2 tensors × 8 × 128 × 16 × 2 bytes
    assert bf16_kv_bytes(shape) == 2 * 8 * 128 * 16 * 2


def test_int4_smaller_than_bf16():
    shape = KvShape(num_kv_heads=8, head_dim=128, num_tokens=128)
    assert int4_kv_bytes(shape) < bf16_kv_bytes(shape)


def test_realized_budget_ratio_rough():
    # All-INT4 should land near ~25–35% of FullKV BF16 when metadata included.
    heads, dim, toks = 8, 128, 4096
    full = bf16_kv_bytes(KvShape(heads, dim, toks))
    mixed = realized_bytes(
        num_bf16_tokens=0,
        num_int4_tokens=toks,
        num_kv_heads=heads,
        head_dim=dim,
    )
    ratio = mixed / full
    assert 0.20 < ratio < 0.45, f"unexpected INT4 ratio {ratio:.3f}"


def test_partial_page_still_counts_positive():
    assert (
        realized_bytes(
            num_bf16_tokens=3,
            num_int4_tokens=0,
            num_kv_heads=2,
            head_dim=64,
        )
        > 0
    )
