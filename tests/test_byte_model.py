"""Extended byte-model / budget tests (CPU)."""

from __future__ import annotations

from prioritykv.byte_model import (
    QWEN3_8B_KV,
    KvShape,
    bf16_kv_bytes,
    budget_table,
    fullkv_bf16_bytes,
    plan_budget,
    realized_bytes,
)


def test_qwen3_fullkv_scales_with_layers():
    one = bf16_kv_bytes(KvShape(QWEN3_8B_KV.num_kv_heads, QWEN3_8B_KV.head_dim, 1024))
    full = fullkv_bf16_bytes(1024)
    assert full == one * QWEN3_8B_KV.num_layers + 16 * ((1024 + 15) // 16)


def test_plan_50_allows_more_bf16_than_30():
    a = plan_budget(32768, 0.50)
    b = plan_budget(32768, 0.30)
    assert a.feasible and b.feasible
    assert a.max_bf16_tokens > b.max_bf16_tokens


def test_all_int4_floor_above_025():
    # Plan note: 25% unreachable for mixed with protected BF16 + INT4 metadata.
    p = plan_budget(32768, 0.30)
    assert p.all_int4_frac > 0.25
    assert p.all_int4_frac < 0.40


def test_budget_table_len():
    rows = budget_table()
    assert len(rows) == 8  # 4 lengths × 2 fracs


def test_partial_mixed_layers():
    r = realized_bytes(
        num_bf16_tokens=128,
        num_int4_tokens=896,
        num_kv_heads=8,
        head_dim=128,
        num_layers=36,
    )
    assert r > 0
