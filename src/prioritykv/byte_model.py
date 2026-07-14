"""Byte accounting for mixed-precision paged KV (W1).

Budgets are defined in *realized* bytes (storage + scales/zero-points + page table overhead).
All-layer FullKV uses the Qwen3-8B GQA shape pinned below.
"""

from __future__ import annotations

from dataclasses import dataclass


# Backend-native physical page size (tokens per page). Plan §1.
PHYSICAL_PAGE_TOKENS = 16
# Allocation unit (tokens). Plan §1; one ablation at 64.
DEFAULT_ALLOC_UNIT_TOKENS = 128
# KIVI-style INT4 group size. Plan §1.
INT4_GROUP_SIZE = 32


@dataclass(frozen=True)
class ModelKvGeom:
    """Per-layer KV geometry for a decoder-only GQA model."""

    num_layers: int
    num_kv_heads: int
    head_dim: int


# Qwen3-8B (matches config.json of the pinned revision).
QWEN3_8B_KV = ModelKvGeom(num_layers=36, num_kv_heads=8, head_dim=128)


@dataclass(frozen=True)
class KvShape:
    """Per-layer KV tensor shape for one sequence (or one page)."""

    num_kv_heads: int
    head_dim: int
    num_tokens: int


def bf16_kv_bytes(shape: KvShape) -> int:
    """Bytes for K and V stored in BF16 (2 bytes/elem each), one layer."""
    elems = shape.num_kv_heads * shape.head_dim * shape.num_tokens
    return 2 * elems * 2  # K + V


def int4_kv_bytes(shape: KvShape, group_size: int = INT4_GROUP_SIZE) -> int:
    """Bytes for K and V in group-wise asymmetric INT4 + scale/zero-point metadata.

    - INT4 payloads: 0.5 bytes/elem
    - Per group: 1× FP16 scale + 1× zp byte (KIVI-style approx)
    Groups along token axis for K, channel axis for V.
    """
    elems_per_tensor = shape.num_kv_heads * shape.head_dim * shape.num_tokens
    payload = int(elems_per_tensor * 0.5)  # K or V payload
    groups_k = _ceil_div(shape.num_tokens, group_size) * shape.num_kv_heads * shape.head_dim
    groups_v = _ceil_div(shape.head_dim, group_size) * shape.num_kv_heads * shape.num_tokens
    meta = (groups_k + groups_v) * 3
    return 2 * payload + meta


def page_table_overhead_bytes(num_pages: int, bytes_per_entry: int = 16) -> int:
    """Controller / page-table metadata bytes (one sequence, all pages)."""
    return num_pages * bytes_per_entry


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def realized_bytes(
    *,
    num_bf16_tokens: int,
    num_int4_tokens: int,
    num_kv_heads: int,
    head_dim: int,
    page_tokens: int = PHYSICAL_PAGE_TOKENS,
    num_layers: int = 1,
) -> int:
    """Realized bytes for mixed BF16/INT4 KV across ``num_layers``."""
    bf16 = bf16_kv_bytes(KvShape(num_kv_heads, head_dim, num_bf16_tokens))
    int4 = int4_kv_bytes(KvShape(num_kv_heads, head_dim, num_int4_tokens))
    pages = _ceil_div(num_bf16_tokens + num_int4_tokens, page_tokens)
    per_layer = bf16 + int4
    # Page table counted once (shared controller view); scales gently with pages.
    return num_layers * per_layer + page_table_overhead_bytes(pages)


def fullkv_bf16_bytes(num_tokens: int, geom: ModelKvGeom = QWEN3_8B_KV) -> int:
    """All-layer FullKV in BF16 for a sequence of ``num_tokens``."""
    return realized_bytes(
        num_bf16_tokens=num_tokens,
        num_int4_tokens=0,
        num_kv_heads=geom.num_kv_heads,
        head_dim=geom.head_dim,
        num_layers=geom.num_layers,
    )


@dataclass(frozen=True)
class BudgetPlan:
    """How many tokens can stay BF16 under a byte-fraction budget of FullKV."""

    seq_len: int
    budget_frac: float
    fullkv_bytes: int
    budget_bytes: int
    max_bf16_tokens: int
    realized_at_max_bf16: int
    all_int4_bytes: int
    all_int4_frac: float
    feasible: bool


def plan_budget(
    seq_len: int,
    budget_frac: float,
    geom: ModelKvGeom = QWEN3_8B_KV,
    page_tokens: int = PHYSICAL_PAGE_TOKENS,
) -> BudgetPlan:
    """Largest BF16 token count such that mixed cache ≤ ``budget_frac`` × FullKV.

    Rest of the sequence is INT4. Binary search over bf16 token count.
    """
    if not 0.0 < budget_frac <= 1.0:
        raise ValueError(f"budget_frac out of range: {budget_frac}")
    if seq_len < 0:
        raise ValueError(f"seq_len must be >= 0, got {seq_len}")

    full = fullkv_bf16_bytes(seq_len, geom)
    budget = int(full * budget_frac)
    all_int4 = realized_bytes(
        num_bf16_tokens=0,
        num_int4_tokens=seq_len,
        num_kv_heads=geom.num_kv_heads,
        head_dim=geom.head_dim,
        page_tokens=page_tokens,
        num_layers=geom.num_layers,
    )
    all_int4_frac = all_int4 / full if full else 0.0

    def cost(bf16_toks: int) -> int:
        return realized_bytes(
            num_bf16_tokens=bf16_toks,
            num_int4_tokens=seq_len - bf16_toks,
            num_kv_heads=geom.num_kv_heads,
            head_dim=geom.head_dim,
            page_tokens=page_tokens,
            num_layers=geom.num_layers,
        )

    if all_int4 > budget:
        # Even all-INT4 exceeds the fraction (can happen for tiny budgets).
        return BudgetPlan(
            seq_len=seq_len,
            budget_frac=budget_frac,
            fullkv_bytes=full,
            budget_bytes=budget,
            max_bf16_tokens=0,
            realized_at_max_bf16=all_int4,
            all_int4_bytes=all_int4,
            all_int4_frac=all_int4_frac,
            feasible=False,
        )

    lo, hi = 0, seq_len
    best = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        if cost(mid) <= budget:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1

    realized = cost(best)
    return BudgetPlan(
        seq_len=seq_len,
        budget_frac=budget_frac,
        fullkv_bytes=full,
        budget_bytes=budget,
        max_bf16_tokens=best,
        realized_at_max_bf16=realized,
        all_int4_bytes=all_int4,
        all_int4_frac=all_int4_frac,
        feasible=True,
    )


def budget_table(
    lengths: tuple[int, ...] = (8192, 16384, 32768, 65536),
    fracs: tuple[float, ...] = (0.50, 0.30),
    geom: ModelKvGeom = QWEN3_8B_KV,
) -> list[BudgetPlan]:
    return [plan_budget(n, f, geom) for n in lengths for f in fracs]
