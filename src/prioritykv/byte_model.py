"""Byte accounting for mixed-precision paged KV (W1).

Budgets are defined in *realized* bytes (storage + scales/zero-points + page table overhead).
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
class KvShape:
    """Per-layer KV tensor shape for one sequence (or one page)."""

    num_kv_heads: int
    head_dim: int
    num_tokens: int


def bf16_kv_bytes(shape: KvShape) -> int:
    """Bytes for K and V stored in BF16 (2 bytes/elem each)."""
    elems = shape.num_kv_heads * shape.head_dim * shape.num_tokens
    return 2 * elems * 2  # K + V


def int4_kv_bytes(shape: KvShape, group_size: int = INT4_GROUP_SIZE) -> int:
    """Bytes for K and V in group-wise asymmetric INT4 + scale/zero-point metadata.

    - INT4 payloads: 0.5 bytes/elem
    - Per group: 1× FP16 scale + 1× INT4 zero-point packed to 1 byte (KIVI-style approx)
    Groups along token axis for K, channel for V — same byte count when dims match.
    """
    elems_per_tensor = shape.num_kv_heads * shape.head_dim * shape.num_tokens
    payload = int(elems_per_tensor * 0.5)  # K or V payload
    # Conservative metadata: 2 bytes scale (FP16) + 1 byte zp per group, ×2 tensors
    groups_k = _ceil_div(shape.num_tokens, group_size) * shape.num_kv_heads * shape.head_dim
    groups_v = _ceil_div(shape.head_dim, group_size) * shape.num_kv_heads * shape.num_tokens
    meta = (groups_k + groups_v) * 3
    return 2 * payload + meta


def page_table_overhead_bytes(num_pages: int, bytes_per_entry: int = 16) -> int:
    """Controller / page-table metadata bytes."""
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
) -> int:
    """Total realized bytes for a mixed BF16/INT4 cache at given token counts."""
    bf16 = bf16_kv_bytes(KvShape(num_kv_heads, head_dim, num_bf16_tokens))
    int4 = int4_kv_bytes(KvShape(num_kv_heads, head_dim, num_int4_tokens))
    pages = _ceil_div(num_bf16_tokens + num_int4_tokens, page_tokens)
    return bf16 + int4 + page_table_overhead_bytes(pages)
