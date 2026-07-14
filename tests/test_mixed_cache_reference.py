"""Tests for mixed BF16/INT4 dequant-then-attend reference."""

from __future__ import annotations

import numpy as np

from prioritykv.mixed_cache_reference import (
    attention_reference,
    gather_kv,
    mixed_attend_kv,
    pages_from_sequence,
)
from prioritykv.page_roles import StorageDtype


def test_all_bf16_matches_dense():
    rng = np.random.default_rng(0)
    kv = rng.standard_normal((64, 8)).astype(np.float32)
    q = rng.standard_normal((3, 8)).astype(np.float32)
    pages = pages_from_sequence(kv, [StorageDtype.BF16] * 10, page_tokens=16)
    out = mixed_attend_kv(q, pages, pages)
    ref = attention_reference(q, kv, kv)
    assert np.allclose(out, ref, atol=1e-5)


def test_all_int4_close_to_dense():
    rng = np.random.default_rng(1)
    kv = rng.standard_normal((48, 16)).astype(np.float32)
    q = rng.standard_normal((2, 16)).astype(np.float32)
    pages = pages_from_sequence(kv, [StorageDtype.INT4] * 10, page_tokens=16)
    out = mixed_attend_kv(q, pages, pages)
    ref = attention_reference(q, kv, kv)
    # INT4 noise — relative still bounded
    rel = np.linalg.norm(out - ref) / (np.linalg.norm(ref) + 1e-8)
    assert rel < 0.35, rel


def test_mixed_equals_full_dequant_attend():
    rng = np.random.default_rng(2)
    kv = rng.standard_normal((80, 8)).astype(np.float32)
    q = rng.standard_normal((1, 8)).astype(np.float32)
    # Alternate page dtypes
    n_pages = (80 + 15) // 16
    dtypes = [StorageDtype.BF16 if i % 2 == 0 else StorageDtype.INT4 for i in range(n_pages)]
    pages = pages_from_sequence(kv, dtypes, page_tokens=16)
    out = mixed_attend_kv(q, pages, pages)
    k_mat = gather_kv(pages)
    ref = attention_reference(q, k_mat, k_mat)
    assert np.allclose(out, ref, atol=1e-5)


def test_empty_int4_group():
    rng = np.random.default_rng(3)
    kv = rng.standard_normal((32, 4)).astype(np.float32)
    q = rng.standard_normal((1, 4)).astype(np.float32)
    pages = pages_from_sequence(kv, [StorageDtype.BF16] * 4, page_tokens=16)
    out = mixed_attend_kv(q, pages, pages)
    assert out.shape == (1, 4)


def test_partial_last_page():
    rng = np.random.default_rng(4)
    kv = rng.standard_normal((20, 8)).astype(np.float32)  # 16 + 4
    pages = pages_from_sequence(kv, [StorageDtype.BF16, StorageDtype.INT4], page_tokens=16)
    assert pages[-1].n_tokens == 4
    mat = gather_kv(pages)
    assert mat.shape[0] == 20
