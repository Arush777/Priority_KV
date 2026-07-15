"""Tests for multi-call LSE merge parity with dense mixed attend."""

from __future__ import annotations

import numpy as np

from prioritykv.mixed_cache_reference import (
    attention_reference,
    mixed_attend_kv,
    mixed_attend_kv_multicall,
    pages_from_sequence,
)
from prioritykv.page_roles import StorageDtype
from prioritykv.linear_risk import fit_ridge, score_page


def test_lse_multicall_matches_dense_mixed():
    rng = np.random.default_rng(7)
    kv = rng.standard_normal((80, 8)).astype(np.float32)
    q = rng.standard_normal((2, 8)).astype(np.float32)
    n_pages = (80 + 15) // 16
    dtypes = [StorageDtype.BF16 if i % 2 == 0 else StorageDtype.INT4 for i in range(n_pages)]
    pages = pages_from_sequence(kv, dtypes, page_tokens=16)
    dense = mixed_attend_kv(q, pages, pages)
    multi = mixed_attend_kv_multicall(q, pages, pages)
    assert np.allclose(dense, multi, atol=1e-5), np.max(np.abs(dense - multi))


def test_lse_multicall_matches_all_bf16():
    rng = np.random.default_rng(8)
    kv = rng.standard_normal((64, 8)).astype(np.float32)
    q = rng.standard_normal((3, 8)).astype(np.float32)
    pages = pages_from_sequence(kv, [StorageDtype.BF16] * 8, page_tokens=16)
    multi = mixed_attend_kv_multicall(q, pages, pages)
    ref = attention_reference(q, kv, kv)
    assert np.allclose(multi, ref, atol=1e-5)


def test_linear_risk_fit_prefers_tool():
    rows = [
        {"is_tool": 1, "is_system": 0, "is_constraint": 0, "is_sink": 0, "is_recent": 0, "token_mass": 16, "score_delta": 1.0},
        {"is_tool": 0, "is_system": 0, "is_constraint": 0, "is_sink": 0, "is_recent": 1, "token_mass": 16, "score_delta": -0.2},
        {"is_tool": 1, "is_system": 0, "is_constraint": 0, "is_sink": 0, "is_recent": 0, "token_mass": 32, "score_delta": 0.8},
        {"is_tool": 0, "is_system": 0, "is_constraint": 0, "is_sink": 0, "is_recent": 0, "token_mass": 16, "score_delta": -0.5},
    ]
    cfg = fit_ridge(rows, l2=1e-3)
    assert score_page({"roles": ["tool"], "n_tokens": 16}, cfg) > score_page(
        {"roles": ["recent"], "n_tokens": 16}, cfg
    )
