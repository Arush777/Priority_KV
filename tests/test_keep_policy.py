"""CPU tests for matched-budget keep policies."""

from __future__ import annotations

import numpy as np

from prioritykv.baselines.keep_policy import (
    KeepPolicyConfig,
    select_random,
    select_structure,
    select_uniform,
)
from prioritykv.page_roles import PageRole


def test_uniform_respects_frac():
    cfg = KeepPolicyConfig(keep_frac=0.25, sink_tokens=16, force_recent=64)
    idx = select_uniform(1000, cfg)
    assert 240 <= len(idx) <= 260
    assert idx[0] == 0
    assert idx[-1] == 999


def test_structure_prefers_protected():
    cfg = KeepPolicyConfig(keep_frac=0.2, sink_tokens=8, force_recent=32)
    roles = [PageRole.FILLER] * 500
    for i in range(50, 80):
        roles[i] = PageRole.TOOL
    idx = select_structure(500, roles, cfg)
    tool_kept = sum(1 for i in idx if 50 <= int(i) < 80)
    assert tool_kept >= 20


def test_random_matched_budget():
    cfg = KeepPolicyConfig(keep_frac=0.3, sink_tokens=16, force_recent=64, seed=1)
    idx = select_random(800, cfg)
    assert abs(len(idx) / 800 - 0.3) < 0.05


def test_apply_keep_indices():
    import pytest

    torch = pytest.importorskip("torch")
    from prioritykv.baselines.keep_policy import apply_keep_indices

    ids = torch.arange(100)
    out = apply_keep_indices(ids, np.array([0, 1, 98, 99]))
    assert list(out.tolist()) == [0, 1, 98, 99]


def test_page_structure_floors_budget():
    from prioritykv.baselines.keep_policy import (
        select_structure_pages,
        select_uniform_pages,
    )

    cfg = KeepPolicyConfig(
        keep_frac=0.25,
        sink_tokens=16,
        force_recent=64,
        page_tokens=16,
        granularity="page",
    )
    n = 1000
    idx_u = select_uniform_pages(n, cfg)
    budget = max(16 + 64, int(round(n * 0.25)))
    assert len(idx_u) <= budget + 16
    roles = [PageRole.FILLER] * n
    for i in range(100, 200):
        roles[i] = PageRole.TOOL
    idx_s = select_structure_pages(n, roles, cfg)
    tool_kept = sum(1 for i in idx_s if 100 <= int(i) < 200)
    assert tool_kept >= 48
