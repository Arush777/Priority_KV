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


def test_structure_risk_prefers_tool_over_other_when_budget_tight():
    from pathlib import Path

    from prioritykv.baselines.keep_policy import (
        select_structure_pages,
        select_structure_risk_pages,
    )
    from prioritykv.linear_risk import load_linear_risk_config

    root = Path(__file__).resolve().parents[1]
    risk = load_linear_risk_config(root / "configs" / "linear_risk_fit.json")
    # Leave residual budget after sink+recent so structure pages compete.
    cfg = KeepPolicyConfig(
        keep_frac=0.35,
        sink_tokens=16,
        force_recent=64,
        page_tokens=16,
        granularity="page",
    )
    n = 512
    roles = [PageRole.FILLER] * n
    # Early OTHER pages (plain structure adds in page-id order first).
    for i in range(80, 160):
        roles[i] = PageRole.OTHER
    # Later TOOL pages — risk should prefer these over OTHER when competing.
    for i in range(240, 320):
        roles[i] = PageRole.TOOL
    idx_q7 = select_structure_pages(n, roles, cfg)
    idx_p2 = select_structure_risk_pages(n, roles, cfg, risk_cfg=risk)
    tool_q7 = sum(1 for i in idx_q7 if 240 <= int(i) < 320)
    tool_p2 = sum(1 for i in idx_p2 if 240 <= int(i) < 320)
    other_p2 = sum(1 for i in idx_p2 if 80 <= int(i) < 160)
    assert tool_p2 >= tool_q7
    assert tool_p2 > other_p2 or tool_p2 >= 48
    assert abs(len(idx_p2) - len(idx_q7)) <= 16
    # P2 should not starve tools relative to plain early-OTHER preference.
    assert tool_p2 >= other_p2 or tool_p2 > tool_q7


def test_load_linear_risk_fit_json():
    from pathlib import Path

    from prioritykv.linear_risk import load_linear_risk_config, score_page

    root = Path(__file__).resolve().parents[1]
    cfg = load_linear_risk_config(root / "configs" / "linear_risk_fit.json")
    assert score_page({"roles": ["tool"], "n_tokens": 16}, cfg) > score_page(
        {"roles": ["filler"], "n_tokens": 16}, cfg
    )


def test_fixed_hot_prefers_prefix():
    from prioritykv.baselines.keep_policy import select_fixed_hot, select_fixed_hot_pages

    cfg = KeepPolicyConfig(keep_frac=0.25, sink_tokens=16, force_recent=64)
    idx = select_fixed_hot(400, cfg)
    # Prefix bias: most early tokens kept beyond bare sink.
    early = sum(1 for i in idx if int(i) < 100)
    late_mid = sum(1 for i in idx if 200 <= int(i) < 300)
    assert early > late_mid
    cfg_p = KeepPolicyConfig(
        keep_frac=0.25, sink_tokens=16, force_recent=64, page_tokens=16, granularity="page"
    )
    idx_p = select_fixed_hot_pages(400, cfg_p)
    assert idx_p[0] == 0
    assert abs(len(idx_p) / 400 - 0.25) < 0.08 or len(idx_p) >= 16 + 64
