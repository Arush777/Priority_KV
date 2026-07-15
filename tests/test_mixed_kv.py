"""Tests for the per-position BF16/INT4 mixed-precision plan (systems half)."""

from __future__ import annotations

import numpy as np

from prioritykv.mixed_kv import MixedPlanConfig, plan_int4_mask
from prioritykv.page_roles import PROTECTED_ROLES, PageRole


def _roles(n: int) -> list[PageRole]:
    # sink | tool(struct) | filler... | short-state OTHER | filler... | recent
    roles = [PageRole.FILLER] * n
    for i in range(16):
        roles[i] = PageRole.SINK
    for i in range(16, 48):
        roles[i] = PageRole.TOOL  # mid-ish structural block
    roles[n // 2] = PageRole.OTHER  # buried short state
    return roles


def test_matched_int4_budget_between_policies():
    n = 512
    roles = _roles(n)
    cfg = MixedPlanConfig(int4_frac=0.6, sink_tokens=16, recent_window=128)
    s = plan_int4_mask(roles, cfg, policy="structure")
    u = plan_int4_mask(roles, cfg, policy="uniform")
    # Byte-fair: same number of INT4 positions.
    assert int(s.sum()) == int(u.sum())
    assert int(s.sum()) > 0


def test_sink_and_recent_never_int4():
    n = 512
    roles = _roles(n)
    cfg = MixedPlanConfig(int4_frac=0.9, sink_tokens=16, recent_window=128)
    for pol in ("structure", "uniform"):
        m = plan_int4_mask(roles, cfg, policy=pol)
        assert not m[:16].any(), f"{pol} demoted a sink position"
        assert not m[n - 128 :].any(), f"{pol} demoted a recent position"


def test_structure_protects_roles_when_budget_allows():
    n = 512
    roles = _roles(n)
    cfg = MixedPlanConfig(int4_frac=0.5, sink_tokens=16, recent_window=128)
    s = plan_int4_mask(roles, cfg, policy="structure")
    u = plan_int4_mask(roles, cfg, policy="uniform")
    struct_pos = [
        i for i, r in enumerate(roles) if r in PROTECTED_ROLES or r == PageRole.OTHER
    ]
    struct_pos = [i for i in struct_pos if r_ok(i, n)]
    # Structure keeps structural positions BF16; uniform (role-blind) hits some.
    s_struct_int4 = sum(int(s[i]) for i in struct_pos)
    u_struct_int4 = sum(int(u[i]) for i in struct_pos)
    assert s_struct_int4 == 0
    assert u_struct_int4 > 0


def r_ok(i: int, n: int) -> bool:
    return 16 <= i < n - 128


def test_budget_zero_when_frac_zero():
    roles = _roles(256)
    cfg = MixedPlanConfig(int4_frac=0.0)
    assert int(plan_int4_mask(roles, cfg, policy="structure").sum()) == 0
    assert int(plan_int4_mask(roles, cfg, policy="uniform").sum()) == 0
