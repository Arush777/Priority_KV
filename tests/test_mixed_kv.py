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


def test_structure_protects_mid_state_under_filler_heavy_budget():
    """Mid-context OTHER stays BF16 under structure when filler can fill the INT4 quota."""
    n = 2048
    roles = [PageRole.FILLER] * n
    for i in range(16):
        roles[i] = PageRole.SINK
    # Gold block in the middle (mirrors relocate_state_to_middle).
    for i in range(900, 940):
        roles[i] = PageRole.OTHER
    cfg = MixedPlanConfig(int4_frac=0.75, sink_tokens=16, recent_window=128)
    s = plan_int4_mask(roles, cfg, policy="structure")
    u = plan_int4_mask(roles, cfg, policy="uniform")
    assert int(s.sum()) == int(u.sum())
    gold = range(900, 940)
    assert not any(s[i] for i in gold), "structure demoted mid-context gold"
    assert any(u[i] for i in gold), "uniform should hit mid-context gold"


def r_ok(i: int, n: int) -> bool:
    return 16 <= i < n - 128


def test_budget_zero_when_frac_zero():
    roles = _roles(256)
    cfg = MixedPlanConfig(int4_frac=0.0)
    assert int(plan_int4_mask(roles, cfg, policy="structure").sum()) == 0
    assert int(plan_int4_mask(roles, cfg, policy="uniform").sum()) == 0


def test_high_frac_still_matched_and_respects_sink_recent():
    n = 1024
    roles = _roles(n)
    cfg = MixedPlanConfig(int4_frac=0.92, sink_tokens=16, recent_window=128)
    s = plan_int4_mask(roles, cfg, policy="structure")
    u = plan_int4_mask(roles, cfg, policy="uniform")
    assert int(s.sum()) == int(u.sum())
    assert not s[:16].any() and not u[:16].any()
    assert not s[n - 128 :].any() and not u[n - 128 :].any()
    # Cap = all demotable positions (sink+recent forced BF16).
    max_int4 = n - 16 - 128
    assert int(s.sum()) == max_int4


def test_nbits2_has_higher_roundtrip_error_than_nbits4():
    from prioritykv.int4_kv import Int4KvConfig, fake_quant_roundtrip

    rng = np.random.default_rng(0)
    x = rng.standard_normal((8, 128)).astype(np.float32)
    e4 = float(np.mean((fake_quant_roundtrip(x, Int4KvConfig(nbits=4)) - x) ** 2))
    e2 = float(np.mean((fake_quant_roundtrip(x, Int4KvConfig(nbits=2)) - x) ** 2))
    assert e2 > e4 * 2.0


def test_flashinfer_script_rejects_illegal_head_dim():
    import subprocess
    import sys
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "scripts" / "run_flashinfer_lse_parity.py"
    proc = subprocess.run(
        [sys.executable, str(script), "--head-dim", "32"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "not in" in (proc.stdout + proc.stderr)
