"""Tests for buried-state transform."""

from __future__ import annotations

from prioritykv.baselines.buried_state import bury_short_state_turns


def test_bury_pads_short_nonfinal():
    msgs = [
        {"role": "user", "content": "Hold ORD-123"},
        {"role": "assistant", "content": "ok ORD-123"},
        {"role": "user", "content": "FINAL: ORDER_ID=<earlier>"},
    ]
    out = bury_short_state_turns(msgs, min_len=520, seed=1)
    assert len(out[0]["content"]) >= 520
    assert "ORD-123" in out[0]["content"]
    assert out[-1]["content"].startswith("FINAL")
    assert len(out[-1]["content"]) < 520


def test_no_final_keying_in_roles():
    from prioritykv.baselines.keep_policy import _message_role_stress
    from prioritykv.page_roles import PageRole

    r = _message_role_stress({"role": "user", "content": "FINAL: please answer now " * 40})
    # Long FINAL-marked filler is FILLER, not auto-RECENT via string match.
    assert r == PageRole.FILLER
