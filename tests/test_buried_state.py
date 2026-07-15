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


def test_relocate_moves_gold_out_of_prefix():
    from prioritykv.baselines.buried_state import (
        _is_filler_turn,
        relocate_state_to_middle,
    )

    msgs = [
        {"role": "system", "content": "You are an order desk agent."},
        {"role": "user", "content": "Create a hold for order ORD-777. Confirm the id."},
        {"role": "assistant", "content": "Hold created for ORD-777."},
        {"role": "user", "content": "[filler/logs] record=1 status=ok " * 40},
        {"role": "assistant", "content": "Acknowledged filler update on logs; ref=1."},
        {"role": "user", "content": "[filler/logs] record=2 status=ok " * 40},
        {"role": "assistant", "content": "Acknowledged filler update on logs; ref=2."},
        {"role": "user", "content": "FINAL: Reply with ORDER_ID=<the hold id>."},
    ]
    out = relocate_state_to_middle(msgs, position=0.5)
    # System stays first, FINAL stays last.
    assert out[0]["role"] == "system"
    assert out[-1]["content"].startswith("FINAL")
    # Gold turn no longer sits immediately after system (prefix now filler).
    assert _is_filler_turn(out[1])
    # Gold content is preserved somewhere in the middle.
    joined = " ".join(m["content"] for m in out)
    assert "ORD-777" in joined
    gold_idx = next(i for i, m in enumerate(out) if "ORD-777" in m["content"])
    assert 0 < gold_idx < len(out) - 1


def test_relocate_preserves_gold_order():
    from prioritykv.baselines.buried_state import relocate_state_to_middle

    msgs = [
        {"role": "system", "content": "Obey the latest constraint."},
        {"role": "user", "content": "Constraint v1: start with [[FMT:A]]."},
        {"role": "assistant", "content": "Understood, [[FMT:A]]."},
        {"role": "user", "content": "Constraint v2 (REPLACES v1): start with [[FMT:B]]."},
        {"role": "assistant", "content": "Updated, [[FMT:B]] only."},
        {"role": "user", "content": "[filler/notes] record=9 status=ok " * 40},
        {"role": "assistant", "content": "Acknowledged filler update on notes; ref=9."},
        {"role": "user", "content": "FINAL: emit the latest format tag."},
    ]
    out = relocate_state_to_middle(msgs, position=0.5)
    order = [m["content"] for m in out if "Constraint" in m["content"]]
    assert order[0].startswith("Constraint v1")
    assert order[1].startswith("Constraint v2")


def test_no_final_keying_in_roles():
    from prioritykv.baselines.keep_policy import _message_role_stress
    from prioritykv.page_roles import PageRole

    r = _message_role_stress({"role": "user", "content": "FINAL: please answer now " * 40})
    # Long FINAL-marked filler is FILLER, not auto-RECENT via string match.
    assert r == PageRole.FILLER
