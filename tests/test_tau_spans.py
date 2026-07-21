"""CPU tests for the generation-free τ-bench gold-span extraction and audit."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from prioritykv.external.tau_spans import (  # noqa: E402
    BURIED_CLASSES,
    SPAN_CLASSES,
    VISIBLE_CLASSES,
    Trajectory,
    aggregate,
    extract_spans,
    measure_retention,
    render,
    sample_for_manual_audit,
    stratified_sample,
)


def make_trajectory(messages, traj_id="m::0", task="airline", model="gpt-4.1"):
    return Trajectory(traj_id=traj_id, task_name=task, source_model=model,
                      messages=messages)


POLICY_TEXT = (
    "# Airline Agent Policy\n"
    "You must always verify the user id before any booking.\n"
    "Agents can never issue a refund without a certificate.\n"
    "Be friendly.\n"
)


def rich_messages():
    return [
        {"role": "system", "content": POLICY_TEXT},
        {"role": "user", "content": "Hi, my user id is mia_li_3668 please book me."},
        {"role": "assistant", "content": None,
         "tool_calls": [{"function": {"name": "get_user_details",
                                      "arguments": '{"user_id":"mia_li_3668"}'},
                         "id": "c1", "type": "function"}]},
        {"role": "tool", "content": '{"email": "mia.li3818@example.com", '
                                    '"reservation_id": "ZFA04Y"}'},
        {"role": "assistant", "content": "I found your profile, mia_li_3668."},
        {"role": "user", "content": "Actually, instead make it a return flight."},
        {"role": "assistant", "content": "Sent to mia.li3818@example.com"},
    ]


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def test_render_offsets_map_back_to_message_text():
    traj = make_trajectory(rich_messages())
    r = render(traj)
    assert len(r.message_offsets) == len(traj.messages)
    for (s, e), msg in zip(r.message_offsets, traj.messages):
        assert r.text[s:e].startswith(f"{msg['role']}:")
    assert r.message_offsets[-1][1] == len(r.text)


def test_render_includes_tool_call_names_and_arguments():
    r = render(make_trajectory(rich_messages()))
    assert "get_user_details" in r.text
    assert "mia_li_3668" in r.text


def test_render_handles_null_content():
    r = render(make_trajectory([{"role": "assistant", "content": None}]))
    assert "assistant:" in r.text


# --------------------------------------------------------------------------- #
# Span extraction
# --------------------------------------------------------------------------- #


def test_extract_finds_every_span_class():
    r = render(make_trajectory(rich_messages()))
    spans = extract_spans(r)
    found = {s.span_class for s in spans}
    for cls in ("tool_name", "tool_call_argument", "reused_identifier",
                "explicit_policy", "correction"):
        assert cls in found, f"missing {cls}; found {found}"


def test_span_offsets_slice_back_to_span_text():
    r = render(make_trajectory(rich_messages()))
    for s in extract_spans(r):
        assert r.text[s.start:s.end] == s.text, f"{s.span_class} offsets are wrong"


def test_explicit_policy_picks_imperatives_not_pleasantries():
    r = render(make_trajectory([{"role": "system", "content": POLICY_TEXT}]))
    policies = [s.text for s in extract_spans(r) if s.span_class == "explicit_policy"]
    assert any("must always verify" in p for p in policies)
    assert any("never issue a refund" in p for p in policies)
    assert not any("Be friendly" in p for p in policies)


def test_policy_lines_only_come_from_system_role():
    msgs = [{"role": "user", "content": "You must always give me a refund."}]
    spans = extract_spans(render(make_trajectory(msgs)))
    assert not [s for s in spans if s.span_class == "explicit_policy"]


def test_correction_requires_a_superseding_cue():
    plain = [{"role": "user", "content": "Please book the 9am flight."}]
    assert not [s for s in extract_spans(render(make_trajectory(plain)))
                if s.span_class == "correction"]
    fixed = [{"role": "user", "content": "Actually, make it the 10am flight."}]
    assert [s for s in extract_spans(render(make_trajectory(fixed)))
            if s.span_class == "correction"]


def test_identifier_must_recur_to_count_as_reused():
    once = [{"role": "user", "content": "my id is solo_user_9999"}]
    assert not [s for s in extract_spans(render(make_trajectory(once)))
                if s.span_class == "reused_identifier"]
    twice = [{"role": "user", "content": "my id is solo_user_9999"},
             {"role": "assistant", "content": "confirmed solo_user_9999"}]
    reused = [s for s in extract_spans(render(make_trajectory(twice)))
              if s.span_class == "reused_identifier"]
    assert len(reused) == 2


def test_tool_result_value_counts_only_when_reused_downstream():
    reused = [
        {"role": "tool", "content": '{"email": "abc.def@example.com"}'},
        {"role": "assistant", "content": "sent to abc.def@example.com"},
    ]
    assert [s for s in extract_spans(render(make_trajectory(reused)))
            if s.span_class == "reused_tool_result_value"]

    unused = [
        {"role": "tool", "content": '{"email": "abc.def@example.com"}'},
        {"role": "assistant", "content": "all done"},
    ]
    assert not [s for s in extract_spans(render(make_trajectory(unused)))
                if s.span_class == "reused_tool_result_value"]


def test_extraction_is_deterministic():
    r = render(make_trajectory(rich_messages()))
    a = [(s.span_class, s.start, s.end) for s in extract_spans(r)]
    b = [(s.span_class, s.start, s.end) for s in extract_spans(r)]
    assert a == b


def test_empty_trajectory_yields_no_spans():
    assert extract_spans(render(make_trajectory([]))) == []


def test_visible_and_buried_classes_partition_all_classes():
    assert VISIBLE_CLASSES | BURIED_CLASSES == set(SPAN_CLASSES)
    assert not (VISIBLE_CLASSES & BURIED_CLASSES)


# --------------------------------------------------------------------------- #
# Retention measurement
# --------------------------------------------------------------------------- #


class FakeSpan:
    def __init__(self, span_class, visible):
        self.span_class = span_class
        self.is_visible_structure = visible


def test_measure_retention_reports_any_all_and_fraction():
    spans = [(FakeSpan("tool_name", True), 0, 4),
             (FakeSpan("correction", False), 10, 14)]
    # Keep tokens 0,1 (partial) and none of 10..13.
    rets = measure_retention(spans, np.array([0, 1]), context_tokens=20)
    kept, dropped = rets[0], rets[1]
    assert kept.any_retained and not kept.all_retained
    assert kept.fraction_retained == pytest.approx(0.5)
    assert not dropped.any_retained
    assert dropped.fraction_retained == 0.0


def test_measure_retention_full_keep_retains_everything():
    spans = [(FakeSpan("tool_name", True), 0, 5)]
    rets = measure_retention(spans, np.arange(20), context_tokens=20)
    assert rets[0].all_retained and rets[0].fraction_retained == 1.0


def test_age_is_measured_from_the_decision_point():
    spans = [(FakeSpan("tool_name", True), 0, 5),
             (FakeSpan("tool_name", True), 90, 100)]
    rets = measure_retention(spans, np.arange(100), context_tokens=100)
    assert rets[0].age_tokens == 95
    assert rets[1].age_tokens == 0


def test_retention_ignores_out_of_range_keep_indices():
    spans = [(FakeSpan("tool_name", True), 0, 2)]
    rets = measure_retention(spans, np.array([-5, 0, 999]), context_tokens=10)
    assert rets[0].n_retained == 1


def test_aggregate_splits_visible_from_buried():
    spans = [(FakeSpan("tool_name", True), 0, 2),
             (FakeSpan("correction", False), 8, 10)]
    out = aggregate(measure_retention(spans, np.array([0, 1]), context_tokens=10))
    assert out["by_visibility"]["visible_structure"]["any_retained_rate"] == 1.0
    assert out["by_visibility"]["buried"]["any_retained_rate"] == 0.0


def test_aggregate_of_nothing_is_empty_not_an_error():
    assert aggregate([])["n_spans"] == 0


# --------------------------------------------------------------------------- #
# Sampling for manual audit
# --------------------------------------------------------------------------- #


def test_manual_sample_is_deterministic_and_unbiased_by_class():
    from prioritykv.external.tau_spans import Span

    spans = [(f"t{i}", Span(SPAN_CLASSES[i % len(SPAN_CLASSES)], f"x{i}", 0, 1, 0, "user"))
             for i in range(500)]
    a = sample_for_manual_audit(spans, n=50, seed=0)
    b = sample_for_manual_audit(spans, n=50, seed=0)
    assert a == b
    assert len(a) == 50
    # Sampling is uniform over all spans, so every class should appear.
    assert len({r["span_class"] for r in a}) == len(SPAN_CLASSES)


def test_manual_sample_rows_await_human_review():
    from prioritykv.external.tau_spans import Span

    spans = [("t0", Span("tool_name", "cd", 0, 2, 0, "assistant"))]
    row = sample_for_manual_audit(spans, n=1, seed=0)[0]
    assert row["correct_extraction"] is None


def test_manual_sample_handles_fewer_spans_than_requested():
    from prioritykv.external.tau_spans import Span

    spans = [("t0", Span("tool_name", "cd", 0, 2, 0, "assistant"))]
    assert len(sample_for_manual_audit(spans, n=100, seed=0)) == 1


def test_stratified_sample_spreads_across_task_and_model():
    trajs = [make_trajectory([], traj_id=f"{m}::{i}", task=t, model=m)
             for m in ("gpt-4.1", "o3-high") for t in ("airline", "retail")
             for i in range(50)]
    picked = stratified_sample(trajs, n=40, seed=0)
    assert len(picked) == 40
    assert len({(t.task_name, t.source_model) for t in picked}) == 4


def test_stratified_sample_is_deterministic():
    trajs = [make_trajectory([], traj_id=f"m::{i}") for i in range(100)]
    a = [t.traj_id for t in stratified_sample(trajs, n=20, seed=0)]
    b = [t.traj_id for t in stratified_sample(trajs, n=20, seed=0)]
    assert a == b
