"""CPU tests for the EXTERNAL_BFCL_PRAJNA_V1 harness.

These must all pass before any GPU time is reserved: they cover work identity,
deterministic sampling, context exclusion, matched keep budgets, the no-SnapKV-
fallback assertion, atomic checkpoint/resume, and paired completeness.
"""

from __future__ import annotations

import json
import signal
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402
from prioritykv.baselines.keep_policy import (  # noqa: E402
    KeepPolicyConfig,
    select_keep_indices,
)
from prioritykv.external.arms import (  # noqa: E402
    SnapKVUnavailableError,
    assert_real_snapkv,
    check_matched_budget,
    keep_budget,
)
from prioritykv.external.bfcl_data import BfclTask, balanced_sample, work_id  # noqa: E402
from prioritykv.external.bfcl_rollout import (  # noqa: E402
    ContextLimitExceeded,
    pad_decoded_to_turns,
    run_rollout,
)
from prioritykv.external.checkpoint import (  # noqa: E402
    ResultStore,
    atomic_write_json,
    build_shards,
    completed_work_ids,
    load_valid_point,
    pending_work_items,
    write_point,
)
from prioritykv.external.stats import (  # noqa: E402
    build_paired_table,
    exact_mcnemar_p,
    paired_bootstrap_ci,
    paired_completeness,
    restrict_to_common,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def make_task(task_id="multi_turn_base_0", category="base", n_turns=2) -> BfclTask:
    return BfclTask(
        task_id=task_id,
        category=category,
        question=[[{"role": "user", "content": f"turn {i}"}] for i in range(n_turns)],
        initial_config={"GorillaFileSystem": {}},
        involved_classes=["GorillaFileSystem"],
        ground_truth=[[f"cd(folder='d{i}')"] for i in range(n_turns)],
        function=[{"name": "cd", "description": "change dir"}],
    )


class FakeGenResult:
    def __init__(self, text, prompt_tokens, requested, realized):
        self.text = text
        self.prompt_tokens = prompt_tokens
        self.requested_keep = requested
        self.realized_keep = realized
        self.kept_indices = None
        self.timings = {"select_s": 0.0, "generate_s": 0.0, "total_s": 0.0}
        self.extra = {}


class FakeGenerator:
    """Emits a scripted sequence of raw model outputs."""

    def __init__(self, arm, outputs, prompt_tokens=100):
        self.arm = arm
        self.outputs = list(outputs)
        self.prompt_tokens = prompt_tokens
        self.calls = 0

    def generate(self, messages, max_new_tokens):
        out = self.outputs[min(self.calls, len(self.outputs) - 1)]
        self.calls += 1
        n = self.prompt_tokens + 10 * self.calls
        return FakeGenResult(out, n, n // 4, n // 4)


def fake_decode(raw: str) -> list[str]:
    raw = raw.strip()
    if not raw or raw == "[]":
        return []
    return [c.strip() for c in raw.strip("[]").split(";") if c.strip()]


def fake_execute(**kwargs):
    return [f"ok:{c}" for c in kwargs["func_call_list"]], {}


def fake_is_empty(decoded) -> bool:
    return len(decoded) == 0


# --------------------------------------------------------------------------- #
# Work identity
# --------------------------------------------------------------------------- #


def _wid(**over):
    base = dict(
        dataset_revision="deadbeef", task_id="multi_turn_base_0",
        model_revision="cafebabe", arm="structure", keep_frac=0.25,
        seed=0, harness_revision="abc123", decision_turn="all",
    )
    base.update(over)
    return work_id(**base)


def test_work_id_is_stable_and_deterministic():
    assert _wid() == _wid()
    assert len(_wid()) == 64


@pytest.mark.parametrize(
    "field,value",
    [
        ("dataset_revision", "other"), ("task_id", "multi_turn_base_1"),
        ("model_revision", "other"), ("arm", "uniform"), ("keep_frac", 0.10),
        ("seed", 1), ("harness_revision", "def456"), ("decision_turn", 3),
    ],
)
def test_work_id_changes_with_every_identity_field(field, value):
    assert _wid(**{field: value}) != _wid()


def test_work_id_keep_frac_is_not_float_fragile():
    assert _wid(keep_frac=0.25) == _wid(keep_frac=0.250000)


# --------------------------------------------------------------------------- #
# Deterministic balanced sampling
# --------------------------------------------------------------------------- #


def _pool(n_per_cat=50):
    return [
        make_task(task_id=f"multi_turn_{c}_{i}", category=c)
        for c in ("base", "miss_param", "miss_func", "long_context")
        for i in range(n_per_cat)
    ]


def test_balanced_sample_is_deterministic_and_balanced():
    quota = {"base": 10, "miss_param": 10, "miss_func": 10, "long_context": 10}
    a = balanced_sample(_pool(), per_category=quota, seed=0)
    b = balanced_sample(_pool(), per_category=quota, seed=0)
    assert [t.task_id for t in a] == [t.task_id for t in b]
    counts = {c: sum(1 for t in a if t.category == c) for c in quota}
    assert counts == quota


def test_balanced_sample_is_insensitive_to_input_order():
    quota = {"base": 5, "miss_param": 5, "miss_func": 5, "long_context": 5}
    pool = _pool()
    a = balanced_sample(pool, per_category=quota, seed=0)
    b = balanced_sample(list(reversed(pool)), per_category=quota, seed=0)
    assert [t.task_id for t in a] == [t.task_id for t in b]


def test_balanced_sample_is_prefix_stable_when_n_grows():
    """Growing 400 -> 600 must not reshuffle the tasks already chosen."""
    small = balanced_sample(_pool(), per_category={"base": 10}, seed=0)
    large = balanced_sample(_pool(), per_category={"base": 20}, seed=0)
    assert {t.task_id for t in small} <= {t.task_id for t in large}


def test_balanced_sample_rejects_oversized_request():
    with pytest.raises(ValueError, match="only"):
        balanced_sample(_pool(n_per_cat=5), per_category={"base": 99}, seed=0)


# --------------------------------------------------------------------------- #
# Matched keep budget
# --------------------------------------------------------------------------- #


def test_keep_budget_is_identical_across_arms():
    cfg = KeepPolicyConfig(keep_frac=0.25, sink_tokens=16, force_recent=128)
    for n in (200, 1000, 4096, 16384):
        budgets = {arm: keep_budget(n, cfg) for arm in ("structure", "uniform",
                                                        "random", "snapkv")}
        assert len(set(budgets.values())) == 1, budgets


def test_keep_budget_never_exceeds_context():
    cfg = KeepPolicyConfig(keep_frac=0.25, sink_tokens=16, force_recent=128)
    for n in (1, 10, 100, 144):
        assert keep_budget(n, cfg) <= n


def test_check_matched_budget_flags_mismatch():
    ok, _ = check_matched_budget({"structure": 100, "uniform": 100, "snapkv": 100})
    assert ok
    bad, msg = check_matched_budget({"structure": 100, "uniform": 100, "snapkv": 87})
    assert not bad and "mismatch" in msg


def test_check_matched_budget_ignores_fullkv():
    ok, _ = check_matched_budget({"full": 4096, "structure": 100, "uniform": 100})
    assert ok


def test_realized_keep_matches_requested_for_token_gather_arms():
    """The three deterministic policies must land on the shared budget exactly."""
    from prioritykv.baselines.keep_policy import select_keep_indices
    from prioritykv.page_roles import PageRole

    cfg = KeepPolicyConfig(keep_frac=0.25, sink_tokens=16, force_recent=128, seed=0)
    n = 4096
    roles = [PageRole.FILLER] * n
    for i in range(0, 512):
        roles[i] = PageRole.TOOL
    want = keep_budget(n, cfg)
    for policy in ("uniform", "random", "structure"):
        idx = select_keep_indices(n, cfg, policy=policy, roles=roles)
        assert len(idx) == want, f"{policy}: {len(idx)} != {want}"
        assert len(set(idx.tolist())) == len(idx), f"{policy} produced duplicates"


# --------------------------------------------------------------------------- #
# No SnapKV fallback
# --------------------------------------------------------------------------- #


def test_assert_real_snapkv_rejects_impostor():
    class DropKeep:
        pass

    with pytest.raises(SnapKVUnavailableError, match="SnapKVPress"):
        assert_real_snapkv(DropKeep())


def test_assert_real_snapkv_accepts_genuine_press():
    kvpress = pytest.importorskip("kvpress")
    assert_real_snapkv(kvpress.SnapKVPress(compression_ratio=0.75))


def test_other_kvpress_presses_are_rejected():
    """A different real press is still not SnapKV."""
    kvpress = pytest.importorskip("kvpress")
    with pytest.raises(SnapKVUnavailableError):
        assert_real_snapkv(kvpress.KnormPress(compression_ratio=0.75))


# --------------------------------------------------------------------------- #
# Rollout: context limit, step limit, turn structure
# --------------------------------------------------------------------------- #


def _rollout(gen, task, **over):
    kwargs = dict(
        system_prompt="sys", decode_execute=fake_decode, execute_calls=fake_execute,
        is_empty_response=fake_is_empty, max_step_limit=20, max_new_tokens=32,
    )
    kwargs.update(over)
    return run_rollout(task, gen, **kwargs)


def test_rollout_produces_one_step_list_per_turn():
    task = make_task(n_turns=3)
    gen = FakeGenerator("full", ["[]"])
    res = _rollout(gen, task)
    assert len(res.model_result_decoded) == 3
    assert res.terminal_status == "success"


def test_rollout_executes_calls_then_ends_turn_on_empty():
    task = make_task(n_turns=1)
    gen = FakeGenerator("full", ["[cd(folder='d0')]", "[]"])
    res = _rollout(gen, task)
    assert res.model_result_decoded[0][0] == ["cd(folder='d0')"]
    assert res.model_result_decoded[0][-1] == []


def test_rollout_raises_rather_than_truncating_over_context():
    task = make_task(n_turns=1)
    gen = FakeGenerator("full", ["[]"], prompt_tokens=50_000)
    with pytest.raises(ContextLimitExceeded) as exc:
        _rollout(gen, task, prompt_token_ceiling=30_720)
    assert exc.value.prompt_tokens > exc.value.limit


def test_rollout_force_quits_at_step_limit():
    task = make_task(n_turns=2)
    gen = FakeGenerator("full", ["[cd(folder='x')]"])  # never returns empty
    res = _rollout(gen, task, max_step_limit=5)
    assert res.force_quit
    assert res.terminal_status == "force_quit"
    assert gen.calls <= 6


def test_rollout_treats_undecodable_output_as_end_of_turn():
    def boom(_raw):
        raise ValueError("cannot parse")

    res = _rollout(FakeGenerator("full", ["garbage"]), make_task(n_turns=1),
                   decode_execute=boom)
    assert res.model_result_decoded == [[[]]]
    assert res.terminal_status == "success"


def test_force_quit_rollout_is_padded_to_full_turn_count_not_dropped():
    """A truncated rollout must be scoreable as a failure, never skipped."""
    padded = pad_decoded_to_turns([[["cd()"]]], 4)
    assert len(padded) == 4
    assert padded[1:] == [[], [], []]


def test_pad_never_lengthens_beyond_expected_turns():
    assert len(pad_decoded_to_turns([[[]], [[]], [[]]], 2)) == 2


# --------------------------------------------------------------------------- #
# Atomic checkpointing and resume
# --------------------------------------------------------------------------- #


def _point(work_id_value="w1", **over):
    p = {
        "work_id": work_id_value, "freeze_id": "EXTERNAL_BFCL_PRAJNA_V1",
        "dataset_revision": "rev", "task_id": "t1", "category": "base",
        "model_id": "Qwen/Qwen3-8B", "model_revision": "mrev", "arm": "structure",
        "keep_frac": 0.25, "seed": 0, "harness_revision": "h",
        "terminal_status": "success",
    }
    p.update(over)
    return p


def test_write_and_reload_point(tmp_path):
    store = ResultStore(tmp_path).ensure()
    write_point(store, _point())
    assert load_valid_point(store.point_path("w1")) is not None
    assert completed_work_ids(store) == {"w1"}


def test_incomplete_point_is_not_treated_as_complete(tmp_path):
    store = ResultStore(tmp_path).ensure()
    with pytest.raises(ValueError, match="invalid point"):
        write_point(store, {"work_id": "w2"})


def test_corrupt_point_is_retried_not_skipped(tmp_path):
    store = ResultStore(tmp_path).ensure()
    store.point_path("w3").write_text("{ truncated json")
    assert load_valid_point(store.point_path("w3")) is None
    assert completed_work_ids(store) == set()
    pending = pending_work_items([{"work_id": "w3"}], store)
    assert [w["work_id"] for w in pending] == ["w3"]


def test_point_missing_required_field_is_invalid(tmp_path):
    store = ResultStore(tmp_path).ensure()
    bad = _point("w4")
    del bad["terminal_status"]
    atomic_write_json(store.point_path("w4"), bad)
    assert load_valid_point(store.point_path("w4")) is None
    assert completed_work_ids(store) == set()


def test_restart_skips_only_complete_points(tmp_path):
    store = ResultStore(tmp_path).ensure()
    write_point(store, _point("done"))
    store.point_path("corrupt").write_text("nope")
    items = [{"work_id": "done"}, {"work_id": "corrupt"}, {"work_id": "fresh"}]
    assert {w["work_id"] for w in pending_work_items(items, store)} == {"corrupt", "fresh"}


def test_atomic_write_leaves_no_tmp_files(tmp_path):
    atomic_write_json(tmp_path / "a.json", {"k": "v"})
    assert list(tmp_path.glob("*.tmp")) == []
    assert json.loads((tmp_path / "a.json").read_text()) == {"k": "v"}


def test_atomic_write_does_not_clobber_on_failure(tmp_path):
    path = tmp_path / "keep.json"
    atomic_write_json(path, {"good": True})

    # Arbitrary objects are now stringified by design (the official checker
    # embeds live API instances), so provoke a real encoder failure instead.
    circular: dict = {}
    circular["self"] = circular

    with pytest.raises(ValueError, match="[Cc]ircular"):
        atomic_write_json(path, circular)
    assert json.loads(path.read_text()) == {"good": True}
    assert list(tmp_path.glob("*.tmp")) == []


def test_unserialisable_objects_are_stringified_not_dropped(tmp_path):
    """A point must survive the checker leaving live objects in its output."""

    class Directory:
        def __repr__(self):
            return "Directory(/tmp)"

    out = atomic_write_json(tmp_path / "p.json",
                            {"verdict": {"obj": Directory(), "s": {"b", "a"}}})
    payload = json.loads(out.read_text())
    assert payload["verdict"]["obj"] == "Directory(/tmp)"
    assert payload["verdict"]["s"] == ["a", "b"]


def test_sigterm_mid_run_leaves_no_partial_point(tmp_path):
    """Kill a writer between points; the store must stay parseable."""
    script = textwrap.dedent(f"""
        import os, signal, sys, time
        sys.path.insert(0, {str(REPO_ROOT / 'src')!r})
        from prioritykv.external.checkpoint import ResultStore, write_point
        store = ResultStore({str(tmp_path)!r}).ensure()
        base = dict(freeze_id="F", dataset_revision="r", task_id="t", category="base",
                    model_id="m", model_revision="mr", arm="structure", keep_frac=0.25,
                    seed=0, harness_revision="h", terminal_status="success")
        for i in range(200):
            write_point(store, dict(base, work_id=f"w{{i}}"))
            print(i, flush=True)
            time.sleep(0.01)
    """)
    proc = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE,
                            text=True)
    # Let it get going, then terminate mid-stream.
    for _ in range(5):
        if proc.stdout.readline() == "":
            break
    proc.send_signal(signal.SIGTERM)
    proc.wait(timeout=30)

    store = ResultStore(tmp_path)
    assert list(store.points.glob("*.tmp")) == []
    for f in store.points.glob("*.json"):
        assert load_valid_point(f) is not None, f"{f} is a torn write"


def test_shards_cover_every_work_item_exactly_once():
    items = [{"work_id": f"w{i}"} for i in range(107)]
    shards = build_shards(items, 25)
    assert len(shards) == 5
    seen = [w["work_id"] for s in shards for w in s.work_items]
    assert seen == [w["work_id"] for w in items]


# --------------------------------------------------------------------------- #
# Paired statistics
# --------------------------------------------------------------------------- #


def test_paired_table_counts_discordance():
    a = {"t1": True, "t2": True, "t3": False, "t4": False}
    b = {"t1": True, "t2": False, "t3": True, "t4": False}
    table = build_paired_table("structure", "snapkv", a, b)
    assert (table.both_pass, table.a_only, table.b_only, table.both_fail) == (1, 1, 1, 1)


def test_paired_table_uses_only_shared_tasks():
    table = build_paired_table("a", "b", {"t1": True, "t2": True}, {"t1": False})
    assert table.n_paired == 1


def test_exact_mcnemar_matches_known_values():
    # The frozen core reports p=0.125 for a 3-vs-0 discordant split.
    assert exact_mcnemar_p(3, 0) == pytest.approx(0.25)
    assert exact_mcnemar_p(4, 0) == pytest.approx(0.125)
    assert exact_mcnemar_p(0, 0) == 1.0
    assert exact_mcnemar_p(5, 5) == 1.0


def test_exact_mcnemar_is_symmetric():
    assert exact_mcnemar_p(7, 2) == exact_mcnemar_p(2, 7)


def test_exact_mcnemar_detects_large_imbalance():
    assert exact_mcnemar_p(20, 2) < 0.05


def test_paired_bootstrap_ci_brackets_the_difference():
    a = {f"t{i}": i % 2 == 0 for i in range(200)}
    b = {f"t{i}": False for i in range(200)}
    out = paired_bootstrap_ci(a, b, n_boot=2000, seed=0)
    assert out["ci_low"] <= out["diff"] <= out["ci_high"]
    assert out["ci_low"] > 0


def test_paired_bootstrap_ci_of_identical_arms_contains_zero():
    a = {f"t{i}": i % 3 == 0 for i in range(100)}
    out = paired_bootstrap_ci(a, dict(a), n_boot=1000, seed=0)
    assert out["diff"] == 0.0
    assert out["ci_low"] <= 0 <= out["ci_high"]


def test_paired_completeness_reports_missing_tasks():
    expected = ["t1", "t2", "t3", "t4"]
    outcomes = {
        "full": {"t1": True, "t2": True, "t3": True, "t4": True},
        "structure": {"t1": True, "t2": False, "t3": True},
    }
    rep = paired_completeness(expected, outcomes)
    assert rep.n_tasks_all_arms == 3
    assert rep.paired_completeness == pytest.approx(0.75)
    assert rep.missing_by_arm["structure"] == ["t4"]


def test_restrict_to_common_equalises_arm_sizes():
    outcomes = {
        "full": {"t1": True, "t2": True, "t3": False},
        "structure": {"t1": False, "t2": True},
        "uniform": {"t2": True, "t3": True},
    }
    common = restrict_to_common(outcomes)
    assert {len(v) for v in common.values()} == {1}
    assert set(common["full"]) == {"t2"}


def test_unknown_realized_keep_is_not_treated_as_matched():
    """An unmeasurable keep count must fail the budget check, not pass silently."""
    ok, msg = check_matched_budget({"structure": 100, "uniform": 100, "snapkv": -1})
    assert not ok and "unknown" in msg


def test_split_reasoning_recovers_the_tool_call():
    """A thinking-mode answer must reach the decoder, not the <think> block."""
    from prioritykv.external.bfcl_rollout import split_reasoning

    raw = "<think>\nI should fill the tank.\n</think>\n\n[fillFuelTank(fuelAmount=30)]"
    reasoning, answer = split_reasoning(raw)
    assert answer == "[fillFuelTank(fuelAmount=30)]"
    assert "fill the tank" in reasoning


def test_split_reasoning_passes_plain_output_through():
    from prioritykv.external.bfcl_rollout import split_reasoning

    assert split_reasoning("[foo(a=1)]") == ("", "[foo(a=1)]")


def test_rollout_decodes_the_answer_not_the_reasoning():
    task = make_task(n_turns=1)
    gen = FakeGenerator("full", ["<think>musing</think>\n[cd(folder='d0')]", "[]"])
    res = _rollout(gen, task)
    assert res.model_result_decoded[0][0] == ["cd(folder='d0')"]


def test_frozen_random_is_actually_uniform():
    """Documents the frozen-core defect the external namespace works around."""
    from prioritykv.baselines.keep_policy import select_keep_indices

    cfg = KeepPolicyConfig(keep_frac=0.25, sink_tokens=16, force_recent=128, seed=0)
    for n in (600, 4096, 40000):
        u = select_keep_indices(n, cfg, policy="uniform")
        r = select_keep_indices(n, cfg, policy="random")
        assert np.array_equal(u, r), f"frozen random unexpectedly differs at n={n}"


def test_external_random_is_genuinely_random_and_matched():
    """The corrected control must differ from uniform yet keep the same count."""
    from prioritykv.external.arms import select_random_external

    cfg = KeepPolicyConfig(keep_frac=0.25, sink_tokens=16, force_recent=128, seed=0)
    for n in (600, 4096, 40000):
        u = select_keep_indices(n, cfg, policy="uniform")
        r = select_random_external(n, cfg)
        assert len(r) == keep_budget(n, cfg), f"budget not matched at n={n}"
        assert not np.array_equal(u, r), f"still identical to uniform at n={n}"
        assert len(set(r.tolist())) == len(r), "duplicate indices"


def test_external_random_is_deterministic_per_seed():
    from prioritykv.external.arms import select_random_external

    cfg = KeepPolicyConfig(keep_frac=0.25, sink_tokens=16, force_recent=128, seed=0)
    other = KeepPolicyConfig(keep_frac=0.25, sink_tokens=16, force_recent=128, seed=1)
    a = select_random_external(4096, cfg)
    assert np.array_equal(a, select_random_external(4096, cfg))
    assert not np.array_equal(a, select_random_external(4096, other))


def test_external_random_keeps_sink_and_recent():
    from prioritykv.external.arms import select_random_external

    cfg = KeepPolicyConfig(keep_frac=0.25, sink_tokens=16, force_recent=128, seed=0)
    n = 4096
    idx = set(select_random_external(n, cfg).tolist())
    assert set(range(16)) <= idx
    assert set(range(n - 128, n)) <= idx
