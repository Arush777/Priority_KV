"""Integration tests against the *official* BFCL scorer on known examples.

The handoff requires proving the official scorer is wired up correctly before
GPU time is spent. The strongest available known-good example is the dataset's
own ground truth: replaying it must score valid, and corrupting it must not.

These skip cleanly when the pinned Gorilla checkout is absent, so the suite still
runs on a machine without the external data staged.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from prioritykv.external.bfcl_data import load_tasks  # noqa: E402
from prioritykv.external.bfcl_official import (  # noqa: E402
    BfclUnavailableError,
    head_revision,
    load_official,
    reset_execution_instances,
)
from prioritykv.external.bfcl_rollout import (  # noqa: E402
    RolloutResult,
    pad_decoded_to_turns,
    score_rollout,
)
from prioritykv.external.config import load_config  # noqa: E402

PINNED_REVISION = "cd9429ccf3d4d04156affe883c495b3b047e6b64"


def _gorilla_root() -> str:
    for candidate in (
        os.environ.get("PKV_GORILLA_ROOT"),
        os.path.expandvars("$PRAJNA_ROOT/gorilla"),
        str(Path.home() / "prioritykv_prajna" / "gorilla"),
    ):
        if candidate and (Path(candidate) / "berkeley-function-call-leaderboard").is_dir():
            return candidate
    pytest.skip("pinned Gorilla checkout not staged")


@pytest.fixture(scope="module")
def official():
    return load_official(_gorilla_root())


@pytest.fixture(scope="module")
def tasks(official):
    return load_tasks(_gorilla_root(),
                      doc_mapping=official["MULTI_TURN_FUNC_DOC_FILE_MAPPING"])


@pytest.fixture(autouse=True)
def _clean_execution_state():
    """Stateful API instances must not leak between scoring runs."""
    reset_execution_instances()
    yield
    reset_execution_instances()


def _rollout_from(task, decoded, status="success"):
    return RolloutResult(task_id=task.task_id, arm="test",
                         model_result_decoded=decoded, terminal_status=status)


def _score(task, decoded, official, name):
    return score_rollout(task, _rollout_from(task, decoded),
                         multi_turn_checker=official["multi_turn_checker"],
                         model_name=name)


def _by_id(tasks, task_id):
    for t in tasks:
        if t.task_id == task_id:
            return t
    pytest.skip(f"{task_id} not in pinned dataset")


# --------------------------------------------------------------------------- #
# Pinning
# --------------------------------------------------------------------------- #


def test_checkout_is_on_the_frozen_revision():
    assert head_revision(_gorilla_root()) == PINNED_REVISION


def test_assert_pinned_revision_rejects_a_different_commit():
    from prioritykv.external.bfcl_official import assert_pinned_revision

    with pytest.raises(BfclUnavailableError, match="refusing to score"):
        assert_pinned_revision(_gorilla_root(), "0" * 40)


def test_v3_has_four_categories_of_two_hundred(tasks):
    """Guards the handoff's incorrect 5-category/1000-case premise."""
    counts: dict[str, int] = {}
    for t in tasks:
        counts[t.category] = counts.get(t.category, 0) + 1
    assert counts == {"base": 200, "miss_param": 200,
                      "miss_func": 200, "long_context": 200}
    assert "composite" not in counts


# --------------------------------------------------------------------------- #
# Known-good / known-bad scoring
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("task_id", ["multi_turn_base_0", "multi_turn_base_1",
                                     "multi_turn_base_2"])
def test_ground_truth_replay_scores_valid(tasks, official, task_id):
    """The canonical known-good example: the dataset's own answer must pass."""
    task = _by_id(tasks, task_id)
    decoded = [[turn] for turn in task.ground_truth]
    verdict = _score(task, decoded, official, f"gt_{task_id}")
    assert verdict.get("valid") is True, verdict


def test_empty_model_output_scores_invalid(tasks, official):
    task = _by_id(tasks, "multi_turn_base_0")
    decoded = [[[]] for _ in task.ground_truth]
    verdict = _score(task, decoded, official, "empty_base_0")
    assert verdict.get("valid") is False
    assert "empty" in str(verdict.get("error_type", "")).lower()


def test_wrong_arguments_score_invalid(tasks, official):
    """Right function, wrong argument values must fail the state check."""
    task = _by_id(tasks, "multi_turn_base_0")
    decoded = [[[c.replace("'", "'wrong_") for c in turn]] for turn in task.ground_truth]
    verdict = _score(task, decoded, official, "wrongargs_base_0")
    assert verdict.get("valid") is False


def test_dropping_a_turn_scores_invalid(tasks, official):
    task = _by_id(tasks, "multi_turn_base_0")
    if len(task.ground_truth) < 2:
        pytest.skip("needs a multi-turn task")
    decoded = [[turn] for turn in task.ground_truth]
    decoded[-1] = [[]]
    verdict = _score(task, decoded, official, "dropturn_base_0")
    assert verdict.get("valid") is False


def test_scoring_is_deterministic_across_repeats(tasks, official):
    """Leaked API state would make a replay's second run disagree."""
    task = _by_id(tasks, "multi_turn_base_0")
    decoded = [[turn] for turn in task.ground_truth]
    first = _score(task, decoded, official, "det_a")
    reset_execution_instances()
    second = _score(task, decoded, official, "det_b")
    assert first.get("valid") == second.get("valid") is True


def test_force_quit_rollout_is_scored_as_failure_not_skipped(tasks, official):
    """A truncated rollout must reach the scorer and fail, never vanish."""
    task = _by_id(tasks, "multi_turn_base_0")
    if len(task.ground_truth) < 2:
        pytest.skip("needs a multi-turn task")
    truncated = [[task.ground_truth[0]]]
    padded = pad_decoded_to_turns(truncated, task.n_turns)
    assert len(padded) == task.n_turns
    verdict = _score(task, padded, official, "forcequit_base_0")
    assert verdict.get("valid") is False


def test_scorer_exception_is_captured_not_raised(tasks, official):
    """A scorer blow-up must be recorded as a failure, not crash the shard."""
    task = _by_id(tasks, "multi_turn_base_0")

    def boom(*_a, **_k):
        raise RuntimeError("scorer exploded")

    verdict = score_rollout(task, _rollout_from(task, [[[]]]),
                            multi_turn_checker=boom, model_name="boom")
    assert verdict["valid"] is False
    assert verdict["error_type"] == "scorer_exception"
    assert "scorer exploded" in verdict["scorer_error"]


# --------------------------------------------------------------------------- #
# Decoder round-trip
# --------------------------------------------------------------------------- #


def test_official_decoder_round_trips_ground_truth(tasks):
    """Ground-truth call strings must survive the model-output decoder."""
    from bfcl_eval.model_handler.utils import default_decode_execute_prompting

    task = _by_id(tasks, "multi_turn_base_0")
    calls = task.ground_truth[0]
    decoded = default_decode_execute_prompting("[" + ", ".join(calls) + "]")
    assert len(decoded) == len(calls)


def test_official_decoder_treats_prose_as_no_calls():
    from bfcl_eval.model_handler.utils import default_decode_execute_prompting

    try:
        decoded = default_decode_execute_prompting("I cannot help with that.")
    except Exception:
        return  # upstream raises; the rollout treats that as end-of-turn
    assert decoded == [] or all(not c.strip() for c in decoded)


def test_miss_func_holds_out_its_function_doc(tasks):
    """The withheld tool must be absent from the visible docs and stashed."""
    task = next((t for t in tasks if t.category == "miss_func" and t.missed_function), None)
    if task is None:
        pytest.skip("no miss_func task with a holdout")
    visible = {f["name"] for f in task.function}
    for _turn, docs in task.missed_function.items():
        for doc in docs:
            assert doc["name"] not in visible, f"{doc['name']} leaked into visible docs"
