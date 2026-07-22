"""Tests for the press-based arms: one mechanism, four policies."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from prioritykv.baselines.keep_policy import KeepPolicyConfig, select_keep_indices  # noqa: E402
from prioritykv.external.arms import keep_budget  # noqa: E402
from prioritykv.external.presses import (  # noqa: E402
    PRESS_ARMS,
    compression_ratio_for_budget,
    expected_kept,
    press_class_name,
    structure_token_scores,
)
from prioritykv.page_roles import PageRole  # noqa: E402

CFG = KeepPolicyConfig(keep_frac=0.25, sink_tokens=16, force_recent=128, seed=0)


def _roles(n):
    roles = [PageRole.FILLER] * n
    for i in range(200, min(700, n)):
        roles[i] = PageRole.TOOL
    for i in range(min(900, n), min(1000, n)):
        roles[i] = PageRole.CONSTRAINT
    return roles


@pytest.mark.parametrize("n", [600, 800, 1024, 2048, 4096, 16384, 40000])
def test_structure_scores_reproduce_the_frozen_policy_selection(n):
    """Switching mechanism must not change *which* tokens the policy keeps."""
    roles = _roles(n)
    budget = keep_budget(n, CFG)
    top = set(np.argsort(-structure_token_scores(n, roles, CFG))[:budget].tolist())
    ref = set(select_keep_indices(n, CFG, policy="structure", roles=roles).tolist())
    assert top == ref, f"press selection diverges from frozen structure at n={n}"


def test_structure_scores_protect_sink_and_recent():
    n = 4096
    top = set(np.argsort(-structure_token_scores(n, _roles(n), CFG))[
        : keep_budget(n, CFG)].tolist())
    assert set(range(16)) <= top
    assert set(range(n - 128, n)) <= top


def test_structure_scores_rank_protected_roles_above_filler():
    n = 4096
    s = structure_token_scores(n, _roles(n), CFG)
    assert s[950] > s[1500], "CONSTRAINT must outrank FILLER"
    assert s[300] > s[1500], "TOOL must outrank FILLER"


def test_protected_roles_share_one_band_like_the_frozen_policy():
    """Frozen select_structure has no preference among protected roles."""
    n = 4096
    roles = [PageRole.FILLER] * n
    roles[300] = PageRole.TOOL
    roles[400] = PageRole.CONSTRAINT
    roles[500] = PageRole.SYSTEM
    roles[600] = PageRole.OTHER
    s = structure_token_scores(n, roles, CFG)
    # Same band => ordering is purely positional (earliest first).
    assert s[300] > s[400] > s[500] > s[600]


def test_mixed_protected_roles_match_frozen_selection_under_tight_budget():
    n = 2048
    roles = [PageRole.FILLER] * n
    for i in range(200, 700):
        roles[i] = PageRole.TOOL
    for i in range(900, 1000):
        roles[i] = PageRole.CONSTRAINT
    for i in range(1100, 1200):
        roles[i] = PageRole.SYSTEM
    budget = keep_budget(n, CFG)
    top = set(np.argsort(-structure_token_scores(n, roles, CFG))[:budget].tolist())
    ref = set(select_keep_indices(n, CFG, policy="structure", roles=roles).tolist())
    assert top == ref


def test_structure_scores_are_deterministic():
    n = 2048
    a = structure_token_scores(n, _roles(n), CFG)
    assert np.array_equal(a, structure_token_scores(n, _roles(n), CFG))


def test_structure_scores_handle_empty_and_tiny_contexts():
    assert structure_token_scores(0, [], CFG).shape == (0,)
    assert structure_token_scores(4, [PageRole.FILLER] * 4, CFG).shape == (4,)


@pytest.mark.parametrize("n", [512, 4096, 40000])
def test_compression_ratio_lands_exactly_on_the_budget(n):
    """Every arm must realise the same keep count from the same ratio."""
    budget = keep_budget(n, CFG)
    ratio = compression_ratio_for_budget(n, budget)
    assert expected_kept(n, ratio) == budget


def test_compression_ratio_is_clamped_to_valid_range():
    # kvpress asserts 0 <= ratio < 1.
    assert 0.0 <= compression_ratio_for_budget(100, 100) < 1.0
    assert 0.0 <= compression_ratio_for_budget(100, 0) < 1.0
    assert compression_ratio_for_budget(0, 0) == 0.0


def test_every_press_arm_names_a_real_press_class():
    for arm in PRESS_ARMS:
        assert press_class_name(arm)
    assert press_class_name("snapkv") == "kvpress.SnapKVPress"


def test_presses_construct_and_are_the_expected_types():
    kvpress = pytest.importorskip("kvpress")
    from prioritykv.external.presses import (
        make_random_press,
        make_snapkv_press_ext,
        make_structure_press,
        make_uniform_press,
    )

    assert isinstance(make_snapkv_press_ext(0.75), kvpress.SnapKVPress)
    assert isinstance(make_uniform_press(0.75), kvpress.StreamingLLMPress)
    assert isinstance(make_random_press(0.75), kvpress.RandomPress)
    assert isinstance(make_structure_press(0.75), kvpress.ScorerPress)


def test_structure_press_rejects_misaligned_scores():
    pytest.importorskip("kvpress")
    torch = pytest.importorskip("torch")
    from prioritykv.external.presses import make_structure_press

    press = make_structure_press(0.75)
    press.token_scores = np.ones(10, dtype=np.float64)
    keys = torch.zeros(1, 2, 99, 4)
    with pytest.raises(RuntimeError, match="alignment"):
        press.score(None, None, keys, keys, None, {})


def test_structure_press_errors_if_scores_never_set():
    pytest.importorskip("kvpress")
    torch = pytest.importorskip("torch")
    from prioritykv.external.presses import make_structure_press

    press = make_structure_press(0.75)
    keys = torch.zeros(1, 2, 8, 4)
    with pytest.raises(RuntimeError, match="token_scores"):
        press.score(None, None, keys, keys, None, {})


def test_structure_press_score_shape_matches_keys():
    pytest.importorskip("kvpress")
    torch = pytest.importorskip("torch")
    from prioritykv.external.presses import make_structure_press

    press = make_structure_press(0.75)
    press.token_scores = np.arange(32, dtype=np.float64)
    keys = torch.zeros(1, 4, 32, 8)
    out = press.score(None, None, keys, keys, None, {})
    assert tuple(out.shape) == (1, 4, 32)
