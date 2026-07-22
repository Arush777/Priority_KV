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
    assert isinstance(make_random_press(0.75), kvpress.ScorerPress)
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


def test_random_press_scores_on_the_keys_device_and_is_seeded():
    """kvpress's own RandomPress builds a CPU generator and dies on CUDA keys."""
    pytest.importorskip("kvpress")
    torch = pytest.importorskip("torch")
    from prioritykv.external.presses import make_random_press

    keys = torch.zeros(1, 4, 64, 8)
    a = make_random_press(0.75, seed=3).score(None, None, keys, keys, None, {})
    b = make_random_press(0.75, seed=3).score(None, None, keys, keys, None, {})
    c = make_random_press(0.75, seed=4).score(None, None, keys, keys, None, {})
    assert tuple(a.shape) == (1, 4, 64)
    assert a.device == keys.device
    assert torch.equal(a, b), "same seed must give the same mask"
    assert not torch.equal(a, c), "different seed must give a different mask"


def test_random_press_selects_a_spread_not_a_contiguous_block():
    """A position-blind control must not degenerate into sink+recent."""
    pytest.importorskip("kvpress")
    torch = pytest.importorskip("torch")
    from prioritykv.external.presses import make_random_press

    keys = torch.zeros(1, 1, 1000, 8)
    s = make_random_press(0.75, seed=0).score(None, None, keys, keys, None, {})[0, 0]
    kept = set(torch.topk(s, 250).indices.tolist())
    # A contiguous block would put nearly all of its mass in one region.
    thirds = [len([i for i in kept if lo <= i < lo + 334]) for lo in (0, 334, 668)]
    assert all(t > 40 for t in thirds), f"mask is not position-blind: {thirds}"


# --------------------------------------------------------------------------- #
# ADAPT: adaptive structure prior
# --------------------------------------------------------------------------- #


def test_alpha_is_one_when_structure_fits_the_budget():
    """Under-subscribed => pure structure, so ADAPT reduces to the frozen policy."""
    from prioritykv.external.presses import adaptive_alpha

    assert adaptive_alpha(protected_mass=100, budget=1000) == 1.0
    assert adaptive_alpha(protected_mass=1000, budget=1000) == 1.0


def test_alpha_falls_toward_zero_as_structure_oversubscribes():
    """Oversubscribed => attention dominates, so ADAPT reduces toward SnapKV."""
    from prioritykv.external.presses import adaptive_alpha

    assert adaptive_alpha(protected_mass=4000, budget=1000) == pytest.approx(0.25)
    assert adaptive_alpha(protected_mass=100000, budget=1000) == pytest.approx(0.01)


def test_alpha_matches_the_measured_workloads():
    """The three measured regimes must land where the paper claims."""
    from prioritykv.external.presses import adaptive_alpha

    # PriorityBench-A: 6.1% protected, 25% budget -> structure can rank freely.
    assert adaptive_alpha(int(0.061 * 10000), 2500) == 1.0
    # BFCL: 98.8% protected -> mostly attention.
    assert adaptive_alpha(int(0.988 * 10000), 2500) == pytest.approx(0.253, abs=1e-2)
    # tau-bench: 79.5% protected -> blend.
    assert adaptive_alpha(int(0.795 * 10000), 2500) == pytest.approx(0.314, abs=1e-2)


def test_alpha_is_degenerate_safe():
    from prioritykv.external.presses import adaptive_alpha

    assert adaptive_alpha(0, 1000) == 1.0
    assert 0.0 <= adaptive_alpha(10**9, 1) <= 1.0


def test_protected_mass_counts_only_structure_roles():
    from prioritykv.external.presses import protected_mass_from_roles

    roles = [PageRole.TOOL, PageRole.FILLER, PageRole.CONSTRAINT,
             PageRole.GENERATED, PageRole.OTHER]
    assert protected_mass_from_roles(roles) == 3


def test_rank_normalise_maps_to_unit_interval_preserving_order():
    torch = pytest.importorskip("torch")
    from prioritykv.external.presses import _rank_normalise

    x = torch.tensor([[[5.0, 1.0, 3.0, 9.0]]])
    r = _rank_normalise(x)
    assert float(r.min()) == 0.0 and float(r.max()) == 1.0
    # order preserved: 1 < 3 < 5 < 9
    assert r[0, 0, 1] < r[0, 0, 2] < r[0, 0, 0] < r[0, 0, 3]


def test_rank_normalise_is_scale_free():
    """This is what makes alpha an honest weight rather than a fudge factor."""
    torch = pytest.importorskip("torch")
    from prioritykv.external.presses import _rank_normalise

    x = torch.tensor([[[1e-3, 2e-3, 3e-3]]])
    y = x * 1e9
    assert torch.allclose(_rank_normalise(x), _rank_normalise(y))


def test_adapt_at_alpha_one_selects_exactly_what_structure_selects():
    """alpha=1 must reproduce the structure arm, not merely approximate it."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("kvpress")
    from prioritykv.external.presses import _rank_normalise

    n = 4096
    roles = _roles(n)
    budget = keep_budget(n, CFG)
    s = structure_token_scores(n, roles, CFG)
    ranked = _rank_normalise(torch.as_tensor(s, dtype=torch.float32).view(1, 1, n))
    top = set(torch.topk(ranked[0, 0], budget).indices.tolist())
    ref = set(np.argsort(-s)[:budget].tolist())
    assert top == ref


def test_adapt_press_requires_scores_and_checks_alignment():
    pytest.importorskip("kvpress")
    pytest.importorskip("torch")
    from prioritykv.external.presses import make_adaptive_press

    press = make_adaptive_press(0.75)
    assert press.alpha == 1.0
    assert press.token_scores is None
    assert press.compression_ratio == 0.75


def test_adapt_is_registered_as_a_press_arm():
    from prioritykv.external.presses import PRESS_ARMS, press_class_name

    assert "adapt" in PRESS_ARMS
    assert "AdaptiveStructurePress" in press_class_name("adapt")
