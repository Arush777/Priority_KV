"""Paired task-level statistics for the external BFCL table.

The unit is the conversation. Every comparison is paired on ``task_id`` and
restricted to tasks where *both* arms produced a scored outcome, so an arm can
never look better by having failed to run somewhere.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class PairedTable:
    """2x2 discordance table for two binary-scored arms."""

    arm_a: str
    arm_b: str
    n_paired: int
    both_pass: int
    a_only: int  # a passed, b failed
    b_only: int  # b passed, a failed
    both_fail: int

    @property
    def n_discordant(self) -> int:
        return self.a_only + self.b_only


def build_paired_table(
    arm_a: str,
    arm_b: str,
    outcomes_a: Mapping[str, bool],
    outcomes_b: Mapping[str, bool],
) -> PairedTable:
    shared = sorted(set(outcomes_a) & set(outcomes_b))
    both_pass = a_only = b_only = both_fail = 0
    for task in shared:
        a, b = bool(outcomes_a[task]), bool(outcomes_b[task])
        if a and b:
            both_pass += 1
        elif a and not b:
            a_only += 1
        elif b and not a:
            b_only += 1
        else:
            both_fail += 1
    return PairedTable(arm_a, arm_b, len(shared), both_pass, a_only, b_only, both_fail)


def exact_mcnemar_p(b: int, c: int) -> float:
    """Two-sided exact McNemar (binomial on the discordant pairs).

    Exact rather than chi-square because discordant counts here are small; a
    chi-square approximation would overstate significance exactly where the
    structure-vs-SnapKV boundary lives.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2.0**n)
    return min(1.0, 2.0 * tail)


def mcnemar(table: PairedTable) -> dict:
    p = exact_mcnemar_p(table.a_only, table.b_only)
    return {
        "arm_a": table.arm_a,
        "arm_b": table.arm_b,
        "n_paired": table.n_paired,
        "both_pass": table.both_pass,
        "a_only": table.a_only,
        "b_only": table.b_only,
        "both_fail": table.both_fail,
        "n_discordant": table.n_discordant,
        "p_exact_mcnemar": p,
        "significant_at_0.05": p < 0.05,
    }


def paired_bootstrap_ci(
    outcomes_a: Mapping[str, bool],
    outcomes_b: Mapping[str, bool],
    *,
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    """Percentile CI for the paired accuracy difference (a - b).

    Resamples *tasks*, keeping each task's two arm outcomes together, which is
    what makes the interval respect the pairing.
    """
    shared = sorted(set(outcomes_a) & set(outcomes_b))
    if not shared:
        return {"n_paired": 0, "diff": None, "ci_low": None, "ci_high": None}
    a = np.array([1.0 if outcomes_a[t] else 0.0 for t in shared])
    b = np.array([1.0 if outcomes_b[t] else 0.0 for t in shared])
    diff = float(a.mean() - b.mean())

    rng = np.random.default_rng(seed)
    n = len(shared)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    lo = float(np.quantile(boot, alpha / 2))
    hi = float(np.quantile(boot, 1 - alpha / 2))
    return {
        "n_paired": n,
        "diff": diff,
        "ci_low": lo,
        "ci_high": hi,
        "n_boot": n_boot,
        "alpha": alpha,
    }


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a single arm's pass rate."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


@dataclass
class CompletenessReport:
    n_tasks_expected: int
    n_tasks_all_arms: int
    per_arm_complete: dict[str, int] = field(default_factory=dict)
    missing_by_arm: dict[str, list[str]] = field(default_factory=dict)

    @property
    def paired_completeness(self) -> float:
        if self.n_tasks_expected == 0:
            return 0.0
        return self.n_tasks_all_arms / self.n_tasks_expected


def paired_completeness(
    expected_task_ids: Sequence[str],
    outcomes_by_arm: Mapping[str, Mapping[str, bool]],
) -> CompletenessReport:
    """How many tasks have a scored outcome in *every* arm."""
    expected = list(dict.fromkeys(expected_task_ids))
    arms = sorted(outcomes_by_arm)
    complete = {arm: sum(1 for t in expected if t in outcomes_by_arm[arm]) for arm in arms}
    missing = {
        arm: [t for t in expected if t not in outcomes_by_arm[arm]] for arm in arms
    }
    all_arms = sum(
        1 for t in expected if all(t in outcomes_by_arm[arm] for arm in arms)
    )
    return CompletenessReport(
        n_tasks_expected=len(expected),
        n_tasks_all_arms=all_arms,
        per_arm_complete=complete,
        missing_by_arm=missing,
    )


def arm_summary(outcomes: Mapping[str, bool]) -> dict:
    n = len(outcomes)
    k = sum(1 for v in outcomes.values() if v)
    lo, hi = wilson_ci(k, n)
    return {
        "n": n,
        "n_pass": k,
        "accuracy": (k / n) if n else 0.0,
        "wilson_ci_low": lo,
        "wilson_ci_high": hi,
    }


def restrict_to_common(
    outcomes_by_arm: Mapping[str, Mapping[str, bool]]
) -> dict[str, dict[str, bool]]:
    """Restrict every arm to the tasks scored in all arms (the paired set)."""
    if not outcomes_by_arm:
        return {}
    common = set.intersection(*(set(v) for v in outcomes_by_arm.values()))
    return {
        arm: {t: bool(v) for t, v in outs.items() if t in common}
        for arm, outs in outcomes_by_arm.items()
    }
