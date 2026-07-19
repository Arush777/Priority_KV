#!/usr/bin/env python3
"""Paired stress stats over structured_stress example_rows / arms_detail.

Computes:
  - Wilson 95% CIs on pass rates per arm
  - McNemar exact (binomial) structure vs uniform (and vs fullkv)
  - Paired permutation p-values on mean score deltas
  - Bootstrap 95% CIs for latency/score ratios when seconds present

Usage:
  python scripts/stats_paired_stress.py jobs/results/<id>/summary.json
  python scripts/stats_paired_stress.py runs/stress_structured/*.json --compare structure uniform
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n <= 0:
        return (float("nan"), float("nan"), float("nan"))
    p = k / n
    den = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = (z / den) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (p, max(0.0, center - half), min(1.0, center + half))


def _mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar on discordant pairs (b, c)."""
    n = b + c
    if n == 0:
        return 1.0
    # P(X <= min(b,c)) + P(X >= max(b,c)) under Bin(n, 0.5)
    from math import comb

    lo, hi = min(b, c), max(b, c)
    p = sum(comb(n, k) for k in range(0, lo + 1)) / (2**n)
    p += sum(comb(n, k) for k in range(hi, n + 1)) / (2**n)
    # double-count when lo==hi and n even mid — for exact two-sided with
    # equal tails this is fine when lo < hi; when lo==hi return 1.
    if lo == hi:
        return 1.0
    return min(1.0, p)


def _paired_perm_pvalue(
    a: np.ndarray, b: np.ndarray, *, n_perm: int = 10_000, seed: int = 0
) -> float:
    """Two-sided permutation test on mean(a-b)."""
    diff = a - b
    obs = float(np.mean(diff))
    rng = np.random.default_rng(seed)
    n = len(diff)
    if n == 0:
        return float("nan")
    count = 0
    for _ in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=n)
        if abs(float(np.mean(signs * diff))) >= abs(obs):
            count += 1
    return (count + 1) / (n_perm + 1)


def _bootstrap_ratio_ci(
    num: np.ndarray,
    den: np.ndarray,
    *,
    n_boot: int = 5_000,
    seed: int = 0,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(num)
    if n == 0 or float(np.mean(den)) == 0.0:
        return (float("nan"), float("nan"), float("nan"))
    point = float(np.mean(num) / np.mean(den))
    ratios = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        d = float(np.mean(den[idx]))
        if d == 0.0:
            continue
        ratios.append(float(np.mean(num[idx]) / d))
    if not ratios:
        return (point, float("nan"), float("nan"))
    lo, hi = np.percentile(ratios, [2.5, 97.5])
    return (point, float(lo), float(hi))


def _load_rows(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("example_rows")
    if not rows:
        # Fall back: rebuild from arms_detail
        arms = data.get("arms_detail") or {}
        policies = list(data.get("policies") or arms.keys())
        by_id: dict[str, dict[str, Any]] = {}
        for pol in policies:
            for r in (arms.get(pol) or {}).get("rows") or []:
                eid = r["example_id"]
                row = by_id.setdefault(
                    eid,
                    {
                        "example_id": eid,
                        "category": r.get("category"),
                        "context_length": r.get("context_length"),
                        "replication_slice": r.get("replication_slice"),
                        "fullkv_score": r.get("fullkv_score"),
                        "fullkv_pass": bool(r.get("fullkv_pass", (r.get("fullkv_score") or 0) >= 1.0)),
                    },
                )
                row[f"{pol}_score"] = r.get("policy_score")
                row[f"{pol}_pass"] = bool(
                    r.get("policy_pass", (r.get("policy_score") or 0) >= 1.0)
                )
        rows = list(by_id.values())
    return data, rows


def summarize(
    data: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    arm_a: str,
    arm_b: str,
    n_perm: int,
) -> dict[str, Any]:
    n = len(rows)
    policies = list(data.get("policies") or [])
    if arm_a not in policies and f"{arm_a}_pass" not in (rows[0] if rows else {}):
        pass

    out: dict[str, Any] = {
        "n": n,
        "manifest_id": data.get("manifest_id"),
        "selection": data.get("selection"),
        "keep_frac": (data.get("keep") or {}).get("keep_frac"),
        "arms": {},
        "comparisons": {},
    }

    def _arm_key(name: str) -> tuple[str, str]:
        if name == "fullkv":
            return ("fullkv_pass", "fullkv_score")
        return (f"{name}_pass", f"{name}_score")

    for name in ["fullkv"] + [p for p in policies if p != "fullkv"]:
        pk, sk = _arm_key(name)
        passes = [bool(r.get(pk)) for r in rows if pk in r]
        scores = [float(r[sk]) for r in rows if sk in r and r[sk] is not None]
        k = sum(passes)
        m = len(passes)
        p, lo, hi = _wilson(k, m)
        out["arms"][name] = {
            "n": m,
            "pass_rate": p,
            "wilson95": [lo, hi],
            "mean_score": float(np.mean(scores)) if scores else float("nan"),
        }

    for a, b in ((arm_a, arm_b), (arm_a, "fullkv"), (arm_b, "fullkv")):
        pa, sa = _arm_key(a)
        pb, sb = _arm_key(b)
        paired = [
            r
            for r in rows
            if pa in r and pb in r and sa in r and sb in r
        ]
        if not paired:
            continue
        a_pass = np.array([bool(r[pa]) for r in paired], dtype=bool)
        b_pass = np.array([bool(r[pb]) for r in paired], dtype=bool)
        # McNemar: b = a fail & b pass; c = a pass & b fail
        discord_b = int(np.sum(~a_pass & b_pass))
        discord_c = int(np.sum(a_pass & ~b_pass))
        a_scores = np.array([float(r[sa]) for r in paired], dtype=float)
        b_scores = np.array([float(r[sb]) for r in paired], dtype=float)
        out["comparisons"][f"{a}_vs_{b}"] = {
            "n": len(paired),
            "mcnemar_discordant": {"a_fail_b_pass": discord_b, "a_pass_b_fail": discord_c},
            "mcnemar_exact_p": _mcnemar_exact(discord_b, discord_c),
            "mean_delta_a_minus_b": float(np.mean(a_scores - b_scores)),
            "paired_perm_p": _paired_perm_pvalue(a_scores, b_scores, n_perm=n_perm),
        }

    secs = data.get("seconds") or {}
    if "structure" in secs and "uniform" in secs and secs["uniform"]:
        # Single ratio (not per-example); report raw + note.
        out["latency_ratio_structure_over_uniform"] = {
            "point": float(secs["structure"]) / float(secs["uniform"]),
            "note": "wall-clock arm totals; bootstrap needs per-example latencies",
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", type=Path)
    ap.add_argument("--compare-a", default="structure")
    ap.add_argument("--compare-b", default="uniform")
    ap.add_argument("--n-perm", type=int, default=10_000)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    reports = []
    for path in args.paths:
        data, rows = _load_rows(path)
        rep = summarize(
            data,
            rows,
            arm_a=args.compare_a,
            arm_b=args.compare_b,
            n_perm=args.n_perm,
        )
        rep["source"] = str(path)
        reports.append(rep)
        print(json.dumps(rep, indent=2))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        payload = reports[0] if len(reports) == 1 else {"reports": reports}
        args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"out={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
