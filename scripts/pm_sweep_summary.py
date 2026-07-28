#!/usr/bin/env python3
"""Aggregate SWEEP_PM_V1 points into per-cell rates, paired tests, and a table.

CPU-only. Reads every point under ``<results_root>/<model>/points``, groups by
(pm_level, keep_frac, arm), and writes:

- ``summaries/pm_sweep_summary.json`` — per-cell pass counts, Wilson CIs, mean
  measured protected fraction, oversubscription rate, paired
  structure-vs-snapkv and adapt-vs-snapkv McNemar per cell;
- a printed markdown table for quick reading.

    uv run python scripts/pm_sweep_summary.py --model qwen
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from prioritykv.external.config import load_config  # noqa: E402
from prioritykv.external.stats import exact_mcnemar_p, wilson_ci  # noqa: E402


def load_points(points_dir: Path) -> list[dict]:
    rows = []
    for p in sorted(points_dir.glob("*.json")):
        try:
            d = json.loads(p.read_text())
        except Exception:  # noqa: BLE001
            continue
        if d.get("terminal_status") == "ok" and "score" in d:
            rows.append(d)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "configs" / "pm_sweep_v1.yaml"))
    ap.add_argument("--model", default="qwen")
    args = ap.parse_args()

    cfg = load_config(args.config)
    root = Path(cfg["paths"]["results_root"]) / args.model
    rows = load_points(root / "points")
    if not rows:
        print(f"no points under {root / 'points'}")
        return 1

    # outcome[(level, kf, arm)][example_id] = bool
    outcome: dict[tuple, dict[str, bool]] = defaultdict(dict)
    frac_by_level: dict[str, list[float]] = defaultdict(list)
    oversub: dict[tuple, list[bool]] = defaultdict(list)
    # Pooling across categories hides opposite behaviours: state in the leading
    # system block survives structure's earliest-first tiebreak at any
    # oversubscription, while mid-conversation state does not. Keep the split.
    by_cat: dict[tuple, list[float]] = defaultdict(list)

    for d in rows:
        lvl = d.get("pm_level")
        kf = d.get("keep_frac")
        arm = d["arm"]
        key = (lvl, kf if kf is not None else "full", arm)
        outcome[key][d["example_id"]] = d["score"] >= 1.0
        by_cat[(lvl, kf if kf is not None else "full", arm, d["category"])].append(
            float(d["score"])
        )
        mf = d.get("measured_protected_frac")
        if mf is not None and arm == "full":
            frac_by_level[lvl].append(mf)
        if mf is not None and kf:
            oversub[(lvl, kf)].append(mf > kf)

    budgets = sorted({k[1] for k in outcome if k[1] != "full"})
    levels = sorted(frac_by_level, key=lambda lv: sum(frac_by_level[lv]) / max(1, len(frac_by_level[lv])))
    arms_press = list(cfg["arms"]["press"])

    cells = []
    for lvl in levels:
        x = sum(frac_by_level[lvl]) / max(1, len(frac_by_level[lvl]))
        full = outcome.get((lvl, "full", "full"), {})
        for kf in budgets:
            cell = {
                "pm_level": lvl,
                "measured_frac_mean": x,
                "keep_frac": kf,
                "oversubscribed_rate": (
                    sum(oversub[(lvl, kf)]) / len(oversub[(lvl, kf)])
                    if oversub[(lvl, kf)] else None
                ),
                "arms": {},
            }
            armmaps = {"full": full}
            for arm in arms_press:
                armmaps[arm] = outcome.get((lvl, kf, arm), {})
            for arm, m in armmaps.items():
                n = len(m)
                k = sum(m.values())
                lo, hi = wilson_ci(k, n) if n else (0.0, 0.0)
                cats = {}
                for (l2, k2, a2, c2), vals in by_cat.items():
                    if l2 == lvl and a2 == arm and k2 == (kf if arm != "full" else "full"):
                        cats[c2] = {"rate": sum(vals) / len(vals), "n": len(vals)}
                cell["arms"][arm] = {"passes": k, "n": n, "rate": k / n if n else None,
                                     "wilson": [lo, hi], "by_category": cats}
            for a, b in (("structure", "snapkv"), ("adapt", "snapkv"),
                         ("adapt", "structure")):
                ma, mb = armmaps.get(a, {}), armmaps.get(b, {})
                common = sorted(set(ma) & set(mb))
                x01 = sum(1 for e in common if ma[e] and not mb[e])
                x10 = sum(1 for e in common if not ma[e] and mb[e])
                cell[f"mcnemar_{a}_vs_{b}"] = {
                    "n_common": len(common), "b": x01, "c": x10,
                    "p": exact_mcnemar_p(x01, x10) if common else None,
                }
            cells.append(cell)

    out = {
        "freeze_id": cfg["freeze_id"],
        "model": args.model,
        "n_points": len(rows),
        "budgets": budgets,
        "levels": {lv: sum(frac_by_level[lv]) / max(1, len(frac_by_level[lv]))
                   for lv in levels},
        "cells": cells,
    }
    outdir = root / "summaries"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "pm_sweep_summary.json").write_text(json.dumps(out, indent=2) + "\n")

    arm_order = ["full", "structure", "snapkv", "adapt", "uniform", "random"]
    print(f"points={len(rows)}  → {outdir / 'pm_sweep_summary.json'}")
    for kf in budgets:
        print(f"\n### keep_frac={kf}")
        print("| level | frac | " + " | ".join(arm_order) + " | McN s-vs-snap |")
        print("|---|---|" + "---|" * (len(arm_order) + 1))
        for c in cells:
            if c["keep_frac"] != kf:
                continue
            vals = []
            for a in arm_order:
                st = c["arms"].get(a, {})
                vals.append(f"{st.get('rate'):.3f}" if st.get("rate") is not None
                            else "—")
            mc = c["mcnemar_structure_vs_snapkv"]
            print(f"| {c['pm_level']} | {c['measured_frac_mean']:.3f} | "
                  + " | ".join(vals)
                  + f" | b={mc['b']} c={mc['c']} p={mc['p'] if mc['p'] is None else round(mc['p'], 4)} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
