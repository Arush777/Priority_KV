#!/usr/bin/env python3
"""Render the SWEEP_PM_V1 dose-response figure from the aggregated summary.

Three panels (one per keep budget), shared y. x = measured protected-role
fraction (mean per level), y = pass rate with Wilson 95% intervals. A dashed
vertical line marks x = keep budget — the predicted oversubscription boundary.

    uv run python scripts/pm_sweep_summary.py --model qwen   # writes summary
    uv run python scripts/make_pm_sweep_figure.py --model qwen
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from prioritykv.external.config import load_config  # noqa: E402

# Validated categorical palette (dataviz reference instance, light mode).
# Color follows the arm across every panel; FullKV is a neutral reference.
ARM_STYLE = {
    "structure": {"color": "#2a78d6", "label": "Structure", "z": 5},
    "snapkv": {"color": "#eb6834", "label": "SnapKV", "z": 4},
    "adapt": {"color": "#1baf7a", "label": "ADAPT", "z": 6},
    "uniform": {"color": "#eda100", "label": "Uniform (sink+recent)", "z": 3},
    "random": {"color": "#e87ba4", "label": "Random", "z": 2},
}
FULL_COLOR = "#52514e"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e6e5e1"


def _render_ratio_collapse(cells, budgets, summary, root, args) -> None:
    """One panel: pass rate against oversubscription ratio (mass / budget).

    The three budget panels are three views of a single quantity. Plotting
    against the ratio collapses them onto one curve, which is the actual claim:
    a structural prior discriminates while its protected set fits the budget and
    stops once it does not, regardless of which budget produced that condition.
    """
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=200)
    marker_for = {b: m for b, m in zip(budgets, ("o", "s", "^", "D", "v"))}

    for arm, st in ARM_STYLE.items():
        xs, ys = [], []
        for c in cells:
            a = c["arms"].get(arm, {})
            if a.get("rate") is None or not c["keep_frac"]:
                continue
            xs.append(c["measured_frac_mean"] / c["keep_frac"])
            ys.append(a["rate"])
        if not xs:
            continue
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        ax.plot([xs[i] for i in order], [ys[i] for i in order],
                color=st["color"], lw=1.8, alpha=0.55, zorder=st["z"])
        # Marker shape carries the budget, so a reader can verify that points
        # from different budgets land on the same curve rather than take it on faith.
        for c in cells:
            a = c["arms"].get(arm, {})
            if a.get("rate") is None or not c["keep_frac"]:
                continue
            ax.plot(c["measured_frac_mean"] / c["keep_frac"], a["rate"],
                    marker=marker_for.get(c["keep_frac"], "o"), ms=5.0,
                    color=st["color"], mfc="white", mew=1.5, ls="none",
                    zorder=st["z"] + 1)

    ax.axvline(1.0, color=TEXT_SECONDARY, lw=1.1, ls=":", zorder=0)
    ax.annotate("protected set exactly fills the budget", xy=(1.0, 0.5),
                xytext=(1.15, 0.46), fontsize=7.5, color=TEXT_SECONDARY,
                rotation=90, va="center")
    ax.set_xscale("log")
    ax.set_xlabel("oversubscription ratio  (protected mass ÷ keep budget)",
                  fontsize=9, color=TEXT_PRIMARY)
    ax.set_ylabel("pass rate", fontsize=9, color=TEXT_PRIMARY)
    ax.set_ylim(-0.03, 1.03)
    ax.grid(True, color=GRID, lw=0.7, zorder=-5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)

    handles = [plt.Line2D([], [], color=st["color"], lw=1.8, marker="o", ms=5,
                          mfc="white", mew=1.5, label=st["label"])
               for st in ARM_STYLE.values()]
    handles += [plt.Line2D([], [], color=TEXT_SECONDARY, ls="none",
                           marker=marker_for.get(b, "o"), ms=5, mfc="white",
                           mew=1.2, label=f"budget {int(round(b * 100))}%")
                for b in budgets]
    ax.legend(handles=handles, frameon=False, fontsize=7.5, ncol=2,
              loc="center left", bbox_to_anchor=(1.01, 0.5))
    ax.set_title("Retention policies collapse onto one boundary",
                 fontsize=11, color=TEXT_PRIMARY)
    fig.tight_layout()
    out = root / "summaries" / f"pm_sweep_ratio_{args.model}.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "configs" / "pm_sweep_v1.yaml"))
    ap.add_argument("--model", default="qwen")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    root = Path(cfg["paths"]["results_root"]) / args.model
    summary = json.loads((root / "summaries" / "pm_sweep_summary.json").read_text())
    budgets = summary["budgets"]
    cells = summary["cells"]

    fig, axes = plt.subplots(
        1, len(budgets), figsize=(4.1 * len(budgets), 3.6), sharey=True, dpi=200
    )
    if len(budgets) == 1:
        axes = [axes]

    for ax, kf in zip(axes, budgets):
        sub = sorted(
            (c for c in cells if c["keep_frac"] == kf),
            key=lambda c: c["measured_frac_mean"],
        )
        xs = [c["measured_frac_mean"] for c in sub]

        # FullKV reference band (difficulty control at each level).
        fy = [c["arms"]["full"]["rate"] for c in sub]
        ax.plot(xs, fy, color=FULL_COLOR, lw=1.6, ls=(0, (4, 2.5)), zorder=1)

        for arm, st in ARM_STYLE.items():
            ys, lo, hi = [], [], []
            for c in sub:
                a = c["arms"].get(arm, {})
                ys.append(a.get("rate"))
                w = a.get("wilson", [None, None])
                # Wilson bounds can land a hair inside a rate of exactly 0 or 1;
                # clamp so the asymmetric error bars stay non-negative.
                lo.append(None if a.get("rate") is None else max(0.0, a["rate"] - w[0]))
                hi.append(None if a.get("rate") is None else max(0.0, w[1] - a["rate"]))
            keep = [i for i, y in enumerate(ys) if y is not None]
            if not keep:
                continue
            ax.errorbar(
                [xs[i] for i in keep],
                [ys[i] for i in keep],
                yerr=[[lo[i] for i in keep], [hi[i] for i in keep]],
                color=st["color"],
                lw=2.0,
                marker="o",
                ms=4.5,
                mfc="white",
                mew=1.6,
                capsize=0,
                elinewidth=1.1,
                alpha=0.95,
                zorder=st["z"],
            )

        ax.axvline(kf, color=TEXT_SECONDARY, lw=1.0, ls=":", zorder=0)
        ax.annotate(
            "budget",
            xy=(kf, 0.02),
            xytext=(kf + 0.015, 0.02),
            fontsize=7.5,
            color=TEXT_SECONDARY,
        )
        ax.set_title(f"keep budget = {int(round(kf * 100))}%", fontsize=10,
                     color=TEXT_PRIMARY)
        ax.set_xlabel("measured protected-role fraction", fontsize=9,
                      color=TEXT_PRIMARY)
        ax.set_xlim(0, 1.0)
        ax.set_ylim(-0.03, 1.03)
        ax.grid(True, color=GRID, lw=0.7, zorder=-5)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)

    axes[0].set_ylabel("pass rate", fontsize=9, color=TEXT_PRIMARY)

    _render_ratio_collapse(cells, budgets, summary, root, args)

    handles = [
        plt.Line2D([], [], color=FULL_COLOR, lw=1.6, ls=(0, (4, 2.5)),
                   label="FullKV (no eviction)")
    ] + [
        plt.Line2D([], [], color=st["color"], lw=2.0, marker="o", ms=4.5,
                   mfc="white", mew=1.6, label=st["label"])
        for st in ARM_STYLE.values()
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(handles),
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.5, -0.04),
    )
    fig.suptitle(
        f"Retention policy vs. protected mass — {summary['model']}, "
        f"n per point ≈ {max((c['arms']['structure']['n'] for c in cells), default=0)}",
        fontsize=11,
        color=TEXT_PRIMARY,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))

    out = Path(args.out) if args.out else root / "summaries" / f"pm_sweep_{args.model}.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
