#!/usr/bin/env python3
"""Figures for EXTERNAL_BFCL_PRAJNA_V1, built from tracked summary JSON only.

Two figures:

``protected_fraction_boundary``
    The paper's central claim. Structure-aware retention can only express a
    preference while protected mass stays under the keep budget. Plotting the
    three measured workloads against structure's outcome shows the boundary.

``external_bfcl_arms``
    The paired five/six-arm BFCL table with Wilson intervals.

    uv run python scripts/make_external_figures.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

OUT = REPO_ROOT / "paper" / "figures"

# Colour-blind safe, consistent across both figures.
C_STRUCT = "#0F7B6C"
C_ATTN = "#2D6BA8"
C_BLIND = "#9AA5B1"
C_FULL = "#1B1F24"
C_ADAPT = "#B85C00"
C_GRID = "#D8DEE4"

ARM_STYLE = {
    "full": (C_FULL, "FullKV (no eviction)"),
    "snapkv": (C_ATTN, "SnapKV (attention)"),
    "adapt": (C_ADAPT, "ADAPT (ours)"),
    "structure": (C_STRUCT, "Structure"),
    "uniform": (C_BLIND, "Uniform"),
    "random": (C_BLIND, "Random"),
}


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#8A94A0")
    ax.grid(axis="y", color=C_GRID, lw=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    ax.tick_params(colors="#4A5560", labelsize=9)


def fig_boundary(pf: dict, structure_outcomes: dict, out: Path) -> Path:
    """Protected fraction vs structure's measured accuracy."""
    pts = []
    for w in pf["workloads"]:
        name = w["workload"]
        if name in structure_outcomes:
            pts.append((w["protected_fraction_mean"], structure_outcomes[name], name))
    pts.sort()

    fig, ax = plt.subplots(figsize=(6.2, 3.9), dpi=200)
    budget = pf.get("keep_frac", 0.25)

    # Everything right of the budget line is oversubscribed: the policy cannot
    # rank within the protected set, so it carries no information.
    ax.axvspan(budget, 1.02, color="#F2B8B5", alpha=0.22, zorder=0)
    ax.axvline(budget, color="#C0392B", lw=1.4, ls="--", zorder=2)
    ax.text(budget + 0.015, 0.965, "oversubscribed\n(protected mass > keep budget)",
            fontsize=8.5, color="#8E2B22", va="top", linespacing=1.35)
    ax.text(budget - 0.015, 0.965, "structure can rank", fontsize=8.5,
            color="#0F7B6C", va="top", ha="right")

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ax.plot(xs, ys, color=C_STRUCT, lw=1.6, alpha=0.55, zorder=3)
    ax.scatter(xs, ys, s=110, color=C_STRUCT, zorder=4,
               edgecolor="white", linewidth=1.6)

    for x, y, name in pts:
        va, dy = ("bottom", 0.035) if y > 0.05 else ("bottom", 0.045)
        ax.annotate(f"{name}\n{x:.1%} protected",
                    (x, y), textcoords="offset points",
                    xytext=(0, 12 if va == "bottom" else -22),
                    ha="center", fontsize=8.5, color="#2B3138", linespacing=1.3)
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                    xytext=(0, -14), ha="center", fontsize=9,
                    color=C_STRUCT, fontweight="bold")
    _ = dy

    ax.set_xlim(0, 1.02)
    ax.set_ylim(-0.06, 1.06)
    ax.set_xlabel("Fraction of context tokens carrying a protected role", fontsize=10)
    ax.set_ylabel("Structure-aware retention accuracy", fontsize=10)
    ax.set_title("Structure-aware KV retention helps only below the keep budget",
                 fontsize=11.5, pad=12, color="#1B1F24")
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    _style(ax)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_arms(summary: dict, out: Path) -> Path:
    """Paired arm accuracies with Wilson intervals."""
    order = [a for a in ("full", "snapkv", "adapt", "structure", "uniform", "random")
             if a in summary["overall"]]
    vals = [summary["overall"][a]["accuracy"] for a in order]
    lo = [summary["overall"][a]["accuracy"] - summary["overall"][a]["wilson_ci_low"]
          for a in order]
    hi = [summary["overall"][a]["wilson_ci_high"] - summary["overall"][a]["accuracy"]
          for a in order]
    colors = [ARM_STYLE[a][0] for a in order]
    labels = [ARM_STYLE[a][1] for a in order]

    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=200)
    x = np.arange(len(order))
    ax.bar(x, vals, color=colors, width=0.62, zorder=3)
    ax.errorbar(x, vals, yerr=[lo, hi], fmt="none", ecolor="#4A5560",
                elinewidth=1.2, capsize=4, zorder=4)
    for xi, v in zip(x, vals):
        ax.text(xi, v + max(hi) * 0.12 + 0.006, f"{v:.3f}", ha="center",
                fontsize=9, color="#1B1F24", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.8, rotation=12, ha="right")
    ax.set_ylabel("BFCL V3 multi-turn accuracy", fontsize=10)
    n = summary.get("n_tasks_paired", 0)
    kf = summary.get("keep_frac", 0.25)
    ax.set_title(f"External BFCL evaluation · Qwen3-8B · {kf:.0%} keep · "
                 f"n={n} paired conversations", fontsize=11, pad=10)
    ax.set_ylim(0, max(vals + [0.01]) * 1.45)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summaries", default=None,
                    help="results summaries dir (default: from config)")
    ap.add_argument("--tag", default="primary")
    args = ap.parse_args()

    if args.summaries:
        sdir = Path(args.summaries)
    else:
        from prioritykv.external.config import load_config

        cfg = load_config(REPO_ROOT / "configs/external_bfcl_prajna_v1.yaml")
        sdir = Path(cfg["paths"]["results_root"]) / "summaries"

    pf = json.loads((sdir / "protected_fraction.json").read_text())
    summary = json.loads((sdir / f"summary_{args.tag}.json").read_text())

    # Structure's measured accuracy per workload. PriorityBench-A is the frozen
    # core result; BFCL is measured here. tau-bench is retention-only (no task
    # success), so it is deliberately absent from the accuracy axis.
    structure_outcomes = {
        "PriorityBench-A": 0.933,
        "BFCL-all": summary["overall"].get("structure", {}).get("accuracy", 0.0),
    }

    written = []
    written.append(fig_boundary(pf, structure_outcomes,
                                OUT / "protected_fraction_boundary.png"))
    written.append(fig_arms(summary, OUT / "external_bfcl_arms.png"))
    for p in written:
        print(f"wrote {p}")
    print("EXTERNAL_FIGURES_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
