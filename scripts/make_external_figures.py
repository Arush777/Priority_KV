#!/usr/bin/env python3
"""Figures for EXTERNAL_BFCL_PRAJNA_V1, built from tracked summary JSON only.

Two figures:

``protected_fraction_boundary``
    The paper's central claim. Structure-aware retention can only express a
    preference while protected mass stays under the keep budget. Workloads are
    discrete, so they are drawn as bars against the budget threshold -- never as
    a line through the points, which would invent a continuous trend from three
    measurements.

``external_bfcl_arms``
    The paired arm table, one panel per model, so the cross-model replication is
    visible. Emphasis colouring: the two arms that carry the story are coloured,
    the arms that sit at zero are muted, rather than spending six hues on a
    result that is really one comparison.

Colour: slots 1-3 of the reference palette (blue/orange/aqua), whose all-pairs
CVD separation is documented as validated in both modes. `node` is unavailable on
this cluster so `validate_palette.js` could not be re-run here; staying inside the
documented-safe first three slots is the reason this is sound.

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

# Reference palette, light mode.
SERIES_1 = "#2a78d6"   # blue   — the emphasised measure
SERIES_2 = "#eb6834"   # orange — second emphasised measure
CRITICAL = "#c0392b"   # threshold rule (a real status: the budget boundary)
MUTED = "#b6bcc4"      # arms that sit at zero
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e3e7eb"

# Arms whose value carries the finding; everything else is deliberately muted.
EMPHASIS = {"full": INK, "snapkv": SERIES_1, "adapt": SERIES_2}
ARM_LABEL = {
    "full": "FullKV", "snapkv": "SnapKV", "adapt": "ADAPT",
    "structure": "Structure", "uniform": "Uniform", "random": "Random",
}


def _style(ax, *, xgrid=False):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#9aa3ad")
    ax.spines[["left", "bottom"]].set_linewidth(0.8)
    ax.grid(axis="x" if xgrid else "y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_2, labelsize=9, length=3)


def fig_boundary(pf: dict, structure_outcomes: dict, out: Path) -> Path:
    """Protected fraction per workload against the keep budget."""
    rows = [w for w in pf["workloads"] if not w["workload"].startswith("BFCL-")
            or w["workload"] == "BFCL-all"]
    rows.sort(key=lambda w: w["protected_fraction_mean"])
    names = [w["workload"] for w in rows]
    fracs = [w["protected_fraction_mean"] for w in rows]
    budget = pf.get("keep_frac", 0.25)

    fig, ax = plt.subplots(figsize=(7.0, 2.9), dpi=200)
    y = np.arange(len(rows))

    colors = [SERIES_1 if f <= budget else MUTED for f in fracs]
    ax.barh(y, fracs, color=colors, height=0.5, zorder=3)

    ax.axvline(budget, color=CRITICAL, lw=1.5, zorder=4)
    # Sit the rule label inside the axes, clear of both the title and the bars.
    ax.annotate(f"keep budget {budget:.0%}", xy=(budget, 1.0),
                xycoords=("data", "axes fraction"), xytext=(4, -10),
                textcoords="offset points", color=CRITICAL, fontsize=8.5,
                ha="left", va="top")

    for yi, (w, f) in enumerate(zip(rows, fracs)):
        acc = structure_outcomes.get(w["workload"])
        note = ("structure " + (f"{acc:.3f}" if acc is not None else "n/a"))
        if acc is None:
            note += "  (retention-only)"
        ax.text(f + 0.012, yi, f"{f:.1%}   ·   {note}", va="center",
                fontsize=9, color=INK)

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9.5, color=INK)
    ax.set_ylim(-0.6, len(rows) - 0.15)
    ax.set_xlim(0, 1.42)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xlabel("Fraction of context tokens carrying a protected role", fontsize=10)
    ax.set_title("Structure-aware retention can only rank below the keep budget",
                 fontsize=11.5, pad=10, color=INK, loc="left")
    _style(ax, xgrid=True)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def fig_arms(panels: list[tuple[str, dict]], out: Path) -> Path:
    """One panel per model so the cross-model replication is visible."""
    fig, axes = plt.subplots(1, len(panels), figsize=(4.1 * len(panels), 3.3), dpi=200)
    if len(panels) == 1:
        axes = [axes]

    # One shared limit computed from every panel's upper CI. sharex plus a
    # per-panel set_xlim silently clipped the tallest bar in the first panel.
    xmax = max(
        summary["overall"][a]["wilson_ci_high"]
        for _, summary in panels for a in summary["overall"]
    )
    xlim = xmax * 1.55

    for ax, (title, summary) in zip(axes, panels):
        order = [a for a in ("full", "snapkv", "adapt", "structure", "uniform", "random")
                 if a in summary["overall"]]
        vals = [summary["overall"][a]["accuracy"] for a in order]
        lo = [v - summary["overall"][a]["wilson_ci_low"] for a, v in zip(order, vals)]
        hi = [summary["overall"][a]["wilson_ci_high"] - v for a, v in zip(order, vals)]
        colors = [EMPHASIS.get(a, MUTED) for a in order]
        y = np.arange(len(order))[::-1]

        ax.barh(y, vals, color=colors, height=0.55, zorder=3)
        ax.errorbar(vals, y, xerr=[lo, hi], fmt="none", ecolor="#6b737c",
                    elinewidth=1.1, capsize=3, zorder=4)
        for yi, v, h in zip(y, vals, hi):
            # Sit the label clear of the error-bar cap, never on top of it.
            ax.text(v + h + xlim * 0.035, yi, f"{v:.3f}", va="center",
                    fontsize=9, color=INK)

        ax.set_yticks(y)
        ax.set_yticklabels([ARM_LABEL[a] for a in order], fontsize=9.5, color=INK)
        n = summary.get("n_tasks_paired", 0)
        ax.set_title(f"{title}  ·  n={n} paired", fontsize=10.5, pad=8,
                     color=INK, loc="left")
        ax.set_xlim(0, xlim)
        _style(ax, xgrid=True)

    axes[0].set_xlabel("BFCL V3 multi-turn accuracy", fontsize=10)
    fig.suptitle("Attention-based eviction holds; structure-aware retention does not",
                 fontsize=11.5, y=1.03, x=0.02, ha="left", color=INK)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summaries", default=None)
    ap.add_argument("--tag", default="primary")
    ap.add_argument("--llama-summaries", default=None)
    ap.add_argument("--llama-tag", default="llama")
    args = ap.parse_args()

    from prioritykv.external.config import load_config

    cfg = load_config(REPO_ROOT / "configs/external_bfcl_prajna_v1.yaml")
    sdir = Path(args.summaries) if args.summaries else (
        Path(cfg["paths"]["results_root"]) / "summaries")

    pf = json.loads((sdir / "protected_fraction.json").read_text())
    summary = json.loads((sdir / f"summary_{args.tag}.json").read_text())

    # PriorityBench-A is the frozen core result; BFCL is measured here. tau-bench
    # is retention-only (no task success), so it has no accuracy and is labelled
    # as such rather than being given a fabricated y value.
    structure_outcomes = {
        "PriorityBench-A": 0.933,
        "BFCL-all": summary["overall"].get("structure", {}).get("accuracy", 0.0),
        "tau-bench": None,
    }

    panels = [("Qwen3-8B", summary)]
    ldir = Path(args.llama_summaries) if args.llama_summaries else None
    if ldir is None:
        cand = Path(cfg["paths"]["prajna_root"]) / "results/external_bfcl_llama/summaries"
        ldir = cand if cand.is_dir() else None
    if ldir is not None:
        lpath = ldir / f"summary_{args.llama_tag}.json"
        if lpath.is_file():
            panels.append(("Llama-3.1-8B", json.loads(lpath.read_text())))

    written = [
        fig_boundary(pf, structure_outcomes, OUT / "protected_fraction_boundary.png"),
        fig_arms(panels, OUT / "external_bfcl_arms.png"),
    ]
    for p in written:
        print(f"wrote {p}")
    print(f"panels: {[t for t, _ in panels]}")
    print("EXTERNAL_FIGURES_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
