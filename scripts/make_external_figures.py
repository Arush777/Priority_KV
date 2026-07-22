#!/usr/bin/env python3
"""Figures for EXTERNAL_BFCL_PRAJNA_V1, built from frozen summary data only.

Two figures:

``protected_fraction_boundary``
    The paper's central claim. Structure-aware retention can only express a
    preference while protected mass stays under the keep budget. Workloads are
    discrete, so they are drawn as bars against the budget threshold -- never as
    a line through the points, which would invent a continuous trend from the
    available workload measurements.

``external_bfcl_arms``
    The paired arm table, one panel per model, so the cross-model replication is
    visible. Emphasis colouring: the two arms that carry the story are coloured,
    the arms that sit at zero are muted, rather than spending six hues on a
    result that is really one comparison.

The rendering follows the same restrained, colourblind-safe palette and
typographic scale as the core paper figures.  The BFCL comparison uses
point-ranges rather than bars because accuracy is an estimate with uncertainty,
not a quantity whose filled area carries meaning.

    uv run python scripts/make_external_figures.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

OUT = REPO_ROOT / "paper" / "figures"

# Shared with make_publication_figures.py.  Hue is reserved for scientific
# distinctions; axes, uncertainty intervals, and zero-valued arms stay neutral.
BLUE = "#2C6DA4"
ORANGE = "#B57A28"
GREEN = "#315A48"
PURPLE = "#72578D"
CRITICAL = "#A44747"
MUTED = "#AEB5BC"
INK = "#20262E"
INK_2 = "#5D6772"
GRID = "#D9DEE3"

# Arms whose value carries the finding; everything else is deliberately muted.
EMPHASIS = {"full": GREEN, "snapkv": PURPLE, "adapt": ORANGE}
MARKERS = {"full": "o", "snapkv": "s", "adapt": "D"}
ARM_LABEL = {
    "full": "FullKV", "snapkv": "SnapKV", "adapt": "ADAPT",
    "structure": "Structure", "uniform": "Uniform", "random": "Random",
}


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Liberation Sans", "DejaVu Sans", "Arial"],
            "font.size": 8.2,
            "axes.labelsize": 8.2,
            "axes.titlesize": 8.8,
            "axes.titleweight": "bold",
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 7.8,
            "axes.edgecolor": INK,
            "axes.linewidth": 0.7,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.55,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
            "pdf.fonttype": 42,
        }
    )


def _wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    radius = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return max(0.0, centre - radius), min(1.0, centre + radius)


def _summary_from_counts(n: int, counts: dict[str, int]) -> dict:
    overall = {}
    for arm, k in counts.items():
        lo, hi = _wilson(k, n)
        overall[arm] = {
            "n": n,
            "n_pass": k,
            "accuracy": k / n,
            "wilson_ci_low": lo,
            "wilson_ci_high": hi,
        }
    return {"n_tasks_paired": n, "overall": overall}


def paper_snapshot() -> tuple[dict, dict, dict]:
    """Load the frozen values reported in paper/body.tex.

    This explicit mode makes the submitted figures reproducible without access
    to the external cluster filesystem.  It is not selected implicitly.
    """
    frozen = json.loads((OUT / "external_paper_snapshot.json").read_text())
    protected = {"keep_frac": frozen["keep_frac"], "workloads": frozen["workloads"]}
    qwen_raw = frozen["models"]["Qwen3-8B"]
    llama_raw = frozen["models"]["Llama-3.1-8B"]
    qwen = _summary_from_counts(qwen_raw["n_tasks_paired"], qwen_raw["n_pass"])
    llama = _summary_from_counts(llama_raw["n_tasks_paired"], llama_raw["n_pass"])
    return protected, qwen, llama


def _style(ax, *, xgrid=False):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(INK)
    ax.spines[["left", "bottom"]].set_linewidth(0.7)
    ax.grid(axis="x" if xgrid else "y", color=GRID, lw=0.55)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_2, length=2.5, width=0.7)


def _save(fig: plt.Figure, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=240, metadata={"Software": "PriorityKV figure generator"})
    fig.savefig(
        out.with_suffix(".pdf"),
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(fig)


def fig_boundary(pf: dict, out: Path) -> Path:
    """Protected fraction per workload against the keep budget."""
    rows = [w for w in pf["workloads"] if not w["workload"].startswith("BFCL-")
            or w["workload"] == "BFCL-all"]
    rows.sort(key=lambda w: w["protected_fraction_mean"])
    names = [w["workload"] for w in rows]
    fracs = [w["protected_fraction_mean"] for w in rows]
    budget = pf.get("keep_frac", 0.25)

    fig, ax = plt.subplots(figsize=(5.8, 1.85))
    y = np.arange(len(rows))

    colors = [BLUE if f <= budget else MUTED for f in fracs]
    ax.barh(y, fracs, color=colors, height=0.46, zorder=3)

    ax.axvline(budget, color=CRITICAL, lw=1.0, zorder=4)
    ax.annotate(f"keep budget = {budget:.0%}", xy=(budget, 1.0),
                xycoords=("data", "axes fraction"), xytext=(4, -4),
                textcoords="offset points", color=CRITICAL, fontsize=7.4,
                ha="left", va="top")

    for yi, (w, f) in enumerate(zip(rows, fracs)):
        value_x = f - 0.012 if f > 0.93 else f + 0.012
        value_ha = "right" if f > 0.93 else "left"
        ax.text(value_x, yi, f"{f:.1%}", va="center", ha=value_ha,
                fontsize=7.7, fontweight="bold", color=INK)

    ax.set_yticks(y)
    ax.set_yticklabels(names, color=INK)
    ax.set_ylim(-0.55, len(rows) - 0.15)
    ax.set_xlim(0, 1.04)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xlabel("Mean protected-role fraction per context")
    _style(ax, xgrid=True)
    fig.subplots_adjust(left=0.17, right=0.985, top=0.93, bottom=0.30)
    _save(fig, out)
    return out


def fig_arms(panels: list[tuple[str, dict]], out: Path) -> Path:
    """Paired-model forest plot with Wilson intervals and exact counts."""
    fig, axes = plt.subplots(1, len(panels), figsize=(3.4 * len(panels), 2.65), sharex=True)
    if len(panels) == 1:
        axes = [axes]

    max_ci = max(
        summary["overall"][a]["wilson_ci_high"]
        for _, summary in panels for a in summary["overall"]
    )
    count_x = max(0.30, max_ci + 0.035)
    xlim = count_x + 0.025

    for ax, (title, summary) in zip(axes, panels):
        order = [a for a in ("full", "snapkv", "adapt", "structure", "uniform", "random")
                 if a in summary["overall"]]
        y = np.arange(len(order))[::-1]

        for yi, arm in zip(y, order):
            arm_data = summary["overall"][arm]
            value = float(arm_data["accuracy"])
            lo = float(arm_data["wilson_ci_low"])
            hi = float(arm_data["wilson_ci_high"])
            color = EMPHASIS.get(arm, MUTED)
            marker = MARKERS.get(arm, "o")
            ax.errorbar(
                value,
                yi,
                xerr=np.maximum(0.0, np.array([[value - lo], [hi - value]])),
                fmt=marker,
                color=color,
                markeredgecolor=INK,
                markeredgewidth=0.45,
                markersize=4.8,
                ecolor=INK_2,
                elinewidth=0.8,
                capsize=2.2,
                capthick=0.8,
                zorder=3,
            )
            ax.text(
                count_x,
                yi,
                f'{int(arm_data["n_pass"])}/{int(arm_data["n"])}',
                va="center",
                ha="right",
                fontsize=7.4,
                fontweight="bold",
                color=color if arm in EMPHASIS else INK_2,
            )

        ax.set_yticks(y)
        ax.set_yticklabels([ARM_LABEL[a] for a in order], color=INK)
        n = summary.get("n_tasks_paired", 0)
        ax.set_title(f"{title} ($n={n}$ paired)", color=INK, loc="left", pad=5)
        ax.set_xlim(-0.012, xlim)
        ax.set_ylim(-0.55, len(order) - 0.45)
        _style(ax, xgrid=True)

    fig.supxlabel("BFCL V3 multi-turn accuracy (Wilson 95% CI)", fontsize=8.2, y=0.035)
    fig.subplots_adjust(left=0.11, right=0.985, top=0.86, bottom=0.20, wspace=0.34)
    _save(fig, out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summaries", default=None)
    ap.add_argument("--tag", default="primary")
    ap.add_argument("--llama-summaries", default=None)
    ap.add_argument("--llama-tag", default="llama")
    ap.add_argument(
        "--paper-snapshot",
        action="store_true",
        help="render the exact frozen counts/fractions reported in paper/body.tex",
    )
    args = ap.parse_args()

    configure_matplotlib()
    if args.paper_snapshot:
        pf, summary, llama_summary = paper_snapshot()
        panels = [("Qwen3-8B", summary), ("Llama-3.1-8B", llama_summary)]
    else:
        cfg = None
        if args.summaries is None or args.llama_summaries is None:
            from prioritykv.external.config import load_config

            cfg = load_config(REPO_ROOT / "configs/external_bfcl_prajna_v1.yaml")
        sdir = Path(args.summaries) if args.summaries else (
            Path(cfg["paths"]["results_root"]) / "summaries")
        pf = json.loads((sdir / "protected_fraction.json").read_text())
        summary = json.loads((sdir / f"summary_{args.tag}.json").read_text())

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
        fig_boundary(pf, OUT / "protected_fraction_boundary.png"),
        fig_arms(panels, OUT / "external_bfcl_arms.png"),
    ]
    for p in written:
        print(f"wrote {p}")
    print(f"panels: {[t for t, _ in panels]}")
    print("EXTERNAL_FIGURES_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
