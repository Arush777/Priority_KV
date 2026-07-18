#!/usr/bin/env python3
"""Generate publication SVG/PDF figures from frozen PriorityKV artifacts.

The script intentionally uses only the Python standard library. If `rsvg-convert`
is available, matching PDF files are emitted for LaTeX/arXiv packaging.

Design notes (paper conventions):
- No in-image titles, subtitles, or footnotes: LaTeX captions carry that text.
- Result charts show Wilson 95% intervals derived from the canonical counts.
- Categorical colors are fixed per entity across figures and were validated for
  color-vision-deficiency separation and lightness/chroma bands on white.
- Text uses ink tokens, never series colors; grids are recessive hairlines.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Sequence


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures"

# Ink and chrome tokens (light/print surface).
SURFACE = "#FFFFFF"
INK = "#0B0B0B"
SEC = "#52514E"
MUTED = "#898781"
GRID = "#E1E0D9"
BASELINE = "#C3C2B7"

# Entity colors -- fixed across every figure (CVD-validated in display order).
C_STRUCT = "#2A78D6"   # structure-aware arm (hero)
C_FULL = "#008300"     # FullKV / keep-all upper bound
C_RB = "#E87BA4"       # role-blind / uniform baseline
C_RAND = "#EDA100"     # random control
C_INT4 = "#9EC5F4"     # demoted-precision pages (light step of the blue ramp)
C_ACCENT = "#EB6834"   # limitation accent (cold scratch)
C_VIOLET = "#4A3AA7"   # constraint role / merge stage
C_AQUA = "#1BAF7A"     # free-form state role
C_FILLER = "#D8D7D2"   # filler spans (deliberate de-emphasis, not a series)

FONT = "DejaVu Sans,Arial,sans-serif"


def esc(text_value: object) -> str:
    return (
        str(text_value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def text(x: float, y: float, value: object, *, size: float = 15, anchor: str = "middle",
         weight: int = 400, fill: str = SEC) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'font-family="{FONT}" font-size="{size:g}" '
        f'font-weight="{weight}" fill="{fill}">{esc(value)}</text>'
    )


def line(x1: float, y1: float, x2: float, y2: float, *, color: str = GRID,
         width: float = 1.0, dash: str | None = None) -> str:
    dashed = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{color}" stroke-width="{width}"{dashed}/>'
    )


def arrow(x1: float, y1: float, x2: float, y2: float, *, color: str = SEC,
          width: float = 2.0) -> str:
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{color}" stroke-width="{width}" marker-end="url(#arrowhead)"/>'
    )


def rect(x: float, y: float, width: float, height: float, color: str,
         *, stroke: str = "none", stroke_width: float = 1.0, rx: float = 0.0,
         opacity: float = 1.0) -> str:
    extra = f' rx="{rx:.1f}"' if rx else ""
    if opacity < 1.0:
        extra += f' fill-opacity="{opacity:.2f}"'
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(0, width):.1f}" '
        f'height="{max(0, height):.1f}" fill="{color}" stroke="{stroke}" '
        f'stroke-width="{stroke_width:g}"{extra}/>'
    )


def bar(x: float, y_top: float, width: float, height: float, color: str,
        *, radius: float = 3.0) -> str:
    """Column with a rounded data-end and a square baseline."""
    if height <= 0.05:
        return ""
    r = min(radius, width / 2, height)
    y_bot = y_top + height
    return (
        f'<path d="M{x:.1f},{y_bot:.1f} L{x:.1f},{y_top + r:.1f} '
        f'Q{x:.1f},{y_top:.1f} {x + r:.1f},{y_top:.1f} '
        f'L{x + width - r:.1f},{y_top:.1f} '
        f'Q{x + width:.1f},{y_top:.1f} {x + width:.1f},{y_top + r:.1f} '
        f'L{x + width:.1f},{y_bot:.1f} Z" fill="{color}"/>'
    )


def wilson(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% score interval for a binomial proportion."""
    if n <= 0:
        return (p, p)
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def svg_document(width: int, height: int, body: Sequence[str], title_value: str) -> str:
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title">',
            f'<title id="title">{esc(title_value)}</title>',
            '<defs><marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" '
            'refY="3.5" orient="auto"><polygon points="0 0, 10 3.5, 0 7" '
            f'fill="{SEC}"/></marker></defs>',
            rect(0, 0, width, height, SURFACE),
            *body,
            "</svg>",
        ]
    )


def write_figure(name: str, svg: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    svg_path = OUT / f"{name}.svg"
    svg_path.write_text(svg + "\n", encoding="utf-8")
    converter = shutil.which("rsvg-convert")
    if converter:
        subprocess.run(
            [converter, "--format=pdf", "--output", str(OUT / f"{name}.pdf"), str(svg_path)],
            check=True,
        )


def grouped_bars(
    *,
    figure_title: str,
    categories: Sequence[str],
    series: Sequence[tuple[str, Sequence[float], str]],
    ci_ns: Optional[Sequence[int]],
    y_label: str,
) -> str:
    """Grouped column chart with optional Wilson intervals (one n per category)."""
    width, height = 1200, 500
    left, right, top = 92, 24, 40
    plot_bottom = 408
    plot_w, plot_h = width - left - right, plot_bottom - top

    body: list[str] = []
    # Recessive grid and tick labels; solid hairlines only.
    for i in range(6):
        v = i / 5
        y = plot_bottom - plot_h * v
        body.append(line(left, y, left + plot_w, y))
        body.append(text(left - 12, y + 5, f"{v:.1f}", size=14, anchor="end", fill=MUTED))
    body.append(line(left, plot_bottom, left + plot_w, plot_bottom, color=BASELINE, width=1.5))
    body.append(
        f'<text x="28" y="{top + plot_h / 2:.1f}" transform="rotate(-90 28 '
        f'{top + plot_h / 2:.1f})" text-anchor="middle" font-family="{FONT}" '
        f'font-size="16" fill="{SEC}">{esc(y_label)}</text>'
    )

    group_w = plot_w / len(categories)
    usable = group_w * 0.64
    slot_w = usable / len(series)
    bar_w = slot_w - 3.0  # 3px surface gap between adjacent bars
    for ci, category in enumerate(categories):
        group_x = left + ci * group_w + (group_w - usable) / 2
        n = ci_ns[ci] if ci_ns else 0
        for si, (_, values, color) in enumerate(series):
            v = float(values[ci])
            x = group_x + si * slot_w + 1.5
            y_top = plot_bottom - plot_h * v
            body.append(bar(x, y_top, bar_w, plot_h * v, color))
            label_top = y_top
            if n:
                lo, hi = wilson(v, n)
                y_lo = plot_bottom - plot_h * lo
                y_hi = plot_bottom - plot_h * hi
                xc = x + bar_w / 2
                body.append(line(xc, y_lo, xc, y_hi, color=SEC, width=1.8))
                body.append(line(xc - 4, y_hi, xc + 4, y_hi, color=SEC, width=1.8))
                body.append(line(xc - 4, y_lo, xc + 4, y_lo, color=SEC, width=1.8))
                label_top = min(label_top, y_hi)
            body.append(
                text(x + bar_w / 2, label_top - 8, f"{v:.3f}", size=13, fill=SEC)
            )
        body.append(text(left + (ci + 0.5) * group_w, plot_bottom + 27, category,
                         size=17, fill=INK))

    # Legend row below the axis labels (identity channel for >= 2 series).
    entries = [(label, color) for label, _, color in series]
    entry_w = [22 + 8.6 * len(label) + 26 for label, _ in entries]
    legend_x = left + (plot_w - sum(entry_w)) / 2
    legend_y = height - 30
    for (label, color), w in zip(entries, entry_w, strict=True):
        body.append(rect(legend_x, legend_y - 13, 15, 15, color, rx=3))
        body.append(text(legend_x + 22, legend_y - 1, label, size=15, anchor="start",
                         fill=SEC))
        legend_x += w
    return svg_document(width, height, body, figure_title)


def reliability_figure() -> None:
    rows = [json.loads(row) for row in
            (ROOT / "docs/atlas_w4_structure_rows.jsonl").read_text().splitlines()]
    manifests = ["w4_structured_paged_015", "w3_structured_paged", "w4_structured_paged_035"]
    methods = ["keep_uniform", "keep_random", "keep_structure", "keep_keep_all"]
    labels = {"keep_uniform": "Role-blind", "keep_random": "Random",
              "keep_structure": "Structure", "keep_keep_all": "Keep all"}
    colors = {"keep_uniform": C_RB, "keep_random": C_RAND,
              "keep_structure": C_STRUCT, "keep_keep_all": C_FULL}
    lookup = {(row["manifest_id"], row["method"]): row["score"] for row in rows}
    series = [
        (labels[m], [lookup[(manifest, m)] for manifest in manifests], colors[m])
        for m in methods
    ]
    write_figure(
        "reliability_keep_sweep",
        grouped_bars(
            figure_title="Matched keep-budget eviction sweep",
            categories=["15% keep", "25% keep", "35% keep"],
            series=series,
            ci_ns=[14, 14, 14],
            y_label="PriorityBench score",
        ),
    )


def quality_figure() -> None:
    data = json.loads(
        (ROOT / "jobs/results/mg_b_lock240_quality_gpu01_r1/summary.json").read_text())
    contexts = ["8000", "16000", "32000"]
    names = [("FullKV", "full", C_FULL),
             ("Role-blind INT4", "uniform", C_RB),
             ("Structure INT4", "structure", C_STRUCT)]
    series = [
        (label, [data["by_context"][ctx][key]["mean"] for ctx in contexts], color)
        for label, key, color in names
    ]
    ns = [data["by_context"][ctx]["full"]["n"] for ctx in contexts]
    write_figure(
        "lock240_quality_by_length",
        grouped_bars(
            figure_title="Lock-240 quality by context length",
            categories=["8k", "16k", "32k"],
            series=series,
            ci_ns=ns,
            y_label="PriorityBench score",
        ),
    )


def systems_figure() -> None:
    peak = json.loads(
        (ROOT / "jobs/results/mg_a_peak_mem_gpu5_r1/summary.json").read_text())
    latency = json.loads(
        (ROOT / "jobs/results/d4_latency_m3c_gpu56_r1/summary.json").read_text())
    arm = peak["arms"]["mixed_structure_fi_shim"]
    memory_values = [
        arm["compression_ratio_modeled_mean"],
        arm["payload_ratio_measured_mean"],
        peak["structure_vs_fullkv_peak_ratio"],
    ]
    ctx = latency["m3"]["ctx_gates"]
    latency_values = [
        ctx["8000"]["e2e_ratio"],
        ctx["16000"]["e2e_ratio"],
        ctx["8000"]["tpot_ratio"],
        ctx["16000"]["tpot_ratio"],
    ]

    width, height = 1200, 400
    y_max = 1.3
    plot_top, plot_bottom = 56, 330
    plot_h = plot_bottom - plot_top
    panels = [
        (86, 470, "KV memory, ratio to FullKV (lower is better)",
         ["Modeled", "Payload", "Peak"], memory_values, True),
        (656, 470, "Latency, ratio to FullKV (lower is better)",
         ["E2E 8k", "E2E 16k", "TPOT 8k", "TPOT 16k"], latency_values, False),
    ]
    body: list[str] = []
    for x0, panel_w, panel_title, cat_labels, values, label_parity in panels:
        body.append(text(x0, 32, panel_title, size=16, anchor="start", weight=600,
                         fill=INK))
        for tick in (0.0, 0.25, 0.5, 0.75, 1.25):
            y = plot_bottom - plot_h * tick / y_max
            body.append(line(x0, y, x0 + panel_w, y))
            body.append(text(x0 - 10, y + 5, f"{tick:.2f}", size=13, anchor="end",
                             fill=MUTED))
        parity_y = plot_bottom - plot_h * 1.0 / y_max
        body.append(line(x0, parity_y, x0 + panel_w, parity_y, color=SEC,
                         width=1.5, dash="7 5"))
        if label_parity:
            body.append(text(x0 + panel_w - 2, parity_y - 7, "FullKV parity",
                             size=13, anchor="end", fill=SEC))
        body.append(text(x0 - 10, parity_y + 5, "1.00", size=13, anchor="end",
                         fill=MUTED))
        body.append(line(x0, plot_bottom, x0 + panel_w, plot_bottom,
                         color=BASELINE, width=1.5))
        slot = panel_w / len(values)
        bar_w = min(64.0, slot * 0.52)
        for i, (label, value) in enumerate(zip(cat_labels, values, strict=True)):
            x = x0 + i * slot + (slot - bar_w) / 2
            y_top = plot_bottom - plot_h * value / y_max
            body.append(bar(x, y_top, bar_w, plot_bottom - y_top, C_STRUCT))
            body.append(text(x + bar_w / 2, y_top - 8, f"{value:.3f}×", size=13,
                             fill=SEC))
            body.append(text(x + bar_w / 2, plot_bottom + 26, label, size=15,
                             fill=INK))
    write_figure("systems_tradeoff",
                 svg_document(width, height, body, "PriorityKV systems tradeoff"))


def panel_box(body: list[str], x: float, y: float, width: float, height: float,
              title_value: str, lines: Sequence[str], *, tint: str | None = None,
              border: str = BASELINE, border_width: float = 1.5) -> None:
    body.append(rect(x, y, width, height, SURFACE, stroke=border,
                     stroke_width=border_width, rx=8))
    if tint:
        body.append(rect(x, y, width, height, tint, rx=8, opacity=0.08))
    body.append(text(x + width / 2, y + 30, title_value, size=17, weight=600,
                     fill=INK))
    for i, value in enumerate(lines):
        body.append(text(x + width / 2, y + 56 + i * 22, value, size=13.5, fill=SEC))


def _ink_for(color: str) -> str:
    """Pick label ink by fill luminance so in-strip labels always read."""
    r, g, b = (int(color[i:i + 2], 16) / 255 for i in (1, 3, 5))
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "#FFFFFF" if lum < 0.45 else INK


def token_strip(body: list[str], x: float, y: float, width: float, height: float,
                segments: Sequence[tuple[str, float, str]]) -> None:
    cursor = x
    for label, fraction, color in segments:
        seg_w = width * fraction
        body.append(rect(cursor + 1, y, seg_w - 2, height, color))
        if label and seg_w >= 44:
            body.append(text(cursor + seg_w / 2, y + height / 2 + 5, label,
                             size=13, fill=_ink_for(color), weight=600))
        cursor += seg_w
    body.append(rect(x, y, width, height, "none", stroke=BASELINE, stroke_width=1.2))


def overview_diagram() -> None:
    width, height = 1300, 640
    body: list[str] = []

    body.append(text(60, 42, "Long agent trace (known message roles)", size=16,
                     anchor="start", weight=600, fill=INK))
    trace = [
        ("System", 0.10, C_FULL),
        ("Tool schema", 0.14, C_STRUCT),
        ("Filler", 0.24, C_FILLER),
        ("Constraint", 0.13, C_VIOLET),
        ("Filler", 0.20, C_FILLER),
        ("State", 0.09, C_AQUA),
        ("Recent", 0.10, C_RAND),
    ]
    token_strip(body, 60, 56, 1180, 50, trace)

    panel_box(body, 60, 190, 300, 130, "Structural tagger",
              ["chat role + schema markers", "sink = first 16 tokens",
               "recent = last 128 tokens"], tint=C_FULL)
    panel_box(body, 500, 190, 300, 130, "Matched allocator",
              ["same keep / INT4 budget as", "every baseline arm; protects",
               "protocol state, demotes filler"], tint=C_STRUCT)
    panel_box(body, 940, 190, 300, 130, "Paged KV cache",
              ["16-token pages", "hot pages: BF16",
               "cold pages: packed INT4"], tint=C_VIOLET)
    body.append(arrow(210, 106, 210, 190))
    body.append(arrow(360, 255, 500, 255))
    body.append(arrow(800, 255, 940, 255))

    body.append(text(60, 396, "Role-blind eviction (sink + recent)", size=15,
                     anchor="start", weight=600, fill=INK))
    role_blind = [
        ("Sink", 0.10, MUTED),
        ("evicted middle", 0.70, "#EFEEEA"),
        ("Recent", 0.20, MUTED),
    ]
    token_strip(body, 60, 410, 540, 46, role_blind)
    body.append(text(60, 486, "agent-critical middle state can be deleted",
                     size=13.5, anchor="start", fill=SEC))

    body.append(text(700, 396, "Structure-aware mixed cache", size=15,
                     anchor="start", weight=600, fill=INK))
    mixed = [
        ("", 0.10, C_FULL),
        ("BF16", 0.14, C_STRUCT),
        ("INT4", 0.24, C_INT4),
        ("BF16", 0.13, C_VIOLET),
        ("INT4", 0.20, C_INT4),
        ("", 0.09, C_AQUA),
        ("", 0.10, C_RAND),
    ]
    token_strip(body, 700, 410, 540, 46, mixed)
    body.append(text(700, 486, "all positions kept; filler demoted to INT4",
                     size=13.5, anchor="start", fill=SEC))

    body.append(line(60, 560, 1240, 560, color=GRID))
    body.append(text(60, 596,
                     "Conceptual diagram (not measured data). The tagger uses prompt "
                     "metadata only, never future attention or benchmark answers.",
                     size=13.5, anchor="start", fill=MUTED))
    write_figure("prioritykv_overview",
                 svg_document(width, height, body, "PriorityKV overview"))


def decode_diagram() -> None:
    width, height = 1400, 560
    body: list[str] = []

    panel_box(body, 60, 100, 250, 120, "Hot cache",
              ["BF16 protected pages", "+ BF16 decode tail", "GPU resident"],
              tint=C_FULL)
    panel_box(body, 60, 330, 250, 120, "Cold cache",
              ["packed INT4 pages", "+ per-group scales", "smaller payload"],
              tint=C_STRUCT)

    panel_box(body, 460, 330, 250, 120, "Cold scratch",
              ["dequantize INT4 to BF16", "full-layer, transient",
               "limits peak; adds TPOT"],
              tint=C_ACCENT, border=C_ACCENT, border_width=2.0)
    panel_box(body, 460, 100, 250, 120, "FlashInfer attention A",
              ["query × hot K/V", "partial output + LSE"], tint=None)
    panel_box(body, 860, 330, 250, 120, "FlashInfer attention B",
              ["query × cold K/V", "partial output + LSE"], tint=None)
    panel_box(body, 860, 100, 250, 120, "LSE merge",
              ["flashinfer.merge_state", "exact softmax over", "hot ∪ cold pages"],
              tint=C_VIOLET)
    panel_box(body, 1210, 100, 130, 120, "Decode",
              ["next-token", "output"], tint=None)

    body.append(arrow(310, 160, 460, 160))
    body.append(arrow(310, 390, 460, 390))
    body.append(arrow(710, 390, 860, 390))
    body.append(arrow(710, 160, 860, 160))
    body.append(arrow(985, 330, 985, 220))
    body.append(arrow(1110, 160, 1210, 160))

    body.append(line(60, 490, 1340, 490, color=GRID))
    body.append(text(60, 524,
                     "Conceptual diagram (not measured data). At most two homogeneous "
                     "FlashInfer calls per layer; no full Hugging Face cache is "
                     "materialized.",
                     size=13.5, anchor="start", fill=MUTED))
    write_figure("flashinfer_decode_dataflow",
                 svg_document(width, height, body,
                              "PriorityKV FlashInfer decode dataflow"))


def main() -> None:
    overview_diagram()
    decode_diagram()
    reliability_figure()
    quality_figure()
    systems_figure()
    print(f"wrote publication figures to {OUT}")


if __name__ == "__main__":
    main()
