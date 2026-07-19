#!/usr/bin/env python3
"""Generate PriorityKV publication figures from tracked result bundles.

Conceptual/architectural figures are emitted as hand-authored SVG and converted
to PDF with ``rsvg-convert``. Measured figures are produced with matplotlib.
For visual quality control, every SVG is also rendered to PNG at a large review
size and at a 3.35-inch, 300-dpi column width under ``/tmp``.
"""

from __future__ import annotations

import html
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures"
QC_OUT = Path("/tmp/prioritykv-figure-qc")

FIGURES = (
    "agent_trace_failure_mode",
    "page_allocation_architecture",
    "decode_memory_lifetime",
    "hypothesis_split",
    "eviction_and_baselines",
    "budget_and_transfer",
    "lock240_quality_by_length",
    "systems_tradeoff",
)

STALE = (
    "prioritykv_overview",
    "flashinfer_decode_dataflow",
    "reliability_keep_sweep",
)

# One semantic palette is used throughout diagrams and plots.
COLORS = {
    "ink": "#20262E",
    "muted": "#5D6772",
    "grid": "#D9DEE3",
    "full": "#315A48",
    "blind": "#777F88",
    "random": "#B57A28",
    "structure": "#2C6DA4",
    "attention": "#72578D",
    "attention_alt": "#A06A8A",
    "h2o": "#99643A",
    "bf16": "#315A48",
    "int4": "#82AFC8",
    "evicted": "#ECEFF1",
    "negative": "#A44747",
    "neutral": "#F6F7F8",
    "system": "#315A48",
    "tool": "#2C6DA4",
    "state": "#72578D",
    "constraint": "#A44747",
    "filler": "#B8BEC5",
    "recent": "#B57A28",
    "sink": "#315A48",
}


def read_json(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def wilson(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        raise ValueError("Wilson interval requires n > 0")
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    radius = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return centre - radius, centre + radius


def count_from_mean(mean: float, n: int) -> int:
    successes = int(round(float(mean) * n))
    if not math.isclose(successes / n, float(mean), abs_tol=1e-8):
        raise ValueError(f"mean {mean} is not an integral count over n={n}")
    return successes


def pooled_arm(files: Sequence[str], arm: str) -> tuple[int, int]:
    successes = 0
    total = 0
    for relative in files:
        data = read_json(relative)
        n = int(data["n"])
        successes += count_from_mean(data["arms"][arm]["mean"], n)
        total += n
    return successes, total


def pooled_full(files: Sequence[str]) -> tuple[int, int]:
    successes = 0
    total = 0
    for relative in files:
        data = read_json(relative)
        n = int(data["n"])
        successes += count_from_mean(data["fullkv_mean"], n)
        total += n
    return successes, total


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Liberation Sans", "DejaVu Sans", "Arial"],
            "font.size": 8.2,
            "axes.labelsize": 8.2,
            "axes.titlesize": 9.2,
            "axes.titleweight": "bold",
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 7.8,
            "legend.fontsize": 7.5,
            "axes.edgecolor": COLORS["ink"],
            "axes.linewidth": 0.7,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.55,
            "grid.alpha": 0.9,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "lines.linewidth": 1.2,
        }
    )


def save_plot(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    svg_path = OUT / f"{name}.svg"
    fig.savefig(svg_path)
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)
    normalize_svg(svg_path)


def normalize_svg(path: Path) -> None:
    """Remove renderer-added line-end spaces for reviewable generated SVG."""
    source = path.read_text(encoding="utf-8")
    path.write_text("\n".join(line.rstrip() for line in source.splitlines()) + "\n", encoding="utf-8")


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def stext(
    x: float,
    y: float,
    value: object,
    *,
    size: float = 15,
    anchor: str = "middle",
    weight: int = 400,
    fill: str | None = None,
    family: str = "Liberation Sans,Arial,sans-serif",
    italic: bool = False,
) -> str:
    style = ' font-style="italic"' if italic else ""
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'font-family="{family}" font-size="{size:.1f}" font-weight="{weight}" '
        f'fill="{fill or COLORS["ink"]}"{style}>{esc(value)}</text>'
    )


def srect(
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    fill: str = "white",
    stroke: str | None = None,
    stroke_width: float = 1.1,
    hatch: bool = False,
) -> str:
    actual_fill = "url(#evictedHatch)" if hatch else fill
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" '
        f'fill="{actual_fill}" stroke="{stroke or COLORS["ink"]}" '
        f'stroke-width="{stroke_width:.1f}"/>'
    )


def sline(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    stroke: str | None = None,
    width: float = 1.2,
    dash: str | None = None,
    arrow: bool = False,
) -> str:
    extra = f' stroke-dasharray="{dash}"' if dash else ""
    if arrow:
        extra += ' marker-end="url(#arrow)"'
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{stroke or COLORS["ink"]}" stroke-width="{width:.1f}"{extra}/>'
    )


def spolyline(
    points: Sequence[tuple[float, float]],
    *,
    stroke: str | None = None,
    width: float = 1.2,
    dash: str | None = None,
    arrow: bool = False,
) -> str:
    point_text = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    extra = f' stroke-dasharray="{dash}"' if dash else ""
    if arrow:
        extra += ' marker-end="url(#arrow)"'
    return (
        f'<polyline points="{point_text}" fill="none" stroke="{stroke or COLORS["ink"]}" '
        f'stroke-width="{width:.1f}"{extra}/>'
    )


def svg_document(width: int, height: int, title: str, body: Sequence[str]) -> str:
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title">',
            f'<title id="title">{esc(title)}</title>',
            "<defs>",
            '<marker id="arrow" markerWidth="8" markerHeight="6" refX="7" refY="3" '
            'orient="auto"><path d="M0,0 L8,3 L0,6 z" fill="#20262E"/></marker>',
            '<pattern id="evictedHatch" width="8" height="8" patternUnits="userSpaceOnUse">'
            '<rect width="8" height="8" fill="#ECEFF1"/>'
            '<path d="M-2,8 L8,-2 M2,10 L10,2" stroke="#AEB5BC" stroke-width="1"/>'
            "</pattern>",
            "</defs>",
            srect(0, 0, width, height, fill="white", stroke="white"),
            *body,
            "</svg>",
        ]
    )


def write_diagram(name: str, width: int, height: int, title: str, body: Sequence[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    svg_path = OUT / f"{name}.svg"
    svg_path.write_text(svg_document(width, height, title, body) + "\n", encoding="utf-8")
    normalize_svg(svg_path)
    converter = shutil.which("rsvg-convert")
    if converter is None:
        raise RuntimeError("rsvg-convert is required to emit diagram PDFs")
    subprocess.run(
        [converter, "--format=pdf", "--output", str(OUT / f"{name}.pdf"), str(svg_path)],
        check=True,
    )


def token_cells(
    body: list[str],
    *,
    x: float,
    y: float,
    roles: Sequence[str],
    kept: set[int] | None = None,
    cell_w: float = 37,
    cell_h: float = 31,
) -> None:
    color_for = {
        "S": COLORS["system"],
        "T": COLORS["tool"],
        "D": COLORS["state"],
        "C": COLORS["constraint"],
        "F": COLORS["filler"],
        "R": COLORS["recent"],
    }
    for i, role in enumerate(roles):
        keep = kept is None or i in kept
        body.append(
            srect(
                x + i * cell_w,
                y,
                cell_w - 1,
                cell_h,
                fill=color_for[role] if keep else COLORS["evicted"],
                stroke="white" if keep else COLORS["blind"],
                hatch=not keep,
            )
        )


def agent_trace_failure_mode() -> None:
    width, height = 1200, 505
    roles = list("SSTTTTFFFFDDFFCCFFFFRRRR")
    blind = {0, 1, 6, 7, 8, 9, 12, 13, 16, 17, 22, 23, 24, 25}
    structure = {0, 1, 2, 3, 4, 5, 10, 11, 14, 15, 22, 23, 24, 25}
    body = [
        stext(42, 34, "WHY STRUCTURE MATTERS", size=13, anchor="start", weight=700, fill=COLORS["structure"]),
        stext(42, 59, "Equal token budgets can preserve very different agent state", size=19, anchor="start", weight=700),
        stext(
            42,
            82,
            "Illustrative trace · hatched cells are evicted",
            size=12.5,
            anchor="start",
            fill=COLORS["muted"],
        ),
    ]
    labels = [
        (0, 2, "system"),
        (2, 6, "tool schema"),
        (6, 10, "filler"),
        (10, 12, "state: ORDER_ID"),
        (12, 14, "filler"),
        (14, 16, "new instruction"),
        (16, 22, "filler"),
        (22, 26, "final request"),
    ]
    x0, cw = 76, 40
    for start, end, label in labels:
        x1 = x0 + start * cw
        x2 = x0 + end * cw - 1
        body.append(sline(x1, 119, x2, 119, width=0.9))
        body.append(sline(x1, 115, x1, 123, width=0.9))
        body.append(sline(x2, 115, x2, 123, width=0.9))
        body.append(stext((x1 + x2) / 2, 108, label, size=11.5, fill=COLORS["muted"]))
    token_cells(body, x=x0, y=130, roles=roles, cell_w=cw, cell_h=28)
    body.append(stext(42, 150, "INPUT", size=11.5, anchor="start", weight=700, fill=COLORS["muted"]))

    body.append(stext(42, 231, "A", size=16, anchor="start", weight=700, fill=COLORS["blind"]))
    body.append(stext(76, 200, "role-blind selection", size=14, anchor="start", weight=700, fill=COLORS["blind"]))
    body.append(stext(1116, 200, "14 / 26 kept", size=12, anchor="end", fill=COLORS["muted"]))
    token_cells(body, x=x0, y=211, roles=roles, kept=blind, cell_w=cw, cell_h=30)

    body.append(stext(42, 339, "B", size=16, anchor="start", weight=700, fill=COLORS["structure"]))
    body.append(stext(76, 308, "PriorityKV role-aware selection", size=14, anchor="start", weight=700, fill=COLORS["structure"]))
    body.append(stext(1116, 308, "14 / 26 kept", size=12, anchor="end", fill=COLORS["muted"]))
    token_cells(body, x=x0, y=319, roles=roles, kept=structure, cell_w=cw, cell_h=30)

    callouts = [
        (116, "schema lost", "invalid call", 3),
        (452, "state lost", "wrong ID", 10),
        (788, "constraint lost", "stale policy", 14),
    ]
    for x, line1, line2, idx in callouts:
        target_x = x0 + (idx + 0.5) * cw
        body.append(spolyline([(target_x, 242), (target_x, 385), (x + 128, 385), (x + 128, 403)], stroke=COLORS["negative"], arrow=True))
        body.append(srect(x, 406, 256, 55, fill="white", stroke=COLORS["negative"], stroke_width=1.2))
        body.append(stext(x + 128, 428, line1, size=12, weight=700, fill=COLORS["negative"]))
        body.append(stext(x + 128, 449, line2, size=13.5, weight=700))

    body.append(srect(76, 478, 15, 15, hatch=True, stroke=COLORS["blind"]))
    body.append(stext(98, 490, "hatched = evicted", size=11.3, anchor="start"))
    body.append(stext(1116, 490, "sink + recent are mandatory in both arms", size=11.3, anchor="end", fill=COLORS["muted"]))
    write_diagram("agent_trace_failure_mode", width, height, "Agent trace failure mode", body)


def _page_allocation_architecture_legacy() -> None:
    width, height = 1180, 720
    body = [
        stext(42, 35, "CONCEPTUAL ARCHITECTURE · policy decisions separated from physical storage", size=18, anchor="start", weight=700),
        stext(42, 60, "Code path: tagging.py → page roles → matched allocation → packed_mixed_cache.py", size=13.5, anchor="start", fill=COLORS["muted"]),
        sline(35, 82, 1145, 82, stroke=COLORS["grid"]),
        stext(42, 107, "POLICY PLANE", size=13, anchor="start", weight=700, fill=COLORS["structure"]),
    ]

    boxes = [
        (45, 130, 210, 132, "1 · chat metadata", ["role_for_message", "schema / constraint hints", "no future attention"], COLORS["system"]),
        (305, 130, 210, 132, "2 · token roles", ["SINK · SYSTEM · TOOL", "CONSTRAINT · RECENT", "FILLER · OTHER"], COLORS["tool"]),
        (565, 130, 210, 132, "3 · page assignment", ["P = 16 tokens", "majority role; protected tie", "contiguous dtype runs"], COLORS["state"]),
        (825, 130, 305, 132, "4 · matched decision", ["B(n,k)=round(k n)", "M=sink ∪ recent", "|A|=max(B, |M|)"], COLORS["structure"]),
    ]
    for x, y, w, h, title, lines, color in boxes:
        body.append(srect(x, y, w, h, fill="white", stroke=color, stroke_width=1.6))
        body.append(stext(x + 12, y + 27, title, size=14, anchor="start", weight=700, fill=color))
        for i, value in enumerate(lines):
            body.append(stext(x + 12, y + 55 + i * 23, value, size=12.5, anchor="start"))
    for x1, x2 in ((255, 305), (515, 565), (775, 825)):
        body.append(sline(x1 + 5, 196, x2 - 5, 196, arrow=True))

    body.append(srect(45, 292, 1085, 82, fill=COLORS["neutral"], stroke=COLORS["grid"]))
    body.append(stext(60, 318, "priority order", size=13, anchor="start", weight=700))
    priority_labels = ["sink", "system", "tool", "constraint", "recent", "other/state", "generated", "filler"]
    for i, label in enumerate(priority_labels):
        x = 170 + i * 115
        key = label.split("/")[0]
        color = COLORS.get(key, COLORS["filler"])
        body.append(srect(x, 304, 99, 28, fill=color, stroke="white"))
        body.append(stext(x + 49.5, 323, label, size=11.7, weight=700, fill="white" if key != "filler" else COLORS["ink"]))
        if i < len(priority_labels) - 1:
            body.append(sline(x + 102, 318, x + 112, 318, width=0.9, arrow=True))
    body.append(stext(170, 357, "protect / retain first", size=11.5, anchor="start", fill=COLORS["muted"]))
    body.append(stext(1090, 357, "evict or demote first", size=11.5, anchor="end", fill=COLORS["muted"]))

    body.append(sline(35, 400, 1145, 400, stroke=COLORS["grid"])),
    body.append(stext(42, 425, "STORAGE PLANE", size=13, anchor="start", weight=700, fill=COLORS["bf16"]))
    headers = ["page", "token span", "role", "policy", "dtype", "physical payload"]
    xs = [52, 132, 275, 425, 585, 700]
    widths = [80, 143, 150, 160, 115, 410]
    for x, w, label in zip(xs, widths, headers, strict=True):
        body.append(srect(x, 447, w, 32, fill="#E9EDF0", stroke=COLORS["ink"], stroke_width=0.8))
        body.append(stext(x + 8, 468, label, size=12, anchor="start", weight=700))
    rows = [
        ("p0", "[0,16)", "SINK", "mandatory M", "BF16", "K_bf16 | V_bf16", COLORS["bf16"]),
        ("p1", "[16,32)", "TOOL", "retain / hot", "BF16", "K_bf16 | V_bf16", COLORS["tool"]),
        ("p2", "[32,48)", "FILLER", "demote", "INT4", "K_uint8 + scale_fp32 | V_uint8 + scale_fp32", COLORS["int4"]),
        ("p3", "[48,64)", "STATE", "retain / hot", "BF16", "K_bf16 | V_bf16", COLORS["state"]),
        ("p4", "[64,80)", "FILLER", "evict or demote", "INT4", "packed values + per-group metadata", COLORS["int4"]),
        ("pN", "recent", "RECENT", "mandatory M", "BF16", "decode-local tail", COLORS["recent"]),
    ]
    for ri, row in enumerate(rows):
        y = 479 + ri * 34
        for ci, (x, w) in enumerate(zip(xs, widths, strict=True)):
            fill = "white" if ci < 4 else (row[6] if ci == 4 else "white")
            body.append(srect(x, y, w, 34, fill=fill, stroke=COLORS["grid"], stroke_width=0.8))
            fill_text = "white" if ci == 4 and row[4] != "INT4" else COLORS["ink"]
            body.append(stext(x + 8, y + 22, row[ci], size=11.7, anchor="start", fill=fill_text, weight=700 if ci in (0, 4) else 400))

    body.append(stext(52, 704, "Eviction experiment: selected A is gathered and re-prefilled.", size=12.5, anchor="start", fill=COLORS["muted"]))
    body.append(stext(1128, 704, "Mixed path: all positions remain; f=0.75 positions are packed INT4.", size=12.5, anchor="end", fill=COLORS["muted"]))
    write_diagram("page_allocation_architecture", width, height, "PriorityKV page allocation architecture", body)


def _decode_memory_lifetime_legacy() -> None:
    width, height = 1160, 680
    body = [
        stext(42, 35, "CONCEPTUAL DATAFLOW + MEASURED ENDPOINTS · frozen full-scratch decode", size=18, anchor="start", weight=700),
        stext(42, 60, "At most two homogeneous FlashInfer calls per layer; P2 streamed-cold is smoke-only and is not shown as a systems result.", size=13, anchor="start", fill=COLORS["muted"]),
    ]
    stages = [
        (45, 115, 140, "HF prefill", ["full BF16 past", "native model path"], COLORS["blind"]),
        (225, 115, 155, "pack", ["hot BF16 pages", "cold uint8 + scales"], COLORS["structure"]),
        (420, 115, 175, "eager cold scratch", ["INT4 → BF16", "all layers; GPU"], COLORS["int4"]),
        (635, 95, 145, "FI call A", ["q × hot+tail", "O_hot, LSE_hot"], COLORS["bf16"]),
        (635, 205, 145, "FI call B", ["q × cold scratch", "O_cold, LSE_cold"], COLORS["int4"]),
        (825, 145, 145, "LSE merge", ["merge_state", "max |Δ|≈4.88e−4"], COLORS["attention"]),
        (1010, 145, 110, "output", ["append KV", "next layer"], COLORS["ink"]),
    ]
    for x, y, w, title, lines, color in stages:
        body.append(srect(x, y, w, 78, fill="white", stroke=color, stroke_width=1.5))
        body.append(stext(x + w / 2, y + 24, title, size=13.5, weight=700, fill=color))
        for i, value in enumerate(lines):
            body.append(stext(x + w / 2, y + 47 + i * 18, value, size=11.3))
    body.append(sline(185, 154, 220, 154, arrow=True))
    body.append(sline(380, 154, 415, 154, arrow=True))
    body.append(spolyline([(595, 154), (615, 154), (615, 134), (630, 134)], arrow=True))
    body.append(spolyline([(595, 154), (615, 154), (615, 244), (630, 244)], arrow=True))
    body.append(spolyline([(780, 134), (803, 134), (803, 169), (820, 169)], arrow=True))
    body.append(spolyline([(780, 244), (803, 244), (803, 193), (820, 193)], arrow=True))
    body.append(sline(970, 181, 1005, 181, arrow=True))

    body.append(stext(45, 335, "allocation lifetime (per request)", size=13.5, anchor="start", weight=700))
    x0, x1 = 80, 1090
    ticks = [(80, "prefill"), (260, "pack"), (445, "prepare"), (650, "decode token 1"), (860, "tokens 2…T"), (1090, "teardown")]
    body.append(sline(x0, 367, x1, 367, stroke=COLORS["ink"], width=1.0, arrow=True))
    for x, label in ticks:
        body.append(sline(x, 359, x, 375, width=0.9))
        body.append(stext(x, 393, label, size=11.5, fill=COLORS["muted"]))

    lifetimes = [
        ("packed payload", 260, 1090, 425, COLORS["structure"], "0.719× vs BF16 model"),
        ("BF16 cold scratch", 445, 1090, 468, COLORS["int4"], "drives 0.868× peak"),
        ("decode tail", 650, 1090, 511, COLORS["recent"], "grows token by token"),
    ]
    for label, start, end, y, color, value in lifetimes:
        body.append(stext(45, y + 5, label, size=11.7, anchor="start", weight=700))
        body.append(srect(start, y - 11, end - start, 20, fill=color, stroke=color))
        body.append(stext((start + end) / 2, y + 5, value, size=11.2, fill="white" if color != COLORS["int4"] else COLORS["ink"], weight=700))
        body.append(sline(end, y - 16, end, y + 14, stroke=COLORS["negative"], width=1.3))
    body.append(stext(1090, 545, "released with decode state", size=11.5, anchor="end", fill=COLORS["negative"]))

    measures = [
        (260, 445, 580, "pack 34.8 / 48.1 ms"),
        (445, 650, 610, "cold scratch 14.2 / 19.7 ms"),
        (80, 650, 640, "E2E TTFT 1.118× / 1.113×"),
        (650, 1090, 580, "TPOT 1.200× / 1.215× (cost)"),
    ]
    for start, end, y, label in measures:
        body.append(sline(start, y, end, y, stroke=COLORS["negative"] if "TPOT" in label else COLORS["ink"], width=1.0))
        body.append(sline(start, y - 6, start, y + 6, width=1.0))
        body.append(sline(end, y - 6, end, y + 6, width=1.0))
        body.append(stext((start + end) / 2, y - 8, label, size=11.4, weight=700, fill=COLORS["negative"] if "TPOT" in label else COLORS["ink"]))
    write_diagram("decode_memory_lifetime", width, height, "FlashInfer decode memory lifetime", body)


def _hypothesis_split_legacy() -> None:
    width, height = 1080, 500
    body = [
        stext(38, 35, "CONCEPTUAL EXPERIMENT LOGIC · measured outcomes remain separate", size=18, anchor="start", weight=700),
        stext(38, 60, "Eviction removes state; mixed precision approximates retained state; systems metrics measure implementation cost.", size=13.5, anchor="start", fill=COLORS["muted"]),
    ]
    panels = [
        (38, "H1 · eviction reliability", "SUPPORTED (SCOPED)", COLORS["structure"], ["Qwen, k=0.25, n=120", "structure 112/120", "uniform/random 1/120", "buried: structure < FullKV"]),
        (382, "H2 · role-aware INT4 quality", "FALSIFIED", COLORS["negative"], ["Qwen, f=0.75, n=240", "FullKV 0.8875", "uniform 0.8792", "structure 0.8833"]),
        (726, "H3 · packed systems path", "BYTES WIN · LATENCY COST", COLORS["attention"], ["H200, single request", "payload 0.719×", "peak 0.868×", "TPOT 1.200–1.215×"]),
    ]
    for x, title, verdict, color, lines in panels:
        body.append(srect(x, 100, 316, 330, fill="white", stroke=color, stroke_width=1.8))
        body.append(srect(x, 100, 316, 46, fill=color, stroke=color))
        body.append(stext(x + 158, 129, title, size=14.5, weight=700, fill="white"))
        body.append(stext(x + 158, 183, verdict, size=14.5, weight=700, fill=color))
        body.append(sline(x + 24, 199, x + 292, 199, stroke=COLORS["grid"]))
        for i, value in enumerate(lines):
            body.append(stext(x + 28, 235 + i * 36, value, size=13.2, anchor="start"))
        body.append(stext(x + 158, 404, "do not transfer this verdict across columns", size=11.3, fill=COLORS["muted"], italic=True))
    body.append(stext(540, 475, "Hybrid complementarity was tested and not supported: hybrid = SnapKV on Qwen k=0.25.", size=13.2, weight=700))
    write_diagram("hypothesis_split", width, height, "PriorityKV hypothesis split", body)


def _eviction_and_baselines_legacy() -> None:
    p0_files = [
        "jobs/results/p0_w5_s0_kf25_token_gpu7_r1/summary.json",
        "jobs/results/p0_w5_s1_kf25_token_gpu7_r1/summary.json",
        "jobs/results/p0_w5_s2_kf25_token_gpu0_r1/summary.json",
    ]
    p1_files = [
        "jobs/results/p1_attn_baselines_s0_kf25_gpu1_r3/summary.json",
        "jobs/results/p1_attn_baselines_s1_kf25_gpu1_r1/summary.json",
        "jobs/results/p1_attn_baselines_s2_kf25_gpu1_r1/summary.json",
    ]
    h2o_files = [
        "jobs/results/p1_h2o_chunked_s0_kf25_gpu1_r1/summary.json",
        p1_files[1],
        p1_files[2],
    ]
    values = [
        ("FullKV", *pooled_full(p0_files), COLORS["full"], ""),
        ("Uniform", *pooled_arm(p0_files, "uniform"), COLORS["blind"], "///"),
        ("Random", *pooled_arm(p0_files, "random"), COLORS["random"], "xx"),
        ("Structure", *pooled_arm(p0_files, "structure"), COLORS["structure"], ""),
        ("SnapKV", *pooled_arm(p1_files, "snapkv"), COLORS["attention"], ".."),
        ("PyramidKV", *pooled_arm(p1_files, "pyramid"), COLORS["attention_alt"], "\\\\"),
        ("Hybrid", *pooled_arm(p1_files, "hybrid"), "#9A83AA", "++"),
        ("H2O*", *pooled_arm(h2o_files, "h2o"), COLORS["h2o"], "--"),
    ]
    labels = [row[0] for row in values][::-1]
    successes = np.array([row[1] for row in values][::-1])
    ns = np.array([row[2] for row in values][::-1])
    proportions = successes / ns
    intervals = np.array([wilson(int(s), int(n)) for s, n in zip(successes, ns, strict=True)])
    colors = [row[3] for row in values][::-1]
    hatches = [row[4] for row in values][::-1]

    fig, ax = plt.subplots(figsize=(3.35, 3.25))
    y = np.arange(len(labels))
    bars = ax.barh(y, proportions, color=colors, edgecolor=COLORS["ink"], linewidth=0.45)
    for bar, hatch in zip(bars, hatches, strict=True):
        bar.set_hatch(hatch)
    ax.errorbar(
        proportions,
        y,
        xerr=np.maximum(
            0.0,
            np.vstack((proportions - intervals[:, 0], intervals[:, 1] - proportions)),
        ),
        fmt="none",
        ecolor=COLORS["ink"],
        elinewidth=0.75,
        capsize=1.8,
        capthick=0.75,
    )
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 1.14)
    ax.set_xlabel("PriorityBench-A pass rate (Wilson 95% CI)")
    ax.set_title("Qwen3-8B, matched 25% token keep (n=120)", loc="left", pad=5)
    ax.grid(axis="y", visible=False)
    ax.spines[["top", "right"]].set_visible(False)
    for yi, p, s, n in zip(y, proportions, successes, ns, strict=True):
        x = min(p + 0.018, 1.025)
        ax.text(x, yi, f"{s}/{n}", va="center", ha="left", fontsize=7.2)
    struct_y = labels.index("Structure")
    snap_y = labels.index("SnapKV")
    bracket_x = 1.075
    ax.plot([bracket_x, bracket_x + 0.014, bracket_x + 0.014, bracket_x], [struct_y, struct_y, snap_y, snap_y], color=COLORS["ink"], lw=0.7)
    ax.text(1.098, (struct_y + snap_y) / 2, "McNemar\np=0.125", va="center", ha="left", fontsize=7.0)
    fig.text(0.01, 0.005, "* H2O is a chunked reimplementation; all attention baselines are repository reimplementations.", fontsize=6.8, color=COLORS["muted"])
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    save_plot(fig, "eviction_and_baselines")


def page_allocation_architecture() -> None:
    """Two-panel policy-to-storage architecture with direct role/dtype labels."""
    width, height = 1200, 582
    body = [
        stext(42, 34, "(a) Policy allocation", size=15, anchor="start", weight=700),
        stext(42, 56, "Metadata determines token roles; the allocator never uses future attention.", size=12, anchor="start", fill=COLORS["muted"]),
    ]

    stages = [
        (44, 88, 220, "chat spans", ("system", "tool schema", "state / constraint"), COLORS["system"]),
        (326, 88, 220, "token tags", ("SINK · SYSTEM · TOOL", "OTHER · CONSTRAINT", "RECENT · FILLER"), COLORS["tool"]),
        (608, 88, 220, "physical pages", ("16 tokens per page", "majority role", "protected-role tie"), COLORS["state"]),
        (890, 88, 266, "budget controller", ("M = sink ∪ recent", "|A| = max(round(kn), |M|)", "demote low-priority pages"), COLORS["structure"]),
    ]
    for x, y, w, title, lines, color in stages:
        body.append(sline(x, y, x + w, y, stroke=color, width=3.0))
        body.append(stext(x, y + 26, title, size=14, anchor="start", weight=700, fill=color))
        for i, value in enumerate(lines):
            body.append(stext(x, y + 52 + i * 22, value, size=11.8, anchor="start"))
    for x1, x2 in ((264, 326), (546, 608), (828, 890)):
        body.append(sline(x1 + 7, 148, x2 - 7, 148, arrow=True))

    body.append(stext(42, 204, "priority", size=11.5, anchor="start", weight=700, fill=COLORS["muted"]))
    priority = [
        ("sink", COLORS["sink"]),
        ("recent", COLORS["recent"]),
        ("system", COLORS["system"]),
        ("tool", COLORS["tool"]),
        ("constraint", COLORS["constraint"]),
        ("other", COLORS["filler"]),
        ("generated", COLORS["filler"]),
        ("filler", COLORS["evicted"]),
    ]
    for i, (label, color) in enumerate(priority):
        x = 122 + i * 130
        body.append(srect(x, 185, 112, 29, fill=color, stroke="white"))
        dark_text = color in {COLORS["filler"], COLORS["evicted"], COLORS["int4"]}
        body.append(stext(x + 56, 205, label, size=11.3, weight=700, fill=COLORS["ink"] if dark_text else "white"))
        if i < len(priority) - 1:
            body.append(sline(x + 116, 199, x + 126, 199, width=0.8, arrow=True))
    body.append(stext(122, 231, "protect first", size=10.8, anchor="start", fill=COLORS["muted"]))
    body.append(stext(1154, 231, "evict / demote first", size=10.8, anchor="end", fill=COLORS["muted"]))

    body.extend([
        sline(38, 258, 1162, 258, stroke=COLORS["grid"]),
        stext(42, 286, "(b) Physical page stack", size=15, anchor="start", weight=700),
        stext(42, 307, "Policy labels become page metadata; storage remains contiguous within each dtype run.", size=12, anchor="start", fill=COLORS["muted"]),
    ])
    page_rows = [
        ("p0", "0–15", "SINK", "BF16", COLORS["sink"]),
        ("p1", "16–31", "TOOL", "BF16", COLORS["tool"]),
        ("p2", "32–47", "FILLER", "INT4", COLORS["int4"]),
        ("p3", "48–63", "OTHER", "BF16", COLORS["state"]),
        ("p4", "64–79", "FILLER", "INT4", COLORS["int4"]),
        ("pN", "tail", "RECENT", "BF16", COLORS["recent"]),
    ]
    for i, (page, span, role, dtype, color) in enumerate(page_rows):
        x = 50 + i * 184
        body.append(srect(x, 337, 166, 88, fill="white", stroke=COLORS["ink"], stroke_width=0.9))
        body.append(srect(x, 337, 166, 23, fill=COLORS["neutral"], stroke=COLORS["ink"], stroke_width=0.9))
        body.append(stext(x + 9, 353, page, size=11.3, anchor="start", weight=700))
        body.append(stext(x + 157, 353, span, size=10.8, anchor="end", fill=COLORS["muted"]))
        body.append(stext(x + 83, 382, role, size=12, weight=700, fill=color if dtype == "BF16" else COLORS["ink"]))
        body.append(srect(x + 8, 393, 150, 24, fill=color, stroke="white"))
        body.append(stext(x + 83, 410, dtype, size=11, weight=700, fill="white" if dtype == "BF16" else COLORS["ink"]))

    body.append(sline(598, 432, 598, 468, arrow=True))
    body.append(spolyline([(598, 468), (310, 468), (310, 485)], arrow=True))
    body.append(spolyline([(598, 468), (890, 468), (890, 485)], arrow=True))
    body.append(stext(310, 478, "EVICTION STUDY", size=10.8, weight=700, fill=COLORS["structure"]))
    body.append(srect(85, 490, 450, 78, fill="white", stroke=COLORS["structure"], stroke_width=1.3))
    body.append(stext(105, 516, "gather selected positions A", size=12.2, anchor="start", weight=700))
    body.append(sline(285, 524, 340, 524, arrow=True))
    body.append(stext(360, 516, "re-prefill shortened trace", size=12.2, anchor="start", weight=700))
    body.append(stext(105, 549, "unselected state is absent", size=11.2, anchor="start", fill=COLORS["muted"]))
    body.append(stext(515, 549, "matched token budget", size=11.2, anchor="end", fill=COLORS["muted"]))

    body.append(stext(890, 478, "MIXED-PRECISION STUDY", size=10.8, weight=700, fill=COLORS["attention"]))
    body.append(srect(665, 490, 450, 78, fill="white", stroke=COLORS["attention"], stroke_width=1.3))
    body.append(srect(682, 507, 130, 28, fill=COLORS["bf16"], stroke="white"))
    body.append(stext(747, 526, "hot K/V · BF16", size=11.3, weight=700, fill="white"))
    body.append(srect(825, 507, 170, 28, fill=COLORS["int4"], stroke="white"))
    body.append(stext(910, 526, "cold K/V · packed INT4", size=11.1, weight=700))
    body.append(srect(1008, 507, 90, 28, fill=COLORS["neutral"], stroke=COLORS["grid"]))
    body.append(stext(1053, 526, "scales", size=11.1, weight=700))
    body.append(stext(682, 555, "all token positions remain; only representation changes", size=11.2, anchor="start", fill=COLORS["muted"]))
    write_diagram("page_allocation_architecture", width, height, "PriorityKV policy and physical page architecture", body)


def decode_memory_lifetime() -> None:
    """Decode lifecycle with persistent, transient, and growing allocations."""
    width, height = 1200, 625
    body = [
        stext(42, 34, "Per-layer mixed decode", size=15, anchor="start", weight=700),
        stext(42, 55, "The packed payload persists; eager BF16 cold scratch is transient implementation overhead.", size=12, anchor="start", fill=COLORS["muted"]),
        stext(42, 91, "PERSISTENT CACHE", size=10.8, anchor="start", weight=700, fill=COLORS["muted"]),
    ]
    body.append(srect(42, 106, 215, 62, fill="white", stroke=COLORS["bf16"], stroke_width=1.3))
    body.append(stext(58, 131, "hot pages + decode tail", size=12.2, anchor="start", weight=700, fill=COLORS["bf16"]))
    body.append(stext(58, 153, "BF16 · GPU resident", size=11.2, anchor="start"))
    body.append(srect(42, 190, 215, 62, fill="white", stroke=COLORS["int4"], stroke_width=1.3))
    body.append(stext(58, 215, "cold pages", size=12.2, anchor="start", weight=700, fill=COLORS["structure"]))
    body.append(stext(58, 237, "packed INT4 + scales", size=11.2, anchor="start"))

    body.append(stext(322, 91, "MATERIALIZE", size=10.8, anchor="start", weight=700, fill=COLORS["muted"]))
    body.append(srect(322, 190, 180, 62, fill=COLORS["neutral"], stroke=COLORS["int4"], stroke_width=1.3))
    body.append(stext(412, 215, "cold scratch", size=12.2, weight=700, fill=COLORS["structure"]))
    body.append(stext(412, 237, "INT4 → BF16", size=11.2))
    body.append(sline(257, 221, 316, 221, arrow=True))

    body.append(stext(566, 91, "≤ 2 HOMOGENEOUS CALLS", size=10.8, anchor="start", weight=700, fill=COLORS["muted"]))
    body.append(srect(566, 106, 188, 62, fill="white", stroke=COLORS["bf16"], stroke_width=1.3))
    body.append(stext(660, 131, "FlashInfer A", size=12.2, weight=700, fill=COLORS["bf16"]))
    body.append(stext(660, 153, "q × hot + tail", size=11.2))
    body.append(srect(566, 190, 188, 62, fill="white", stroke=COLORS["int4"], stroke_width=1.3))
    body.append(stext(660, 215, "FlashInfer B", size=12.2, weight=700, fill=COLORS["structure"]))
    body.append(stext(660, 237, "q × cold scratch", size=11.2))
    body.append(sline(257, 137, 560, 137, arrow=True))
    body.append(sline(502, 221, 560, 221, arrow=True))
    body.append(stext(534, 171, "q", size=12, weight=700))
    body.append(spolyline([(534, 176), (534, 221), (560, 221)], arrow=True))

    body.append(srect(824, 148, 154, 70, fill="white", stroke=COLORS["attention"], stroke_width=1.4))
    body.append(stext(901, 176, "LSE merge", size=12.8, weight=700, fill=COLORS["attention"]))
    body.append(stext(901, 199, "two partial states", size=11.2))
    body.append(spolyline([(754, 137), (790, 137), (790, 174), (818, 174)], arrow=True))
    body.append(spolyline([(754, 221), (790, 221), (790, 192), (818, 192)], arrow=True))
    body.append(srect(1042, 148, 116, 70, fill="white", stroke=COLORS["ink"], stroke_width=1.3))
    body.append(stext(1100, 176, "append K/V", size=12.2, weight=700))
    body.append(stext(1100, 199, "next token", size=11.2))
    body.append(sline(978, 183, 1036, 183, arrow=True))
    body.append(spolyline([(1100, 218), (1100, 275), (150, 275), (150, 174)], dash="5,4", arrow=True))
    body.append(stext(621, 292, "decode loop: tail grows by one token", size=11.2, fill=COLORS["muted"]))

    body.extend([
        sline(38, 314, 1162, 314, stroke=COLORS["grid"]),
        stext(42, 340, "Request lifetime", size=15, anchor="start", weight=700),
    ])
    x0, x1 = 205, 1142
    ticks = [(205, "prefill"), (390, "pack"), (565, "prepare"), (750, "decode 1"), (960, "decode 2…T"), (1142, "free")]
    body.append(sline(x0, 371, x1, 371, width=0.9, arrow=True))
    for x, label in ticks:
        body.append(sline(x, 365, x, 378, width=0.8))
        body.append(stext(x, 394, label, size=10.8, fill=COLORS["muted"]))
    lifetimes = [
        ("packed payload", 390, 1142, 424, COLORS["structure"], "persistent · 0.719× payload"),
        ("BF16 cold scratch", 565, 1142, 470, COLORS["int4"], "transient · drives 0.868× peak"),
        ("decode tail", 750, 1142, 516, COLORS["recent"], "grows each step"),
    ]
    for label, start, end, y, color, value in lifetimes:
        body.append(stext(42, y + 5, label, size=11.5, anchor="start", weight=700))
        body.append(srect(start, y - 12, end - start, 22, fill=color, stroke=color))
        body.append(stext((start + end) / 2, y + 5, value, size=11, weight=700, fill="white" if color != COLORS["int4"] else COLORS["ink"]))
    body.append(sline(1142, 406, 1142, 532, stroke=COLORS["negative"], width=1.2))
    body.append(stext(1142, 552, "request state released", size=10.8, anchor="end", fill=COLORS["negative"]))
    body.append(stext(42, 596, "Measured per-token latency: 1.200–1.215× FullKV", size=11.8, anchor="start", fill=COLORS["negative"], weight=700))
    write_diagram("decode_memory_lifetime", width, height, "PriorityKV mixed decode and memory lifetime", body)


def hypothesis_split() -> None:
    """Compact common-input DAG for the three independent hypotheses."""
    width, height = 1200, 445
    body = [
        stext(42, 35, "One frozen benchmark, three independent questions", size=17, anchor="start", weight=700),
        srect(42, 75, 190, 80, fill="white", stroke=COLORS["ink"], stroke_width=1.2),
        stext(137, 103, "COMMON INPUT", size=11.2, weight=700, fill=COLORS["muted"]),
        stext(137, 127, "PriorityBench-A", size=13.2, weight=700),
        stext(137, 146, "frozen model + seeds", size=10.8),
        sline(232, 115, 275, 115, arrow=True),
        sline(275, 115, 275, 345, width=1.0),
    ]
    rows = [
        (82, "H1", "Which tokens survive?", "evict to k=0.25", "blind + attention", "112 vs 1 (blind)", "108 attention · p=.125", COLORS["structure"]),
        (198, "H2", "How are tokens stored?", "pack f=0.75 INT4", "uniform placement", "0.8833 vs 0.8792", "no quality gain", COLORS["negative"]),
        (314, "H3", "What does it cost?", "packed H200 path", "FullKV", "0.719× bytes", "1.20× TPOT", COLORS["attention"]),
    ]
    for y, tag, question, intervention, comparator, result, verdict, color in rows:
        body.append(sline(275, y + 40, 307, y + 40, arrow=True))
        body.append(srect(312, y, 78, 80, fill=color, stroke=color))
        body.append(stext(351, y + 47, tag, size=16, weight=700, fill="white"))
        body.append(srect(390, y, 240, 80, fill="white", stroke=COLORS["grid"], stroke_width=1.0))
        body.append(stext(408, y + 29, question, size=12.2, anchor="start", weight=700))
        body.append(stext(408, y + 55, intervention, size=11.1, anchor="start", fill=COLORS["muted"]))
        body.append(sline(630, y + 40, 662, y + 40, arrow=True))
        body.append(srect(667, y, 196, 80, fill="white", stroke=COLORS["grid"], stroke_width=1.0))
        body.append(stext(682, y + 27, "against", size=10.2, anchor="start", fill=COLORS["muted"])),
        body.append(stext(682, y + 54, comparator, size=12.2, anchor="start", weight=700))
        body.append(sline(863, y + 40, 895, y + 40, arrow=True))
        body.append(srect(900, y, 258, 80, fill="white", stroke=color, stroke_width=1.3))
        body.append(stext(920, y + 29, result, size=13, anchor="start", weight=700, fill=color))
        body.append(stext(920, y + 57, verdict, size=11.5, anchor="start", weight=700))
    body.append(stext(42, 423, "The eviction verdict does not imply an INT4-quality or latency verdict.", size=11.5, anchor="start", fill=COLORS["muted"]))
    body.append(stext(1158, 423, "Hybrid = SnapKV at k=0.25", size=11.5, anchor="end", fill=COLORS["muted"]))
    write_diagram("hypothesis_split", width, height, "Three independent PriorityKV hypotheses", body)


def eviction_and_baselines() -> None:
    """Exact-count forest plot for the matched-budget Qwen comparison."""
    p0_files = [
        "jobs/results/p0_w5_s0_kf25_token_gpu7_r1/summary.json",
        "jobs/results/p0_w5_s1_kf25_token_gpu7_r1/summary.json",
        "jobs/results/p0_w5_s2_kf25_token_gpu0_r1/summary.json",
    ]
    p1_files = [
        "jobs/results/p1_attn_baselines_s0_kf25_gpu1_r3/summary.json",
        "jobs/results/p1_attn_baselines_s1_kf25_gpu1_r1/summary.json",
        "jobs/results/p1_attn_baselines_s2_kf25_gpu1_r1/summary.json",
    ]
    h2o_files = [
        "jobs/results/p1_h2o_chunked_s0_kf25_gpu1_r1/summary.json",
        p1_files[1],
        p1_files[2],
    ]
    values = [
        ("FullKV", *pooled_full(p0_files), COLORS["full"], "o"),
        ("Uniform", *pooled_arm(p0_files, "uniform"), COLORS["blind"], "s"),
        ("Random", *pooled_arm(p0_files, "random"), COLORS["random"], "D"),
        ("Structure", *pooled_arm(p0_files, "structure"), COLORS["structure"], "o"),
        ("SnapKV", *pooled_arm(p1_files, "snapkv"), COLORS["attention"], "s"),
        ("PyramidKV", *pooled_arm(p1_files, "pyramid"), COLORS["attention_alt"], "D"),
        ("Hybrid", *pooled_arm(p1_files, "hybrid"), "#80648C", "^"),
        ("H2O*", *pooled_arm(h2o_files, "h2o"), COLORS["h2o"], "v"),
    ]
    ypos = np.array([8.6, 7.25, 6.25, 4.85, 3.85, 2.85, 1.85, 0.85])
    fig, ax = plt.subplots(figsize=(5.8, 3.05))
    for y, (label, successes, n, color, marker) in zip(ypos, values, strict=True):
        value = successes / n
        lo, hi = wilson(successes, n)
        ax.errorbar(
            value,
            y,
            xerr=np.array([[value - lo], [hi - value]]),
            fmt=marker,
            color=color,
            markeredgecolor=COLORS["ink"],
            markeredgewidth=0.45,
            markersize=5.2,
            ecolor=COLORS["ink"],
            elinewidth=0.8,
            capsize=2.4,
            capthick=0.8,
            zorder=3,
        )
        ax.text(1.11, y, f"{successes}/{n}", va="center", ha="right", fontsize=7.5, fontweight="bold")
    ax.set_yticks(ypos, [row[0] for row in values])
    ax.set_xlim(-0.02, 1.28)
    ax.set_ylim(0.25, 9.18)
    ax.set_xlabel("PriorityBench-A pass rate (Wilson 95% CI)")
    ax.grid(axis="y", visible=False)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.axhline(7.78, color=COLORS["grid"], linewidth=0.6)
    ax.axhline(5.55, color=COLORS["grid"], linewidth=0.6)
    ax.text(-0.018, 8.98, "reference", fontsize=6.7, color=COLORS["muted"], va="bottom")
    ax.text(-0.018, 7.66, "role blind", fontsize=6.7, color=COLORS["muted"], va="bottom")
    ax.text(-0.018, 5.43, "selectors", fontsize=6.7, color=COLORS["muted"], va="bottom")
    ax.text(1.11, 8.98, "passes", fontsize=6.7, color=COLORS["muted"], va="bottom", ha="right")
    bracket_x = 1.145
    struct_y, snap_y = ypos[3], ypos[4]
    ax.plot([bracket_x - 0.012, bracket_x, bracket_x, bracket_x - 0.012], [struct_y, struct_y, snap_y, snap_y], color=COLORS["ink"], lw=0.75)
    ax.text(1.158, (struct_y + snap_y) / 2, "paired\np=.125", va="center", ha="left", fontsize=6.8)
    fig.subplots_adjust(left=0.19, right=0.985, top=0.98, bottom=0.19)
    save_plot(fig, "eviction_and_baselines")


def _budget_and_transfer_legacy() -> None:
    qwen_files = [
        "jobs/results/p1_attn_baselines_s0_kf25_gpu1_r3/summary.json",
        "jobs/results/p1_attn_baselines_s1_kf25_gpu1_r1/summary.json",
        "jobs/results/p1_attn_baselines_s2_kf25_gpu1_r1/summary.json",
    ]
    llama25_files = [
        "jobs/results/p3_llama31_attn_s0_kf25_gpu1_r1/summary.json",
        "jobs/results/p3_llama31_attn_s1_kf25_gpu1_r1/summary.json",
        "jobs/results/p3_llama31_attn_s2_kf25_gpu1_r1/summary.json",
    ]
    llama05_files = [
        "jobs/results/p3_llama31_attn_s0_kf05_gpu1_r1/summary.json",
        "jobs/results/p3_llama31_attn_s1_kf05_gpu1_r1/summary.json",
    ]
    scenarios = [
        ("Qwen\nk=.25", qwen_files),
        ("Llama\nk=.25", llama25_files),
        ("Llama\nk=.05 s0", [llama05_files[0]]),
        ("Llama\nk=.05 s1", [llama05_files[1]]),
    ]
    arms = [
        ("Structure", "structure", COLORS["structure"], ""),
        ("SnapKV", "snapkv", COLORS["attention"], ".."),
        ("Hybrid", "hybrid", "#9A83AA", "++"),
    ]
    fig, ax = plt.subplots(figsize=(3.35, 3.25))
    x = np.arange(len(scenarios))
    width = 0.24
    for ai, (label, arm, color, hatch) in enumerate(arms):
        points = [pooled_arm(files, arm) for _, files in scenarios]
        successes = np.array([point[0] for point in points])
        ns = np.array([point[1] for point in points])
        means = successes / ns
        intervals = np.array([wilson(int(s), int(n)) for s, n in points])
        xpos = x + (ai - 1) * width
        bars = ax.bar(
            xpos,
            means,
            width,
            label=label,
            color=color,
            edgecolor=COLORS["ink"],
            linewidth=0.45,
            hatch=hatch,
        )
        ax.errorbar(
            xpos,
            means,
            yerr=np.maximum(
                0.0,
                np.vstack((means - intervals[:, 0], intervals[:, 1] - means)),
            ),
            fmt="none",
            ecolor=COLORS["ink"],
            elinewidth=0.7,
            capsize=1.5,
        )
        for bar, success, n in zip(bars, successes, ns, strict=True):
            ax.text(bar.get_x() + bar.get_width() / 2, max(0.04, bar.get_height() - 0.075), f"{success}/{n}", ha="center", va="top", fontsize=6.4, rotation=90, color="white" if bar.get_height() > 0.75 else COLORS["ink"], fontweight="bold")
    ax.set_ylim(0, 1.10)
    ax.set_ylabel("PriorityBench-A pass rate")
    ax.set_xticks(x, [label for label, _ in scenarios])
    fig.suptitle(
        "Budget and model dependence (Wilson 95% CI)",
        x=0.04,
        y=0.99,
        ha="left",
        fontsize=9.2,
        fontweight="bold",
    )
    handles, legend_labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        ncols=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        handlelength=1.2,
        columnspacing=0.8,
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", visible=False)
    fig.text(
        0.01,
        0.035,
        "SnapKV > structure on both 5% slices (40/40 vs 35/40 and 36/40).",
        fontsize=6.8,
        color=COLORS["negative"],
        fontweight="bold",
    )
    fig.text(
        0.01,
        0.008,
        "No uniform arm was run for Llama; k=.25 is an easy-task ceiling, not a universal transfer win.",
        fontsize=6.5,
        color=COLORS["muted"],
    )
    fig.tight_layout(rect=(0, 0.105, 1, 0.84))
    save_plot(fig, "budget_and_transfer")


def _reliability_keep_sweep_legacy() -> None:
    rows = [
        json.loads(line)
        for line in (ROOT / "docs/atlas_w4_structure_rows.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    manifests = ["w4_structured_paged_015", "w3_structured_paged", "w4_structured_paged_035"]
    x = np.array([0.15, 0.25, 0.35])
    methods = [
        ("Role-blind", "keep_uniform", COLORS["blind"], "o", "--"),
        ("Random", "keep_random", COLORS["random"], "s", ":"),
        ("Structure", "keep_structure", COLORS["structure"], "D", "-"),
        ("Keep all", "keep_keep_all", COLORS["full"], "^", "-."),
    ]
    lookup = {(row["manifest_id"], row["method"]): float(row["score"]) for row in rows}
    fig, ax = plt.subplots(figsize=(3.35, 2.95))
    for label, method, color, marker, linestyle in methods:
        means = np.array([lookup[(manifest, method)] for manifest in manifests])
        successes = np.array([count_from_mean(value, 14) for value in means])
        intervals = np.array([wilson(int(success), 14) for success in successes])
        ax.errorbar(
            x,
            means,
            yerr=np.maximum(
                0.0,
                np.vstack((means - intervals[:, 0], intervals[:, 1] - means)),
            ),
            label=label,
            color=color,
            marker=marker,
            linestyle=linestyle,
            markersize=4.2,
            capsize=2.0,
            elinewidth=0.7,
        )
    ax.set_xlim(0.13, 0.37)
    ax.set_ylim(-0.03, 1.05)
    ax.set_xticks(x, ["15%", "25%", "35%"])
    ax.set_xlabel("Page keep budget")
    ax.set_ylabel("PriorityBench-A pass rate")
    ax.set_title("Legacy page sweep: one fixed Qwen slice (n=14)", loc="left", pad=5)
    handles, legend_labels = ax.get_legend_handles_labels()
    ax.spines[["top", "right"]].set_visible(False)
    fig.legend(
        handles,
        legend_labels,
        ncols=2,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        columnspacing=1.0,
    )
    fig.tight_layout(rect=(0, 0.20, 1, 1))
    save_plot(fig, "reliability_keep_sweep")


def _lock240_quality_by_length_legacy() -> None:
    data = read_json("jobs/results/mg_b_lock240_quality_gpu01_r1/summary.json")
    contexts = ["8000", "16000", "32000"]
    x = np.arange(3)
    arms = [
        ("FullKV", "full", COLORS["full"], "o", "-"),
        ("Uniform INT4", "uniform", COLORS["blind"], "s", "--"),
        ("Structure INT4", "structure", COLORS["structure"], "D", ":"),
    ]
    fig, ax = plt.subplots(figsize=(3.35, 2.75))
    for label, key, color, marker, linestyle in arms:
        means = np.array([float(data["by_context"][ctx][key]["mean"]) for ctx in contexts])
        ns = np.array([int(data["by_context"][ctx][key]["n"]) for ctx in contexts])
        successes = np.array([count_from_mean(mean, int(n)) for mean, n in zip(means, ns, strict=True)])
        intervals = np.array([wilson(int(s), int(n)) for s, n in zip(successes, ns, strict=True)])
        ax.errorbar(
            x,
            means,
            yerr=np.maximum(
                0.0,
                np.vstack((means - intervals[:, 0], intervals[:, 1] - means)),
            ),
            label=label,
            color=color,
            marker=marker,
            linestyle=linestyle,
            markersize=4.2,
            capsize=2.0,
            elinewidth=0.7,
        )
    ax.set_ylim(0.48, 1.04)
    ax.set_xticks(x, ["8k\nn=83", "16k\nn=81", "32k\nn=76"])
    ax.set_ylabel("PriorityBench-A pass rate")
    ax.set_title("Locked quality, f=0.75 (Wilson 95% CI)", loc="left", pad=5)
    ax.legend(loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save_plot(fig, "lock240_quality_by_length")


def _systems_tradeoff_legacy() -> None:
    peak = read_json("jobs/results/mg_a_peak_mem_gpu5_r1/summary.json")
    latency = read_json("jobs/results/d4_latency_m3c_gpu56_r1/summary.json")
    struct_peak = peak["arms"]["mixed_structure_fi_shim"]
    memory_labels = ["modeled", "payload", "peak"]
    memory_values = [
        float(struct_peak["compression_ratio_modeled_mean"]),
        float(struct_peak["payload_ratio_measured_mean"]),
        float(peak["structure_vs_fullkv_peak_ratio"]),
    ]
    latency_labels = ["E2E 8k", "E2E 16k", "TPOT 8k", "TPOT 16k"]
    latency_values = []
    for metric in ("e2e_ttft_ms_mean", "tpot_ms_mean"):
        for ctx in ("8000", "16000"):
            full = latency["by_context"][ctx]["fullkv_sdpa"][metric]
            struct = latency["by_context"][ctx]["mixed_structure_fi_shim"][metric]
            latency_values.append(float(struct) / float(full))

    fig, axes = plt.subplots(2, 1, figsize=(3.35, 3.55), sharex=True)
    panels = [
        (axes[0], memory_labels, memory_values, [COLORS["attention"], COLORS["structure"], COLORS["negative"]], ["..", "", "//"], "Memory / bytes ratio"),
        (axes[1], latency_labels, latency_values, [COLORS["attention"], COLORS["attention"], COLORS["negative"], COLORS["negative"]], ["..", "..", "//", "//"], "Latency ratio"),
    ]
    for ax, labels, values, colors, hatches, title in panels:
        y = np.arange(len(labels))
        bars = ax.barh(y, values, color=colors, edgecolor=COLORS["ink"], linewidth=0.45)
        for bar, hatch in zip(bars, hatches, strict=True):
            bar.set_hatch(hatch)
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.axvline(1.0, color=COLORS["ink"], lw=0.8, ls="--")
        ax.set_title(title, loc="left", fontsize=8.5, pad=2)
        ax.grid(axis="y", visible=False)
        ax.spines[["top", "right"]].set_visible(False)
        for yi, value in zip(y, values, strict=True):
            ax.text(value + 0.025, yi, f"{value:.3f}×", va="center", fontsize=7.2)
    axes[1].set_xlim(0, 1.32)
    axes[1].set_xlabel("Structure-aware packed path / FullKV")
    fig.suptitle("Measured H200 systems trade-off (single request)", x=0.02, ha="left", fontsize=9.2, fontweight="bold")
    fig.text(
        0.01,
        0.012,
        "No uncertainty bars: canonical jobs were not independently repeated; ratios >1 are regressions.",
        fontsize=6.6,
        color=COLORS["muted"],
    )
    fig.subplots_adjust(left=0.27, right=0.97, top=0.88, bottom=0.16, hspace=0.60)
    save_plot(fig, "systems_tradeoff")


def lock240_quality_by_length() -> None:
    """Offset point-ranges avoid implying interpolation across context lengths."""
    data = read_json("jobs/results/mg_b_lock240_quality_gpu01_r1/summary.json")
    contexts = ["8000", "16000", "32000"]
    arms = [
        ("FullKV", "full", COLORS["full"], "o", -0.16),
        ("Uniform INT4", "uniform", COLORS["blind"], "s", 0.0),
        ("Structure INT4", "structure", COLORS["structure"], "D", 0.16),
    ]
    fig, ax = plt.subplots(figsize=(4.9, 2.35))
    for label, key, color, marker, offset in arms:
        for ci, context in enumerate(contexts):
            mean = float(data["by_context"][context][key]["mean"])
            n = int(data["by_context"][context][key]["n"])
            successes = count_from_mean(mean, n)
            lo, hi = wilson(successes, n)
            xpos = ci + offset
            ax.errorbar(
                xpos,
                mean,
                yerr=np.array([[mean - lo], [hi - mean]]),
                fmt=marker,
                color=color,
                markeredgecolor=COLORS["ink"],
                markeredgewidth=0.45,
                markersize=5.0,
                ecolor=COLORS["ink"],
                elinewidth=0.75,
                capsize=2.0,
                capthick=0.75,
                label=label if ci == 0 else None,
            )
            if context == "32000":
                yoffset = {"full": 11, "uniform": -16, "structure": 0}[key]
                ax.annotate(
                    f"{successes}/{n}",
                    (xpos, mean),
                    xytext=(0, yoffset),
                    textcoords="offset points",
                    ha="center",
                    va="bottom" if yoffset >= 0 else "top",
                    fontsize=6.8,
                    fontweight="bold",
                    color=color,
                )
    for ci, context in enumerate(contexts[:2]):
        n = int(data["by_context"][context]["full"]["n"])
        ax.text(ci, 1.055, f"all {n}/{n}", ha="center", va="bottom", fontsize=7.1, fontweight="bold")
    ax.set_xlim(-0.42, 2.42)
    ax.set_ylim(0.43, 1.10)
    ax.set_xticks(range(3), ["8k", "16k", "32k"])
    ax.set_xlabel("Prompt length")
    ax.set_ylabel("Pass rate (Wilson 95% CI)")
    ax.grid(axis="x", visible=False)
    ax.spines[["top", "right"]].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, ncols=3, loc="upper center", bbox_to_anchor=(0.52, 0.985), handletextpad=0.35, columnspacing=1.0)
    fig.subplots_adjust(left=0.115, right=0.985, top=0.79, bottom=0.22)
    save_plot(fig, "lock240_quality_by_length")


def systems_tradeoff() -> None:
    """Compact, directly labeled memory and latency ratios."""
    peak = read_json("jobs/results/mg_a_peak_mem_gpu5_r1/summary.json")
    latency = read_json("jobs/results/d4_latency_m3c_gpu56_r1/summary.json")
    struct_peak = peak["arms"]["mixed_structure_fi_shim"]
    memory = [
        ("modeled", float(struct_peak["compression_ratio_modeled_mean"]), COLORS["attention"], "D"),
        ("payload", float(struct_peak["payload_ratio_measured_mean"]), COLORS["structure"], "o"),
        ("peak", float(peak["structure_vs_fullkv_peak_ratio"]), COLORS["negative"], "s"),
    ]
    latency_points = []
    for label, metric, context in (
        ("E2E · 8k", "e2e_ttft_ms_mean", "8000"),
        ("E2E · 16k", "e2e_ttft_ms_mean", "16000"),
        ("TPOT · 8k", "tpot_ms_mean", "8000"),
        ("TPOT · 16k", "tpot_ms_mean", "16000"),
    ):
        full = float(latency["by_context"][context]["fullkv_sdpa"][metric])
        struct = float(latency["by_context"][context]["mixed_structure_fi_shim"][metric])
        latency_points.append((label, struct / full))

    fig, axes = plt.subplots(1, 2, figsize=(5.65, 2.05))
    ax = axes[0]
    y = np.arange(len(memory))[::-1]
    for yi, (label, value, color, marker) in zip(y, memory, strict=True):
        ax.plot([value, 1.0], [yi, yi], color=COLORS["grid"], lw=1.4, zorder=1)
        ax.plot(value, yi, marker=marker, color=color, markeredgecolor=COLORS["ink"], markeredgewidth=0.45, markersize=5.5, zorder=2)
        text_x = value - 0.018 if value > 0.82 else value + 0.018
        text_ha = "right" if value > 0.82 else "left"
        ax.text(text_x, yi, f"{value:.3f}×", ha=text_ha, va="center", fontsize=7.3, fontweight="bold")
    ax.axvline(1.0, color=COLORS["ink"], lw=0.7, ls="--")
    ax.set_yticks(y, [row[0] for row in memory])
    ax.set_xlim(0.36, 1.08)
    ax.set_title("(a) Memory / bytes", loc="left", fontsize=8.6, pad=4)
    ax.set_xlabel("ratio vs FullKV")
    ax.grid(axis="y", visible=False)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)

    ax = axes[1]
    y = np.arange(len(latency_points))[::-1]
    for yi, (label, value) in zip(y, latency_points, strict=True):
        color = COLORS["negative"] if label.startswith("TPOT") else COLORS["attention"]
        marker = "s" if label.startswith("TPOT") else "o"
        ax.plot([1.0, value], [yi, yi], color=COLORS["grid"], lw=1.4, zorder=1)
        ax.plot(value, yi, marker=marker, color=color, markeredgecolor=COLORS["ink"], markeredgewidth=0.45, markersize=5.5, zorder=2)
        ax.text(value + 0.006, yi, f"{value:.3f}×", ha="left", va="center", fontsize=7.3, fontweight="bold")
    ax.axvline(1.0, color=COLORS["ink"], lw=0.7, ls="--")
    ax.set_yticks(y, [row[0] for row in latency_points])
    ax.set_xlim(0.98, 1.27)
    ax.set_title("(b) Latency", loc="left", fontsize=8.6, pad=4)
    ax.set_xlabel("ratio vs FullKV")
    ax.grid(axis="y", visible=False)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.subplots_adjust(left=0.105, right=0.98, top=0.86, bottom=0.23, wspace=0.48)
    save_plot(fig, "systems_tradeoff")


def budget_and_transfer() -> None:
    """Two-panel budget/model transfer comparison with direct counts."""
    qwen_files = [
        "jobs/results/p1_attn_baselines_s0_kf25_gpu1_r3/summary.json",
        "jobs/results/p1_attn_baselines_s1_kf25_gpu1_r1/summary.json",
        "jobs/results/p1_attn_baselines_s2_kf25_gpu1_r1/summary.json",
    ]
    llama25_files = [
        "jobs/results/p3_llama31_attn_s0_kf25_gpu1_r1/summary.json",
        "jobs/results/p3_llama31_attn_s1_kf25_gpu1_r1/summary.json",
        "jobs/results/p3_llama31_attn_s2_kf25_gpu1_r1/summary.json",
    ]
    llama05_files = [
        "jobs/results/p3_llama31_attn_s0_kf05_gpu1_r1/summary.json",
        "jobs/results/p3_llama31_attn_s1_kf05_gpu1_r1/summary.json",
    ]
    arms = [
        ("Structure", "structure", COLORS["structure"], "o", -0.17),
        ("SnapKV", "snapkv", COLORS["attention"], "s", 0.0),
        ("Hybrid", "hybrid", "#80648C", "^", 0.17),
    ]
    panels = [
        ("(a) 25%: Qwen gap, Llama ceiling", [("Qwen", qwen_files), ("Llama", llama25_files)]),
        ("(b) Llama 5%: SnapKV wins both", [("slice 0", [llama05_files[0]]), ("slice 1", [llama05_files[1]])]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(5.75, 2.55), sharey=True)
    for panel_idx, (ax, (title, scenarios)) in enumerate(zip(axes, panels, strict=True)):
        for label, key, color, marker, offset in arms:
            for scenario_idx, (_, files) in enumerate(scenarios):
                successes, n = pooled_arm(files, key)
                mean = successes / n
                lo, hi = wilson(successes, n)
                xpos = scenario_idx + offset
                ax.errorbar(
                    xpos,
                    mean,
                    yerr=np.array([[mean - lo], [hi - mean]]),
                    fmt=marker,
                    color=color,
                    markeredgecolor=COLORS["ink"],
                    markeredgewidth=0.45,
                    markersize=5.0,
                    ecolor=COLORS["ink"],
                    elinewidth=0.75,
                    capsize=2.0,
                    capthick=0.75,
                    label=label if panel_idx == 0 and scenario_idx == 0 else None,
                )
                is_ceiling = panel_idx == 0 and scenario_idx == 1
                if not is_ceiling:
                    offsets = {
                        (0, 0, "structure"): 9,
                        (0, 0, "snapkv"): -13,
                        (0, 0, "hybrid"): 9,
                        (1, 0, "structure"): 8,
                        (1, 0, "snapkv"): 8,
                        (1, 0, "hybrid"): -13,
                        (1, 1, "structure"): 8,
                        (1, 1, "snapkv"): 8,
                        (1, 1, "hybrid"): -13,
                    }
                    yoffset = offsets[(panel_idx, scenario_idx, key)]
                    ax.annotate(
                        f"{successes}/{n}",
                        (xpos, mean),
                        xytext=(0, yoffset),
                        textcoords="offset points",
                        ha="center",
                        va="bottom" if yoffset >= 0 else "top",
                        fontsize=6.5,
                        fontweight="bold",
                        color=color,
                    )
        if panel_idx == 0:
            ceiling_n = pooled_arm(llama25_files, "structure")[1]
            ax.text(1.0, 1.055, f"all {ceiling_n}/{ceiling_n}", ha="center", va="bottom", fontsize=7.0, fontweight="bold")
        ax.set_title(title, loc="left", fontsize=8.4, pad=4)
        ax.set_xticks(range(len(scenarios)), [name for name, _ in scenarios])
        ax.set_xlim(-0.42, 1.42)
        ax.set_ylim(0.39, 1.10)
        ax.grid(axis="x", visible=False)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Pass rate (Wilson 95% CI)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncols=3, loc="upper center", bbox_to_anchor=(0.53, 0.985), handletextpad=0.35, columnspacing=1.0)
    fig.subplots_adjust(left=0.10, right=0.985, top=0.78, bottom=0.16, wspace=0.30)
    save_plot(fig, "budget_and_transfer")


def remove_stale_outputs() -> None:
    for name in STALE:
        for suffix in (".svg", ".pdf"):
            path = OUT / f"{name}{suffix}"
            if path.exists():
                path.unlink()


def validate_outputs() -> None:
    expected = {f"{name}.{suffix}" for name in FIGURES for suffix in ("svg", "pdf")}
    actual = {path.name for path in OUT.iterdir() if path.suffix in {".svg", ".pdf"}}
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise RuntimeError(f"figure output mismatch; missing={missing}, extra={extra}")


def render_qc_pngs(names: Iterable[str]) -> None:
    converter = shutil.which("rsvg-convert")
    if converter is None:
        raise RuntimeError("rsvg-convert is required for PNG quality-control renders")
    if QC_OUT.exists():
        shutil.rmtree(QC_OUT)
    QC_OUT.mkdir(parents=True)
    for name in names:
        source = OUT / f"{name}.svg"
        for label, width in (("full", 1800), ("column", 1005)):
            subprocess.run(
                [
                    converter,
                    "--format=png",
                    "--keep-aspect-ratio",
                    "--width",
                    str(width),
                    "--output",
                    str(QC_OUT / f"{name}_{label}.png"),
                    str(source),
                ],
                check=True,
            )


def main() -> None:
    configure_matplotlib()
    OUT.mkdir(parents=True, exist_ok=True)
    remove_stale_outputs()
    agent_trace_failure_mode()
    page_allocation_architecture()
    decode_memory_lifetime()
    hypothesis_split()
    eviction_and_baselines()
    budget_and_transfer()
    lock240_quality_by_length()
    systems_tradeoff()
    validate_outputs()
    render_qc_pngs(FIGURES)
    print(f"wrote {len(FIGURES)} SVG/PDF figures to {OUT}")
    print(f"wrote full-size and column-width PNG review renders to {QC_OUT}")


if __name__ == "__main__":
    main()
