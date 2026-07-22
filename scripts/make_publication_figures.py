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
import os
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
SVG_FONT_SCALE = 1.35

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

EXTERNAL_FIGURES = (
    "protected_fraction_boundary",
    "external_bfcl_arms",
)

EXTERNAL_FIGURES = (
    "protected_fraction_boundary",
    "external_bfcl_arms",
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
            "svg.hashsalt": "prioritykv-publication-figures",
            "pdf.fonttype": 42,
            "lines.linewidth": 1.2,
        }
    )


def save_plot(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    svg_path = OUT / f"{name}.svg"
    fig.savefig(svg_path, metadata={"Date": None})
    fig.savefig(
        OUT / f"{name}.pdf",
        metadata={"CreationDate": None, "ModDate": None},
    )
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
    size *= SVG_FONT_SCALE
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
    env = dict(os.environ)
    env.setdefault("SOURCE_DATE_EPOCH", "0")
    subprocess.run(
        [converter, "--format=pdf", "--output", str(OUT / f"{name}.pdf"), str(svg_path)],
        check=True,
        env=env,
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
    roles = list("SSTTTTFFFFDDFFCCFFFFFFRRRR")
    blind = {0, 1, 6, 7, 8, 9, 12, 13, 16, 17, 22, 23, 24, 25}
    structure = {0, 1, 2, 3, 4, 5, 10, 11, 14, 15, 22, 23, 24, 25}
    assert len(roles) == 26
    assert len(blind) == len(structure) == 14
    assert max(blind | structure) < len(roles)
    body = [
        stext(42, 39, "Illustrative agent trace", size=15, anchor="start", weight=700),
        stext(42, 70, "Matched 14/26-token budgets; hatched cells are evicted.", size=12.5, anchor="start", fill=COLORS["muted"]),
    ]
    labels = [
        (0, 2, "system"),
        (2, 6, "tool schema"),
        (6, 10, "filler"),
        (10, 12, "ORDER_ID"),
        (12, 14, "filler"),
        (14, 16, "constraint"),
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
    body.append(stext(68, 150, "INPUT", size=11.5, anchor="end", weight=700, fill=COLORS["muted"]))

    body.append(stext(42, 231, "A", size=16, anchor="start", weight=700, fill=COLORS["blind"]))
    body.append(stext(76, 200, "position-blind selection", size=14, anchor="start", weight=700, fill=COLORS["blind"]))
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
        ("p4", "[64,80)", "FILLER", "evict or demote", "INT4", "uint8 codes + per-group metadata", COLORS["int4"]),
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
    body.append(stext(1128, 704, "Mixed path: all positions remain; f=0.75 use uint8-backed INT4 codes.", size=12.5, anchor="end", fill=COLORS["muted"]))
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
        ("implemented payload", 260, 1090, 425, COLORS["structure"], "0.719× vs FullKV"),
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
        (726, "H3 · mixed-code systems path", "BYTES WIN · LATENCY COST", COLORS["attention"], ["H200, single request", "payload 0.719×", "peak 0.868×", "TPOT 1.200–1.215×"]),
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
    """Keep retention and mixed-precision controllers visually separate."""
    width, height = 1200, 545
    body = [
        stext(42, 34, "Two uses of the same structural tags", size=19, anchor="start", weight=700),
        stext(42, 61, "Eviction removes positions; mixed precision retains positions and changes storage.", size=15, anchor="start", fill=COLORS["muted"]),
        sline(600, 84, 600, 508, stroke=COLORS["grid"], width=1.2),
    ]

    panels = [
        (
            42,
            "(a) PriorityBench-A eviction path",
            COLORS["structure"],
            [
                ("Tag tokens", "sink · recent · system · tool · constraint · other · filler"),
                ("Set token budget", "B = max(round(kn), |M|);  M = sink + recent"),
                ("Rank and select B", "M → structural roles / other → recent-edge filler"),
                ("Gather selected positions", "Re-prefill the shortened trace; omitted KV is absent"),
            ],
        ),
        (
            632,
            "(b) Mixed-precision page path",
            COLORS["attention"],
            [
                ("Tag and group", "≤16-token pages; protected role wins ties"),
                ("Assign representation", "Prefer structural roles in BF16; low-risk → codes"),
                ("Enforce byte budget", "Demote by risk; sink + recent remain BF16"),
                ("Decode from dtype runs", "FP32 scale / zero point; all positions remain"),
            ],
        ),
    ]
    for x, title, color, rows in panels:
        body.append(stext(x, 105, title, size=18, anchor="start", weight=700, fill=color))
        for i, (head, detail) in enumerate(rows):
            y = 132 + i * 92
            body.append(srect(x, y, 526, 72, fill="white", stroke=color, stroke_width=1.25))
            body.append(srect(x, y, 12, 72, fill=color, stroke=color))
            body.append(stext(x + 31, y + 28, f"{i + 1}. {head}", size=16.2, anchor="start", weight=700))
            body.append(stext(x + 31, y + 55, detail, size=14.8, anchor="start", fill=COLORS["muted"]))
            if i < len(rows) - 1:
                body.append(sline(x + 263, y + 73, x + 263, y + 88, arrow=True))
    body.append(stext(600, 530, "evaluated independently", size=14.8, weight=700, fill=COLORS["muted"]))
    write_diagram("page_allocation_architecture", width, height, "PriorityKV policy and physical page architecture", body)


def decode_memory_lifetime() -> None:
    """Decode ordering and allocation lifetimes, matching the FI shim."""
    width, height = 1200, 590
    body = [
        stext(42, 35, "Mixed decode: append, attend twice, then merge", size=19, anchor="start", weight=700),
        stext(42, 63, "The new token joins the BF16 tail before either attention call.", size=15, anchor="start", fill=COLORS["muted"]),
    ]

    # Implemented per-layer order.
    steps = [
        (42, 94, 190, "1  project q, k, v", "Qwen3 norms + RoPE", COLORS["ink"]),
        (270, 94, 210, "2  append k / v", "BF16 decode tail", COLORS["recent"]),
        (836, 133, 172, "4  LSE merge", "partial states", COLORS["attention"]),
        (1046, 133, 112, "5  output", "next layer", COLORS["ink"]),
    ]
    for x, y, w, head, detail, color in steps:
        body.append(srect(x, y, w, 76, fill="white", stroke=color, stroke_width=1.35))
        body.append(stext(x + w / 2, y + 31, head, size=16.2, weight=700, fill=color))
        body.append(stext(x + w / 2, y + 58, detail, size=14.8, fill=COLORS["muted"]))
    body.append(sline(232, 132, 264, 132, arrow=True))

    body.append(srect(524, 84, 250, 76, fill="white", stroke=COLORS["bf16"], stroke_width=1.35))
    body.append(stext(649, 115, "3a  FlashInfer hot", size=16.2, weight=700, fill=COLORS["bf16"]))
    body.append(stext(649, 142, "q × (hot pages + tail)", size=14.8, fill=COLORS["muted"]))
    body.append(srect(524, 190, 250, 76, fill="white", stroke=COLORS["structure"], stroke_width=1.35))
    body.append(stext(649, 221, "3b  FlashInfer cold", size=16.2, weight=700, fill=COLORS["structure"]))
    body.append(stext(649, 248, "q × BF16 cold scratch", size=14.8, fill=COLORS["muted"]))
    body.append(spolyline([(480, 132), (501, 132), (501, 122), (518, 122)], arrow=True))
    body.append(spolyline([(480, 132), (501, 132), (501, 228), (518, 228)], arrow=True))
    body.append(spolyline([(774, 122), (804, 122), (804, 157), (830, 157)], arrow=True))
    body.append(spolyline([(774, 228), (804, 228), (804, 185), (830, 185)], arrow=True))
    body.append(sline(1008, 171, 1040, 171, arrow=True))
    body.append(stext(42, 226, "Cold cache", size=15.5, anchor="start", weight=700, fill=COLORS["structure"]))
    body.append(stext(42, 251, "uint8 codes + FP32 scale / zero point", size=14.8, anchor="start", fill=COLORS["muted"]))
    body.append(sline(332, 238, 518, 238, arrow=True))
    body.append(stext(42, 286, "The sequence length is committed after all layers; the loop then advances one token.", size=14.8, anchor="start", fill=COLORS["muted"]))

    body.extend([
        sline(38, 315, 1162, 315, stroke=COLORS["grid"]),
        stext(42, 345, "Request-lifetime allocations", size=18, anchor="start", weight=700),
    ])
    x0, x1 = 270, 1140
    ticks = [(270, "prefill"), (450, "pack"), (630, "prepare"), (810, "decode"), (1140, "free")]
    body.append(sline(x0, 376, x1, 376, width=1.0, arrow=True))
    for x, label in ticks:
        body.append(sline(x, 369, x, 384, width=0.9))
        body.append(stext(x, 403, label, size=14.8, fill=COLORS["muted"]))
    lifetimes = [
        ("implemented code + metadata", 450, 1140, 438, COLORS["structure"], "persistent · payload ratio 0.719×"),
        ("BF16 cold scratch", 630, 1140, 490, COLORS["int4"], "transient · allocated peak ratio 0.868×"),
        ("BF16 decode tail", 810, 1140, 542, COLORS["recent"], "grows one token per step"),
    ]
    for label, start, end, y, color, value in lifetimes:
        body.append(stext(42, y + 6, label, size=15, anchor="start", weight=700))
        body.append(srect(start, y - 14, end - start, 27, fill=color, stroke=color))
        body.append(stext((start + end) / 2, y + 6, value, size=14.5, weight=700, fill="white" if color != COLORS["int4"] else COLORS["ink"]))
    write_diagram("decode_memory_lifetime", width, height, "PriorityKV mixed decode and memory lifetime", body)


def hypothesis_split() -> None:
    """Compact claim matrix for the three independently evaluated studies."""
    width, height = 1200, 385
    body = [
        stext(42, 35, "Three experiment families answer three different questions", size=19, anchor="start", weight=700),
        stext(42, 67, "Study", size=15, anchor="start", weight=700, fill=COLORS["muted"]),
        stext(190, 67, "Question and comparison", size=15, anchor="start", weight=700, fill=COLORS["muted"]),
        stext(636, 67, "Measured endpoint", size=15, anchor="start", weight=700, fill=COLORS["muted"]),
        stext(922, 67, "Interpretation", size=15, anchor="start", weight=700, fill=COLORS["muted"]),
        sline(42, 80, 1158, 80, stroke=COLORS["ink"], width=1.1),
    ]
    rows = [
        (91, "H1", "Retention · W5, n=120", "Which positions survive at k=0.25?", "112/120 vs 1/120", "In-regime structural win", COLORS["structure"]),
        (183, "H2", "INT4 quality · W3, n=240", "Does role-aware placement beat uniform?", "0.8833 vs 0.8792", "No quality separation", COLORS["negative"]),
        (275, "H3", "Systems · H200, n=18", "Mixed FlashInfer vs FullKV/SDPA", "0.719× / 1.200–1.215×", "Payload–latency trade-off", COLORS["attention"]),
    ]
    for y, tag, study, question, endpoint, verdict, color in rows:
        body.append(srect(42, y, 120, 76, fill=color, stroke=color))
        body.append(stext(102, y + 47, tag, size=20, weight=700, fill="white"))
        body.append(stext(190, y + 28, study, size=16, anchor="start", weight=700))
        body.append(stext(190, y + 57, question, size=15, anchor="start", fill=COLORS["muted"]))
        body.append(stext(636, y + 28, endpoint, size=15.5, anchor="start", weight=700, fill=color))
        body.append(stext(636, y + 57, "fixed manifests and seeds", size=15, anchor="start", fill=COLORS["muted"]))
        body.append(stext(922, y + 42, verdict, size=15.2, anchor="start", weight=700))
        if y != rows[-1][0]:
            body.append(sline(42, y + 84, 1158, y + 84, stroke=COLORS["grid"], width=0.9))
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
    uniform = pooled_arm(p0_files, "uniform")
    random = pooled_arm(p0_files, "random")
    if uniform != random:
        raise AssertionError("released position-blind controls must have identical counts")
    values = [
        ("FullKV", *pooled_full(p0_files), COLORS["full"], "o"),
        ("Position-blind†", *uniform, COLORS["blind"], "s"),
        ("Structure", *pooled_arm(p0_files, "structure"), COLORS["structure"], "o"),
        ("SnapKV", *pooled_arm(p1_files, "snapkv"), COLORS["attention"], "s"),
        ("PyramidKV", *pooled_arm(p1_files, "pyramid"), COLORS["attention_alt"], "D"),
        ("Hybrid", *pooled_arm(p1_files, "hybrid"), "#80648C", "^"),
        ("H2O*", *pooled_arm(h2o_files, "h2o"), COLORS["h2o"], "v"),
    ]
    ypos = np.array([7.6, 6.25, 4.85, 3.85, 2.85, 1.85, 0.85])
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
        ax.text(1.04, y, f"{successes}/{n}", transform=ax.get_yaxis_transform(), clip_on=False, va="center", ha="left", fontsize=7.5, fontweight="bold")
    ax.set_yticks(ypos, [row[0] for row in values])
    ax.set_xlim(-0.02, 1.0)
    ax.set_ylim(0.25, 8.18)
    ax.set_xlabel("PriorityBench-A pass rate (Wilson 95% CI)")
    ax.grid(axis="y", visible=False)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.axhline(6.88, color=COLORS["grid"], linewidth=0.6)
    ax.axhline(5.55, color=COLORS["grid"], linewidth=0.6)
    ax.text(-0.018, 7.98, "reference", fontsize=6.7, color=COLORS["muted"], va="bottom")
    ax.text(-0.018, 6.76, "position blind", fontsize=6.7, color=COLORS["muted"], va="bottom")
    ax.text(-0.018, 5.43, "selectors", fontsize=6.7, color=COLORS["muted"], va="bottom")
    ax.text(1.04, 7.98, "passes", transform=ax.get_yaxis_transform(), clip_on=False, fontsize=6.7, color=COLORS["muted"], va="bottom", ha="left")
    struct_y, snap_y = ypos[2], ypos[3]
    trans = ax.get_yaxis_transform()
    ax.plot([1.22, 1.24, 1.24, 1.22], [struct_y, struct_y, snap_y, snap_y], transform=trans, clip_on=False, color=COLORS["ink"], lw=0.75)
    ax.text(1.255, (struct_y + snap_y) / 2, "paired\np=.125", transform=trans, clip_on=False, va="center", ha="left", fontsize=6.8)
    fig.subplots_adjust(left=0.25, right=0.74, top=0.97, bottom=0.19)
    fig.text(0.01, 0.012, "† Released Uniform and Random select byte-identical indices.", fontsize=6.6, color=COLORS["muted"])
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
    axes[1].set_xlabel("Structure-aware mixed-code path / FullKV")
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
    ax.set_xlabel("Nominal context stratum")
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
        ("Idealized nibble KV", float(struct_peak["compression_ratio_modeled_mean"]), COLORS["attention"], "D"),
        ("Implemented payload", float(struct_peak["payload_ratio_measured_mean"]), COLORS["structure"], "o"),
        ("CUDA allocated peak", float(peak["structure_vs_fullkv_peak_ratio"]), COLORS["negative"], "s"),
    ]
    latency_points = []
    for label, metric, context in (
        ("E2E · nominal 8k", "e2e_ttft_ms_mean", "8000"),
        ("E2E · nominal 16k", "e2e_ttft_ms_mean", "16000"),
        ("TPOT · nominal 8k", "tpot_ms_mean", "8000"),
        ("TPOT · nominal 16k", "tpot_ms_mean", "16000"),
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
    ax.set_title("(a) Storage and memory", loc="left", fontsize=8.6, pad=4)
    ax.set_xlabel("ratio vs BF16 KV / FullKV peak")
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
    ax.set_xlabel("mixed FlashInfer / FullKV SDPA")
    ax.grid(axis="y", visible=False)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.subplots_adjust(left=0.205, right=0.98, top=0.86, bottom=0.25, wspace=0.58)
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
        ("(a) Qwen and Llama, k=0.25", [("Qwen", qwen_files), ("Llama", llama25_files)]),
        ("(b) Llama, k=0.05", [("slice 0", [llama05_files[0]]), ("slice 1", [llama05_files[1]])]),
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
    expected |= {f"{name}.{suffix}" for name in EXTERNAL_FIGURES for suffix in ("png", "pdf")}
    actual = {path.name for path in OUT.iterdir() if path.suffix in {".svg", ".png", ".pdf"}}
    missing = sorted(expected - actual)
    if missing:
        raise RuntimeError(f"figure output mismatch; missing={missing}")


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
    # The external cluster filesystem is not part of the submission checkout.
    # Render the exact tracked submission snapshot in this complete build.
    from make_external_figures import fig_arms, fig_boundary, paper_snapshot

    protected, qwen, llama = paper_snapshot()
    fig_boundary(protected, OUT / "protected_fraction_boundary.png")
    fig_arms([("Qwen3-8B", qwen), ("Llama-3.1-8B", llama)], OUT / "external_bfcl_arms.png")
    validate_outputs()
    render_qc_pngs(FIGURES)
    print(f"wrote all {len(FIGURES) + len(EXTERNAL_FIGURES)} publication figures to {OUT}")
    print(f"wrote full-size and column-width PNG review renders to {QC_OUT}")


if __name__ == "__main__":
    main()
