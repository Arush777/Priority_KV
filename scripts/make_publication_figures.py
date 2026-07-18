#!/usr/bin/env python3
"""Generate publication SVG/PDF figures from frozen PriorityKV artifacts.

The script intentionally uses only the Python standard library. If `rsvg-convert`
is available, matching PDF files are emitted for LaTeX/arXiv packaging.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures"

COLORS = {
    "full": "#2F6B4F",
    "uniform": "#737B86",
    "random": "#D18B2C",
    "structure": "#2673B8",
    "payload": "#2673B8",
    "modeled": "#55A868",
    "peak": "#C44E52",
    "e2e": "#8172B2",
    "tpot": "#CC6B35",
}


def esc(text: object) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def text(x: float, y: float, value: object, *, size: int = 22, anchor: str = "middle",
         weight: int = 400, fill: str = "#20252B") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'font-family="DejaVu Sans,Arial,sans-serif" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}">{esc(value)}</text>'
    )


def line(x1: float, y1: float, x2: float, y2: float, *, color: str = "#D7DCE1",
         width: float = 1.5, dash: str | None = None) -> str:
    dashed = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{color}" stroke-width="{width}"{dashed}/>'
    )


def arrow(x1: float, y1: float, x2: float, y2: float, *, color: str = "#4E5964",
          width: float = 2.5) -> str:
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{color}" stroke-width="{width}" marker-end="url(#arrowhead)"/>'
    )


def rect(x: float, y: float, width: float, height: float, color: str,
         *, stroke: str = "none") -> str:
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(0, width):.1f}" '
        f'height="{max(0, height):.1f}" fill="{color}" stroke="{stroke}"/>'
    )


def svg_document(width: int, height: int, body: Sequence[str], title_value: str) -> str:
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title">',
            f'<title id="title">{esc(title_value)}</title>',
            '<defs><marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" '
            'refY="3.5" orient="auto"><polygon points="0 0, 10 3.5, 0 7" '
            'fill="#4E5964"/></marker></defs>',
            rect(0, 0, width, height, "#FFFFFF"),
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
    title_value: str,
    subtitle: str,
    categories: Sequence[str],
    series: Sequence[tuple[str, Sequence[float], str]],
    y_max: float,
    y_label: str,
    note: str,
) -> str:
    width, height = 1200, 700
    left, right, top, bottom = 115, 40, 120, 125
    plot_w, plot_h = width - left - right, height - top - bottom
    body = [text(left, 48, title_value, size=30, anchor="start", weight=700)]
    body.append(text(left, 82, subtitle, size=18, anchor="start", fill="#56606B"))

    for i in range(6):
        value = y_max * i / 5
        y = top + plot_h - plot_h * value / y_max
        body.append(line(left, y, left + plot_w, y))
        body.append(text(left - 16, y + 7, f"{value:.1f}", size=17, anchor="end", fill="#56606B"))
    body.append(line(left, top, left, top + plot_h, color="#6D747C", width=2))
    body.append(line(left, top + plot_h, left + plot_w, top + plot_h, color="#6D747C", width=2))
    body.append(
        f'<text x="30" y="{top + plot_h / 2:.1f}" transform="rotate(-90 30 '
        f'{top + plot_h / 2:.1f})" text-anchor="middle" font-family="DejaVu Sans,Arial,sans-serif" '
        f'font-size="19" fill="#343B43">{esc(y_label)}</text>'
    )

    group_w = plot_w / len(categories)
    usable = group_w * 0.76
    bar_w = usable / len(series)
    for ci, category in enumerate(categories):
        group_x = left + ci * group_w + (group_w - usable) / 2
        for si, (name, values, color) in enumerate(series):
            value = float(values[ci])
            bar_h = plot_h * value / y_max
            x = group_x + si * bar_w + 2
            y = top + plot_h - bar_h
            body.append(rect(x, y, bar_w - 5, bar_h, color))
            body.append(text(x + (bar_w - 5) / 2, y - 9, f"{value:.3f}", size=14))
        body.append(text(left + (ci + 0.5) * group_w, top + plot_h + 32, category, size=19))

    legend_y = height - 65
    total_legend_w = len(series) * 205
    legend_x = left + (plot_w - total_legend_w) / 2
    for si, (name, _, color) in enumerate(series):
        x = legend_x + si * 205
        body.append(rect(x, legend_y - 17, 24, 17, color))
        body.append(text(x + 34, legend_y - 2, name, size=17, anchor="start"))
    body.append(text(left, height - 18, note, size=15, anchor="start", fill="#606A75"))
    return svg_document(width, height, body, title_value)


def reliability_figure() -> None:
    rows = [json.loads(line) for line in (ROOT / "docs/atlas_w4_structure_rows.jsonl").read_text().splitlines()]
    manifests = ["w4_structured_paged_015", "w3_structured_paged", "w4_structured_paged_035"]
    methods = ["keep_uniform", "keep_random", "keep_structure", "keep_keep_all"]
    labels = {"keep_uniform": "Role-blind", "keep_random": "Random", "keep_structure": "Structure", "keep_keep_all": "Keep all"}
    color_keys = {"keep_uniform": "uniform", "keep_random": "random", "keep_structure": "structure", "keep_keep_all": "full"}
    lookup = {(row["manifest_id"], row["method"]): row["score"] for row in rows}
    series = [
        (labels[method], [lookup[(manifest, method)] for manifest in manifests], COLORS[color_keys[method]])
        for method in methods
    ]
    write_figure(
        "reliability_keep_sweep",
        grouped_bars(
            title_value="Structure-aware retention preserves agent-critical state",
            subtitle="Matched 16-token page budgets on the same 14-example Qwen3-8B stress slice",
            categories=["15% keep", "25% keep", "35% keep"],
            series=series,
            y_max=1.0,
            y_label="PriorityBench score",
            note="The sweep is an ablation over one fixed slice, not three independent replications.",
        ),
    )


def quality_figure() -> None:
    data = json.loads((ROOT / "jobs/results/mg_b_lock240_quality_gpu01_r1/summary.json").read_text())
    contexts = ["8000", "16000", "32000"]
    names = [("FullKV", "full", "full"), ("Role-blind INT4", "uniform", "uniform"), ("Structure INT4", "structure", "structure")]
    series = [
        (label, [data["by_context"][ctx][key]["mean"] for ctx in contexts], COLORS[color])
        for label, key, color in names
    ]
    write_figure(
        "lock240_quality_by_length",
        grouped_bars(
            title_value="Soft INT4 remains close to FullKV on the locked benchmark",
            subtitle="Qwen3-8B, int4_frac=0.75; n=83/81/76 for 8k/16k/32k",
            categories=["8k", "16k", "32k"],
            series=series,
            y_max=1.0,
            y_label="PriorityBench score",
            note="Overall means: FullKV 0.8875, role-blind 0.8792, structure-aware 0.8833 (n=240).",
        ),
    )


def systems_figure() -> None:
    peak = json.loads((ROOT / "jobs/results/mg_a_peak_mem_gpu5_r1/summary.json").read_text())
    latency = json.loads((ROOT / "jobs/results/d4_latency_m3c_gpu56_r1/summary.json").read_text())
    memory_values = [
        peak["arms"]["mixed_structure_fi_shim"]["compression_ratio_modeled_mean"],
        peak["arms"]["mixed_structure_fi_shim"]["payload_ratio_measured_mean"],
        peak["structure_vs_fullkv_peak_ratio"],
    ]
    ctx = latency["m3"]["ctx_gates"]
    latency_values = [
        ctx["8000"]["e2e_ratio"],
        ctx["16000"]["e2e_ratio"],
        ctx["8000"]["tpot_ratio"],
        ctx["16000"]["tpot_ratio"],
    ]

    width, height = 1200, 690
    body = [text(70, 48, "Packed bytes do not translate directly to peak or latency", size=30, anchor="start", weight=700)]
    body.append(text(70, 82, "Structure-aware mixed cache relative to FullKV on H200", size=18, anchor="start", fill="#56606B"))

    panels = [
        (70, 130, 500, 430, "Memory ratios (lower is better)", ["Modeled", "Payload", "Peak"], memory_values,
         [COLORS["modeled"], COLORS["payload"], COLORS["peak"]], 1.30),
        (640, 130, 500, 430, "Latency ratios (lower is better)", ["E2E 8k", "E2E 16k", "TPOT 8k", "TPOT 16k"], latency_values,
         [COLORS["e2e"], COLORS["e2e"], COLORS["tpot"], COLORS["tpot"]], 1.30),
    ]
    for x0, y0, panel_w, panel_h, panel_title, labels, values, colors, y_max in panels:
        body.append(text(x0, y0 - 20, panel_title, size=21, anchor="start", weight=700))
        for i in range(6):
            value = y_max * i / 5
            y = y0 + panel_h - panel_h * value / y_max
            body.append(line(x0, y, x0 + panel_w, y))
            body.append(text(x0 - 12, y + 6, f"{value:.2f}", size=14, anchor="end", fill="#56606B"))
        parity_y = y0 + panel_h - panel_h / y_max
        body.append(line(x0, parity_y, x0 + panel_w, parity_y, color="#303840", width=2, dash="8 6"))
        body.append(line(x0, y0, x0, y0 + panel_h, color="#6D747C", width=2))
        body.append(line(x0, y0 + panel_h, x0 + panel_w, y0 + panel_h, color="#6D747C", width=2))
        slot = panel_w / len(values)
        for i, (label, value, color) in enumerate(zip(labels, values, colors, strict=True)):
            bar_w = slot * 0.58
            x = x0 + i * slot + (slot - bar_w) / 2
            bar_h = panel_h * value / y_max
            y = y0 + panel_h - bar_h
            body.append(rect(x, y, bar_w, bar_h, color))
            body.append(text(x + bar_w / 2, y - 9, f"{value:.3f}x", size=15))
            body.append(text(x + bar_w / 2, y0 + panel_h + 28, label, size=15))
        body.append(text(x0 + panel_w - 4, parity_y - 9, "FullKV parity", size=14, anchor="end", fill="#303840"))
    body.append(text(70, 625, "Measured payload includes packed values and scale metadata; modeled uses ideal 4-bit values.", size=16, anchor="start", fill="#606A75"))
    body.append(text(70, 652, "Cold INT4 pages expand to BF16 scratch before attention, limiting peak savings and increasing TPOT.", size=16, anchor="start", fill="#606A75"))
    write_figure("systems_tradeoff", svg_document(width, height, body, "PriorityKV systems tradeoff"))


def labeled_box(body: list[str], x: float, y: float, width: float, height: float,
                title_value: str, lines: Sequence[str], *, fill: str, stroke: str) -> None:
    body.append(
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" '
        f'rx="6" fill="{fill}" stroke="{stroke}" stroke-width="2"/>'
    )
    body.append(text(x + width / 2, y + 34, title_value, size=20, weight=700))
    for i, value in enumerate(lines):
        body.append(text(x + width / 2, y + 66 + i * 25, value, size=15, fill="#3D4650"))


def token_strip(body: list[str], x: float, y: float, width: float, height: float,
                segments: Sequence[tuple[str, float, str]], *, labels: bool = True) -> None:
    cursor = x
    for label, fraction, color in segments:
        segment_w = width * fraction
        body.append(rect(cursor, y, segment_w, height, color, stroke="#FFFFFF"))
        if labels and segment_w >= 58:
            body.append(text(cursor + segment_w / 2, y + height / 2 + 6, label, size=13, fill="#FFFFFF", weight=700))
        cursor += segment_w
    body.append(
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" '
        f'fill="none" stroke="#3F4852" stroke-width="1.5"/>'
    )


def overview_diagram() -> None:
    width, height = 1400, 830
    body = [text(60, 50, "PriorityKV: application structure becomes a cache-allocation prior", size=30, anchor="start", weight=700)]
    body.append(text(60, 84, "Original diagram; visual conventions follow token-strip workflows used by H2O, SnapKV, and KIVI", size=16, anchor="start", fill="#5E6873"))

    body.append(text(70, 145, "Long agent trace", size=20, anchor="start", weight=700))
    trace = [
        ("System", 0.10, "#2F6B4F"),
        ("Tool schema", 0.14, "#2673B8"),
        ("Filler", 0.24, "#B7BDC5"),
        ("Constraint", 0.13, "#C44E52"),
        ("Filler", 0.20, "#B7BDC5"),
        ("State", 0.09, "#8172B2"),
        ("Recent", 0.10, "#D18B2C"),
    ]
    token_strip(body, 70, 170, 1260, 58, trace)
    body.append(text(70, 252, "Known message roles", size=15, anchor="start", fill="#4E5964"))

    labeled_box(body, 80, 320, 300, 155, "Structural tagger",
                ["role + schema markers", "sink = 16 tokens", "recent = 128 tokens"],
                fill="#EFF6F2", stroke=COLORS["full"])
    labeled_box(body, 550, 320, 300, 155, "Matched allocator",
                ["same keep / INT4 budget", "protect protocol state", "demote filler first"],
                fill="#EEF5FB", stroke=COLORS["structure"])
    labeled_box(body, 1020, 320, 300, 155, "Paged KV cache",
                ["hot pages: BF16", "cold pages: packed INT4", "16-token pages"],
                fill="#F7F2FA", stroke=COLORS["e2e"])
    body.append(arrow(380, 397, 550, 397))
    body.append(arrow(850, 397, 1020, 397))
    body.append(arrow(230, 228, 230, 320))

    body.append(text(70, 555, "Role-blind eviction", size=18, anchor="start", weight=700))
    role_blind = [
        ("Sink", 0.10, "#737B86"),
        ("Dropped", 0.70, "#E5E7EA"),
        ("Recent", 0.20, "#737B86"),
    ]
    token_strip(body, 70, 575, 540, 55, role_blind)
    body.append(text(340, 658, "Agent-critical middle state can disappear", size=15, fill="#A33D42", weight=700))

    body.append(text(790, 555, "Structure-aware mixed cache", size=18, anchor="start", weight=700))
    mixed = [
        ("BF16", 0.10, "#2F6B4F"),
        ("BF16", 0.14, "#2673B8"),
        ("INT4", 0.24, "#9DC3DF"),
        ("BF16", 0.13, "#C44E52"),
        ("INT4", 0.20, "#9DC3DF"),
        ("BF16", 0.09, "#8172B2"),
        ("BF16", 0.10, "#D18B2C"),
    ]
    token_strip(body, 790, 575, 540, 55, mixed)
    body.append(text(1060, 658, "All positions retained; precision follows role", size=15, fill="#245F91", weight=700))

    body.append(line(60, 715, 1340, 715, color="#CED3D8"))
    body.append(text(70, 755, "Eviction result", size=17, anchor="start", weight=700))
    body.append(text(230, 755, "structure >> role-blind at matched keep", size=17, anchor="start", fill=COLORS["structure"]))
    body.append(text(750, 755, "INT4 result", size=17, anchor="start", weight=700))
    body.append(text(875, 755, "no quality gap at 75%; evaluate bytes + latency", size=17, anchor="start", fill=COLORS["full"]))
    body.append(text(70, 800, "The tagger uses prompt metadata, not future attention or benchmark answers.", size=15, anchor="start", fill="#5E6873"))
    write_figure("prioritykv_overview", svg_document(width, height, body, "PriorityKV overview"))


def decode_diagram() -> None:
    width, height = 1400, 760
    body = [text(60, 50, "FlashInfer decode over heterogeneous KV pages", size=30, anchor="start", weight=700)]
    body.append(text(60, 84, "Original dataflow diagram following the modular system convention used by FlashInfer and KIVI", size=16, anchor="start", fill="#5E6873"))

    labeled_box(body, 65, 165, 250, 145, "Hot cache",
                ["BF16 protected pages", "+ BF16 decode tail", "GPU resident"],
                fill="#EFF6F2", stroke=COLORS["full"])
    labeled_box(body, 65, 430, 250, 145, "Cold cache",
                ["packed INT4 pages", "+ group scales", "smaller payload"],
                fill="#EEF5FB", stroke=COLORS["structure"])

    labeled_box(body, 470, 430, 250, 145, "Cold scratch",
                ["dequantize INT4", "to BF16 on GPU", "accepted limitation"],
                fill="#FFF3EE", stroke=COLORS["tpot"])
    body.append(arrow(315, 503, 470, 503))

    labeled_box(body, 470, 165, 250, 145, "FI attention A",
                ["query x hot K/V", "state O_hot", "LSE_hot"],
                fill="#F2F4F6", stroke="#66717C")
    labeled_box(body, 865, 430, 250, 145, "FI attention B",
                ["query x cold K/V", "state O_cold", "LSE_cold"],
                fill="#F2F4F6", stroke="#66717C")
    body.append(arrow(315, 237, 470, 237))
    body.append(arrow(720, 503, 865, 503))

    labeled_box(body, 865, 165, 250, 145, "LSE merge",
                ["flashinfer.merge_state", "combine two chunks", "max abs ~4.88e-4"],
                fill="#F7F2FA", stroke=COLORS["e2e"])
    body.append(arrow(720, 237, 865, 237))
    body.append(arrow(990, 430, 990, 310))

    labeled_box(body, 1205, 165, 150, 145, "Output",
                ["attention", "state", "decode"],
                fill="#F6F7F8", stroke="#4E5964")
    body.append(arrow(1115, 237, 1205, 237))

    body.append(line(60, 640, 1340, 640, color="#CED3D8"))
    body.append(text(70, 683, "Payload win", size=17, anchor="start", weight=700))
    body.append(text(205, 683, "0.72x measured", size=17, anchor="start", fill=COLORS["structure"]))
    body.append(text(520, 683, "Peak limitation", size=17, anchor="start", weight=700))
    body.append(text(680, 683, "BF16 scratch -> 0.87x peak", size=17, anchor="start", fill=COLORS["peak"]))
    body.append(text(1030, 683, "TPOT cost", size=17, anchor="start", weight=700))
    body.append(text(1145, 683, "~1.20x", size=17, anchor="start", fill=COLORS["tpot"]))
    body.append(text(70, 728, "At most two homogeneous FlashInfer calls per layer; no full Hugging Face cache materialization.", size=15, anchor="start", fill="#5E6873"))
    write_figure("flashinfer_decode_dataflow", svg_document(width, height, body, "PriorityKV FlashInfer decode dataflow"))


def main() -> None:
    overview_diagram()
    decode_diagram()
    reliability_figure()
    quality_figure()
    systems_figure()
    print(f"wrote publication figures to {OUT}")


if __name__ == "__main__":
    main()
