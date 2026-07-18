#!/usr/bin/env python3
"""Fold W4 structure denser-sweep JSONs into one atlas JSONL (CPU)."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DEFAULTS = [
    "stress_structured/w3_structured_paged_r1.json",
    "stress_structured/w4_structured_paged_015_r1.json",
    "stress_structured/w4_structured_paged_035_r1.json",
]


def _rows_from_pilot(pilot: dict) -> list[dict]:
    spec = importlib.util.spec_from_file_location(
        "atlas_collect", ROOT / "scripts" / "atlas_collect.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.rows_from_pilot(pilot)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--runs-root",
        type=Path,
        default=None,
        help="Directory with stress_structured/*.json (default: scratch or scratch_mirror)",
    )
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    scratch = os.environ.get("PRIORITYKV_SCRATCH")
    if args.runs_root is not None:
        runs = args.runs_root
    elif scratch:
        runs = Path(scratch) / "runs"
    else:
        runs = ROOT / "scratch_mirror" / "runs"
    out = args.out
    if out is None:
        if scratch:
            out = Path(scratch) / "runs" / "atlas" / "w4_structure_rows.jsonl"
        else:
            out = ROOT / "docs" / "atlas_w4_structure_rows.jsonl"

    rows: list[dict] = []
    for rel in DEFAULTS:
        p = runs / rel
        if not p.is_file():
            print(f"skip missing {p}", flush=True)
            continue
        pilot = json.loads(p.read_text(encoding="utf-8"))
        part = _rows_from_pilot(pilot)
        for r in part:
            r["source_file"] = rel
        rows.extend(part)
        print(f"added {len(part)} from {rel}", flush=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"n_rows={len(rows)} out={out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
