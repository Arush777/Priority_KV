#!/usr/bin/env python3
"""Normalize a quality-pilot JSON into failure-atlas JSONL rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def rows_from_pilot(pilot: dict) -> list[dict]:
    out = []
    mid = pilot.get("manifest_id", "unknown")
    # Triple pilot rows
    for r in pilot.get("rows", []):
        eid = r["example_id"]
        cat = r.get("category", "")
        base = {
            "manifest_id": mid,
            "example_id": eid,
            "category": cat,
            "context_length": r.get("context_length"),
        }
        fk = r.get("fullkv_score")
        for method, key in (
            ("fullkv", "fullkv_score"),
            ("fp8", "fp8_score"),
            ("int4", "int4_score"),
        ):
            sc = r.get(key)
            if sc is None:
                continue
            try:
                if sc != sc:  # NaN
                    continue
            except Exception:
                pass
            delta = None
            if fk is not None and fk == fk and method != "fullkv":
                delta = float(sc) - float(fk)
            out.append(
                {
                    **base,
                    "method": method,
                    "score": float(sc),
                    "delta_vs_fullkv": delta,
                }
            )
    # Structured stress arms
    arms = pilot.get("arms") or pilot.get("arms_detail") or {}
    for policy, arm in arms.items():
        for r in arm.get("rows", []) if isinstance(arm, dict) else []:
            eid = r.get("example_id") or r.get("id")
            if not eid:
                continue
            sc = r.get("policy_score", r.get("score"))
            fk = r.get("fullkv_score")
            if sc is None:
                continue
            delta = None
            if fk is not None and fk == fk:
                try:
                    delta = float(sc) - float(fk)
                except Exception:
                    delta = None
            out.append(
                {
                    "manifest_id": mid,
                    "example_id": eid,
                    "category": r.get("category", ""),
                    "context_length": r.get("context_length"),
                    "method": f"keep_{policy}",
                    "score": float(sc),
                    "delta_vs_fullkv": delta,
                }
            )
        # Aggregate-only arm without rows
        if isinstance(arm, dict) and "mean" in arm and not arm.get("rows"):
            out.append(
                {
                    "manifest_id": mid,
                    "example_id": "_arm_mean_",
                    "category": "",
                    "method": f"keep_{policy}",
                    "score": float(arm["mean"]),
                    "delta_vs_fullkv": arm.get("delta_minus_full"),
                }
            )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    pilot = json.loads(args.pilot.read_text(encoding="utf-8"))
    rows = rows_from_pilot(pilot)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"n_rows={len(rows)} out={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
