#!/usr/bin/env python3
"""Byte-matched mixed BF16/INT4 KV: FullKV vs uniform-INT4 vs structure-mixed."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

_dotenv = ROOT / ".env"
if _dotenv.exists():
    for line in _dotenv.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from prioritykv.mixed_serve import run_mixed_serve  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "w6_mixed_serve.yaml"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--out-tag", default=None)
    args = ap.parse_args()
    out_path = Path(args.out) if args.out else None
    if out_path is None and args.out_tag:
        scratch = os.environ.get("PRIORITYKV_SCRATCH")
        base = (
            Path(scratch) / "runs" / "mixed_serve"
            if scratch
            else ROOT / "runs" / "mixed_serve"
        )
        out_path = base / f"{args.out_tag}.json"
    result = run_mixed_serve(Path(args.config), out_path)
    full = result.get("fullkv_mean")
    print(
        f"n={result['n']} full={full if full is None else round(full, 3)} "
        f"int4_frac={result['mixed']['int4_frac']} "
        f"cold_attend={result['mixed'].get('cold_attend')} "
        f"out={result['out_path']}"
    )
    peaks = result.get("peak_alloc_gib") or {}
    for p, arm in result["arms"].items():
        cats = " ".join(
            f"{k}:{v['policy_mean']:.2f}" for k, v in arm["by_category"].items()
        )
        d = arm.get("delta_minus_full")
        peak = peaks.get(p)
        print(
            f"  {p:10s} mean={arm['mean']:.3f} "
            f"d={'' if d is None else f'{d:+.3f}'} "
            f"int4_real={arm['int4_frac_realized']:.2f} "
            f"peak_gib={peak} cats[{cats}]"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
