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
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "6,7")

from prioritykv.mixed_serve import run_mixed_serve  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "w6_mixed_serve.yaml"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    result = run_mixed_serve(
        Path(args.config), Path(args.out) if args.out else None
    )
    full = result.get("fullkv_mean")
    print(
        f"n={result['n']} full={full if full is None else round(full, 3)} "
        f"int4_frac={result['mixed']['int4_frac']} out={result['out_path']}"
    )
    for p, arm in result["arms"].items():
        cats = " ".join(
            f"{k}:{v['policy_mean']:.2f}" for k, v in arm["by_category"].items()
        )
        d = arm.get("delta_minus_full")
        print(
            f"  {p:10s} mean={arm['mean']:.3f} "
            f"d={'' if d is None else f'{d:+.3f}'} "
            f"int4_real={arm['int4_frac_realized']:.2f} cats[{cats}]"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
