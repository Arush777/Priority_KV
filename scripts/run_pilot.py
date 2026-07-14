#!/usr/bin/env python3
"""PriorityBench FullKV vs FP8 quality pilot. Usage: python scripts/run_pilot.py"""

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

from prioritykv.bench_pilot import run_quality_pilot  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "w2_pb_quality.yaml"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = Path(args.out) if args.out else None
    result = run_quality_pilot(Path(args.config), out_path=out)
    cats = " ".join(
        f"{k}:{v['fullkv_mean']:.2f}/{v['fp8_mean']:.2f}"
        for k, v in result["by_category"].items()
    )
    print(
        f"n={result['n']} full={result['fullkv_mean']:.3f} "
        f"fp8={result['fp8_mean']:.3f} delta={result['delta_fp8_minus_full']:+.3f} "
        f"cats[{cats}] out={result['out_path']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
