#!/usr/bin/env python3
"""PriorityBench FullKV vs FP8 vs INT4 quality pilot. Usage: python scripts/run_pilot3.py"""

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

from prioritykv.bench_pilot import run_triple_pilot  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "w2c_pb_quality_16k.yaml"))
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--reuse",
        default=None,
        help="Prior FullKV/FP8 pilot JSON with fullkv_text/fp8_text (skip vLLM)",
    )
    args = ap.parse_args()
    out = Path(args.out) if args.out else None
    reuse = Path(args.reuse) if args.reuse else None
    result = run_triple_pilot(Path(args.config), out_path=out, reuse_path=reuse)
    cats = " ".join(
        f"{k}:{v.get('fullkv_mean', float('nan')):.2f}/"
        f"{v.get('fp8_mean', float('nan')):.2f}/"
        f"{v.get('int4_mean', float('nan')):.2f}"
        for k, v in result["by_category"].items()
    )
    d4 = result.get("delta_int4_minus_full")
    d4s = f"{d4:+.3f}" if d4 is not None else "n/a"
    print(
        f"n={result['n']} full={result['fullkv_mean']:.3f} "
        f"fp8={result['fp8_mean']:.3f} int4={result['int4_mean']:.3f} "
        f"d_int4={d4s} modes={result.get('int4_modes_seen')} "
        f"cats[{cats}] out={result['out_path']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
