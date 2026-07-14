#!/usr/bin/env python3
"""FullKV vs FP8 KV greedy compare. Usage: python scripts/cmp_fp8.py"""

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

from prioritykv.fp8_baseline import compare_fullkv_fp8  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "w1_fp8.yaml"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = Path(args.out) if args.out else None
    result = compare_fullkv_fp8(Path(args.config), out_path=out)
    print(
        f"n={result['n']} exact={result['exact_match_rate']:.3f} "
        f"tok={result['mean_token_agree']:.3f} pass={int(result['passed'])} "
        f"out={result['out_path']}"
    )
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
