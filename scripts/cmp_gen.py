#!/usr/bin/env python3
"""Greedy generation compare across two backends. Usage: python scripts/cmp_gen.py"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# load .env / CUDA cap
import os

_env = ROOT / "scripts" / "_env.sh"
# soft-load .env without printing
_dotenv = ROOT / ".env"
if _dotenv.exists():
    for line in _dotenv.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "6,7")

from prioritykv.fullkv_compare import compare  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument(
        "--config",
        default=str(ROOT / "configs" / "w1_fullkv.yaml"),
        help="manifest path",
    )
    ap.add_argument("--out", default=None, help="optional json out path")
    args = ap.parse_args()
    out = Path(args.out) if args.out else None
    result = compare(Path(args.config), out_path=out)
    # minimal stdout — no project slogans
    print(
        f"n={result['n']} exact={result['exact_match_rate']:.3f} "
        f"tok={result['mean_token_agree']:.3f} pass={int(result['passed'])} "
        f"out={result['out_path']}"
    )
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
