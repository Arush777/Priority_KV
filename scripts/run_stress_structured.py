#!/usr/bin/env python3
"""Matched keep_frac: FullKV vs uniform / structure / random / keep_all."""

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

from prioritykv.structured_stress import run_structured_stress  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config", default=str(ROOT / "configs" / "stress_structured_25.yaml")
    )
    ap.add_argument("--out", default=None)
    ap.add_argument("--reuse-full", default=None)
    ap.add_argument(
        "--buried",
        action="store_true",
        help="Embed gold turns in long filler (adversarial scope check)",
    )
    args = ap.parse_args()
    result = run_structured_stress(
        Path(args.config),
        Path(args.out) if args.out else None,
        reuse_full_path=Path(args.reuse_full) if args.reuse_full else None,
        buried=True if args.buried else None,
    )
    print(
        f"n={result['n']} full={result['fullkv_mean']:.3f} "
        f"keep_frac={result['keep']['keep_frac']} "
        f"buried={result.get('buried_state')} out={result['out_path']}"
    )
    for p, arm in result["arms"].items():
        cats = " ".join(
            f"{k}:{v['policy_mean']:.2f}" for k, v in arm["by_category"].items()
        )
        lens = " ".join(
            f"{k}:{v['policy_mean']:.2f}" for k, v in arm["by_context_length"].items()
        )
        print(
            f"  {p:10s} mean={arm['mean']:.3f} d={arm['delta_minus_full']:+.3f} "
            f"cats[{cats}] len[{lens}]"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
