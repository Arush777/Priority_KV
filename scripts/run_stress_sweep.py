#!/usr/bin/env python3
"""Keep-budget sweep: FullKV once, DropKeep across recent window sizes."""

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

from prioritykv.stress_pilot import run_stress_sweep  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "stress_dropkeep_sweep.yaml"))
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--reuse-full",
        default=None,
        help="Prior stress JSON with fullkv_text (skip FullKV vLLM)",
    )
    args = ap.parse_args()
    result = run_stress_sweep(
        Path(args.config),
        Path(args.out) if args.out else None,
        reuse_full_path=Path(args.reuse_full) if args.reuse_full else None,
    )
    print(
        f"n={result['n']} full={result['fullkv_mean']:.3f} "
        f"sweep={result['recent_tokens_sweep']} out={result['out_path']}"
    )
    for p in result["curve"]:
        cats = " ".join(
            f"{k}:{v['dropkeep_mean']:.2f}" for k, v in p["by_category"].items()
        )
        print(
            f"  recent={p['recent_tokens']:>4} "
            f"drop={p['dropkeep_mean']:.3f} "
            f"d={p['delta_drop_minus_full']:+.3f} "
            f"x≈{p['mean_compression_x']:.1f} cats[{cats}]"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
