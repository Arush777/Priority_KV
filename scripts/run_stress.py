#!/usr/bin/env python3
"""Decisive stress: FullKV vs ~10–60× DropKeep. Expect multi_turn collapse."""

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

from prioritykv.stress_pilot import run_stress_pilot  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "stress_dropkeep_16k.yaml"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    result = run_stress_pilot(Path(args.config), Path(args.out) if args.out else None)
    cats = " ".join(
        f"{k}:{v['fullkv_mean']:.2f}/{v['dropkeep_mean']:.2f}"
        for k, v in result["by_category"].items()
    )
    cx = result["dropkeep"].get("mean_compression_x")
    cxs = f"{cx:.1f}x" if cx else "?"
    print(
        f"n={result['n']} full={result['fullkv_mean']:.3f} "
        f"drop={result['dropkeep_mean']:.3f} "
        f"d_drop={result['delta_drop_minus_full']:+.3f} "
        f"compression≈{cxs} cats[{cats}] out={result['out_path']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
