#!/usr/bin/env python3
"""Q3 matched-byte SnapKV quality pilot (FullKV vs DropKeep vs SnapKVPress)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "w4_snapkv_matched.yaml",
    )
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    from prioritykv.snapkv_quality import run_snapkv_quality

    run_snapkv_quality(args.config, out_path=args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
