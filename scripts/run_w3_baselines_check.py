#!/usr/bin/env python3
"""W3 baseline wiring check: quanto INT4 assert mode + SnapKV loud-skip.

CPU-safe. Does not load models.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from prioritykv.baselines.snapkv import status as snapkv_status  # noqa: E402
from prioritykv.int4_kv import status as int4_status  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--require-quanto", action="store_true")
    ap.add_argument("--require-snapkv", action="store_true")
    args = ap.parse_args()

    i4 = int4_status()
    sk = snapkv_status()
    print(json.dumps({"int4": i4, "snapkv": sk}, indent=2))

    rc = 0
    quanto_ok = bool(i4.get("quanto_available"))
    if not quanto_ok:
        print(
            "LOUD SKIP: quanto INT4 not installed — "
            "run_transformers_int4(..., allow_fake_fallback=False) will RAISE "
            "until `pip install optimum-quanto` (or quanto).",
            file=sys.stderr,
        )
        if args.require_quanto:
            rc = 1
    else:
        print("quanto INT4: READY (allow_fake_fallback=False on W3 quality runs)")

    if not sk.get("implemented"):
        print(
            "LOUD SKIP: SnapKV (kvpress.SnapKVPress) not installed — "
            "Q3 deferred; DropKeep remains interim eviction (G1 freeze).",
            file=sys.stderr,
        )
        if args.require_snapkv:
            rc = 1
    else:
        print("SnapKV: READY")

    return rc


if __name__ == "__main__":
    raise SystemExit(main())
