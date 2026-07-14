#!/usr/bin/env python3
"""Optional oneshot FP8 KV calib (llmcompressor). Usage: python scripts/prep_fp8.py"""

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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        default=None,
        help="BF16 model dir (default: $PRIORITYKV_SCRATCH/models/Qwen3-8B)",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Output dir (default: $PRIORITYKV_SCRATCH/models/Qwen3-8B-fp8kv)",
    )
    ap.add_argument("--n-calib", type=int, default=64)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--strategy", default="tensor", choices=["tensor", "attn_head"])
    args = ap.parse_args()

    scratch = os.environ.get("PRIORITYKV_SCRATCH", "")
    model = args.model or (
        str(Path(scratch) / "models" / "Qwen3-8B") if scratch else None
    )
    out = args.out or (
        str(Path(scratch) / "models" / "Qwen3-8B-fp8kv") if scratch else None
    )
    if not model or not out:
        print("need --model/--out or PRIORITYKV_SCRATCH", file=sys.stderr)
        return 2

    try:
        import llmcompressor  # noqa: F401
    except ImportError:
        print("missing llmcompressor — run: uv pip install llmcompressor", file=sys.stderr)
        return 2

    from prioritykv.fp8_baseline import oneshot_calibrate_fp8

    path = oneshot_calibrate_fp8(
        model_path=model,
        save_dir=Path(out),
        n_calib=args.n_calib,
        max_seq_len=args.max_seq_len,
        strategy=args.strategy,
    )
    print(f"saved={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
