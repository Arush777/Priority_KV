#!/usr/bin/env python3
"""PriorityBench FullKV vs FP8 vs INT4 quality pilot. Usage: python scripts/run_pilot3.py"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# MUST run before quanto/transformers JIT-build Marlin kernels (same process).
import prioritykv.cxx20_cuda_ext  # noqa: E402,F401

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
        help="Prior FullKV/FP8 JSON with fullkv_text/fp8_text (skip vLLM)",
    )
    ap.add_argument(
        "--modes",
        default=None,
        choices=["all", "skip_fp8", "int4_only", "vllm_only"],
        help="Override config int4.modes (int4_only + --reuse skips vLLM)",
    )
    args = ap.parse_args()
    cfg_path = Path(args.config)
    if args.modes is not None:
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        cfg.setdefault("int4", {})["modes"] = args.modes
        tmp = cfg_path.with_name(f"{cfg_path.stem}_{args.modes}.yaml")
        tmp.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        cfg_path = tmp
    out = Path(args.out) if args.out else None
    reuse = Path(args.reuse) if args.reuse else None
    result = run_triple_pilot(cfg_path, out_path=out, reuse_path=reuse)
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
    # Surface first quanto failure reason if we fell back.
    for r in result.get("rows", []):
        meta = r.get("int4_meta") or {}
        for k in ("quanto_impl_error", "quanto_obj_error"):
            if meta.get(k):
                print(f"int4_fallback_reason[{k}]={meta[k]}")
                break
        else:
            continue
        break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
