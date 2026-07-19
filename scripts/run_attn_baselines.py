#!/usr/bin/env python3
"""P1 matched-budget attention baselines vs structure (SnapKV / H2O / Pyramid / hybrid)."""

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
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


def _ensure_kvpress() -> None:
    """Install kvpress into the active venv if SnapKVPress is missing."""
    try:
        from kvpress import SnapKVPress  # noqa: F401

        return
    except Exception:
        pass
    import shutil
    import subprocess

    uv = shutil.which("uv") or "uv"
    cmd = [uv, "sync", "--extra", "gpu", "--extra", "kvpress", "--extra", "dev", "-q"]
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(ROOT))


from prioritykv.attn_baselines_quality import run_attn_baselines  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default=str(ROOT / "configs" / "p1_attn_baselines_s0_kf25.yaml"),
    )
    ap.add_argument("--out", default=None)
    ap.add_argument("--out-tag", default=None)
    args = ap.parse_args()
    _ensure_kvpress()
    out_path = Path(args.out) if args.out else None
    if out_path is None and args.out_tag:
        scratch = os.environ.get("PRIORITYKV_SCRATCH")
        base = (
            Path(scratch) / "runs" / "attn_baselines"
            if scratch
            else ROOT / "runs" / "attn_baselines"
        )
        out_path = base / f"{args.out_tag}.json"

    result = run_attn_baselines(Path(args.config), out_path)
    print(
        f"n={result['n']} full={result['fullkv_mean']:.3f} "
        f"decision={result['decision'].split('—')[0].strip()}"
    )
    for name, arm in result["arms"].items():
        m = arm.get("mean")
        err = arm.get("error")
        if err:
            print(f"  {name:10s} ERROR {err}")
        else:
            print(
                f"  {name:10s} mean={m:.3f} d={arm.get('delta_minus_full', float('nan')):+.3f}"
            )
    print(f"out={result['out_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
