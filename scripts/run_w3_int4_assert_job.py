#!/usr/bin/env python3
"""H200 job wrapper for W3 INT4 assert (HANDOFF_W3_INT4 §B steps 2–5).

Sets CUDA toolkit env, clears stale torch JIT cache, ensures bench lock present,
then runs ``run_pilot3.py --modes int4_only`` under allow_fake_fallback=False.
Does not weaken assert mode. Does not retune the locked bench.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _prep_env() -> None:
    cuda_home = os.environ.get("CUDA_HOME") or "/usr/local/cuda"
    os.environ["CUDA_HOME"] = cuda_home
    os.environ["PATH"] = f"{cuda_home}/bin:" + os.environ.get("PATH", "")
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "6,7")
    os.environ.setdefault(
        "PRIORITYKV_SCRATCH", "/data/anupam/scratch/prioritykv"
    )
    print(
        f"prep CUDA_HOME={cuda_home} "
        f"TORCH_CUDA_ARCH_LIST={os.environ['TORCH_CUDA_ARCH_LIST']} "
        f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}"
    )
    ext = Path.home() / ".cache" / "torch_extensions"
    if ext.exists():
        print(f"clearing stale JIT cache: {ext}")
        shutil.rmtree(ext, ignore_errors=True)


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(ROOT))


def main() -> int:
    _prep_env()
    py = sys.executable
    # Soft: rebuild gitignored JSONL if missing; audit must PASS.
    _run([py, "scripts/mk_bench.py", "--mode", "w3_lock"])
    _run([py, "scripts/audit_bench.py"])
    _run([py, "scripts/run_w3_baselines_check.py"])
    # Force C++20 for Marlin JIT (torch List_inl.h fails under default c++17).
    _run([py, "scripts/prebuild_quanto_cuda.py"])
    # Also wipe package-local stale objects left by prior c++17 failures.
    # THE JOB — full stderr/stdout go to remote_worker tee.
    _run(
        [
            py,
            "scripts/run_pilot3.py",
            "--config",
            "configs/w3_int4_assert.yaml",
            "--modes",
            "int4_only",
        ]
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as e:
        print(f"FAIL: command exited {e.returncode}", file=sys.stderr)
        raise SystemExit(e.returncode or 1)
