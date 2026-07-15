#!/usr/bin/env python3
"""Prebuild optimum-quanto ``quanto_cuda`` with ``-std=c++20``.

H200/torch 2.11 + nvcc host compile of Marlin sources fails under the default
``-std=c++17`` (ATen ``List_inl.h`` needs C++20 typename rules). Forcing
C++20 makes ``gptq_marlin_repack.cu`` compile. Does not change torch pins or
weaken INT4 assert mode. Uses only packages already in the uv lock.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import prioritykv.cxx20_cuda_ext  # noqa: E402,F401


def _prep_env() -> None:
    cuda_home = os.environ.get("CUDA_HOME") or "/usr/local/cuda"
    os.environ["CUDA_HOME"] = cuda_home
    os.environ["PATH"] = f"{cuda_home}/bin:" + os.environ.get("PATH", "")
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0")
    print(
        f"prebuild CUDA_HOME={cuda_home} "
        f"TORCH_CUDA_ARCH_LIST={os.environ['TORCH_CUDA_ARCH_LIST']}"
    )


def main() -> int:
    _prep_env()

    # Importing quanto registers the Extension as "quanto_cuda" (not "cuda").
    import optimum.quanto  # noqa: F401
    import optimum.quanto.library.extensions.cuda as qcuda
    from optimum.quanto.library.extensions import get_extension

    cuda_root = Path(qcuda.__file__).resolve().parent
    build_dir = cuda_root / "build"
    if build_dir.exists():
        print(f"clearing stale quanto build: {build_dir}")
        shutil.rmtree(build_dir, ignore_errors=True)

    try:
        ext = get_extension("quanto_cuda")
    except KeyError:
        ext = qcuda.ext
    print(f"building extension name={getattr(ext, 'name', '?')}")
    lib = ext.lib  # triggers JIT / ninja build
    so = build_dir / "quanto_cuda.so"
    print(f"OK: quanto_cuda loaded lib={lib} so_exists={so.exists()} so={so}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:  # noqa: BLE001
        print(f"FAIL: prebuild_quanto_cuda: {e}", file=sys.stderr)
        raise SystemExit(1)
