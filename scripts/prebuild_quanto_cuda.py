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


def _prep_env() -> None:
    cuda_home = os.environ.get("CUDA_HOME") or "/usr/local/cuda"
    os.environ["CUDA_HOME"] = cuda_home
    os.environ["PATH"] = f"{cuda_home}/bin:" + os.environ.get("PATH", "")
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0")
    print(
        f"prebuild CUDA_HOME={cuda_home} "
        f"TORCH_CUDA_ARCH_LIST={os.environ['TORCH_CUDA_ARCH_LIST']}"
    )


def _force_cxx20_on_cpp_extension_load() -> None:
    import torch.utils.cpp_extension as cpp_ext

    _orig = cpp_ext.load

    def load(*args, **kwargs):  # type: ignore[no-untyped-def]
        flags = list(kwargs.get("extra_cuda_cflags") or [])
        flags = [f for f in flags if not str(f).startswith("-std=")]
        flags.append("-std=c++20")
        kwargs["extra_cuda_cflags"] = flags
        cflags = list(kwargs.get("extra_cflags") or [])
        cflags = [f for f in cflags if not str(f).startswith("-std=")]
        cflags.append("-std=c++20")
        kwargs["extra_cflags"] = cflags
        print(f"cpp_extension.load name={kwargs.get('name') or (args[0] if args else '?')} std=c++20")
        return _orig(*args, **kwargs)

    cpp_ext.load = load  # type: ignore[assignment]


def main() -> int:
    _prep_env()
    _force_cxx20_on_cpp_extension_load()

    # Importing quanto registers the Extension as "quanto_cuda" (not "cuda").
    import optimum.quanto  # noqa: F401
    import optimum.quanto.library.extensions.cuda as qcuda
    from optimum.quanto.library.extensions import get_extension

    cuda_root = Path(qcuda.__file__).resolve().parent
    build_dir = cuda_root / "build"
    if build_dir.exists():
        print(f"clearing stale quanto build: {build_dir}")
        shutil.rmtree(build_dir, ignore_errors=True)

    # Prefer registered name; fall back to module-local Extension object.
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
