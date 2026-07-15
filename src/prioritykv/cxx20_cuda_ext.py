"""Force -std=c++20 for torch JIT CUDA extensions (quanto Marlin).

torch 2.11 + nvcc host-compiles ATen headers that need C++20 typename rules;
default extension flags use c++17 and fail on List_inl.h. Import this module
before anything that may JIT-build quanto_cuda (same process).
"""
from __future__ import annotations

_PATCHED = False


def apply() -> None:
    global _PATCHED
    if _PATCHED:
        return
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
        name = kwargs.get("name") or (args[0] if args else "?")
        print(f"[cxx20_cuda_ext] cpp_extension.load name={name} std=c++20", flush=True)
        return _orig(*args, **kwargs)

    cpp_ext.load = load  # type: ignore[assignment]
    _PATCHED = True


apply()
