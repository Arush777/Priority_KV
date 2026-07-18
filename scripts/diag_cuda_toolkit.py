#!/usr/bin/env python3
"""Require nvcc and torch.version.cuda to use the same CUDA major version."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys


def main() -> int:
    cuda_home = os.environ.get("CUDA_HOME") or "/usr/local/cuda"
    os.environ["CUDA_HOME"] = cuda_home
    os.environ["PATH"] = f"{cuda_home}/bin:" + os.environ.get("PATH", "")
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0")

    print(f"CUDA_HOME={cuda_home}")
    print(f"PATH_has_nvcc_dir={os.path.isdir(os.path.join(cuda_home, 'bin'))}")
    print(f"TORCH_CUDA_ARCH_LIST={os.environ.get('TORCH_CUDA_ARCH_LIST')}")
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")

    nvcc = shutil.which("nvcc")
    print(f"nvcc_which={nvcc}")
    if not nvcc:
        print("FAIL: nvcc missing — install/link CUDA 13.x toolkit; do not JIT yet")
        return 1

    out = subprocess.check_output(["nvcc", "--version"], text=True)
    print("--- nvcc --version ---")
    print(out.rstrip())
    print("---")

    import torch

    print(f"torch={torch.__version__}")
    print(f"torch.version.cuda={torch.version.cuda}")
    if torch.cuda.is_available():
        print(f"device_count={torch.cuda.device_count()}")
        print(f"cap={torch.cuda.get_device_capability()}")
        print(f"name0={torch.cuda.get_device_name(0)}")
    else:
        print("WARN: torch.cuda.is_available() is False")

    tmaj = str(torch.version.cuda or "").split(".")[0]
    m = re.search(r"release (\d+)\.", out)
    nmaj = m.group(1) if m else "?"
    print(f"gate: torch_cuda_major={tmaj} nvcc_major={nmaj}")
    if not tmaj or nmaj == "?":
        print("FAIL: could not parse torch CUDA or nvcc release")
        return 1
    if nmaj != tmaj:
        print(f"FAIL: nvcc major {nmaj} != torch.cuda major {tmaj}")
        print("Get a matching CUDA 13.x toolkit or CUDA-matched torch wheel; do not build until they match.")
        return 1

    print("OK: toolkit/torch CUDA major match")
    try:
        import optimum.quanto as q  # type: ignore

        print(f"optimum.quanto={getattr(q, '__version__', '?')}")
    except Exception as e:  # noqa: BLE001
        print(f"WARN: optimum.quanto import failed: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
