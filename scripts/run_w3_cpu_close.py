#!/usr/bin/env python3
"""W3 CPU close bundle: rebuild lock JSONL, dual audit, page-perturb labels."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(ROOT))


def main() -> int:
    py = sys.executable
    _run([py, "scripts/mk_bench.py", "--mode", "w3_lock"])
    _run([py, "scripts/audit_bench.py"])
    _run([py, "scripts/dual_audit_w3.py"])
    _run([py, "scripts/label_page_perturb.py", "--n", "40"])
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as e:
        raise SystemExit(e.returncode or 1)
