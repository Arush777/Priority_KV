#!/usr/bin/env python3
"""GPU diag for the git job queue (agent-visible via worker push).

Writes a small JSON under $PRIORITYKV_SCRATCH/runs/diag_nvidia_smi/ and prints
``out=…`` so remote_worker can ship it to jobs/results/.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-tag", default="r1")
    args = ap.parse_args()

    scratch = Path(os.environ.get("PRIORITYKV_SCRATCH", ROOT / "runs"))
    out_dir = scratch / "runs" / "diag_nvidia_smi"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"diag_nvidia_smi_{args.out_tag}.json"

    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    t0 = time.time()
    try:
        proc = subprocess.run(
            ["nvidia-smi"],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
        smi = (proc.stdout or "") + (proc.stderr or "")
        rc = int(proc.returncode)
    except FileNotFoundError:
        smi = "nvidia-smi not found on PATH"
        rc = 127
    except Exception as exc:  # noqa: BLE001
        smi = f"nvidia-smi failed: {exc}"
        rc = 1

    # Queryable one-liner summary (CSV) when available.
    query = ""
    try:
        q = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if q.returncode == 0:
            query = (q.stdout or "").strip()
    except Exception:  # noqa: BLE001
        pass

    result = {
        "job": "diag_nvidia_smi",
        "tag": args.out_tag,
        "decision": "OK" if rc == 0 else "FAIL",
        "pass": rc == 0,
        "exit": rc,
        "cuda_visible_devices": cvd,
        "query_csv": query,
        "nvidia_smi": smi[-12000:],  # keep status JSON-ish small
        "seconds": round(time.time() - t0, 3),
    }
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("decision", "pass", "cuda_visible_devices", "query_csv")}, indent=2))
    print(f"out={path}")
    return 0 if rc == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
