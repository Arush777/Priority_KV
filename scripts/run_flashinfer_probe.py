#!/usr/bin/env python3
"""W6 FlashInfer probe — import + optional tiny CUDA smoke; never silent.

Runs on H200 via the job queue (human worker only). Coding agents stay off H200.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    from prioritykv.flashinfer_multicall import probe, status

    st = status()
    st["timestamp"] = datetime.now(timezone.utc).isoformat()
    st["probe"] = probe()
    decision = st["probe"].get("decision", "UNKNOWN")
    print(json.dumps(st, indent=2))
    scratch = os.environ.get("PRIORITYKV_SCRATCH", str(ROOT / "runs"))
    out = Path(scratch) / "baselines" / "flashinfer_probe_r1.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(st, indent=2) + "\n", encoding="utf-8")
    print(f"decision={decision} out={out}", flush=True)
    # exit 0 even on SKIP — loud decision is in JSON (like SnapKV attempt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
