#!/usr/bin/env python3
"""Q3 SnapKV ≤4-day attempt (W3 close).

Tries ``uv sync --extra kvpress`` then instantiates SnapKVPress. On failure,
exits loudly so DropKeep can be locked as the permanent eviction interim.
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    import shutil

    uv = shutil.which("uv") or "uv"
    cmd = [
        uv,
        "sync",
        "--extra",
        "gpu",
        "--extra",
        "kvpress",
        "--extra",
        "dev",
        "-q",
    ]
    print("+", " ".join(cmd), flush=True)
    try:
        subprocess.check_call(cmd, cwd=str(ROOT))
    except Exception as e:  # noqa: BLE001
        print(f"uv sync kvpress failed: {e}", file=sys.stderr)

    from prioritykv.baselines.snapkv import make_press, status

    st = status()
    st["timestamp"] = datetime.now(timezone.utc).isoformat()
    press = make_press()
    if press is None:
        st["attempt"] = "FAIL"
        st["decision"] = (
            "LOCK_Q_DROPKEEP — SnapKVPress unavailable after uv sync --extra kvpress; "
            "G1 interim DropKeep remains permanent eviction baseline (Q3 open→closed as substituted)."
        )
        print(json.dumps(st, indent=2))
        print("SnapKV: LOUD SKIP → DropKeep locked", file=sys.stderr)
        out = Path(
            __import__("os").environ.get(
                "PRIORITYKV_SCRATCH", str(ROOT / "runs")
            )
        ) / "baselines" / "snapkv_attempt_r1.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(st, indent=2) + "\n", encoding="utf-8")
        print(f"out={out}")
        return 0

    st["attempt"] = "IMPORT_OK"
    st["decision"] = (
        "IMPORT_OK — run matched-byte quality pilot next; not yet claiming Q3 green."
    )
    print(json.dumps(st, indent=2))
    out = Path(
        __import__("os").environ.get("PRIORITYKV_SCRATCH", str(ROOT / "runs"))
    ) / "baselines" / "snapkv_attempt_r1.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(st, indent=2) + "\n", encoding="utf-8")
    print(f"SnapKV: READY out={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
