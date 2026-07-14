#!/usr/bin/env python3
"""RULER + SCBench guardrail harness stub (W2 close).

No model calls this week — exits SKIPPED with reasons. Must run for real before G2.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        default=None,
        help="Write skip manifest JSON",
    )
    args = ap.parse_args()
    status = {
        "manifest_id": "guardrails_stub",
        "rev": 1,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "SKIPPED",
        "tasks": {
            "ruler_niah": {
                "status": "SKIPPED",
                "reason": "W2 stub — harness entry only; full RULER 2-task matrix deferred to pre-G2",
            },
            "ruler_vt": {
                "status": "SKIPPED",
                "reason": "W2 stub — harness entry only; full RULER 2-task matrix deferred to pre-G2",
            },
            "scbench_mt": {
                "status": "SKIPPED",
                "reason": "W2 stub — SCBench 2-task multi-turn harness deferred to pre-G2",
            },
            "scbench_choice": {
                "status": "SKIPPED",
                "reason": "W2 stub — SCBench 2-task harness deferred to pre-G2",
            },
        },
        "note": "G2 requires guardrail movement <1pt; this stub cannot survive into W4.",
    }
    text = json.dumps(status, indent=2)
    print(text)
    out = args.out
    if out is None:
        scratch = os.environ.get("PRIORITYKV_SCRATCH")
        if scratch:
            out = str(Path(scratch) / "runs" / "guardrails" / "stub_r1.json")
        else:
            out = str(ROOT / "runs" / "guardrails" / "stub_r1.json")
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")
    print(f"status=SKIPPED out={path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
