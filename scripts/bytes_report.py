#!/usr/bin/env python3
"""Print KV byte-budget table (CPU). Usage: python scripts/bytes_report.py"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from prioritykv.byte_model import QWEN3_8B_KV, budget_table  # noqa: E402


def main() -> int:
    rows = budget_table()
    # TSV for humans; also dump json if --json
    as_json = "--json" in sys.argv
    payload = []
    for p in rows:
        payload.append(
            {
                "seq_len": p.seq_len,
                "budget_frac": p.budget_frac,
                "fullkv_gib": round(p.fullkv_bytes / (1024**3), 4),
                "budget_gib": round(p.budget_bytes / (1024**3), 4),
                "max_bf16_tokens": p.max_bf16_tokens,
                "max_bf16_frac": round(p.max_bf16_tokens / p.seq_len, 4)
                if p.seq_len
                else 0.0,
                "all_int4_frac": round(p.all_int4_frac, 4),
                "feasible": p.feasible,
            }
        )
    if as_json:
        print(json.dumps({"geom": QWEN3_8B_KV.__dict__, "rows": payload}, indent=2))
    else:
        print(
            "seq\tbudget\tfullGiB\tbudGiB\tmaxBF16\tbf16Frac\tint4Floor\tok"
        )
        for r in payload:
            print(
                f"{r['seq_len']}\t{r['budget_frac']:.2f}\t{r['fullkv_gib']:.3f}\t"
                f"{r['budget_gib']:.3f}\t{r['max_bf16_tokens']}\t{r['max_bf16_frac']:.3f}\t"
                f"{r['all_int4_frac']:.3f}\t{int(r['feasible'])}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
