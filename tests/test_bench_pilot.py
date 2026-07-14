"""CPU tests for PriorityBench quality pilot selection."""

from __future__ import annotations

import json
from pathlib import Path

from prioritykv.bench_pilot import select_rows

ROOT = Path(__file__).resolve().parents[1]


def test_select_cal_8k_mix():
    bench = json.loads(
        (ROOT / "data" / "prioritybench" / "manifests" / "w2_pilot.json").read_text()
    )
    rows = select_rows(
        bench,
        {
            "split": "calibration",
            "context_length": 8000,
            "n_tool_schema": 10,
            "n_instruction_supersession": 5,
        },
    )
    assert len(rows) == 15
    assert sum(1 for r in rows if r["category"] == "tool_schema") == 10
    assert sum(1 for r in rows if r["category"] == "instruction_supersession") == 5
