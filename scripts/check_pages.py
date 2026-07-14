#!/usr/bin/env python3
"""CPU page-manager smoke. Usage: python scripts/check_pages.py"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from prioritykv.page_manager import PageManager, PageManagerConfig  # noqa: E402
from prioritykv.page_roles import StorageDtype  # noqa: E402


def main() -> int:
    pm = PageManager(PageManagerConfig(budget_frac=0.50))
    messages = [
        {
            "role": "system",
            "content": "Available tools (JSON Schema): [{\"name\":\"search_docs\"}]",
        },
        {"role": "user", "content": "Confirm schemas loaded."},
        {"role": "assistant", "content": "Schemas loaded."},
        {"role": "user", "content": ("filler note " * 8000)},
        {"role": "assistant", "content": ("ack " * 1000)},
        {"role": "user", "content": "FINAL: use search_docs"},
    ]
    pm.build_from_messages(messages)
    pm.append_generated_tokens(256)
    errs = pm.check_invariants()
    counts = {k.value: v for k, v in pm.dtype_token_counts().items()}
    summary = {
        "seq_len": pm.seq_len,
        "n_pages": len(pm.pages),
        "dtype_tokens": counts,
        "realized": pm.realized_bytes(),
        "budget": pm.budget_bytes(),
        "within_budget": pm.within_budget(),
        "invariants_ok": len(errs) == 0,
        "errors": errs,
    }
    print(json.dumps(summary, separators=(",", ":")))
    return 0 if not errs else 2


if __name__ == "__main__":
    raise SystemExit(main())
