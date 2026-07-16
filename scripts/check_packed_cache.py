#!/usr/bin/env python3
"""CPU smoke: packed mixed cache bytes + invariants.

Usage: python scripts/check_packed_cache.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from dataclasses import replace

from prioritykv.byte_model import QWEN3_8B_KV  # noqa: E402
from prioritykv.packed_mixed_cache import (  # noqa: E402
    PackedMixedCache,
    ingest_synthetic_layer,
)
from prioritykv.page_manager import PageManager, PageManagerConfig  # noqa: E402


def main() -> int:
    pm = PageManager(PageManagerConfig(budget_frac=0.50))
    messages = [
        {
            "role": "system",
            "content": 'Tools: [{"name":"search_docs"}]',
        },
        {"role": "user", "content": "Load schemas."},
        {"role": "assistant", "content": "Done."},
        {"role": "user", "content": ("context " * 4000)},
        {"role": "assistant", "content": ("reply " * 800)},
        {"role": "user", "content": "FINAL: search_docs"},
    ]
    pm.build_from_messages(messages)
    pm.append_generated_tokens(128)

    heads, dim = QWEN3_8B_KV.num_kv_heads, QWEN3_8B_KV.head_dim
    rng = np.random.default_rng(0)
    k = rng.standard_normal((heads, pm.seq_len, dim)).astype(np.float32)
    v = rng.standard_normal((heads, pm.seq_len, dim)).astype(np.float32)

    layers = [
        ingest_synthetic_layer(k, v, pm)
        for _ in range(min(4, QWEN3_8B_KV.num_layers))
    ]
    cache = PackedMixedCache(
        page_manager=pm,
        layers=layers,
        geom=replace(QWEN3_8B_KV, num_layers=len(layers)),
    )
    demoted = cache.sync_dtypes_from_manager()
    errs = cache.check_invariants()

    summary = {
        "seq_len": pm.seq_len,
        "n_pages": len(pm.pages),
        "n_layers": len(layers),
        "demoted_pages": demoted,
        "payload_bytes": cache.payload_bytes(),
        "realized_bytes": cache.realized_bytes(),
        "fullkv_bf16_bytes": cache.fullkv_bf16_bytes(),
        "compression_ratio": round(cache.compression_ratio(), 4),
        "within_budget": pm.within_budget(),
        "invariants_ok": len(errs) == 0,
        "errors": errs,
    }
    print(json.dumps(summary, separators=(",", ":")))
    return 0 if not errs else 2


if __name__ == "__main__":
    raise SystemExit(main())
