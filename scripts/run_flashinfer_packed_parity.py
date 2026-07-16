#!/usr/bin/env python3
"""FlashInfer multicall parity over mixed BF16/INT4 packed pages.

Builds a synthetic PackedMixedCache (some pages INT4), runs
``attend_packed_layer_flashinfer`` vs dense FlashInfer prefill, and checks
``merge_state`` parity. Coding agents stay off H200 — enqueue via jobs/pending.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--num-kv-heads", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--tq", type=int, default=4)
    ap.add_argument("--int4-frac", type=float, default=0.75)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--out-tag", default="r1")
    args = ap.parse_args()

    import numpy as np

    from prioritykv.byte_model import QWEN3_8B_KV
    from prioritykv.flashinfer_multicall import (
        ALLOWED_HEAD_DIMS,
        packed_layer_parity,
        try_import_flashinfer,
        verify_packed_cache_flashinfer,
    )
    from prioritykv.int4_kv import Int4KvConfig
    from prioritykv.packed_mixed_cache import (
        PackedMixedCache,
        ingest_synthetic_layer,
        page_manager_from_int4_mask,
    )
    from prioritykv.page_roles import PageRole

    result: dict = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "head_dim": args.head_dim,
        "num_kv_heads": args.num_kv_heads,
        "seq_len": args.seq_len,
        "tq": args.tq,
        "int4_frac": args.int4_frac,
        "n_layers": args.n_layers,
    }
    if args.head_dim not in ALLOWED_HEAD_DIMS:
        result["decision"] = "REJECT_HEAD_DIM"
        print(json.dumps(result, indent=2))
        return 2

    fi = try_import_flashinfer()
    result["flashinfer"] = bool(fi is not None)
    if fi is None:
        result["decision"] = "SKIP_NO_PACKAGE"
        print(json.dumps(result, indent=2))
        return 0

    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        result["decision"] = "SKIP_NO_TORCH"
        result["error"] = str(exc)
        print(json.dumps(result, indent=2))
        return 0
    if not torch.cuda.is_available():
        result["decision"] = "SKIP_NO_CUDA"
        print(json.dumps(result, indent=2))
        return 0

    n = args.seq_len
    roles = [PageRole.FILLER] * n
    for i in range(min(16, n)):
        roles[i] = PageRole.SINK
    for i in range(max(0, n - 32), n):
        roles[i] = PageRole.RECENT
    mask = np.zeros(n, dtype=bool)
    # Demote middle filler band to INT4 (leave sink/recent BF16).
    demotable = [i for i in range(n) if roles[i] == PageRole.FILLER]
    n_int4 = int(round(args.int4_frac * n))
    n_int4 = min(n_int4, len(demotable))
    for i in demotable[:n_int4]:
        mask[i] = True

    geom = replace(
        QWEN3_8B_KV,
        num_layers=args.n_layers,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
    )
    pm = page_manager_from_int4_mask(roles, mask, geom=geom)
    rng = np.random.default_rng(0)
    layers = []
    for _ in range(args.n_layers):
        k = rng.standard_normal(
            (args.num_kv_heads, n, args.head_dim)
        ).astype(np.float32)
        v = rng.standard_normal(
            (args.num_kv_heads, n, args.head_dim)
        ).astype(np.float32)
        layers.append(
            ingest_synthetic_layer(k, v, pm, int4_cfg=Int4KvConfig(group_size=32))
        )
    cache = PackedMixedCache(
        page_manager=pm, layers=layers, geom=geom, int4_cfg=Int4KvConfig()
    )
    inv = cache.check_invariants()
    result["invariants_ok"] = len(inv) == 0
    result["errors"] = inv
    result["n_pages"] = len(pm.pages)
    result["int4_tokens"] = int(mask.sum())
    result["payload_bytes"] = cache.payload_bytes()
    result["compression_ratio"] = round(cache.compression_ratio(), 6)

    layer0 = packed_layer_parity(cache.layers[0], tq=args.tq)
    multi = verify_packed_cache_flashinfer(cache, tq=args.tq)
    result["layer0"] = layer0
    result["cache_parity"] = multi
    result["device"] = torch.cuda.get_device_name(0)
    result["flashinfer_version"] = getattr(fi, "__version__", None)
    result["decision"] = multi.get("decision", layer0.get("decision", "FAIL"))
    result["pass"] = multi.get("pass")

    print(json.dumps(result, indent=2))
    scratch = os.environ.get("PRIORITYKV_SCRATCH", str(ROOT / "runs"))
    out = Path(scratch) / "baselines" / f"flashinfer_packed_parity_{args.out_tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"decision={result['decision']} out={out}", flush=True)
    if result["decision"] == "PARITY_PASS":
        return 0
    if str(result["decision"]).startswith("SKIP"):
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
