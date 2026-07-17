#!/usr/bin/env python3
"""Stage-1a FI decode smoke: FiMixedDecodeState parity without materialize_hf_past.

H200: enqueue via jobs/pending. Loud-skips without CUDA/FlashInfer.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import prioritykv.cxx20_cuda_ext  # noqa: F401

import numpy as np

from prioritykv.byte_model import ModelKvGeom
from prioritykv.fi_mixed_decode import build_from_packed_cache, verify_state_flashinfer
from prioritykv.int4_kv import Int4KvConfig
from prioritykv.packed_mixed_cache import PackedMixedCache, ingest_synthetic_layer, page_manager_from_int4_mask
from prioritykv.page_roles import PageRole


def _synth_cache(n: int = 256, h: int = 8, d: int = 128, layers: int = 4):
    roles = [PageRole.SYSTEM] * 32 + [PageRole.FILLER] * (n - 32)
    mask = np.zeros(n, dtype=bool)
    mask[32:] = True
    geom = ModelKvGeom(num_layers=layers, num_kv_heads=h, head_dim=d)
    pm = page_manager_from_int4_mask(roles, mask, page_tokens=16, geom=geom)
    cache = PackedMixedCache(page_manager=pm, geom=geom, int4_cfg=Int4KvConfig(group_size=32))
    rng = np.random.default_rng(7)
    for _ in range(layers):
        k = rng.standard_normal((h, n, d)).astype(np.float16)
        v = rng.standard_normal((h, n, d)).astype(np.float16)
        cache.layers.append(ingest_synthetic_layer(k, v, pm, int4_cfg=cache.int4_cfg))
    return cache


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-tag", default="r1")
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--atol", type=float, default=5e-2)
    args = ap.parse_args()

    scratch = Path(os.environ.get("PRIORITYKV_SCRATCH", ROOT / "runs"))
    out_dir = scratch / "runs" / "fi_decode_smoke"
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    result: dict = {
        "job": "fi_decode_smoke",
        "tag": args.out_tag,
        "used_materialize_hf_past": False,
    }

    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        result.update({"decision": "SKIP_NO_TORCH", "error": str(exc), "pass": None})
        path = out_dir / f"fi_decode_smoke_{args.out_tag}.json"
        path.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        return 0

    if not torch.cuda.is_available():
        result.update({"decision": "SKIP_NO_CUDA", "pass": None})
        path = out_dir / f"fi_decode_smoke_{args.out_tag}.json"
        path.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        return 0

    cache = _synth_cache(args.seq, args.heads, args.head_dim, args.layers)
    device = torch.device("cuda:0")
    state = build_from_packed_cache(
        cache, device=device, dtype=torch.float16, decode_tail_cap=64
    )
    state.assert_no_materialize_path(False)

    torch.cuda.reset_peak_memory_stats(device)
    verify = verify_state_flashinfer(state, atol=args.atol, tq=1)
    peak = int(torch.cuda.max_memory_allocated(device))

    result.update(
        {
            "decision": verify.get("decision"),
            "pass": verify.get("pass"),
            "verify": verify,
            "seq": args.seq,
            "layers": args.layers,
            "head_dim": args.head_dim,
            "cuda_peak_bytes": peak,
            "device": torch.cuda.get_device_name(0),
            "seconds": round(time.time() - t0, 3),
        }
    )
    path = out_dir / f"fi_decode_smoke_{args.out_tag}.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("decision", "pass", "cuda_peak_bytes", "seconds")}, indent=2))
    print(f"out={path}")
    if verify.get("pass") is False:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
