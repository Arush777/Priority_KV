#!/usr/bin/env python3
"""W6 FlashInfer LSE multi-call parity vs CPU oracle.

Uses flashinfer.single_prefill_with_kv_cache(..., return_lse=True) on page chunks,
merges with flashinfer.merge_state (the native LSE contract), and checks vs dense
prefill + the natural-log CPU oracle.

SM90 Hopper kernels only support head_dim ∈ {64, 128, 256} — default 128 (Qwen3).
Coding agents stay off H200 — enqueue via jobs/pending only.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Hopper SM90 single_prefill supports these VO dims (static_assert in flashinfer).
_ALLOWED_HEAD_DIMS = (64, 128, 256)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--head-dim",
        type=int,
        default=128,
        help="Attention head dim (SM90 allows 64/128/256; Qwen3-8B=128)",
    )
    ap.add_argument("--tq", type=int, default=4, help="Query length")
    ap.add_argument("--tk", type=int, default=64, help="KV length (must divide page_tokens)")
    ap.add_argument("--page-tokens", type=int, default=16)
    ap.add_argument("--out-tag", default="r2", help="Artifact filename tag")
    args = ap.parse_args()
    if args.head_dim not in _ALLOWED_HEAD_DIMS:
        print(
            f"ERROR: head_dim={args.head_dim} not in {_ALLOWED_HEAD_DIMS} "
            "(SM90 FlashInfer static_assert). Refusing to JIT a broken kernel.",
            flush=True,
        )
        return 2
    if args.tk % args.page_tokens != 0:
        print("ERROR: --tk must be divisible by --page-tokens", flush=True)
        return 2

    import numpy as np
    import torch

    from prioritykv.flashinfer_multicall import (
        attend_pages_flashinfer,
        dense_prefill_flashinfer,
        try_import_flashinfer,
    )
    from prioritykv.mixed_cache_reference import (
        attention_reference,
        mixed_attend_kv_multicall,
        pages_from_sequence,
    )
    from prioritykv.page_roles import StorageDtype

    fi = try_import_flashinfer()
    result: dict = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "flashinfer": bool(fi is not None),
        "head_dim": args.head_dim,
        "tq": args.tq,
        "tk": args.tk,
        "page_tokens": args.page_tokens,
    }
    if fi is None:
        result["decision"] = "SKIP_NO_PACKAGE"
        print(json.dumps(result, indent=2))
        return 0
    if not torch.cuda.is_available():
        result["decision"] = "SKIP_NO_CUDA"
        print(json.dumps(result, indent=2))
        return 0

    rng = np.random.default_rng(0)
    tq, tk, d = args.tq, args.tk, args.head_dim
    q_np = rng.standard_normal((tq, d)).astype(np.float32)
    kv_np = rng.standard_normal((tk, d)).astype(np.float32)
    page_tokens = args.page_tokens
    n_pages = tk // page_tokens
    dtypes = [StorageDtype.BF16] * n_pages
    pages = pages_from_sequence(kv_np, dtypes, page_tokens=page_tokens)
    cpu_dense = attention_reference(q_np, kv_np, kv_np)
    cpu_multi = mixed_attend_kv_multicall(q_np, pages, pages)
    cpu_err = float(np.max(np.abs(cpu_dense - cpu_multi)))
    result["cpu_multicall_vs_dense_max_abs"] = cpu_err

    device = torch.device("cuda:0")
    q = torch.as_tensor(q_np, device=device, dtype=torch.float16)
    q_fi = q.view(tq, 1, d)
    k_pages = []
    v_pages = []
    for p in pages:
        chunk = p.materialize().astype(np.float32)
        k = torch.as_tensor(chunk, device=device, dtype=torch.float16).view(-1, 1, d)
        k_pages.append(k)
        v_pages.append(k.clone())

    fi_out_t = attend_pages_flashinfer(q_fi, k_pages, v_pages, fi=fi)
    fi_out = fi_out_t.detach().float().cpu().numpy().reshape(tq, d)

    k_all = torch.as_tensor(kv_np, device=device, dtype=torch.float16).view(tk, 1, d)
    o_dense = dense_prefill_flashinfer(q_fi, k_all, k_all.clone(), fi=fi)
    fi_dense = o_dense.detach().float().cpu().numpy().reshape(tq, d)

    err_vs_cpu = float(np.max(np.abs(fi_out - cpu_dense)))
    err_vs_fi_dense = float(np.max(np.abs(fi_out - fi_dense)))
    err_fi_dense_vs_cpu = float(np.max(np.abs(fi_dense - cpu_dense)))
    result.update(
        {
            "device": torch.cuda.get_device_name(0),
            "flashinfer_version": getattr(fi, "__version__", None),
            "merge_impl": "flashinfer.merge_state",
            "lse_contract": "flashinfer-native (historically base-2)",
            "fi_multicall_vs_cpu_dense_max_abs": err_vs_cpu,
            "fi_multicall_vs_fi_dense_max_abs": err_vs_fi_dense,
            "fi_dense_vs_cpu_dense_max_abs": err_fi_dense_vs_cpu,
            "pass_fi_merge": err_vs_fi_dense < 5e-2,
            "pass_vs_cpu": err_vs_cpu < 5e-2,
        }
    )
    result["decision"] = (
        "PARITY_PASS"
        if result["pass_fi_merge"] and result["cpu_multicall_vs_dense_max_abs"] < 1e-4
        else "PARITY_FAIL"
    )
    print(json.dumps(result, indent=2))
    scratch = os.environ.get("PRIORITYKV_SCRATCH", str(ROOT / "runs"))
    out = Path(scratch) / "baselines" / f"flashinfer_lse_parity_{args.out_tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"decision={result['decision']} out={out}", flush=True)
    return 0 if result["decision"] == "PARITY_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
