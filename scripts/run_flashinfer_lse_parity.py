#!/usr/bin/env python3
"""W6 FlashInfer LSE multi-call tiny parity vs CPU oracle.

Uses flashinfer.single_prefill_with_kv_cache(..., return_lse=True) on page chunks,
merges with prioritykv.lse_merge_pair, and checks vs dense prefill + CPU multicall.
Coding agents stay off H200 — enqueue via jobs/pending only.
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
    import numpy as np
    import torch

    from prioritykv.flashinfer_multicall import try_import_flashinfer
    from prioritykv.mixed_cache_reference import (
        attention_reference,
        attention_with_lse,
        lse_merge_pair,
        mixed_attend_kv_multicall,
        pages_from_sequence,
    )
    from prioritykv.page_roles import StorageDtype

    fi = try_import_flashinfer()
    result: dict = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "flashinfer": bool(fi is not None),
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
    # Small single-head case.
    tq, tk, d = 4, 64, 32
    q_np = rng.standard_normal((tq, d)).astype(np.float32)
    kv_np = rng.standard_normal((tk, d)).astype(np.float32)
    page_tokens = 16
    n_pages = tk // page_tokens
    dtypes = [StorageDtype.BF16] * n_pages
    pages = pages_from_sequence(kv_np, dtypes, page_tokens=page_tokens)
    cpu_dense = attention_reference(q_np, kv_np, kv_np)
    cpu_multi = mixed_attend_kv_multicall(q_np, pages, pages)
    cpu_err = float(np.max(np.abs(cpu_dense - cpu_multi)))
    result["cpu_multicall_vs_dense_max_abs"] = cpu_err

    device = torch.device("cuda:0")
    q = torch.as_tensor(q_np, device=device, dtype=torch.float16)
    # FlashInfer expects q: (qo_len, num_heads, head_dim); use 1 head.
    q_fi = q.view(tq, 1, d)
    outs = []
    lses = []
    for p in pages:
        chunk = p.materialize().astype(np.float32)
        k = torch.as_tensor(chunk, device=device, dtype=torch.float16).view(-1, 1, d)
        v = k.clone()
        o, lse = fi.single_prefill_with_kv_cache(
            q_fi, k, v, causal=False, return_lse=True
        )
        # o: (tq, 1, d) · lse: (tq, 1) or (tq,)
        o_np = o.detach().float().cpu().numpy().reshape(tq, d)
        lse_np = lse.detach().float().cpu().numpy().reshape(tq)
        outs.append(o_np)
        lses.append(lse_np)

    fi_out, fi_lse = outs[0], lses[0]
    for ob, lb in zip(outs[1:], lses[1:]):
        fi_out, fi_lse = lse_merge_pair(fi_out, fi_lse, ob, lb)

    # Also dense flashinfer once
    k_all = torch.as_tensor(kv_np, device=device, dtype=torch.float16).view(tk, 1, d)
    o_dense, _ = fi.single_prefill_with_kv_cache(
        q_fi, k_all, k_all.clone(), causal=False, return_lse=True
    )
    fi_dense = o_dense.detach().float().cpu().numpy().reshape(tq, d)

    err_vs_cpu = float(np.max(np.abs(fi_out - cpu_dense)))
    err_vs_fi_dense = float(np.max(np.abs(fi_out - fi_dense)))
    err_fi_dense_vs_cpu = float(np.max(np.abs(fi_dense - cpu_dense)))
    result.update(
        {
            "device": torch.cuda.get_device_name(0),
            "flashinfer_version": getattr(fi, "__version__", None),
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
    out = Path(scratch) / "baselines" / "flashinfer_lse_parity_r1.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"decision={result['decision']} out={out}", flush=True)
    return 0 if result["decision"] == "PARITY_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
