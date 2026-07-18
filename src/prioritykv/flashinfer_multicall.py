"""FlashInfer multi-call attention over homogeneous page chunks (W6→D3).

Each page is attended with ``single_prefill_with_kv_cache(..., return_lse=True)``,
then partials are merged with ``flashinfer.merge_state`` (native LSE contract —
historically base-2; never feed FI LSE into the NumPy natural-log oracle).

CPU oracles remain in ``mixed_cache_reference``. This module loud-skips when
FlashInfer / CUDA is absent so laptop CI stays green.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

# Hopper SM90 single_prefill VO dims (static_assert in flashinfer).
ALLOWED_HEAD_DIMS = (64, 128, 256)


def flashinfer_available() -> bool:
    try:
        import flashinfer  # noqa: F401

        return True
    except Exception:
        return False


def try_import_flashinfer() -> Optional[Any]:
    if not flashinfer_available():
        return None
    import flashinfer

    return flashinfer


def require_head_dim(head_dim: int) -> None:
    if head_dim not in ALLOWED_HEAD_DIMS:
        raise ValueError(
            f"head_dim={head_dim} not in {ALLOWED_HEAD_DIMS} "
            "(SM90 FlashInfer static_assert)"
        )


def attend_pages_flashinfer(
    q: Any,
    k_pages: Sequence[Any],
    v_pages: Sequence[Any],
    *,
    causal: bool = False,
    fi: Any = None,
) -> Any:
    """Multi-call FlashInfer attention + ``merge_state``.

    Parameters
    ----------
    q:
        ``(tq, num_qo_heads, head_dim)`` float16/bfloat16 CUDA tensor.
    k_pages / v_pages:
        Sequences of ``(tk_i, num_kv_heads, head_dim)`` tensors (same dtype/device).
        Empty pages are skipped.
    """
    fi = fi or try_import_flashinfer()
    if fi is None:
        raise RuntimeError("flashinfer not installed")
    if not k_pages or not v_pages:
        raise ValueError("empty k/v pages")
    if len(k_pages) != len(v_pages):
        raise ValueError("k_pages/v_pages length mismatch")
    if q.dim() != 3:
        raise ValueError(f"q must be (tq, heads, dim), got {tuple(q.shape)}")
    require_head_dim(int(q.shape[-1]))

    states: list[tuple[Any, Any]] = []
    for k, v in zip(k_pages, v_pages):
        if k is None or v is None:
            continue
        if k.shape[0] == 0:
            continue
        if k.dim() != 3 or v.dim() != 3:
            raise ValueError(
                f"page K/V must be (tk, kv_heads, dim), got {tuple(k.shape)} / {tuple(v.shape)}"
            )
        o, lse = fi.single_prefill_with_kv_cache(
            q, k, v, causal=causal, return_lse=True
        )
        states.append((o, lse.float()))
    if not states:
        raise ValueError("all pages empty")
    out_t, lse_t = states[0]
    for ob, lb in states[1:]:
        out_t, lse_t = fi.merge_state(out_t, lse_t, ob, lb)
    return out_t


def page_tensors_from_packed_layer(
    layer: Any,
    *,
    device: Any,
    dtype: Any = None,
    coalesce_by_dtype: bool = True,
) -> tuple[list[Any], list[Any]]:
    """Materialize each ``KvPagePayload`` to FI layout ``(tk, heads, dim)``.

    When ``coalesce_by_dtype`` is True (default), merge all BF16 pages into one
    chunk and all INT4 pages into one chunk. Homogeneous 2-way multicall avoids
    FlashInfer TMA / merge_state failures on many 1-token run-length pages.
    """
    import torch

    from prioritykv.page_roles import StorageDtype

    dtype = dtype or torch.float16
    bf16_k: list[Any] = []
    bf16_v: list[Any] = []
    int4_k: list[Any] = []
    int4_v: list[Any] = []
    raw_k: list[Any] = []
    raw_v: list[Any] = []
    for payload in layer.pages:
        k_np, v_np = payload.materialize_kv()  # (heads, tk, dim)
        k = torch.from_numpy(k_np).to(device=device, dtype=dtype).permute(1, 0, 2).contiguous()
        v = torch.from_numpy(v_np).to(device=device, dtype=dtype).permute(1, 0, 2).contiguous()
        if not coalesce_by_dtype:
            raw_k.append(k)
            raw_v.append(v)
            continue
        if payload.dtype == StorageDtype.INT4:
            int4_k.append(k)
            int4_v.append(v)
        else:
            bf16_k.append(k)
            bf16_v.append(v)
    if not coalesce_by_dtype:
        return raw_k, raw_v
    k_pages: list[Any] = []
    v_pages: list[Any] = []
    # Stable order: BF16 first, then INT4 (matches "hot then cold" intuition).
    if bf16_k:
        k_pages.append(torch.cat(bf16_k, dim=0))
        v_pages.append(torch.cat(bf16_v, dim=0))
    if int4_k:
        k_pages.append(torch.cat(int4_k, dim=0))
        v_pages.append(torch.cat(int4_v, dim=0))
    return k_pages, v_pages


def attend_packed_layer_flashinfer(
    q: Any,
    layer: Any,
    *,
    causal: bool = False,
    fi: Any = None,
    dtype: Any = None,
    coalesce_by_dtype: bool = True,
) -> Any:
    """FlashInfer multicall over one ``PackedMixedLayer`` (per-page dequant)."""
    device = q.device
    dtype = dtype or q.dtype
    k_pages, v_pages = page_tensors_from_packed_layer(
        layer,
        device=device,
        dtype=dtype,
        coalesce_by_dtype=coalesce_by_dtype,
    )
    return attend_pages_flashinfer(
        q, k_pages, v_pages, causal=causal, fi=fi
    )


def dense_prefill_flashinfer(
    q: Any,
    k: Any,
    v: Any,
    *,
    causal: bool = False,
    fi: Any = None,
) -> Any:
    """Single FlashInfer prefill over a dense (already-concatenated) KV."""
    fi = fi or try_import_flashinfer()
    if fi is None:
        raise RuntimeError("flashinfer not installed")
    require_head_dim(int(q.shape[-1]))
    o, _lse = fi.single_prefill_with_kv_cache(
        q, k, v, causal=causal, return_lse=True
    )
    return o


def packed_layer_parity(
    layer: Any,
    *,
    tq: int = 4,
    device: Any = None,
    seed: int = 0,
    atol: float = 5e-2,
) -> dict[str, Any]:
    """Compare FI page-multicall vs FI dense on one packed layer.

    Returns a result dict with ``pass`` / errors. Loud-skips without CUDA/FI.
    """
    import numpy as np

    fi = try_import_flashinfer()
    out: dict[str, Any] = {
        "flashinfer": bool(fi is not None),
        "n_pages": len(layer.pages),
        "n_tokens": sum(p.n_tokens for p in layer.pages),
    }
    if fi is None:
        out["decision"] = "SKIP_NO_PACKAGE"
        out["pass"] = None
        return out
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        out["decision"] = "SKIP_NO_TORCH"
        out["error"] = str(exc)
        out["pass"] = None
        return out
    if not torch.cuda.is_available():
        out["decision"] = "SKIP_NO_CUDA"
        out["pass"] = None
        return out

    device = device or torch.device("cuda:0")
    k_full, v_full = layer.materialize()  # (heads, seq, dim)
    heads, seq, dim = k_full.shape
    require_head_dim(dim)
    rng = np.random.default_rng(seed)
    # GQA-style: use same head count for Q as KV for the parity probe.
    q_np = rng.standard_normal((tq, heads, dim)).astype(np.float32)
    q = torch.as_tensor(q_np, device=device, dtype=torch.float16)
    k_dense = (
        torch.from_numpy(k_full)
        .to(device=device, dtype=torch.float16)
        .permute(1, 0, 2)
        .contiguous()
    )
    v_dense = (
        torch.from_numpy(v_full)
        .to(device=device, dtype=torch.float16)
        .permute(1, 0, 2)
        .contiguous()
    )
    with torch.no_grad():
        fi_multi = attend_packed_layer_flashinfer(q, layer, fi=fi, dtype=torch.float16)
        fi_dense = dense_prefill_flashinfer(q, k_dense, v_dense, fi=fi)
    err = float(torch.max(torch.abs(fi_multi.float() - fi_dense.float())).item())
    out.update(
        {
            "device": torch.cuda.get_device_name(0),
            "flashinfer_version": getattr(fi, "__version__", None),
            "head_dim": dim,
            "num_kv_heads": heads,
            "tq": tq,
            "fi_multicall_vs_fi_dense_max_abs": err,
            "atol": atol,
            "pass": err < atol,
            "decision": "PARITY_PASS" if err < atol else "PARITY_FAIL",
            "merge_impl": "flashinfer.merge_state",
        }
    )
    return out


def verify_packed_cache_flashinfer(
    cache: Any,
    *,
    layer_indices: Sequence[int] | None = None,
    tq: int = 4,
    atol: float = 5e-2,
) -> dict[str, Any]:
    """Run packed-layer FI parity on selected layers of a ``PackedMixedCache``."""
    n = len(cache.layers)
    if layer_indices is None:
        layer_indices = (0, n // 2, n - 1) if n >= 3 else tuple(range(n))
    layer_indices = [i for i in layer_indices if 0 <= i < n]
    results = []
    all_pass = True
    any_run = False
    for li in layer_indices:
        r = packed_layer_parity(cache.layers[li], tq=tq, atol=atol)
        r["layer"] = li
        results.append(r)
        if r.get("pass") is False:
            all_pass = False
        if r.get("pass") is True:
            any_run = True
    decision = "SKIP"
    if any(r.get("decision", "").startswith("SKIP") for r in results) and not any_run:
        decision = results[0].get("decision", "SKIP")
    elif any_run and all_pass:
        decision = "PARITY_PASS"
    elif any_run:
        decision = "PARITY_FAIL"
    return {
        "decision": decision,
        "pass": (True if decision == "PARITY_PASS" else False if decision == "PARITY_FAIL" else None),
        "layers": results,
    }


def probe() -> dict[str, Any]:
    """Import + tiny CUDA smoke if package present; never silently claim CUDA."""
    fi = try_import_flashinfer()
    if fi is None:
        return {
            "available": False,
            "decision": "SKIP_NO_PACKAGE",
            "note": (
                "flashinfer not installed; CPU lse_merge_pair / "
                "mixed_attend_kv_multicall remain the correctness oracle. "
                "Install only via uv extra on H200 when wiring kernels."
            ),
        }
    out: dict[str, Any] = {
        "available": True,
        "package": getattr(fi, "__file__", str(fi)),
        "version": getattr(fi, "__version__", None),
    }
    try:
        import torch

        if not torch.cuda.is_available():
            out["decision"] = "IMPORT_OK_NO_CUDA"
            out["note"] = "flashinfer importable but CUDA unavailable in this process"
            return out
        x = torch.zeros(1, device="cuda")
        out["cuda_device"] = torch.cuda.get_device_name(0)
        out["cuda_touch_ok"] = bool(x.numel() == 1)
        out["decision"] = "IMPORT_OK_CUDA_TOUCH"
        out["note"] = (
            "Package present; use attend_pages_flashinfer / "
            "attend_packed_layer_flashinfer for multicall + merge_state."
        )
        out["api"] = [
            "attend_pages_flashinfer",
            "attend_packed_layer_flashinfer",
            "verify_packed_cache_flashinfer",
        ]
        return out
    except Exception as exc:  # noqa: BLE001
        out["decision"] = "IMPORT_OK_CUDA_FAIL"
        out["error"] = f"{type(exc).__name__}: {exc}"
        return out


def status() -> dict[str, Any]:
    ok = flashinfer_available()
    return {
        "name": "flashinfer_multicall",
        "available": ok,
        "allowed_head_dims": list(ALLOWED_HEAD_DIMS),
        "lse_merge": "flashinfer.merge_state",
        "cpu_oracle": "prioritykv.mixed_cache_reference.lse_merge_pair",
        "parity_oracle": "mixed_attend_kv_multicall vs mixed_attend_kv",
        "decision": "WIRED" if ok else "SKIP_NO_PACKAGE",
        "next": (
            "H200 packed mixed BF16/INT4 parity + mixed_kv_run attn_backend=flashinfer"
            if ok
            else "uv sync --extra flashinfer on H200; CPU LSE oracle stays gating"
        ),
    }
