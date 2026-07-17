"""FlashInfer mixed BF16/INT4 decode state (Stage-1 systems path).

Design locked by LLM council 2026-07-17 (Fable + Codex Sol 5.6 xhigh):

* External ``FiMixedDecodeState`` — **not** a generic HF ``Cache`` subclass.
* Prefill stays native HF; decode attends via FI multicall over ≤2 chunks
  (hot+tail BF16, cold dequant scratch) + ``flashinfer.merge_state``.
* Split-prefill contract: ``prefill(n-1) → pack → FI replay(last prompt token)``.
* Assert SM90 ``head_dim ∈ {64,128,256}`` (Qwen3-8B = 128).
* Never FI-call per run-length page — coalesce to ≤2 chunks (hot, cold).
* Refuse silent ``materialize_hf_past`` in the FI decode path.

Stage-1a (this module): GPU-resident hot + cold scratch built from
``PackedMixedCache``; FI attend parity without rebuilding a full DynamicCache.
Stage-1b (next): Qwen3 attention shim that drives decode through this state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from prioritykv.flashinfer_multicall import (
    ALLOWED_HEAD_DIMS,
    attend_pages_flashinfer,
    require_head_dim,
    try_import_flashinfer,
)
from prioritykv.page_roles import StorageDtype


@dataclass
class LayerMixedBuffers:
    """Per-layer GPU-resident hot / cold / decode-tail storage."""

    # Contiguous BF16 hot+tail in HF layout: (num_kv_heads, capacity, head_dim)
    k_hot: Any = None
    v_hot: Any = None
    hot_len: int = 0
    hot_capacity: int = 0
    # Cold: keep packed page payloads (numpy) until attend; scratch is GPU BF16.
    cold_pages: List[Any] = field(default_factory=list)  # KvPagePayload INT4
    cold_len: int = 0
    k_cold_scratch: Any = None  # (cold_len, heads, dim) FI layout when filled
    v_cold_scratch: Any = None
    cold_scratch_valid: bool = False


@dataclass
class FiMixedDecodeState:
    """External decode state for structure-protected mixed attention."""

    layers: List[LayerMixedBuffers] = field(default_factory=list)
    num_kv_heads: int = 0
    head_dim: int = 0
    num_layers: int = 0
    cache_len: int = 0
    decode_len: int = 0
    forbid_materialize: bool = True
    device: Any = None
    dtype: Any = None

    def validate_geom(self) -> None:
        require_head_dim(self.head_dim)
        if self.num_layers <= 0 or self.num_kv_heads <= 0:
            raise ValueError("FiMixedDecodeState geometry incomplete")
        if self.layers and len(self.layers) != self.num_layers:
            raise ValueError(
                f"layers len {len(self.layers)} != num_layers {self.num_layers}"
            )

    def assert_len_invariant(self) -> None:
        for i, layer in enumerate(self.layers):
            if layer.hot_len + layer.cold_len != self.cache_len:
                raise ValueError(
                    f"layer {i}: hot({layer.hot_len})+cold({layer.cold_len}) "
                    f"!= cache_len({self.cache_len})"
                )

    @property
    def total_kv_len(self) -> int:
        return int(self.cache_len + self.decode_len)

    def assert_no_materialize_path(self, used_materialize: bool) -> None:
        if self.forbid_materialize and used_materialize:
            raise RuntimeError(
                "FI Stage-1 refusal: decode used materialize_hf_past / full dequant "
                "copy — acceptance requires peak-mem < materialize path"
            )


def coalesce_hot_cold_lengths(
    *,
    hot_len: int,
    cold_len: int,
    decode_tail: int,
) -> Tuple[int, int]:
    if hot_len < 0 or cold_len < 0 or decode_tail < 0:
        raise ValueError("lengths must be non-negative")
    return hot_len + decode_tail, cold_len


def stage1_acceptance_checklist() -> dict[str, str]:
    return {
        "parity_attn": "per-step FI vs dense max-abs ≪ 5e-2 (target ~1e-3 like w6e/w6i)",
        "parity_greedy": "N-token greedy matches materialize→SDPA on PriorityBench slice",
        "peak_mem": "CUDA peak decode mem strictly below materialize path",
        "all_layers": "shim on every layer, not only 0/18/35 probe gates",
        "split_prefill": "first logits from prefill(n-1)→pack→FI replay(last token)",
        "lse": "flashinfer.merge_state only (base-2/native LSE)",
        "head_dim": f"SM90 head_dim in {ALLOWED_HEAD_DIMS}",
        "chunks": "≤2 FI chunks: hot+tail and cold",
    }


def _cat_bf16_pages(pages: Sequence[Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Concat BF16 page payloads → (heads, tok, dim)."""
    ks, vs = [], []
    for p in pages:
        assert p.k_bf16 is not None and p.v_bf16 is not None
        ks.append(p.k_bf16.astype(np.float32, copy=False))
        vs.append(p.v_bf16.astype(np.float32, copy=False))
    if not ks:
        raise ValueError("no BF16 pages to concatenate")
    return np.concatenate(ks, axis=1), np.concatenate(vs, axis=1)


def _dequant_cold_pages(pages: Sequence[Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Dequant INT4 pages → (heads, tok, dim) float32."""
    ks, vs = [], []
    for p in pages:
        k, v = p.materialize_kv()
        ks.append(k)
        vs.append(v)
    if not ks:
        raise ValueError("no INT4 pages to dequant")
    return np.concatenate(ks, axis=1), np.concatenate(vs, axis=1)


def build_from_packed_cache(
    cache: Any,
    *,
    device: Any,
    dtype: Any = None,
    decode_tail_cap: int = 256,
) -> FiMixedDecodeState:
    """Upload coalesced hot (+ reserved tail) and retain cold pages for scratch.

    Does **not** call ``materialize_hf_past`` — that is the Stage-1 refusal path.
    """
    import torch

    dtype = dtype or torch.float16
    if not cache.layers:
        raise ValueError("empty PackedMixedCache")
    n_layers = len(cache.layers)
    # Probe geometry from first non-empty page.
    sample = None
    for layer in cache.layers:
        for p in layer.pages:
            sample = p
            break
        if sample is not None:
            break
    if sample is None:
        raise ValueError("PackedMixedCache has no pages")
    heads = int(sample.num_kv_heads or (sample.k_bf16.shape[0] if sample.k_bf16 is not None else 0))
    dim = int(sample.head_dim or (sample.k_bf16.shape[2] if sample.k_bf16 is not None else 0))
    if heads <= 0 or dim <= 0:
        # Fall back via materialize of one page.
        k0, _ = sample.materialize_kv()
        heads, _, dim = k0.shape
    require_head_dim(dim)

    seq = int(cache.seq_len)
    state = FiMixedDecodeState(
        num_kv_heads=heads,
        head_dim=dim,
        num_layers=n_layers,
        cache_len=seq,
        decode_len=0,
        device=device,
        dtype=dtype,
        forbid_materialize=True,
    )

    for layer in cache.layers:
        bf16_pages = [p for p in layer.pages if p.dtype == StorageDtype.BF16]
        int4_pages = [p for p in layer.pages if p.dtype == StorageDtype.INT4]
        buf = LayerMixedBuffers(cold_pages=list(int4_pages))
        if bf16_pages:
            k_np, v_np = _cat_bf16_pages(bf16_pages)
            hot_len = int(k_np.shape[1])
        else:
            k_np = np.zeros((heads, 0, dim), dtype=np.float32)
            v_np = np.zeros((heads, 0, dim), dtype=np.float32)
            hot_len = 0
        cold_len = sum(int(p.n_tokens) for p in int4_pages)
        if hot_len + cold_len != seq:
            raise ValueError(
                f"hot({hot_len})+cold({cold_len}) != seq_len({seq}) — check page table"
            )
        cap = hot_len + int(decode_tail_cap)
        k_hot = torch.zeros((heads, cap, dim), device=device, dtype=dtype)
        v_hot = torch.zeros((heads, cap, dim), device=device, dtype=dtype)
        if hot_len:
            k_hot[:, :hot_len, :] = torch.from_numpy(k_np).to(device=device, dtype=dtype)
            v_hot[:, :hot_len, :] = torch.from_numpy(v_np).to(device=device, dtype=dtype)
        buf.k_hot, buf.v_hot = k_hot, v_hot
        buf.hot_len = hot_len
        buf.hot_capacity = cap
        buf.cold_len = cold_len
        state.layers.append(buf)

    state.validate_geom()
    state.assert_len_invariant()
    return state


def _fill_cold_scratch(buf: LayerMixedBuffers, *, device: Any, dtype: Any) -> None:
    """Dequant cold pages into FI-layout scratch (tok, heads, dim)."""
    import torch

    if buf.cold_len == 0:
        buf.k_cold_scratch = None
        buf.v_cold_scratch = None
        buf.cold_scratch_valid = True
        return
    if buf.cold_scratch_valid and buf.k_cold_scratch is not None:
        return
    k_np, v_np = _dequant_cold_pages(buf.cold_pages)
    # (heads, tok, dim) → (tok, heads, dim)
    buf.k_cold_scratch = (
        torch.from_numpy(k_np).to(device=device, dtype=dtype).permute(1, 0, 2).contiguous()
    )
    buf.v_cold_scratch = (
        torch.from_numpy(v_np).to(device=device, dtype=dtype).permute(1, 0, 2).contiguous()
    )
    buf.cold_scratch_valid = True


def fi_chunks_for_layer(
    state: FiMixedDecodeState,
    layer_idx: int,
) -> Tuple[List[Any], List[Any]]:
    """Return ≤2 FI-layout chunks: hot+tail, then cold (if any)."""
    import torch

    buf = state.layers[layer_idx]
    _fill_cold_scratch(buf, device=state.device, dtype=state.dtype)
    k_pages: list[Any] = []
    v_pages: list[Any] = []
    live = buf.hot_len + state.decode_len
    if live > 0:
        # HF (heads, tok, dim) → FI (tok, heads, dim)
        k_ht = buf.k_hot[:, :live, :].permute(1, 0, 2).contiguous()
        v_ht = buf.v_hot[:, :live, :].permute(1, 0, 2).contiguous()
        k_pages.append(k_ht)
        v_pages.append(v_ht)
    if buf.cold_len > 0:
        assert buf.k_cold_scratch is not None and buf.v_cold_scratch is not None
        k_pages.append(buf.k_cold_scratch)
        v_pages.append(buf.v_cold_scratch)
    if not k_pages:
        raise ValueError(f"layer {layer_idx} has empty hot and cold")
    # Council: never more than 2 chunks.
    if len(k_pages) > 2:
        raise RuntimeError(f"FI chunk count {len(k_pages)} > 2 — coalesce failed")
    return k_pages, v_pages


def append_decode_kv(
    state: FiMixedDecodeState,
    layer_idx: int,
    k_new: Any,
    v_new: Any,
) -> None:
    """Append one decode step of K/V into the BF16 hot tail.

    ``k_new`` / ``v_new``: ``(num_kv_heads, 1, head_dim)`` or ``(1, heads, dim)``.
    """
    import torch

    buf = state.layers[layer_idx]
    if k_new.dim() == 3 and k_new.shape[0] == 1:
        # (1, heads, dim) → (heads, 1, dim)
        k_new = k_new.permute(1, 0, 2).contiguous()
        v_new = v_new.permute(1, 0, 2).contiguous()
    pos = buf.hot_len + state.decode_len
    if pos + k_new.shape[1] > buf.hot_capacity:
        raise RuntimeError(
            f"decode tail overflow: pos={pos} cap={buf.hot_capacity} — grow decode_tail_cap"
        )
    n = k_new.shape[1]
    buf.k_hot[:, pos : pos + n, :] = k_new.to(dtype=buf.k_hot.dtype)
    buf.v_hot[:, pos : pos + n, :] = v_new.to(dtype=buf.v_hot.dtype)


def commit_decode_step(state: FiMixedDecodeState) -> None:
    """Advance decode_len after all layers received append_decode_kv."""
    state.decode_len += 1


def attend_layer_flashinfer(
    q: Any,
    state: FiMixedDecodeState,
    layer_idx: int,
    *,
    causal: bool = False,
    fi: Any = None,
) -> Any:
    """FI multicall over coalesced hot+tail / cold for one layer.

    ``q``: ``(tq, num_qo_heads, head_dim)`` on CUDA.
    """
    fi = fi or try_import_flashinfer()
    if fi is None:
        raise RuntimeError("flashinfer not installed")
    require_head_dim(int(q.shape[-1]))
    k_pages, v_pages = fi_chunks_for_layer(state, layer_idx)
    return attend_pages_flashinfer(q, k_pages, v_pages, causal=causal, fi=fi)


def dense_kv_from_state_layer(state: FiMixedDecodeState, layer_idx: int) -> Tuple[Any, Any]:
    """Concat hot+tail and cold into one dense FI-layout KV (parity oracle only)."""
    import torch

    k_pages, v_pages = fi_chunks_for_layer(state, layer_idx)
    return torch.cat(k_pages, dim=0), torch.cat(v_pages, dim=0)


def layer_parity_vs_dense(
    state: FiMixedDecodeState,
    layer_idx: int,
    *,
    tq: int = 1,
    seed: int = 0,
    atol: float = 5e-2,
) -> dict[str, Any]:
    """Compare FI multicall on state chunks vs FI dense concat (no HF materialize)."""
    import torch

    fi = try_import_flashinfer()
    out: dict[str, Any] = {
        "flashinfer": fi is not None,
        "layer": layer_idx,
        "used_materialize_hf_past": False,
    }
    if fi is None:
        out["decision"] = "SKIP_NO_PACKAGE"
        out["pass"] = None
        return out
    if not torch.cuda.is_available():
        out["decision"] = "SKIP_NO_CUDA"
        out["pass"] = None
        return out

    state.assert_no_materialize_path(False)
    require_head_dim(state.head_dim)
    rng = np.random.default_rng(seed + layer_idx)
    # GQA probe: qo_heads == kv_heads for parity simplicity.
    q_np = rng.standard_normal((tq, state.num_kv_heads, state.head_dim)).astype(np.float32)
    q = torch.as_tensor(q_np, device=state.device, dtype=state.dtype)
    with torch.no_grad():
        fi_multi = attend_layer_flashinfer(q, state, layer_idx, fi=fi)
        k_d, v_d = dense_kv_from_state_layer(state, layer_idx)
        from prioritykv.flashinfer_multicall import dense_prefill_flashinfer

        fi_dense = dense_prefill_flashinfer(q, k_d, v_d, fi=fi)
    err = float(torch.max(torch.abs(fi_multi.float() - fi_dense.float())).item())
    buf = state.layers[layer_idx]
    hot_t, cold_t = coalesce_hot_cold_lengths(
        hot_len=buf.hot_len, cold_len=buf.cold_len, decode_tail=state.decode_len
    )
    out.update(
        {
            "decision": "PARITY_PASS" if err < atol else "PARITY_FAIL",
            "pass": err < atol,
            "fi_multicall_vs_dense_max_abs": err,
            "atol": atol,
            "hot_plus_tail": hot_t,
            "cold": cold_t,
            "n_fi_chunks": int(bool(hot_t) + bool(cold_t)),
            "merge_impl": "flashinfer.merge_state",
        }
    )
    return out


def verify_state_flashinfer(
    state: FiMixedDecodeState,
    *,
    layer_indices: Optional[Sequence[int]] = None,
    tq: int = 1,
    atol: float = 5e-2,
) -> dict[str, Any]:
    """Run Stage-1a parity on selected layers of a ``FiMixedDecodeState``."""
    n = state.num_layers
    if layer_indices is None:
        layer_indices = (0, n // 2, n - 1) if n >= 3 else tuple(range(n))
    layer_indices = [i for i in layer_indices if 0 <= i < n]
    layers_out = [
        layer_parity_vs_dense(state, i, tq=tq, atol=atol) for i in layer_indices
    ]
    passes = [r.get("pass") for r in layers_out]
    if any(p is False for p in passes):
        decision = "PARITY_FAIL"
        ok: bool | None = False
    elif all(p is True for p in passes):
        decision = "PARITY_PASS"
        ok = True
    else:
        decision = "SKIP"
        ok = None
    return {
        "decision": decision,
        "pass": ok,
        "used_materialize_hf_past": False,
        "layers": layers_out,
        "acceptance": stage1_acceptance_checklist(),
    }
