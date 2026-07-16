"""True packed BF16/INT4 mixed KV cache (D3 systems path).

``PageManager`` owns dtype/role metadata; this module owns the actual bytes:
BF16 tensors for hot pages, ``PackedInt4Page`` payloads for demoted pages.

Quality-forward mixed runs still round-trip BF16 tensors in-place. This is the
storage layer that saves bytes and will feed FlashInfer multicall next.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from prioritykv.byte_model import (
    PHYSICAL_PAGE_TOKENS,
    ModelKvGeom,
    QWEN3_8B_KV,
    realized_bytes,
)
from prioritykv.int4_kv import Int4KvConfig
from prioritykv.int4_path import PackedInt4Page, append_quantize
from prioritykv.page_manager import Page, PageManager, PageManagerConfig
from prioritykv.page_roles import PageRole, StorageDtype


@dataclass
class KvPagePayload:
    """One physical page of K/V for a single transformer layer."""

    meta: Page
    dtype: StorageDtype
    n_tokens: int
    num_kv_heads: int = 0
    head_dim: int = 0
    k_bf16: Optional[np.ndarray] = None  # (num_kv_heads, n_tokens, head_dim)
    v_bf16: Optional[np.ndarray] = None
    k_packed: Optional[PackedInt4Page] = None
    v_packed: Optional[PackedInt4Page] = None

    def storage_payload_bytes(self) -> int:
        """Bytes actually held in this page (K+V payloads only)."""
        if self.dtype == StorageDtype.BF16:
            assert self.k_bf16 is not None and self.v_bf16 is not None
            return int(self.k_bf16.nbytes + self.v_bf16.nbytes)
        assert self.k_packed is not None and self.v_packed is not None
        return self.k_packed.payload_bytes() + self.v_packed.payload_bytes()

    def materialize_kv(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return float32 K/V shaped (num_kv_heads, n_tokens, head_dim)."""
        if self.dtype == StorageDtype.BF16:
            assert self.k_bf16 is not None and self.v_bf16 is not None
            return (
                self.k_bf16.astype(np.float32, copy=False),
                self.v_bf16.astype(np.float32, copy=False),
            )
        assert self.k_packed is not None and self.v_packed is not None
        h, d = self.num_kv_heads, self.head_dim
        k = self.k_packed.dequant().astype(np.float32).reshape(h, d, self.n_tokens)
        v = self.v_packed.dequant().astype(np.float32).reshape(h, d, self.n_tokens)
        return np.transpose(k, (0, 2, 1)), np.transpose(v, (0, 2, 1))

    def demote_to_int4(self, cfg: Int4KvConfig) -> None:
        """Pack current BF16 payload → INT4; drop BF16 arrays."""
        if self.dtype == StorageDtype.INT4 and self.k_packed is not None:
            return
        assert self.k_bf16 is not None and self.v_bf16 is not None
        self.num_kv_heads = int(self.k_bf16.shape[0])
        self.head_dim = int(self.k_bf16.shape[2])
        # Quantize along token axis: (h, d, t) layout for groupwise INT4.
        k_tok = np.transpose(self.k_bf16.astype(np.float32), (0, 2, 1))
        v_tok = np.transpose(self.v_bf16.astype(np.float32), (0, 2, 1))
        h, t, d = k_tok.shape
        k_flat = k_tok.reshape(h * d, t)
        v_flat = v_tok.reshape(h * d, t)
        self.k_packed = append_quantize(k_flat, cfg=cfg)
        self.v_packed = append_quantize(v_flat, cfg=cfg)
        self.k_bf16 = None
        self.v_bf16 = None
        self.dtype = StorageDtype.INT4
        self.meta.dtype = StorageDtype.INT4


@dataclass
class PackedMixedLayer:
    """All pages for one layer, aligned 1:1 with ``PageManager.pages``."""

    pages: List[KvPagePayload] = field(default_factory=list)

    def materialize(self) -> Tuple[np.ndarray, np.ndarray]:
        """Concat pages → (num_kv_heads, seq_len, head_dim) K and V."""
        if not self.pages:
            raise ValueError("empty layer")
        ks, vs = [], []
        for p in self.pages:
            k, v = p.materialize_kv()
            ks.append(k)
            vs.append(v)
        return np.concatenate(ks, axis=1), np.concatenate(vs, axis=1)

    def payload_bytes(self) -> int:
        return sum(p.storage_payload_bytes() for p in self.pages)


@dataclass
class PackedMixedCache:
    """Multi-layer packed cache driven by a shared ``PageManager``."""

    page_manager: PageManager
    layers: List[PackedMixedLayer] = field(default_factory=list)
    geom: ModelKvGeom = QWEN3_8B_KV
    int4_cfg: Int4KvConfig = field(default_factory=Int4KvConfig)
    _head_shape: Optional[Tuple[int, int]] = None  # (num_kv_heads, head_dim)

    @property
    def seq_len(self) -> int:
        return self.page_manager.seq_len

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    def sync_dtypes_from_manager(self) -> int:
        """Demote pages whose metadata says INT4. Returns pages demoted."""
        demoted = 0
        for layer in self.layers:
            for payload in layer.pages:
                if (
                    payload.meta.dtype == StorageDtype.INT4
                    and payload.dtype == StorageDtype.BF16
                ):
                    payload.demote_to_int4(self.int4_cfg)
                    demoted += 1
        return demoted

    def dtype_token_counts(self) -> dict[StorageDtype, int]:
        return self.page_manager.dtype_token_counts()

    def payload_bytes(self) -> int:
        """Sum of actual K/V payloads across all layers (no page-table overhead)."""
        return sum(layer.payload_bytes() for layer in self.layers)

    def realized_bytes(self) -> int:
        """Plan §1 byte model (layers + page-table overhead)."""
        c = self.dtype_token_counts()
        return realized_bytes(
            num_bf16_tokens=c[StorageDtype.BF16],
            num_int4_tokens=c[StorageDtype.INT4],
            num_kv_heads=self.geom.num_kv_heads,
            head_dim=self.geom.head_dim,
            page_tokens=self.page_manager.config.page_tokens,
            num_layers=self.geom.num_layers,
        )

    def fullkv_bf16_bytes(self) -> int:
        c = self.dtype_token_counts()
        n = c[StorageDtype.BF16] + c[StorageDtype.INT4]
        return realized_bytes(
            num_bf16_tokens=n,
            num_int4_tokens=0,
            num_kv_heads=self.geom.num_kv_heads,
            head_dim=self.geom.head_dim,
            page_tokens=self.page_manager.config.page_tokens,
            num_layers=self.geom.num_layers,
        )

    def compression_ratio(self) -> float:
        full = self.fullkv_bf16_bytes()
        return self.realized_bytes() / full if full else 1.0

    def check_invariants(self) -> List[str]:
        errs = list(self.page_manager.check_invariants())
        if len(self.layers) != self.geom.num_layers:
            errs.append(
                f"layer count {len(self.layers)} != geom {self.geom.num_layers}"
            )
        for li, layer in enumerate(self.layers):
            if len(layer.pages) != len(self.page_manager.pages):
                errs.append(
                    f"layer {li} page count {len(layer.pages)} != "
                    f"manager {len(self.page_manager.pages)}"
                )
            for pi, (payload, meta) in enumerate(
                zip(layer.pages, self.page_manager.pages, strict=True)
            ):
                if payload.meta.page_id != meta.page_id:
                    errs.append(f"L{li}P{pi} page_id mismatch")
                if payload.n_tokens != meta.n_tokens:
                    errs.append(f"L{li}P{pi} n_tokens mismatch")
                if payload.dtype != meta.dtype:
                    errs.append(
                        f"L{li}P{pi} dtype {payload.dtype} != meta {meta.dtype}"
                    )
                if payload.dtype == StorageDtype.BF16:
                    if payload.k_bf16 is None or payload.v_bf16 is None:
                        errs.append(f"L{li}P{pi} BF16 missing tensors")
                elif payload.k_packed is None or payload.v_packed is None:
                    errs.append(f"L{li}P{pi} INT4 missing packed")
        return errs


def _slice_page_kv(
    k: np.ndarray, v: np.ndarray, start: int, end: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Slice (heads, seq, dim) arrays along seq for [start, end)."""
    return k[:, start:end, :].copy(), v[:, start:end, :].copy()


def _layer_tensors_from_hf(past: Any, layer_idx: int) -> Tuple[np.ndarray, np.ndarray]:
    """Extract layer K/V as numpy (heads, seq, dim) from HF DynamicCache."""
    import torch

    layers = getattr(past, "layers", None)
    if layers is not None:
        layer = layers[layer_idx]
        for attr_k, attr_v in (
            ("keys", "values"),
            ("key_cache", "value_cache"),
            ("key", "value"),
        ):
            kt = getattr(layer, attr_k, None)
            vt = getattr(layer, attr_v, None)
            if torch.is_tensor(kt) and torch.is_tensor(vt):
                return (
                    kt[0].detach().float().cpu().numpy(),
                    vt[0].detach().float().cpu().numpy(),
                )
    kc = getattr(past, "key_cache", None)
    vc = getattr(past, "value_cache", None)
    if isinstance(kc, list) and isinstance(vc, list):
        kt, vt = kc[layer_idx], vc[layer_idx]
        if torch.is_tensor(kt) and torch.is_tensor(vt):
            return (
                kt[0].detach().float().cpu().numpy(),
                vt[0].detach().float().cpu().numpy(),
            )
    raise TypeError(f"unsupported past_key_values for layer {layer_idx}: {type(past)}")


def _num_hf_layers(past: Any) -> int:
    layers = getattr(past, "layers", None)
    if layers is not None:
        return len(layers)
    kc = getattr(past, "key_cache", None)
    if isinstance(kc, list):
        return len(kc)
    raise TypeError(f"unsupported past_key_values: {type(past)}")


def page_manager_from_int4_mask(
    roles: Sequence[PageRole],
    int4_mask: np.ndarray,
    *,
    page_tokens: int = PHYSICAL_PAGE_TOKENS,
    geom: ModelKvGeom = QWEN3_8B_KV,
) -> PageManager:
    """Build a page table that preserves the per-token INT4 mask exactly.

    Contiguous same-dtype runs are packed into pages of at most ``page_tokens``.
    Page role is majority-vote within the page (ties prefer protected roles).
    """
    n = int(len(int4_mask))
    if len(roles) != n:
        raise ValueError(f"roles len {len(roles)} != mask len {n}")
    pm = PageManager(
        PageManagerConfig(page_tokens=page_tokens, budget_frac=1.0, geom=geom)
    )
    pm.pages.clear()
    pm._next_id = 0
    i = 0
    while i < n:
        want_int4 = bool(int4_mask[i])
        end = i + 1
        while (
            end < n
            and bool(int4_mask[end]) == want_int4
            and (end - i) < page_tokens
        ):
            end += 1
        chunk_roles = list(roles[i:end])
        role = _majority_role(chunk_roles)
        pm.pages.append(
            Page(
                page_id=pm._next_id,
                start_token=i,
                n_tokens=end - i,
                role=role,
                dtype=StorageDtype.INT4 if want_int4 else StorageDtype.BF16,
            )
        )
        pm._next_id += 1
        i = end
    return pm


def _majority_role(chunk: Sequence[PageRole]) -> PageRole:
    from prioritykv.page_roles import HARD_PROTECTED_ROLES, PROTECTED_ROLES

    counts: dict[PageRole, int] = {}
    for r in chunk:
        counts[r] = counts.get(r, 0) + 1

    def key(item: tuple[PageRole, int]) -> tuple:
        role, c = item
        return (
            c,
            1 if role in PROTECTED_ROLES else 0,
            1 if role in HARD_PROTECTED_ROLES else 0,
        )

    return max(counts.items(), key=key)[0]


def build_from_hf_prefill(
    past: Any,
    page_manager: PageManager,
    *,
    geom: ModelKvGeom = QWEN3_8B_KV,
    int4_cfg: Optional[Int4KvConfig] = None,
) -> PackedMixedCache:
    """Split an HF prefill cache into pages and apply manager dtypes."""
    cfg = int4_cfg or Int4KvConfig()
    n_layers = _num_hf_layers(past)
    k0, _v0 = _layer_tensors_from_hf(past, 0)
    geom = ModelKvGeom(
        num_layers=n_layers,
        num_kv_heads=int(k0.shape[0]),
        head_dim=int(k0.shape[2]),
    )
    # Keep page-manager geom in sync for byte accounting.
    page_manager.config.geom = geom
    cache = PackedMixedCache(page_manager=page_manager, geom=geom, int4_cfg=cfg)
    cache._head_shape = (geom.num_kv_heads, geom.head_dim)
    for li in range(n_layers):
        k_full, v_full = _layer_tensors_from_hf(past, li)
        if k_full.shape[1] != page_manager.seq_len:
            raise ValueError(
                f"layer {li} seq {k_full.shape[1]} != page table {page_manager.seq_len}"
            )
        layer = PackedMixedLayer()
        for meta in page_manager.pages:
            k_slice, v_slice = _slice_page_kv(
                k_full, v_full, meta.start_token, meta.end_token
            )
            payload = KvPagePayload(
                meta=meta,
                dtype=meta.dtype,
                n_tokens=meta.n_tokens,
                num_kv_heads=int(k_slice.shape[0]),
                head_dim=int(k_slice.shape[2]),
                k_bf16=k_slice.astype(np.float16),
                v_bf16=v_slice.astype(np.float16),
            )
            if meta.dtype == StorageDtype.INT4:
                payload.demote_to_int4(cfg)
            layer.pages.append(payload)
        cache.layers.append(layer)
    return cache


def apply_packed_int4_to_hf_past(
    past: Any,
    roles: Sequence[PageRole],
    int4_mask: np.ndarray,
    *,
    int4_cfg: Optional[Int4KvConfig] = None,
    device: Any = "cpu",
    dtype: Any = None,
) -> tuple[Any, PackedMixedCache]:
    """Pack demoted positions into INT4 pages, then rebuild HF past (dequant path).

    Returns ``(past_key_values, packed_cache)``. The packed cache retains true
    INT4 payloads for byte accounting; the returned past is dequantized so the
    Transformers SDPA decode path can continue until FlashInfer is wired.
    """
    cfg = int4_cfg or Int4KvConfig()
    pm = page_manager_from_int4_mask(roles, int4_mask)
    cache = build_from_hf_prefill(past, pm, int4_cfg=cfg)
    past_out = materialize_hf_past(cache, device=device, dtype=dtype)
    return past_out, cache


def materialize_hf_past(
    cache: PackedMixedCache,
    *,
    device: str = "cpu",
    dtype: Any = None,
) -> Any:
    """Rebuild HF-compatible past_key_values from packed storage (dequant path)."""
    import torch

    dtype = dtype or torch.bfloat16
    layer_objs = []
    for layer in cache.layers:
        k, v = layer.materialize()
        kt = torch.from_numpy(k).to(device=device, dtype=dtype).unsqueeze(0)
        vt = torch.from_numpy(v).to(device=device, dtype=dtype).unsqueeze(0)
        layer_objs.append((kt, vt))

    try:
        from transformers.cache_utils import DynamicCache

        dc = DynamicCache()
        for i, (kt, vt) in enumerate(layer_objs):
            dc.update(kt, vt, layer_idx=i)
        return dc
    except Exception:
        keys = [t[0] for t in layer_objs]
        vals = [t[1] for t in layer_objs]
        return type("LegacyPast", (), {"key_cache": keys, "value_cache": vals})()


def ingest_synthetic_layer(
    k: np.ndarray,
    v: np.ndarray,
    page_manager: PageManager,
    *,
    int4_cfg: Optional[Int4KvConfig] = None,
) -> PackedMixedLayer:
    """Build one layer from numpy (heads, seq, dim) — for CPU tests."""
    cfg = int4_cfg or Int4KvConfig()
    layer = PackedMixedLayer()
    for meta in page_manager.pages:
        k_slice, v_slice = _slice_page_kv(k, v, meta.start_token, meta.end_token)
        payload = KvPagePayload(
            meta=meta,
            dtype=meta.dtype,
            n_tokens=meta.n_tokens,
            num_kv_heads=int(k_slice.shape[0]),
            head_dim=int(k_slice.shape[2]),
            k_bf16=k_slice.astype(np.float16),
            v_bf16=v_slice.astype(np.float16),
        )
        if meta.dtype == StorageDtype.INT4:
            payload.demote_to_int4(cfg)
        layer.pages.append(payload)
    return layer
