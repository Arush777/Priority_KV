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
from prioritykv.int4_path import PackedInt4Page, append_quantize, append_quantize_torch
from prioritykv.page_manager import Page, PageManager, PageManagerConfig
from prioritykv.page_roles import PageRole, StorageDtype


def _is_torch(x: Any) -> bool:
    try:
        import torch

        return torch.is_tensor(x)
    except Exception:
        return False


def _nbytes(x: Any) -> int:
    if _is_torch(x):
        return int(x.numel() * x.element_size())
    return int(x.nbytes)


@dataclass
class KvPagePayload:
    """One physical page of K/V for a single transformer layer."""

    meta: Page
    dtype: StorageDtype
    n_tokens: int
    num_kv_heads: int = 0
    head_dim: int = 0
    k_bf16: Optional[Any] = None  # (num_kv_heads, n_tokens, head_dim) np or torch
    v_bf16: Optional[Any] = None
    k_packed: Optional[PackedInt4Page] = None
    v_packed: Optional[PackedInt4Page] = None

    def storage_payload_bytes(self) -> int:
        """Bytes actually held in this page (K+V payloads only)."""
        if self.dtype == StorageDtype.BF16:
            assert self.k_bf16 is not None and self.v_bf16 is not None
            return _nbytes(self.k_bf16) + _nbytes(self.v_bf16)
        assert self.k_packed is not None and self.v_packed is not None
        return self.k_packed.payload_bytes() + self.v_packed.payload_bytes()

    def materialize_kv(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return float32 K/V shaped (num_kv_heads, n_tokens, head_dim) on host."""
        k, v = self.materialize_kv_device(device="cpu", dtype=None)
        if _is_torch(k):
            return (
                k.detach().float().cpu().numpy(),
                v.detach().float().cpu().numpy(),
            )
        return (
            np.asarray(k, dtype=np.float32),
            np.asarray(v, dtype=np.float32),
        )

    def materialize_kv_device(
        self, *, device: Any = None, dtype: Any = None
    ) -> Tuple[Any, Any]:
        """Return K/V (heads, tok, dim) on ``device`` (torch) or host numpy."""
        if self.dtype == StorageDtype.BF16:
            assert self.k_bf16 is not None and self.v_bf16 is not None
            if _is_torch(self.k_bf16):
                k = self.k_bf16
                v = self.v_bf16
                if device is not None:
                    k = k.to(device=device, dtype=dtype or k.dtype)
                    v = v.to(device=device, dtype=dtype or v.dtype)
                return k, v
            k = self.k_bf16.astype(np.float32, copy=False)
            v = self.v_bf16.astype(np.float32, copy=False)
            if device is None or str(device) == "cpu":
                return k, v
            import torch

            dt = dtype or torch.float32
            return (
                torch.from_numpy(np.ascontiguousarray(k)).to(device=device, dtype=dt),
                torch.from_numpy(np.ascontiguousarray(v)).to(device=device, dtype=dt),
            )
        assert self.k_packed is not None and self.v_packed is not None
        h, d = self.num_kv_heads, self.head_dim
        k_raw = self.k_packed.dequant()
        v_raw = self.v_packed.dequant()
        if _is_torch(k_raw):
            k = k_raw.reshape(h, d, self.n_tokens).permute(0, 2, 1).contiguous()
            v = v_raw.reshape(h, d, self.n_tokens).permute(0, 2, 1).contiguous()
            if device is not None:
                k = k.to(device=device, dtype=dtype or k.dtype)
                v = v.to(device=device, dtype=dtype or v.dtype)
            return k, v
        k = np.asarray(k_raw, dtype=np.float32).reshape(h, d, self.n_tokens)
        v = np.asarray(v_raw, dtype=np.float32).reshape(h, d, self.n_tokens)
        k = np.transpose(k, (0, 2, 1))
        v = np.transpose(v, (0, 2, 1))
        if device is None or str(device) == "cpu":
            return k, v
        import torch

        dt = dtype or torch.float32
        return (
            torch.from_numpy(np.ascontiguousarray(k)).to(device=device, dtype=dt),
            torch.from_numpy(np.ascontiguousarray(v)).to(device=device, dtype=dt),
        )

    def demote_to_int4(self, cfg: Int4KvConfig) -> None:
        """Pack current BF16 payload → INT4; drop BF16 arrays."""
        if self.dtype == StorageDtype.INT4 and self.k_packed is not None:
            return
        assert self.k_bf16 is not None and self.v_bf16 is not None
        self.num_kv_heads = int(self.k_bf16.shape[0])
        self.head_dim = int(self.k_bf16.shape[2])
        # Match legacy layout: (h, T, D) → (h, D, T) → reshape(h*T, D).
        # Names mirror original numpy path (t=head_dim, d=n_tokens).
        if _is_torch(self.k_bf16):
            k_tok = self.k_bf16.float().permute(0, 2, 1).contiguous()
            v_tok = self.v_bf16.float().permute(0, 2, 1).contiguous()
            h, t, d = k_tok.shape
            self.k_packed = append_quantize_torch(k_tok.reshape(h * d, t), cfg=cfg)
            self.v_packed = append_quantize_torch(v_tok.reshape(h * d, t), cfg=cfg)
        else:
            k_tok = np.transpose(self.k_bf16.astype(np.float32), (0, 2, 1))
            v_tok = np.transpose(self.v_bf16.astype(np.float32), (0, 2, 1))
            h, t, d = k_tok.shape
            self.k_packed = append_quantize(k_tok.reshape(h * d, t), cfg=cfg)
            self.v_packed = append_quantize(v_tok.reshape(h * d, t), cfg=cfg)
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


def _layer_kv_torch_raw(past: Any, layer_idx: int) -> Tuple[Any, Any]:
    """Extract layer K/V tensors (batch stripped) without host copy."""
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
                return kt[0].detach(), vt[0].detach()
    kc = getattr(past, "key_cache", None)
    vc = getattr(past, "value_cache", None)
    if isinstance(kc, list) and isinstance(vc, list):
        kt, vt = kc[layer_idx], vc[layer_idx]
        if torch.is_tensor(kt) and torch.is_tensor(vt):
            return kt.detach(), vt.detach()
    raise TypeError(f"unsupported past_key_values for layer {layer_idx}: {type(past)}")


def _layer_tensors_from_hf(past: Any, layer_idx: int) -> Tuple[np.ndarray, np.ndarray]:
    """Extract layer K/V as numpy (heads, seq, dim) from HF DynamicCache."""
    kt, vt = _layer_kv_torch_raw(past, layer_idx)
    return kt.float().cpu().numpy(), vt.float().cpu().numpy()


def _layer_tensors_from_hf_torch(past: Any, layer_idx: int) -> Tuple[Any, Any]:
    """Extract layer K/V as CUDA/CPU torch (heads, seq, dim) — no host roundtrip."""
    kt, vt = _layer_kv_torch_raw(past, layer_idx)
    # HF may be (batch, heads, seq, dim) already stripped, or (heads, seq, dim).
    if kt.dim() == 4:
        kt, vt = kt[0], vt[0]
    return kt.contiguous(), vt.contiguous()


def _hf_past_on_cuda(past: Any) -> bool:
    try:
        kt, _ = _layer_kv_torch_raw(past, 0)
        return bool(kt.is_cuda)
    except Exception:
        return False


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
    prefer_gpu: bool = True,
) -> PackedMixedCache:
    """Split an HF prefill cache into pages and apply manager dtypes.

    M2: when HF past lives on CUDA, keep slices/quant on device (no D2H).
    """
    cfg = int4_cfg or Int4KvConfig()
    n_layers = _num_hf_layers(past)
    use_gpu = bool(prefer_gpu and _hf_past_on_cuda(past))
    if use_gpu:
        k0, _v0 = _layer_tensors_from_hf_torch(past, 0)
    else:
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
        if use_gpu:
            k_full, v_full = _layer_tensors_from_hf_torch(past, li)
        else:
            k_full, v_full = _layer_tensors_from_hf(past, li)
        if int(k_full.shape[1]) != page_manager.seq_len:
            raise ValueError(
                f"layer {li} seq {k_full.shape[1]} != page table {page_manager.seq_len}"
            )
        layer = PackedMixedLayer()
        for meta in page_manager.pages:
            if use_gpu:
                k_slice = k_full[:, meta.start_token : meta.end_token, :].contiguous()
                v_slice = v_full[:, meta.start_token : meta.end_token, :].contiguous()
                # Stay in model dtype (bf16) on device until demote.
                payload = KvPagePayload(
                    meta=meta,
                    dtype=meta.dtype,
                    n_tokens=meta.n_tokens,
                    num_kv_heads=int(k_slice.shape[0]),
                    head_dim=int(k_slice.shape[2]),
                    k_bf16=k_slice,
                    v_bf16=v_slice,
                )
            else:
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


def build_from_hf_prefill_batched(
    past: Any,
    int4_mask: np.ndarray,
    *,
    geom: ModelKvGeom = QWEN3_8B_KV,
    int4_cfg: Optional[Int4KvConfig] = None,
) -> PackedMixedCache:
    """M2b: one gather + one INT4 quant per layer (no per-page Python loop).

    Hot tokens are gathered into a single BF16 page; cold tokens into a single
    INT4 page. Token *order within each chunk* is gather order (fine for decode
    attention over already-RoPE'd keys). Byte accounting uses the two pages.
    """
    import torch

    cfg = int4_cfg or Int4KvConfig()
    mask_np = np.asarray(int4_mask, dtype=bool).reshape(-1)
    n = int(mask_np.size)
    hot_n = int((~mask_np).sum())
    cold_n = int(mask_np.sum())
    if hot_n + cold_n != n:
        raise ValueError("int4_mask must cover every token")

    n_layers = _num_hf_layers(past)
    k0, _ = _layer_tensors_from_hf_torch(past, 0)
    if k0.dim() == 4:
        k0 = k0[0]
    geom = ModelKvGeom(
        num_layers=n_layers,
        num_kv_heads=int(k0.shape[0]),
        head_dim=int(k0.shape[2]),
    )
    pm = PageManager(
        PageManagerConfig(page_tokens=max(n, 1), budget_frac=1.0, geom=geom)
    )
    pm.pages.clear()
    pm._next_id = 0
    if hot_n:
        pm.pages.append(
            Page(
                page_id=pm._next_id,
                start_token=0,
                n_tokens=hot_n,
                role=PageRole.RECENT,
                dtype=StorageDtype.BF16,
            )
        )
        pm._next_id += 1
    if cold_n:
        pm.pages.append(
            Page(
                page_id=pm._next_id,
                start_token=hot_n,
                n_tokens=cold_n,
                role=PageRole.FILLER,
                dtype=StorageDtype.INT4,
            )
        )
        pm._next_id += 1

    cache = PackedMixedCache(page_manager=pm, geom=geom, int4_cfg=cfg)
    cache._head_shape = (geom.num_kv_heads, geom.head_dim)

    # Mask on same device as KV for index_select.
    device = k0.device
    mask_t = torch.as_tensor(mask_np, device=device)
    hot_idx = torch.nonzero(~mask_t, as_tuple=False).flatten()
    cold_idx = torch.nonzero(mask_t, as_tuple=False).flatten()

    for li in range(n_layers):
        k_full, v_full = _layer_tensors_from_hf_torch(past, li)
        if k_full.dim() == 4:
            k_full, v_full = k_full[0], v_full[0]
        if int(k_full.shape[1]) != n:
            raise ValueError(
                f"layer {li} seq {k_full.shape[1]} != mask len {n}"
            )
        layer = PackedMixedLayer()
        page_i = 0
        if hot_n:
            meta = pm.pages[page_i]
            page_i += 1
            k_hot = k_full.index_select(1, hot_idx).contiguous()
            v_hot = v_full.index_select(1, hot_idx).contiguous()
            layer.pages.append(
                KvPagePayload(
                    meta=meta,
                    dtype=StorageDtype.BF16,
                    n_tokens=hot_n,
                    num_kv_heads=int(k_hot.shape[0]),
                    head_dim=int(k_hot.shape[2]),
                    k_bf16=k_hot,
                    v_bf16=v_hot,
                )
            )
        if cold_n:
            meta = pm.pages[page_i]
            k_cold = k_full.index_select(1, cold_idx).contiguous()
            v_cold = v_full.index_select(1, cold_idx).contiguous()
            payload = KvPagePayload(
                meta=meta,
                dtype=StorageDtype.BF16,
                n_tokens=cold_n,
                num_kv_heads=int(k_cold.shape[0]),
                head_dim=int(k_cold.shape[2]),
                k_bf16=k_cold,
                v_bf16=v_cold,
            )
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
