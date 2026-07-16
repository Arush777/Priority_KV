"""Tests for true packed BF16/INT4 mixed cache."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from prioritykv.byte_model import QWEN3_8B_KV
from prioritykv.int4_kv import Int4KvConfig
from prioritykv.packed_mixed_cache import (
    PackedMixedCache,
    ingest_synthetic_layer,
)
from prioritykv.page_manager import Page, PageManager, PageManagerConfig
from prioritykv.page_roles import PageRole, StorageDtype


HEADS = QWEN3_8B_KV.num_kv_heads
DIM = QWEN3_8B_KV.head_dim
ONE_LAYER = replace(QWEN3_8B_KV, num_layers=1)


def _synthetic_kv(seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    k = rng.standard_normal((HEADS, seq_len, DIM)).astype(np.float32)
    v = rng.standard_normal((HEADS, seq_len, DIM)).astype(np.float32)
    return k, v


def _manual_manager(
    n_tokens: int,
    *,
    int4_page_ids: set[int] | None = None,
    budget_frac: float = 1.0,
) -> PageManager:
    """Deterministic page table for unit tests (one token per page slot for simplicity)."""
    int4_page_ids = int4_page_ids or set()
    pm = PageManager(
        PageManagerConfig(budget_frac=budget_frac, geom=ONE_LAYER, page_tokens=16)
    )
    pm.pages.clear()
    pid = 0
    for start in range(0, n_tokens, 16):
        n = min(16, n_tokens - start)
        dtype = (
            StorageDtype.INT4 if pid in int4_page_ids else StorageDtype.BF16
        )
        pm.pages.append(
            Page(
                page_id=pid,
                start_token=start,
                n_tokens=n,
                role=PageRole.FILLER,
                dtype=dtype,
            )
        )
        pid += 1
    return pm


def test_int4_payload_smaller_than_bf16_for_same_page():
    pm = _manual_manager(64, int4_page_ids=set())
    k, v = _synthetic_kv(64)
    layer_bf16 = ingest_synthetic_layer(k, v, pm)
    bf16_bytes = layer_bf16.pages[0].storage_payload_bytes()
    payload = layer_bf16.pages[0]
    payload.demote_to_int4(Int4KvConfig())
    int4_bytes = payload.storage_payload_bytes()
    assert int4_bytes < bf16_bytes


def test_materialize_roundtrip_all_bf16():
    pm = _manual_manager(80, int4_page_ids=set())
    k, v = _synthetic_kv(80)
    layer = ingest_synthetic_layer(k, v, pm)
    km, vm = layer.materialize()
    assert km.shape == k.shape
    assert np.allclose(km, k, rtol=0.02, atol=0.02)
    assert np.allclose(vm, v, rtol=0.02, atol=0.02)


def test_sync_demotes_when_manager_says_int4():
    pm = _manual_manager(128, int4_page_ids=set(), budget_frac=1.0)
    k, v = _synthetic_kv(128)
    layer = ingest_synthetic_layer(k, v, pm)
    # Manager demotes half the pages under a tight budget.
    for p in pm.pages[4:]:
        p.dtype = StorageDtype.INT4
    cache = PackedMixedCache(page_manager=pm, layers=[layer], geom=ONE_LAYER)
    demoted = cache.sync_dtypes_from_manager()
    assert demoted == len(pm.pages) - 4
    assert cache.check_invariants() == []
    for payload in layer.pages[4:]:
        assert payload.k_bf16 is None
        assert payload.k_packed is not None


def test_realized_bytes_matches_byte_model_one_layer():
    pm = _manual_manager(256, int4_page_ids={2, 3, 4, 5})
    k, v = _synthetic_kv(256)
    layer = ingest_synthetic_layer(k, v, pm)
    cache = PackedMixedCache(page_manager=pm, layers=[layer], geom=ONE_LAYER)
    assert cache.realized_bytes() == pm.realized_bytes()


def test_mixed_materialize_close_to_dense():
    pm = _manual_manager(64, int4_page_ids={1, 2})
    k, v = _synthetic_kv(64)
    layer = ingest_synthetic_layer(k, v, pm)
    km, vm = layer.materialize()
    rel_k = np.linalg.norm(km - k) / (np.linalg.norm(k) + 1e-8)
    rel_v = np.linalg.norm(vm - v) / (np.linalg.norm(v) + 1e-8)
    assert rel_k < 0.25
    assert rel_v < 0.25


def test_compression_ratio_below_one_when_mostly_int4():
    pm = _manual_manager(256, int4_page_ids=set(range(2, 16)))
    k, v = _synthetic_kv(256)
    layer = ingest_synthetic_layer(k, v, pm)
    cache = PackedMixedCache(page_manager=pm, layers=[layer], geom=ONE_LAYER)
    assert cache.compression_ratio() < 0.85
