"""CPU tests for FI mixed-decode Stage-1 (incl. build_from_packed_cache)."""

from __future__ import annotations

import numpy as np
import pytest

from prioritykv.fi_mixed_decode import (
    FiMixedDecodeState,
    build_from_packed_cache,
    coalesce_hot_cold_lengths,
    stage1_acceptance_checklist,
)
from prioritykv.flashinfer_multicall import require_head_dim
from prioritykv.int4_kv import Int4KvConfig
from prioritykv.packed_mixed_cache import ingest_synthetic_layer, page_manager_from_int4_mask
from prioritykv.page_roles import PageRole
from prioritykv.byte_model import ModelKvGeom


def test_coalesce_two_chunks_only():
    hot_t, cold = coalesce_hot_cold_lengths(hot_len=100, cold_len=400, decode_tail=8)
    assert hot_t == 108
    assert cold == 400


def test_head_dim_gate():
    require_head_dim(128)
    with pytest.raises(ValueError):
        require_head_dim(32)


def test_forbid_materialize_gate():
    st = FiMixedDecodeState(
        num_layers=1,
        num_kv_heads=8,
        head_dim=128,
        cache_len=10,
        forbid_materialize=True,
        layers=[],  # empty layers allowed until buffers are attached
    )
    st.validate_geom()  # geom ok with empty layers
    st.assert_no_materialize_path(False)
    with pytest.raises(RuntimeError):
        st.assert_no_materialize_path(True)
    st.num_layers = 0
    with pytest.raises(ValueError):
        st.validate_geom()


def test_acceptance_checklist_keys():
    keys = stage1_acceptance_checklist()
    for k in ("parity_attn", "peak_mem", "split_prefill", "lse", "chunks"):
        assert k in keys


def test_build_from_packed_cache_cpu_skip_without_cuda():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for build_from_packed_cache GPU upload")

    n, h, d = 64, 4, 128
    roles = [PageRole.SYSTEM] * 16 + [PageRole.FILLER] * 48
    mask = np.zeros(n, dtype=bool)
    mask[16:] = True  # demote filler
    geom = ModelKvGeom(num_layers=1, num_kv_heads=h, head_dim=d)
    pm = page_manager_from_int4_mask(roles, mask, page_tokens=16, geom=geom)
    rng = np.random.default_rng(0)
    k = rng.standard_normal((h, n, d)).astype(np.float16)
    v = rng.standard_normal((h, n, d)).astype(np.float16)
    layer = ingest_synthetic_layer(k, v, pm, int4_cfg=Int4KvConfig(group_size=32))
    # Wrap as fake PackedMixedCache-like object
    from prioritykv.packed_mixed_cache import PackedMixedCache

    cache = PackedMixedCache(page_manager=pm, geom=geom)
    cache.layers = [layer]
    device = torch.device("cuda:0")
    state = build_from_packed_cache(cache, device=device, decode_tail_cap=32)
    assert state.cache_len == n
    assert state.num_layers == 1
    assert state.layers[0].hot_len == 16
    assert state.layers[0].cold_len == 48
    state.assert_len_invariant()
    state.assert_no_materialize_path(False)
