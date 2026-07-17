"""CPU unit tests for Stage-1b shim helpers (no GPU / no model weights)."""

from __future__ import annotations

import pytest

from prioritykv.fi_mixed_decode import FiMixedDecodeState, append_decode_kv, commit_decode_step
from prioritykv.qwen3_fi_shim import FiSeqLenCache


def test_fi_seq_len_cache_reports_committed_only():
    st = FiMixedDecodeState(
        num_layers=2,
        num_kv_heads=8,
        head_dim=128,
        cache_len=10,
        decode_len=3,
        step_kv_len=1,
    )
    stub = FiSeqLenCache(st)
    assert stub.get_seq_length() == 13  # cache+decode, not step
    assert stub.get_mask_sizes(1) == (14, 0)
    assert len(stub) == 2
    with pytest.raises(RuntimeError):
        stub.update(None, None, 0)


def test_step_kv_len_append_commit_positions():
    torch = pytest.importorskip("torch")
    st = FiMixedDecodeState(
        num_layers=2,
        num_kv_heads=2,
        head_dim=4,
        cache_len=0,
        decode_len=0,
        device="cpu",
        dtype=torch.float32,
    )
    for _ in range(2):
        from prioritykv.fi_mixed_decode import LayerMixedBuffers

        buf = LayerMixedBuffers(
            k_hot=torch.zeros(2, 8, 4),
            v_hot=torch.zeros(2, 8, 4),
            hot_len=0,
            hot_capacity=8,
            cold_len=0,
        )
        st.layers.append(buf)
    st.cache_len = 0  # synthetic: all hot is decode tail
    # Pretend hot prefix length 2 already filled as "cache"
    for buf in st.layers:
        buf.hot_len = 2
        buf.k_hot[:, :2] = 1.0
        buf.v_hot[:, :2] = 1.0
    st.cache_len = 2

    k0 = torch.ones(2, 1, 4) * 2
    v0 = torch.ones(2, 1, 4) * 2
    append_decode_kv(st, 0, k0, v0)
    assert st.step_kv_len == 1
    append_decode_kv(st, 1, k0, v0)
    # Both layers wrote the same slot (index 2)
    assert torch.allclose(st.layers[0].k_hot[:, 2:3], k0)
    assert torch.allclose(st.layers[1].k_hot[:, 2:3], k0)
    commit_decode_step(st)
    assert st.decode_len == 1
    assert st.step_kv_len == 0
    assert st.total_kv_len == 3
