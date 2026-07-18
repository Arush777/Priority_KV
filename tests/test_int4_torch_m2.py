"""Torch vs numpy INT4 groupwise parity (M2)."""

from __future__ import annotations

import numpy as np
import pytest

from prioritykv.int4_kv import Int4KvConfig, dequantize_groupwise, quantize_groupwise
from prioritykv.int4_path import (
    append_quantize,
    append_quantize_torch,
    dequantize_groupwise_torch,
    quantize_groupwise_torch,
)


torch = pytest.importorskip("torch")


def test_quantize_groupwise_torch_matches_numpy():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((8, 64)).astype(np.float32)
    q_np, s_np, z_np = quantize_groupwise(x, group_size=32, nbits=4)
    xt = torch.from_numpy(x)
    q_t, s_t, z_t = quantize_groupwise_torch(xt, group_size=32, nbits=4)
    assert np.array_equal(q_t.numpy(), q_np)
    assert np.allclose(s_t.numpy(), s_np, rtol=1e-5, atol=1e-5)
    assert np.allclose(z_t.numpy(), z_np, rtol=1e-5, atol=1e-5)
    y_np = dequantize_groupwise(q_np, s_np, z_np, group_size=32)
    y_t = dequantize_groupwise_torch(q_t, s_t, z_t, group_size=32).numpy()
    assert np.allclose(y_t, y_np, rtol=1e-5, atol=1e-5)


def test_append_quantize_torch_roundtrip_close():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((16, 48)).astype(np.float32)
    cfg = Int4KvConfig(group_size=16)
    p_np = append_quantize(x, cfg=cfg)
    p_t = append_quantize_torch(torch.from_numpy(x), cfg=cfg)
    y_np = p_np.dequant()
    y_t = p_t.dequant().numpy()
    assert np.allclose(y_t, y_np, rtol=1e-5, atol=1e-5)
    assert p_t.payload_bytes() == p_np.payload_bytes()
