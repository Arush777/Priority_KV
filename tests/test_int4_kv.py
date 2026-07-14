"""INT4 groupwise unit tests (CPU)."""

from __future__ import annotations

import numpy as np

from prioritykv.int4_kv import (
    Int4KvConfig,
    dequantize_groupwise,
    fake_quant_roundtrip,
    quantize_groupwise,
    status,
)


def test_quantize_roundtrip_small_error():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 4, 96)).astype(np.float32)
    y = fake_quant_roundtrip(x, Int4KvConfig(group_size=32, nbits=4))
    rel = np.linalg.norm(y - x) / (np.linalg.norm(x) + 1e-8)
    assert rel < 0.15, rel


def test_quantize_shapes():
    x = np.linspace(-1, 1, 40, dtype=np.float32).reshape(2, 20)
    q, scale, zp = quantize_groupwise(x, group_size=8, nbits=4)
    assert q.shape == x.shape
    assert scale.shape[-1] == 3  # 20 padded to 24 → 3 groups
    y = dequantize_groupwise(q, scale, zp, group_size=8)
    assert y.shape == x.shape


def test_status_keys():
    s = status()
    assert s["baseline_id"] == "Q2"
    assert s["implemented"] is True
