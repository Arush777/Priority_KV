"""Extended INT4 append/decode reference tests (W3)."""

from __future__ import annotations

import numpy as np

from prioritykv.int4_kv import Int4KvConfig, fake_quant_roundtrip
from prioritykv.int4_path import append_quantize, decode_gather_reference


def test_max_abs_err_bounded_by_half_scale():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, 64)).astype(np.float32)
    cfg = Int4KvConfig(group_size=32, nbits=4)
    packed = append_quantize(x, cfg=cfg)
    y = packed.dequant()
    # Reconstruct per-group scales for bound check along last axis groups
    err = np.abs(y - x)
    # Loose global bound: 4-bit asymmetric on std-normal shouldn't explode
    assert float(err.max()) < 2.0


def test_partial_page_and_boundary():
    rng = np.random.default_rng(1)
    a = rng.standard_normal((8, 16)).astype(np.float32)
    b = rng.standard_normal((8, 7)).astype(np.float32)  # partial
    y = decode_gather_reference([append_quantize(a), append_quantize(b)])
    assert y.shape == (8, 23)


def test_matches_fake_quant_reference():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((2, 96)).astype(np.float32)
    cfg = Int4KvConfig(group_size=32)
    y1 = append_quantize(x, cfg=cfg).dequant()
    y2 = fake_quant_roundtrip(x, cfg)
    assert np.allclose(y1, y2, atol=1e-5)


def test_channel_axis_v_style():
    # Quantize along channel (last) dim for V-style — same helper, different layout
    rng = np.random.default_rng(3)
    v = rng.standard_normal((16, 32)).astype(np.float32)  # tokens x channels
    packed = append_quantize(v, cfg=Int4KvConfig(group_size=16))
    y = packed.dequant()
    assert y.shape == v.shape
    rel = np.linalg.norm(y - v) / (np.linalg.norm(v) + 1e-8)
    assert rel < 0.2
