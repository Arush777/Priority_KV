"""Tests for W3 INT4 append/decode reference."""

from __future__ import annotations

import numpy as np

from prioritykv.int4_path import append_quantize, decode_gather_reference


def test_append_decode_roundtrip():
    rng = np.random.default_rng(1)
    a = rng.standard_normal((4, 48)).astype(np.float32)
    b = rng.standard_normal((4, 32)).astype(np.float32)
    pa = append_quantize(a)
    pb = append_quantize(b)
    y = decode_gather_reference([pa, pb])
    x = np.concatenate([a, b], axis=-1)
    rel = np.linalg.norm(y - x) / (np.linalg.norm(x) + 1e-8)
    assert y.shape == x.shape
    assert rel < 0.15, rel
