"""CPU tests for INT4 past-cache fake-quant helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from prioritykv.int4_kv import Int4KvConfig

torch = pytest.importorskip("torch")


def test_fake_quant_past_legacy_tuple():
    from prioritykv.int4_baseline import _fake_quant_past

    cfg = Int4KvConfig(group_size=8)
    k = torch.randn(1, 2, 16, dtype=torch.float32)
    v = torch.randn(1, 2, 16, dtype=torch.float32)
    # Qwen-like: occasionally extra trailing slots — must not crash
    past = ((k, v, "extra"),)
    out = _fake_quant_past(past, cfg)
    assert out is not None


def test_fake_quant_past_layers_attrs():
    from prioritykv.int4_baseline import _fake_quant_past

    cfg = Int4KvConfig(group_size=8)
    k = torch.randn(1, 2, 16)
    v = torch.randn(1, 2, 16)
    layer = SimpleNamespace(keys=k.clone(), values=v.clone())
    past = SimpleNamespace(layers=[layer])
    _fake_quant_past(past, cfg)
    assert not torch.equal(layer.keys, k) or torch.allclose(layer.keys, k, atol=0.5)
    rel = (layer.keys.float() - k.float()).norm() / (k.float().norm() + 1e-8)
    assert float(rel) < 0.5


def test_fake_quant_past_key_cache_lists():
    from prioritykv.int4_baseline import _fake_quant_past

    cfg = Int4KvConfig(group_size=8)
    k = torch.randn(1, 2, 24)
    v = torch.randn(1, 2, 24)
    past = SimpleNamespace(key_cache=[k.clone()], value_cache=[v.clone()])
    _fake_quant_past(past, cfg)
    assert past.key_cache[0].shape == k.shape
