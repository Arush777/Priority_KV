"""CPU tests for DropKeep eviction."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from prioritykv.baselines.drop_keep import DropKeepConfig, drop_keep_past, realized_keep_frac


def test_keep_frac():
    assert abs(realized_keep_frac(16000, DropKeepConfig(sink_tokens=16, recent_tokens=256)) - 272 / 16000) < 1e-9


def test_drop_keep_shortens_layers():
    cfg = DropKeepConfig(sink_tokens=4, recent_tokens=8)
    k = torch.randn(1, 2, 100, 8)
    v = torch.randn(1, 2, 100, 8)
    layer = SimpleNamespace(keys=k.clone(), values=v.clone())
    past = SimpleNamespace(layers=[layer])
    drop_keep_past(past, cfg)
    assert layer.keys.shape[-2] == 12
    # head preserved
    assert torch.equal(layer.keys[:, :, :4], k[:, :, :4])
    # tail preserved
    assert torch.equal(layer.keys[:, :, -8:], k[:, :, -8:])
