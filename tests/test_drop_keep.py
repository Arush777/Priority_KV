"""CPU tests for prompt-level DropKeep."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from prioritykv.baselines.drop_keep import (
    DropKeepConfig,
    apply_drop_keep_ids,
    realized_keep_frac,
)


def test_keep_frac():
    assert abs(realized_keep_frac(16000, DropKeepConfig(sink_tokens=16, recent_tokens=256)) - 272 / 16000) < 1e-9


def test_apply_short_is_noop():
    ids = torch.arange(100)
    out, meta = apply_drop_keep_ids(ids, DropKeepConfig(sink_tokens=16, recent_tokens=256))
    assert meta["dropped"] is False
    assert torch.equal(out, ids)


def test_apply_keeps_head_and_tail():
    ids = torch.arange(1000)
    out, meta = apply_drop_keep_ids(ids, DropKeepConfig(sink_tokens=4, recent_tokens=8))
    assert meta["dropped"] is True
    assert meta["kept_tokens"] == 12
    assert torch.equal(out[:4], ids[:4])
    assert torch.equal(out[-8:], ids[-8:])


def test_huge_recent_is_noop():
    ids = torch.arange(5000)
    out, meta = apply_drop_keep_ids(
        ids, DropKeepConfig(sink_tokens=16, recent_tokens=999999)
    )
    assert meta["dropped"] is False
    assert meta["keep_frac"] == 1.0
    assert torch.equal(out, ids)
