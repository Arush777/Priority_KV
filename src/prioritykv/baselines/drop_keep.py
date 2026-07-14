"""Aggressive KV eviction baselines that *must* erase early-turn facts.

Motivation: uniform INT4 / gentle SnapKV often keep PriorityBench perfect.
StreamingLLM-style keep(sink + recent) at ~10–60× compression deletes the
early tool/schema/ID pages — the agent-reliability failure we need for G2.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class DropKeepConfig:
    """Keep only sink prefix + recent suffix; drop the middle (StreamingLLM-like)."""

    sink_tokens: int = 16
    recent_tokens: int = 256
    # Optional absolute keep budget; if set, recent = keep_tokens - sink.
    keep_tokens: Optional[int] = None


def realized_keep_frac(seq_len: int, cfg: DropKeepConfig) -> float:
    keep = cfg.keep_tokens
    if keep is None:
        keep = cfg.sink_tokens + cfg.recent_tokens
    return min(1.0, keep / max(seq_len, 1))


def _slice_tensor(t, keep_idx):
    import torch

    # t: [batch, heads, seq, dim] or [batch, seq, ...]
    if t is None or not torch.is_tensor(t):
        return t
    if t.ndim >= 3:
        return t.index_select(-2, keep_idx)
    return t.index_select(-1, keep_idx)


def drop_keep_past(past, cfg: DropKeepConfig):
    """In-place / reconstructed past with only sink+recent tokens retained."""
    import torch

    if past is None:
        return past

    sink = int(cfg.sink_tokens)
    if cfg.keep_tokens is not None:
        recent = max(0, int(cfg.keep_tokens) - sink)
    else:
        recent = int(cfg.recent_tokens)

    def _seq_len_from_tensor(t) -> int:
        if t is None or not torch.is_tensor(t):
            return 0
        if t.ndim >= 3:
            return int(t.shape[-2])
        return int(t.shape[-1])

    def _keep_index(seq_len: int, device):
        if seq_len <= sink + recent:
            return torch.arange(seq_len, device=device)
        head = torch.arange(0, sink, device=device)
        tail = torch.arange(seq_len - recent, seq_len, device=device)
        return torch.cat([head, tail])

    layers = getattr(past, "layers", None)
    if layers is not None:
        for layer in layers:
            for attr_k, attr_v in (
                ("keys", "values"),
                ("key_cache", "value_cache"),
                ("key", "value"),
            ):
                k = getattr(layer, attr_k, None)
                v = getattr(layer, attr_v, None)
                if k is None or v is None or not torch.is_tensor(k):
                    continue
                idx = _keep_index(_seq_len_from_tensor(k), k.device)
                setattr(layer, attr_k, _slice_tensor(k, idx))
                setattr(layer, attr_v, _slice_tensor(v, idx))
                break
        # Some caches track seen_tokens / _seen_tokens
        for attr in ("_seen_tokens", "seen_tokens"):
            if hasattr(past, attr):
                try:
                    setattr(past, attr, sink + recent)
                except Exception:
                    pass
        return past

    kc = getattr(past, "key_cache", None)
    vc = getattr(past, "value_cache", None)
    if isinstance(kc, list) and isinstance(vc, list) and kc:
        for i in range(len(kc)):
            if kc[i] is None:
                continue
            idx = _keep_index(_seq_len_from_tensor(kc[i]), kc[i].device)
            kc[i] = _slice_tensor(kc[i], idx)
            vc[i] = _slice_tensor(vc[i], idx)
        return past

    if isinstance(past, (tuple, list)):
        out = []
        for layer in past:
            if isinstance(layer, (tuple, list)) and len(layer) >= 2:
                k, v = layer[0], layer[1]
                if torch.is_tensor(k):
                    idx = _keep_index(_seq_len_from_tensor(k), k.device)
                    k2, v2 = _slice_tensor(k, idx), _slice_tensor(v, idx)
                    out.append((k2, v2) + tuple(layer[2:]))
                else:
                    out.append(layer)
            else:
                out.append(layer)
        try:
            from transformers import DynamicCache

            return DynamicCache.from_legacy_cache(tuple(out))
        except Exception:
            return tuple(out)

    raise TypeError(f"unsupported past type for drop_keep: {type(past)}")


def status() -> dict[str, Any]:
    cfg = DropKeepConfig()
    return {
        "baseline_id": "Q_dropkeep",
        "name": "StreamingLLM-style sink+recent",
        "implemented": True,
        "config": asdict(cfg),
        "note": "At 16k with sink=16 recent=256 keep_frac≈0.017 (~60×). Guarantees early-ID loss.",
    }
