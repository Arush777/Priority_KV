"""Aggressive sink+recent compression (StreamingLLM-style).

IMPORTANT: apply at the *token/prompt* level, then run normal generate.
Surgically slicing a RoPE'd KV cache without fixing positions produces
garbage for every budget (false flat zeros). Prompt concat is the correct
quality evaluation path.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class DropKeepConfig:
    """Keep only sink prefix + recent suffix; drop the middle."""

    sink_tokens: int = 16
    recent_tokens: int = 256
    keep_tokens: Optional[int] = None


def keep_budget(cfg: DropKeepConfig) -> int:
    if cfg.keep_tokens is not None:
        return int(cfg.keep_tokens)
    return int(cfg.sink_tokens) + int(cfg.recent_tokens)


def realized_keep_frac(seq_len: int, cfg: DropKeepConfig) -> float:
    return min(1.0, keep_budget(cfg) / max(seq_len, 1))


def apply_drop_keep_ids(ids, cfg: DropKeepConfig):
    """Return (compressed_ids, meta) — contiguous tokens, correct for RoPE."""
    import torch

    if not torch.is_tensor(ids):
        ids = torch.tensor(ids)
    flat = ids.view(-1)
    n = int(flat.numel())
    sink = int(cfg.sink_tokens)
    if cfg.keep_tokens is not None:
        recent = max(0, int(cfg.keep_tokens) - sink)
    else:
        recent = int(cfg.recent_tokens)

    if n <= sink + recent:
        return flat, {
            "prompt_tokens": n,
            "kept_tokens": n,
            "dropped": False,
            "keep_frac": 1.0,
            "approx_compression_x": 1.0,
        }

    head = flat[:sink]
    tail = flat[-recent:]
    out = torch.cat([head, tail])
    kept = int(out.numel())
    return out, {
        "prompt_tokens": n,
        "kept_tokens": kept,
        "dropped": True,
        "keep_frac": kept / n,
        "approx_compression_x": n / max(kept, 1),
    }


def status() -> dict[str, Any]:
    cfg = DropKeepConfig()
    return {
        "baseline_id": "Q_dropkeep",
        "name": "StreamingLLM-style sink+recent (prompt-level)",
        "implemented": True,
        "config": asdict(cfg),
        "note": (
            "Must apply on token ids then regenerate; KV-cache slicing without "
            "RoPE fix is invalid for quality numbers."
        ),
    }
