"""Hybrid: structure-protected positions ∪ SnapKV scores for remaining budget.

Wraps kvpress SnapKVPress and boosts structure/sink/recent positions so they
are never pruned, then lets SnapKV fill the residual budget — tests whether
structural priors are complementary to attention-based selection.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Set


def protected_indices_from_roles(
    roles: Sequence,
    *,
    n: int,
    sink_tokens: int = 16,
    force_recent: int = 128,
) -> set[int]:
    """Sink + recent + structure/OTHER roles (same spirit as select_structure)."""
    from prioritykv.page_roles import PROTECTED_ROLES, PageRole

    must = set(range(min(sink_tokens, n))) | set(range(max(0, n - force_recent), n))
    for i, r in enumerate(roles):
        if i >= n:
            break
        if r in PROTECTED_ROLES or r == PageRole.OTHER:
            must.add(i)
    return must


def make_hybrid_press(
    *,
    compression_ratio: float,
    protected: Optional[Set[int]] = None,
    window_size: int = 64,
    kernel_size: int = 5,
) -> Any:
    try:
        from dataclasses import dataclass, field

        import torch
        from torch import nn
        from kvpress import SnapKVPress  # type: ignore
    except Exception:
        return None

    @dataclass
    class HybridStructureSnapKVPress(SnapKVPress):  # type: ignore
        protected: Set[int] = field(default_factory=set)
        boost: float = 1.0e6

        def score(
            self,
            module: nn.Module,
            hidden_states: torch.Tensor,
            keys: torch.Tensor,
            values: torch.Tensor,
            attentions: torch.Tensor,
            kwargs,
        ) -> torch.Tensor:
            scores = super().score(
                module, hidden_states, keys, values, attentions, kwargs
            )
            if self.protected:
                seq = int(scores.shape[-1])
                for idx in self.protected:
                    if 0 <= int(idx) < seq:
                        scores[..., int(idx)] = scores[..., int(idx)] + self.boost
            return scores

    try:
        return HybridStructureSnapKVPress(
            compression_ratio=compression_ratio,
            window_size=window_size,
            kernel_size=kernel_size,
            protected=set(protected or ()),
        )
    except TypeError:
        return HybridStructureSnapKVPress(
            compression_ratio=compression_ratio,
            protected=set(protected or ()),
        )
