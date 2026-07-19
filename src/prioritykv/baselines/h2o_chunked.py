"""Chunked prefill H2O (Heavy-Hitter Oracle) as a kvpress ScorerPress.

Literature (Zhang et al., NeurIPS'23): accumulate per-key attention mass and keep
heavy hitters + a recent window. kvpress ``ObservedAttentionPress`` forces eager
attention and materializes the full S×S map → OOM at ~16–20k on H200 (~55 GiB).

This press recomputes scores from Q/K in query chunks of size ``chunk_size``
(exact row-wise softmax; peak ~H×C×S, not H×S×S). The model forward stays on
SDPA; eviction uses the same KV-drop path as SnapKV/Pyramid.

Caveat: prefill one-shot KV eviction (kvpress setting), not decode-time streaming
eviction from the original H2O systems paper.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


def make_chunked_h2o_press(
    compression_ratio: float = 0.75,
    chunk_size: int = 1024,
    recent_frac: float = 0.5,
) -> Any:
    """Return a kvpress ``ScorerPress`` instance (requires kvpress installed)."""
    import torch
    from torch import nn
    from kvpress.presses.scorer_press import ScorerPress
    from kvpress.utils import get_prerope_query_states
    from transformers.models.llama.modeling_llama import repeat_kv, rotate_half

    @dataclass
    class ChunkedH2OPress(ScorerPress):
        compression_ratio: float = 0.75
        chunk_size: int = 1024
        recent_frac: float = 0.5

        def score(
            self,
            module: nn.Module,
            hidden_states: torch.Tensor,
            keys: torch.Tensor,
            values: torch.Tensor,
            attentions: torch.Tensor,
            kwargs,
        ) -> torch.Tensor:
            del values, attentions
            bsz, num_kv_heads, k_len, head_dim = keys.shape
            num_heads = int(module.config.num_attention_heads)
            num_kv_groups = num_heads // num_kv_heads

            query_states = get_prerope_query_states(module, hidden_states)
            cos, sin = kwargs["position_embeddings"]
            query_states = (query_states * cos.unsqueeze(1)) + (
                rotate_half(query_states) * sin.unsqueeze(1)
            )
            key_states = repeat_kv(keys, num_kv_groups)
            scores = torch.zeros(
                bsz, num_heads, k_len, device=keys.device, dtype=torch.float32
            )
            scale = head_dim**-0.5

            for start in range(0, k_len, self.chunk_size):
                end = min(start + self.chunk_size, k_len)
                qi = query_states[:, :, start:end, :].float()
                ki = key_states[:, :, :end, :].float()
                logits = torch.matmul(qi, ki.transpose(-1, -2)) * scale
                q_pos = torch.arange(start, end, device=keys.device)[:, None]
                k_pos = torch.arange(end, device=keys.device)[None, :]
                logits = logits.masked_fill(
                    (k_pos > q_pos).view(1, 1, end - start, end), float("-inf")
                )
                attn = torch.softmax(logits, dim=-1)
                scores[:, :, :end] += attn.sum(dim=2)
                del logits, attn, qi, ki

            scores = scores.view(bsz, num_kv_heads, num_kv_groups, k_len).mean(2)

            # Force recent window into top-k (≈50% of keep budget).
            n_kept = max(1, int(k_len * (1.0 - self.compression_ratio)))
            recent = max(1, int(round(n_kept * self.recent_frac)))
            recent = min(recent, k_len)
            scores = scores.clone()
            if k_len > recent:
                floor = scores[..., :-recent].amax(dim=-1, keepdim=True)
            else:
                floor = scores.amax(dim=-1, keepdim=True)
            scores[..., -recent:] = floor + 1.0
            return scores

    return ChunkedH2OPress(
        compression_ratio=float(compression_ratio),
        chunk_size=int(chunk_size),
        recent_frac=float(recent_frac),
    )


def make_h2o_press_from_cfg(cfg) -> Optional[Any]:
    try:
        return make_chunked_h2o_press(
            compression_ratio=float(getattr(cfg, "compression_ratio", 0.75)),
            chunk_size=int(getattr(cfg, "h2o_chunk_size", 1024)),
            recent_frac=float(getattr(cfg, "h2o_recent_frac", 0.5)),
        )
    except Exception:
        return None
