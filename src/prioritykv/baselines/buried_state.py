"""Adversarial buried-state transforms (W2 close scope check).

Embeds short gold-bearing turns inside long filler so length-based structure
tagging can no longer find them. Fable: if structure still scores ~1.0 here,
there's a leak; if it drops toward uniform, W2 claim is correctly scoped.
"""

from __future__ import annotations

import random
from typing import Dict, List, Sequence


_PAD = (
    "background log notes on routing latency cache pages filler update "
    "scheduler ticks unrelated telemetry paste " * 8
)


def bury_short_state_turns(
    messages: Sequence[dict],
    *,
    min_len: int = 520,
    seed: int = 0,
) -> List[Dict[str, str]]:
    """Pad non-final short turns to >min_len with filler wrapping the original text."""
    rng = random.Random(seed)
    out: List[Dict[str, str]] = []
    n = len(messages)
    for i, m in enumerate(messages):
        content = m.get("content") or ""
        is_last = i == n - 1
        if is_last or len(content) >= min_len:
            out.append(dict(m))
            continue
        # Wrap gold text mid-filler so it's not length-separable.
        left = _PAD
        right = _PAD
        while len(left) + len(content) + len(right) < min_len:
            left += f" pad{rng.randint(1000, 9999)} "
        buried = f"{left} <<STATE>> {content} <</STATE>> {right}"
        out.append({"role": m["role"], "content": buried})
    return out
