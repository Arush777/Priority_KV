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


def _is_filler_turn(msg: Mapping[str, str]) -> bool:
    """Filler turns produced by ``pad_with_filler_turns`` (position-only pad)."""
    content = msg.get("content") or ""
    role = (msg.get("role") or "").lower()
    if role == "user":
        return "[filler/" in content
    if role == "assistant":
        return content.startswith("Acknowledged filler")
    return False


def relocate_state_to_middle(
    messages: Sequence[dict],
    *,
    position: float = 0.5,
    seed: int = 0,
) -> List[Dict[str, str]]:
    """Move genuine state turns into the middle of the filler band.

    PriorityBench templates emit ``[system, <short gold turns>, <filler…>, FINAL]``
    so structure-critical state sits in the *prefix*. A trivial FixedHot baseline
    (sink + prefix + recent) then catches the same pages a role-aware policy does,
    which is why buried-in-place could not separate FixedHot from P2.

    This transform keeps leading system turns and the final ask fixed, then
    re-inserts the gold block at ``position`` (fraction) through the filler
    sequence. Gold turn order is preserved (supersession / multi-turn semantics
    intact); only unrelated filler is moved around it. After relocation, prefix
    pages are filler, so FixedHot and uniform miss the mid-context state while a
    role/risk-aware keep still retrieves it.
    """
    del seed  # deterministic; kept for signature parity with bury.
    msgs = [dict(m) for m in messages]
    if not msgs:
        return msgs
    n = len(msgs)
    lead = 0
    while lead < n and (msgs[lead].get("role") or "").lower() == "system":
        lead += 1
    if lead >= n - 1:
        return msgs  # nothing between system and final
    head = msgs[:lead]
    final = msgs[-1]
    body = msgs[lead:-1]
    filler = [m for m in body if _is_filler_turn(m)]
    gold = [m for m in body if not _is_filler_turn(m)]
    if not gold or not filler:
        return msgs  # cannot relocate meaningfully
    frac = min(max(position, 0.0), 1.0)
    cut = int(round(len(filler) * frac))
    new_body = filler[:cut] + gold + filler[cut:]
    return head + new_body + [final]
