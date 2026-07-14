"""Matched-budget keep policies for uniform / structured / random arms (prompt-level)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Mapping, Sequence

import numpy as np

from prioritykv.page_roles import PROTECTED_ROLES, PageRole
from prioritykv.tagging import role_for_message


@dataclass(frozen=True)
class KeepPolicyConfig:
    keep_frac: float = 0.25
    sink_tokens: int = 16
    # Always retain this many trailing tokens (FINAL ask lives here).
    force_recent: int = 128
    seed: int = 0


def _message_role_stress(msg: Mapping[str, str]) -> PageRole:
    """Like role_for_message but short early user/asst turns are kept as state (OTHER).

    Deliberately does NOT key on the word FINAL (benchmark markup / oracle smell).
    Trailing ask is retained via force_recent, not string match.
    """
    base = role_for_message(msg)
    content = msg.get("content") or ""
    role = (msg.get("role") or "").lower()
    if base == PageRole.FILLER and len(content) < 500:
        # Short turn establishing IDs/prefs — treat as structure, not pad.
        return PageRole.OTHER
    if role == "assistant" and len(content) < 500 and base == PageRole.GENERATED:
        return PageRole.OTHER
    return base


def assign_token_roles(
    tokenizer,
    messages: Sequence[Mapping[str, str]],
    *,
    chat_kwargs: dict,
) -> list[PageRole]:
    """Per-token roles aligned to the full chat-templated prompt (+ gen prompt)."""
    full_text = tokenizer.apply_chat_template(
        list(messages),
        tokenize=False,
        add_generation_prompt=True,
        **chat_kwargs,
    )
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    n = len(full_ids)
    roles: list[PageRole] = [PageRole.FILLER] * n
    prev = 0
    for i, msg in enumerate(messages):
        prefix = list(messages[: i + 1])
        text = tokenizer.apply_chat_template(
            prefix,
            tokenize=False,
            add_generation_prompt=(i == len(messages) - 1),
            **chat_kwargs,
        )
        plen = len(tokenizer(text, add_special_tokens=False)["input_ids"])
        plen = min(max(plen, prev), n)
        r = _message_role_stress(msg)
        for t in range(prev, plen):
            roles[t] = r
        prev = plen
    # Any trailing template tokens (gen prompt) → RECENT.
    for t in range(prev, n):
        roles[t] = PageRole.RECENT
    # Overlay hard sink.
    for t in range(min(16, n)):
        roles[t] = PageRole.SINK
    return roles


def _finalize(indices: List[int], n: int) -> np.ndarray:
    uniq = sorted(set(i for i in indices if 0 <= i < n))
    if not uniq:
        return np.arange(n, dtype=np.int64)
    return np.asarray(uniq, dtype=np.int64)


def select_uniform(n: int, cfg: KeepPolicyConfig) -> np.ndarray:
    budget = max(cfg.sink_tokens + cfg.force_recent, int(round(n * cfg.keep_frac)))
    budget = min(budget, n)
    if budget >= n:
        return np.arange(n, dtype=np.int64)
    recent = max(cfg.force_recent, budget - cfg.sink_tokens)
    recent = min(recent, budget)
    sink = budget - recent
    idx = list(range(sink)) + list(range(n - recent, n))
    return _finalize(idx, n)


def select_random(n: int, cfg: KeepPolicyConfig) -> np.ndarray:
    budget = max(cfg.sink_tokens + cfg.force_recent, int(round(n * cfg.keep_frac)))
    budget = min(budget, n)
    if budget >= n:
        return np.arange(n, dtype=np.int64)
    # Match uniform's recent mass so FINAL isn't uniquely handicapped vs uniform.
    recent = max(cfg.force_recent, budget - cfg.sink_tokens)
    recent = min(recent, budget)
    sink = budget - recent
    must = set(range(sink)) | set(range(n - recent, n))
    remaining_budget = budget - len(must)
    middle = [i for i in range(n) if i not in must]
    rng = np.random.default_rng(cfg.seed)
    if remaining_budget > 0 and middle:
        pick = rng.choice(middle, size=min(remaining_budget, len(middle)), replace=False)
        must.update(int(x) for x in pick)
    return _finalize(list(must), n)


def select_structure(n: int, roles: Sequence[PageRole], cfg: KeepPolicyConfig) -> np.ndarray:
    budget = max(cfg.sink_tokens + cfg.force_recent, int(round(n * cfg.keep_frac)))
    budget = min(budget, n)
    if budget >= n:
        return np.arange(n, dtype=np.int64)

    must = set(range(min(cfg.sink_tokens, n))) | set(range(max(0, n - cfg.force_recent), n))
    # Structure: protected roles + OTHER (short state turns)
    struct = {
        i
        for i, r in enumerate(roles)
        if r in PROTECTED_ROLES or r == PageRole.OTHER
    }
    # Prefer structure over random middle; add in index order (early first).
    for i in sorted(struct):
        if len(must) >= budget:
            break
        must.add(i)
    # Fill remainder walking left from recent edge.
    # May under-fill only for tiny n; under-fill disadvantages structure (conservative).
    pos = n - cfg.force_recent - 1
    while len(must) < budget and pos >= 0:
        must.add(pos)
        pos -= 1
    return _finalize(list(must), n)


def apply_keep_indices(ids, indices: np.ndarray):
    """Gather tokens into a contiguous prompt (RoPE-safe regenerate path)."""
    import torch

    if not torch.is_tensor(ids):
        ids = torch.tensor(ids)
    flat = ids.view(-1)
    idx = torch.as_tensor(indices, dtype=torch.long)
    return flat.index_select(0, idx)
