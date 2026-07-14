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
    # Page-level stress (W3): majority-role then whole-page keep.
    page_tokens: int = 16
    granularity: str = "token"  # token | page


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


def page_bounds(n: int, page_tokens: int) -> list[tuple[int, int]]:
    """Inclusive-start exclusive-end page spans covering ``n`` tokens."""
    if page_tokens <= 0:
        raise ValueError("page_tokens must be > 0")
    return [(s, min(s + page_tokens, n)) for s in range(0, n, page_tokens)]


def majority_page_role(roles: Sequence[PageRole], start: int, end: int) -> PageRole:
    counts: dict[PageRole, int] = {}
    for r in roles[start:end]:
        counts[r] = counts.get(r, 0) + 1
    # Prefer protected / OTHER on ties (conservative for structure stress).
    def _key(item: tuple[PageRole, int]):
        role, c = item
        prefer = 2 if role in PROTECTED_ROLES else (1 if role == PageRole.OTHER else 0)
        return (c, prefer)

    return max(counts.items(), key=_key)[0]


def _token_budget(n: int, cfg: KeepPolicyConfig) -> int:
    budget = max(cfg.sink_tokens + cfg.force_recent, int(round(n * cfg.keep_frac)))
    return min(budget, n)


def _pages_covering_tokens(spans: Sequence[tuple[int, int]], token_indices: set[int]) -> set[int]:
    keep: set[int] = set()
    for pi, (s, e) in enumerate(spans):
        if any(t in token_indices for t in range(s, e)):
            keep.add(pi)
    return keep


def _expand_pages_to_tokens(spans: Sequence[tuple[int, int]], page_ids: set[int]) -> list[int]:
    toks: list[int] = []
    for pi in sorted(page_ids):
        s, e = spans[pi]
        toks.extend(range(s, e))
    return toks


def select_uniform_pages(n: int, cfg: KeepPolicyConfig) -> np.ndarray:
    """Whole-page sink+recent keep, floored to token budget (Fable W3)."""
    spans = page_bounds(n, cfg.page_tokens)
    budget = _token_budget(n, cfg)
    if budget >= n:
        return np.arange(n, dtype=np.int64)
    must_toks = set(range(min(cfg.sink_tokens, n))) | set(
        range(max(0, n - cfg.force_recent), n)
    )
    keep_pages = _pages_covering_tokens(spans, must_toks)
    # Floor: never exceed token budget; drop middle pages first (highest page id before recent).
    def _n_toks(pids: set[int]) -> int:
        return sum(spans[p][1] - spans[p][0] for p in pids)

    # If sink+recent pages alone exceed budget, still keep them (RoPE guardrail) but log later.
    if _n_toks(keep_pages) > budget:
        return _finalize(_expand_pages_to_tokens(spans, keep_pages), n)
    # Add middle pages from the recent edge leftward until next page would exceed budget.
    for pi in range(len(spans) - 1, -1, -1):
        if pi in keep_pages:
            continue
        trial = set(keep_pages) | {pi}
        if _n_toks(trial) <= budget:
            keep_pages = trial
    return _finalize(_expand_pages_to_tokens(spans, keep_pages), n)


def select_random_pages(n: int, cfg: KeepPolicyConfig) -> np.ndarray:
    spans = page_bounds(n, cfg.page_tokens)
    budget = _token_budget(n, cfg)
    if budget >= n:
        return np.arange(n, dtype=np.int64)
    must_toks = set(range(min(cfg.sink_tokens, n))) | set(
        range(max(0, n - cfg.force_recent), n)
    )
    keep_pages = _pages_covering_tokens(spans, must_toks)

    def _n_toks(pids: set[int]) -> int:
        return sum(spans[p][1] - spans[p][0] for p in pids)

    middle = [pi for pi in range(len(spans)) if pi not in keep_pages]
    rng = np.random.default_rng(cfg.seed)
    rng.shuffle(middle)
    for pi in middle:
        trial = set(keep_pages) | {pi}
        if _n_toks(trial) <= budget:
            keep_pages = trial
    return _finalize(_expand_pages_to_tokens(spans, keep_pages), n)


def select_structure_pages(
    n: int, roles: Sequence[PageRole], cfg: KeepPolicyConfig
) -> np.ndarray:
    spans = page_bounds(n, cfg.page_tokens)
    budget = _token_budget(n, cfg)
    if budget >= n:
        return np.arange(n, dtype=np.int64)
    must_toks = set(range(min(cfg.sink_tokens, n))) | set(
        range(max(0, n - cfg.force_recent), n)
    )
    keep_pages = _pages_covering_tokens(spans, must_toks)
    page_roles = [majority_page_role(roles, s, e) for s, e in spans]
    struct_pages = [
        pi
        for pi, r in enumerate(page_roles)
        if r in PROTECTED_ROLES or r == PageRole.OTHER
    ]

    def _n_toks(pids: set[int]) -> int:
        return sum(spans[p][1] - spans[p][0] for p in pids)

    for pi in struct_pages:
        if pi in keep_pages:
            continue
        trial = set(keep_pages) | {pi}
        if _n_toks(trial) <= budget:
            keep_pages = trial
    # Fill remainder from right (near recent) without exceeding budget.
    for pi in range(len(spans) - 1, -1, -1):
        if pi in keep_pages:
            continue
        trial = set(keep_pages) | {pi}
        if _n_toks(trial) <= budget:
            keep_pages = trial
    return _finalize(_expand_pages_to_tokens(spans, keep_pages), n)


def select_keep_indices(
    n: int,
    cfg: KeepPolicyConfig,
    *,
    policy: str,
    roles: Sequence[PageRole] | None = None,
) -> np.ndarray:
    """Dispatch token- or page-granularity keep selection."""
    page = cfg.granularity == "page"
    if policy == "uniform":
        return select_uniform_pages(n, cfg) if page else select_uniform(n, cfg)
    if policy == "random":
        return select_random_pages(n, cfg) if page else select_random(n, cfg)
    if policy == "structure":
        if roles is None:
            raise ValueError("structure policy requires roles")
        return (
            select_structure_pages(n, roles, cfg)
            if page
            else select_structure(n, roles, cfg)
        )
    raise ValueError(f"unknown policy {policy}")



def apply_keep_indices(ids, indices: np.ndarray):
    """Gather tokens into a contiguous prompt (RoPE-safe regenerate path)."""
    import torch

    if not torch.is_tensor(ids):
        ids = torch.tensor(ids)
    flat = ids.view(-1)
    idx = torch.as_tensor(indices, dtype=torch.long)
    return flat.index_select(0, idx)
