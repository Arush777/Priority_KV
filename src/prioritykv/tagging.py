"""Structural span tagging → per-token roles for page assignment.

W2: heuristic chat-role tagging (no model required). Later we refine with
tokenizer offsets from the pinned Qwen3 chat template.
"""

from __future__ import annotations

import re
from typing import List, Mapping, Sequence

from prioritykv.page_roles import PageRole

_TOOL_HINT = re.compile(
    r"tools?\s*\(|json schema|available tools|tool call|\"name\"\s*:\s*\"",
    re.IGNORECASE,
)
_CONSTRAINT_HINT = re.compile(
    r"\b(must|never|only|constraint|forbidden|always answer|latest instruction)\b",
    re.IGNORECASE,
)


def role_for_message(message: Mapping[str, str]) -> PageRole:
    """Map one chat message to a structural page role."""
    role = (message.get("role") or "").lower()
    content = message.get("content") or ""

    if role == "system":
        if _TOOL_HINT.search(content):
            return PageRole.TOOL
        if _CONSTRAINT_HINT.search(content):
            return PageRole.CONSTRAINT
        return PageRole.SYSTEM

    if role == "tool":
        return PageRole.TOOL

    if role == "assistant":
        if _TOOL_HINT.search(content) or content.strip().startswith("{"):
            return PageRole.TOOL
        return PageRole.GENERATED

    if role == "user":
        if _CONSTRAINT_HINT.search(content):
            return PageRole.CONSTRAINT
        # FINAL asks are still user text but sit near the end → often RECENT after windowing.
        return PageRole.FILLER

    return PageRole.OTHER


def tag_messages(
    messages: Sequence[Mapping[str, str]],
    *,
    approx_tokens_fn=None,
) -> List[tuple[PageRole, int]]:
    """Return (role, approx_token_count) spans in message order.

    Token counts are approximate (chars/4) unless ``approx_tokens_fn`` is provided.
    """
    if approx_tokens_fn is None:

        def approx_tokens_fn(text: str) -> int:  # type: ignore[misc]
            return max(1, len(text) // 4)

    spans: List[tuple[PageRole, int]] = []
    for msg in messages:
        spans.append((role_for_message(msg), int(approx_tokens_fn(msg.get("content", "")))))
    return spans


def expand_token_roles(
    spans: Sequence[tuple[PageRole, int]],
    *,
    recent_window: int = 128,
    sink_tokens: int = 16,
) -> List[PageRole]:
    """Expand spans to a flat per-token role list; overlay sink + recent window."""
    roles: List[PageRole] = []
    for role, n in spans:
        roles.extend([role] * max(0, n))

    if not roles:
        return roles

    # Attention sinks: first physical page worth of tokens.
    for i in range(min(sink_tokens, len(roles))):
        roles[i] = PageRole.SINK

    # Newest W tokens are RECENT (overwrites filler/generated at the tail).
    for i in range(max(0, len(roles) - recent_window), len(roles)):
        if roles[i] not in (PageRole.SINK, PageRole.SYSTEM, PageRole.TOOL, PageRole.CONSTRAINT):
            roles[i] = PageRole.RECENT
        else:
            # Keep structural protections even inside the recent window.
            pass
    return roles


def tag_chat_to_token_roles(
    messages: Sequence[Mapping[str, str]],
    *,
    recent_window: int = 128,
    sink_tokens: int = 16,
) -> List[PageRole]:
    spans = tag_messages(messages)
    return expand_token_roles(spans, recent_window=recent_window, sink_tokens=sink_tokens)
