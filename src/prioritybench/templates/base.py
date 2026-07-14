"""Shared template types and filler helpers for PriorityBench-A."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Sequence

from prioritybench.schema import Category, CONTEXT_LENGTHS

# Rough English / JSON filler ≈ 4 chars per token (W1; precise counts use
# the pinned Qwen3 tokenizer later in the eval harness).
CHARS_PER_TOKEN = 4

FILLER_TOPICS: tuple[str, ...] = (
    "invoice reconciliation notes",
    "cluster job scheduler logs",
    "customer support ticket drafts",
    "database migration checklists",
    "incident postmortem fragments",
    "inventory SKU updates",
    "calendar scheduling conflicts",
    "API rate-limit telemetry",
)


@dataclass(frozen=True)
class TemplateSpec:
    """One generative template (plan §3.2: 12–15 per category eventually)."""

    template_id: str
    category: Category
    # Build messages + scoring for a given RNG and target context length.
    build: Callable[[random.Random, int], tuple[List[Dict[str, str]], Mapping[str, Any]]]


def approx_token_len(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


def messages_approx_tokens(messages: Sequence[Mapping[str, str]]) -> int:
    return sum(approx_token_len(m.get("content", "")) for m in messages)


def filler_paragraph(rng: random.Random, topic: str, min_chars: int) -> str:
    """Deterministic filler independent of the target tool-call span."""
    chunks: List[str] = []
    n = 0
    while n < min_chars:
        sid = rng.randint(1000, 9999)
        line = (
            f"[filler/{topic}] record={sid} status=ok note="
            f"reviewed batch {rng.randint(1, 50)} with checksum "
            f"{hashlib.md5(f'{topic}-{sid}'.encode()).hexdigest()[:12]}."
        )
        chunks.append(line)
        n += len(line) + 1
    return "\n".join(chunks)


def pad_with_filler_turns(
    messages: List[Dict[str, str]],
    rng: random.Random,
    target_tokens: int,
) -> List[Dict[str, str]]:
    """Insert interleaved user/assistant filler until ≈ target_tokens.

    Filler is sampled from topics independent of the gold tool call so
    page-position confounds are controlled (plan §3.2).
    """
    if target_tokens not in CONTEXT_LENGTHS:
        raise ValueError(f"target_tokens {target_tokens} not in {CONTEXT_LENGTHS}")

    out = list(messages[:-1])  # keep final user ask last
    final = messages[-1]
    topic = rng.choice(FILLER_TOPICS)

    # Leave headroom so the final ask + early schema remain.
    budget = int(target_tokens * 0.92)
    guard = 0
    while messages_approx_tokens(out) + messages_approx_tokens([final]) < budget:
        need = budget - messages_approx_tokens(out) - messages_approx_tokens([final])
        # Split need across a user filler turn and a short assistant ack.
        user_chars = max(200, min(need * CHARS_PER_TOKEN // 2, 4000))
        user_text = filler_paragraph(rng, topic, user_chars)
        asst_text = (
            f"Acknowledged filler update on {topic}; continuing without "
            f"changing tool contracts. ref={rng.randint(10_000, 99_999)}."
        )
        out.append({"role": "user", "content": user_text})
        out.append({"role": "assistant", "content": asst_text})
        guard += 1
        if guard > 500:
            break
    out.append(final)
    return out
