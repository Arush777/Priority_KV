"""multi_turn_state templates: reuse early-turn IDs/paths/prefs verbatim later.

v1: FINAL echoed the gold value (too easy under any KV compression).
v2: FINAL does not echo gold; decoys added — use for G2 stress pilots.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Mapping, Tuple

from prioritybench.schema import Category
from prioritybench.templates.base import TemplateSpec, pad_with_filler_turns


def _decoy_orders(rng: random.Random, real: str, n: int = 3) -> list[str]:
    out = []
    while len(out) < n:
        cand = f"ORD-{rng.randint(100000, 999999)}"
        if cand != real and cand not in out:
            out.append(cand)
    return out


def _build_order_id_v1(
    rng: random.Random, target_tokens: int
) -> Tuple[List[Dict[str, str]], Mapping[str, Any]]:
    order_id = f"ORD-{rng.randint(100000, 999999)}"
    messages: List[Dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "You are an order desk agent. When asked later, repeat IDs "
                "exactly with no reformatting."
            ),
        },
        {
            "role": "user",
            "content": f"Create a hold for order {order_id}. Confirm the id.",
        },
        {"role": "assistant", "content": f"Hold created for {order_id}."},
        {
            "role": "user",
            "content": f"FINAL: Reply with exactly one line: ORDER_ID={order_id}",
        },
    ]
    messages = pad_with_filler_turns(messages, rng, target_tokens)
    scoring = {
        "required_slots": {"order_id": order_id, "line": f"ORDER_ID={order_id}"},
        "mode": "strict",
    }
    return messages, scoring


def _build_file_path_v1(
    rng: random.Random, target_tokens: int
) -> Tuple[List[Dict[str, str]], Mapping[str, Any]]:
    path = rng.choice(
        [
            "/mnt/scratch/prioritybench/cal.jsonl",
            "/var/data/traces/session_a.jsonl",
            "/home/agent/cache/manifest.yaml",
        ]
    )
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": "Preserve file paths exactly when recalling them."},
        {"role": "user", "content": f"The audit log lives at {path}. Remember it."},
        {"role": "assistant", "content": f"Noted path {path}."},
        {"role": "user", "content": f"FINAL: Print exactly PATH={path}"},
    ]
    messages = pad_with_filler_turns(messages, rng, target_tokens)
    scoring = {
        "required_slots": {"path": path, "line": f"PATH={path}"},
        "mode": "strict",
    }
    return messages, scoring


def _build_user_pref_v1(
    rng: random.Random, target_tokens: int
) -> Tuple[List[Dict[str, str]], Mapping[str, Any]]:
    pref = rng.choice(["timezone=UTC", "theme=dark", "units=metric"])
    handle = rng.choice(["user_alpha", "user_bravo", "user_charlie"])
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": "Recall user preferences verbatim when asked."},
        {"role": "user", "content": f"Account {handle} prefers {pref}. Store that."},
        {"role": "assistant", "content": f"Stored preference for {handle}: {pref}."},
        {
            "role": "user",
            "content": f"FINAL: Output exactly PREF={pref} HANDLE={handle}",
        },
    ]
    messages = pad_with_filler_turns(messages, rng, target_tokens)
    scoring = {
        "required_slots": {
            "pref": pref,
            "handle": handle,
            "line": f"PREF={pref} HANDLE={handle}",
        },
        "mode": "strict",
    }
    return messages, scoring


def _build_order_id_v2(
    rng: random.Random, target_tokens: int
) -> Tuple[List[Dict[str, str]], Mapping[str, Any]]:
    order_id = f"ORD-{rng.randint(100000, 999999)}"
    decoys = _decoy_orders(rng, order_id)
    messages: List[Dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "You are an order desk agent. When asked later, repeat IDs "
                "exactly with no reformatting. Ignore decoy IDs in filler."
            ),
        },
        {
            "role": "user",
            "content": f"Create a hold for order {order_id}. Confirm the id.",
        },
        {"role": "assistant", "content": f"Hold created for {order_id}."},
        {
            "role": "user",
            "content": (
                "FINAL: Reply with exactly one line ORDER_ID=<the hold id "
                "created earlier>. Do not invent a new id."
            ),
        },
    ]
    messages = pad_with_filler_turns(messages, rng, target_tokens)
    for i, m in enumerate(messages):
        if m["role"] == "assistant" and "filler" in m["content"].lower():
            messages[i] = {
                "role": "assistant",
                "content": (
                    m["content"]
                    + f" Note unrelated cancelled tickets {', '.join(decoys)}."
                ),
            }
            break
    scoring = {
        "required_slots": {"order_id": order_id, "line": f"ORDER_ID={order_id}"},
        "mode": "strict",
    }
    return messages, scoring


def _build_file_path_v2(
    rng: random.Random, target_tokens: int
) -> Tuple[List[Dict[str, str]], Mapping[str, Any]]:
    path = rng.choice(
        [
            "/mnt/scratch/prioritybench/cal.jsonl",
            "/var/data/traces/session_a.jsonl",
            "/home/agent/cache/manifest.yaml",
        ]
    )
    decoy = rng.choice(
        [
            "/tmp/eval/run_0042/config.json",
            "/var/data/traces/session_b.jsonl",
            "/mnt/scratch/prioritybench/val.jsonl",
        ]
    )
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": "Preserve file paths exactly when recalling them."},
        {"role": "user", "content": f"The audit log lives at {path}. Remember it."},
        {"role": "assistant", "content": f"Noted path {path}."},
        {
            "role": "user",
            "content": (
                "FINAL: Print exactly PATH=<the audit log path remembered "
                f"earlier>. Do not use {decoy}."
            ),
        },
    ]
    messages = pad_with_filler_turns(messages, rng, target_tokens)
    scoring = {
        "required_slots": {"path": path, "line": f"PATH={path}"},
        "mode": "strict",
    }
    return messages, scoring


def _build_user_pref_v2(
    rng: random.Random, target_tokens: int
) -> Tuple[List[Dict[str, str]], Mapping[str, Any]]:
    pref = rng.choice(["timezone=UTC", "theme=dark", "units=metric"])
    handle = rng.choice(["user_alpha", "user_bravo", "user_charlie"])
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": "Recall user preferences verbatim when asked."},
        {"role": "user", "content": f"Account {handle} prefers {pref}. Store that."},
        {"role": "assistant", "content": f"Stored preference for {handle}: {pref}."},
        {
            "role": "user",
            "content": (
                "FINAL: Output exactly one line "
                "PREF=<stored preference> HANDLE=<account handle> "
                "using values stored earlier (no synonyms)."
            ),
        },
    ]
    messages = pad_with_filler_turns(messages, rng, target_tokens)
    scoring = {
        "required_slots": {
            "pref": pref,
            "handle": handle,
            "line": f"PREF={pref} HANDLE={handle}",
        },
        "mode": "strict",
    }
    return messages, scoring


MULTI_TURN_STATE_TEMPLATES_V1: tuple[TemplateSpec, ...] = (
    TemplateSpec("multi_turn_state.order_id.v1", Category.MULTI_TURN_STATE, _build_order_id_v1),
    TemplateSpec("multi_turn_state.file_path.v1", Category.MULTI_TURN_STATE, _build_file_path_v1),
    TemplateSpec("multi_turn_state.user_pref.v1", Category.MULTI_TURN_STATE, _build_user_pref_v1),
)

MULTI_TURN_STATE_TEMPLATES_V2: tuple[TemplateSpec, ...] = (
    TemplateSpec("multi_turn_state.order_id.v2", Category.MULTI_TURN_STATE, _build_order_id_v2),
    TemplateSpec("multi_turn_state.file_path.v2", Category.MULTI_TURN_STATE, _build_file_path_v2),
    TemplateSpec("multi_turn_state.user_pref.v2", Category.MULTI_TURN_STATE, _build_user_pref_v2),
)

# Default registry for new pilots = v2 (non-leaking).
MULTI_TURN_STATE_TEMPLATES: tuple[TemplateSpec, ...] = MULTI_TURN_STATE_TEMPLATES_V2
