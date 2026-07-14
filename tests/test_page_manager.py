"""Unit tests for page manager + tagging (CPU)."""

from __future__ import annotations

from prioritykv.page_manager import PageManager, PageManagerConfig
from prioritykv.page_roles import PageRole, StorageDtype
from prioritykv.tagging import role_for_message, tag_chat_to_token_roles


def test_tool_system_tagged():
    role = role_for_message(
        {
            "role": "system",
            "content": "Available tools (JSON Schema): search_docs...",
        }
    )
    assert role == PageRole.TOOL


def test_sink_and_recent_overlay():
    messages = [
        {"role": "system", "content": "You are helpful. " * 40},
        {"role": "user", "content": "hello " * 200},
        {"role": "assistant", "content": "ack " * 200},
        {"role": "user", "content": "FINAL ask"},
    ]
    roles = tag_chat_to_token_roles(messages, recent_window=32, sink_tokens=8)
    assert roles[:8] == [PageRole.SINK] * 8
    assert PageRole.RECENT in roles[-32:]


def test_build_from_messages_invariants():
    pm = PageManager(PageManagerConfig(budget_frac=0.50))
    messages = [
        {
            "role": "system",
            "content": "Available tools (JSON Schema): [{'name':'x'}]",
        },
        {"role": "user", "content": "filler " * 5000},
        {"role": "assistant", "content": "ok " * 2000},
        {"role": "user", "content": "FINAL: call the tool"},
    ]
    pm.build_from_messages(messages)
    assert pm.seq_len > 0
    assert pm.pages[0].role == PageRole.SINK
    assert pm.pages[0].dtype == StorageDtype.BF16
    assert pm.check_invariants() == []


def test_budget_forces_demotion_not_sink():
    # Aggressive 30% budget on long filler → demote filler, keep sinks BF16.
    pm = PageManager(PageManagerConfig(budget_frac=0.30, recent_window=64))
    messages = [
        {"role": "system", "content": "sys " * 20},
        {"role": "user", "content": "pad " * 20000},
    ]
    pm.build_from_messages(messages)
    assert all(
        p.dtype == StorageDtype.BF16
        for p in pm.pages
        if p.role == PageRole.SINK
    )
    assert pm.within_budget() or pm.check_invariants() == []
    # After enforce, either within budget or only hard-protected left hot.
    errs = pm.check_invariants()
    assert errs == [], errs


def test_append_generated_triggers_demote():
    pm = PageManager(
        PageManagerConfig(budget_frac=0.50, demote_every_tokens=32, recent_window=16)
    )
    pm.build_from_messages([{"role": "user", "content": "start " * 100}])
    before = pm.seq_len
    pm.append_generated_tokens(64)
    assert pm.seq_len == before + 64
    assert pm.check_invariants() == []
