"""W3 locked-structure check: same byte budget, different protected sets."""

from __future__ import annotations

from prioritykv.page_manager import PageManager, PageManagerConfig
from prioritykv.page_roles import PageRole, StorageDtype


def _tool_heavy_trace():
    return [
        {
            "role": "system",
            "content": "Available tools (JSON Schema): "
            + ("search_docs " * 80)
            + '[{"name":"search_docs"}]',
        },
        {"role": "user", "content": "noise " * 8000},
        {"role": "assistant", "content": "ack " * 500},
        {"role": "user", "content": "FINAL call search_docs"},
    ]


def test_structure_protects_tool_vs_uniform_pressure():
    """At tight budget, structural tagging keeps TOOL/SINK hotter than filler."""
    msgs = _tool_heavy_trace()
    structured = PageManager(PageManagerConfig(budget_frac=0.35, recent_window=64))
    structured.build_from_messages(msgs)

    # Uniform pressure simulation: rematerialize then demote TOOL pages first.
    uniform = PageManager(PageManagerConfig(budget_frac=0.35, recent_window=64))
    uniform.build_from_messages(msgs)
    for p in uniform.pages:
        if p.role == PageRole.TOOL:
            p.dtype = StorageDtype.INT4
    uniform.enforce_budget()

    tool_bf16_s = sum(
        p.n_tokens
        for p in structured.pages
        if p.role == PageRole.TOOL and p.dtype == StorageDtype.BF16
    )
    tool_bf16_u = sum(
        p.n_tokens
        for p in uniform.pages
        if p.role == PageRole.TOOL and p.dtype == StorageDtype.BF16
    )
    assert structured.check_invariants() == []
    # Structural policy should retain at least as many TOOL tokens in BF16.
    assert tool_bf16_s >= tool_bf16_u
