"""Page roles and storage dtype for mixed-precision paged KV."""

from __future__ import annotations

from enum import Enum


class StorageDtype(str, Enum):
    BF16 = "bf16"
    INT4 = "int4"


class PageRole(str, Enum):
    """Structural roles used by ProtectedRole / PriorityKV allocators."""

    SINK = "sink"  # early attention sinks / BOS-like pages — never demote
    SYSTEM = "system"
    TOOL = "tool"
    CONSTRAINT = "constraint"
    RECENT = "recent"  # newest W-token window
    GENERATED = "generated"
    FILLER = "filler"
    OTHER = "other"


# Roles that must stay BF16 unless the byte budget forces a demotion tie-break.
PROTECTED_ROLES: frozenset[PageRole] = frozenset(
    {
        PageRole.SINK,
        PageRole.SYSTEM,
        PageRole.TOOL,
        PageRole.CONSTRAINT,
        PageRole.RECENT,
    }
)

# Sinks are hard-protected: never demoted below BF16 (plan §4.1).
HARD_PROTECTED_ROLES: frozenset[PageRole] = frozenset({PageRole.SINK})
