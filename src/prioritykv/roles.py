"""Page roles and protection policy for PriorityKV."""

from __future__ import annotations

from enum import Enum


class StorageDtype(str, Enum):
    BF16 = "bf16"
    INT4 = "int4"


class PageRole(str, Enum):
    """Structural roles used by ProtectedRole / P2 policies."""

    SINK = "sink"  # early attention-sink tokens; never demote
    SYSTEM = "system"
    TOOL_SCHEMA = "tool_schema"
    CONSTRAINT = "constraint"
    RECENT = "recent"  # newest W-token window
    GENERATED = "generated"
    FILLER = "filler"
    OTHER = "other"


# Roles that must stay BF16 unless the byte budget *forces* demotion (then
# linear risk score breaks ties — W4+). W2 enforces "never prefer demoting these".
PROTECTED_ROLES: frozenset[PageRole] = frozenset(
    {
        PageRole.SINK,
        PageRole.SYSTEM,
        PageRole.TOOL_SCHEMA,
        PageRole.CONSTRAINT,
        PageRole.RECENT,
    }
)

# Demotion preference (first demoted → last). Protected roles omitted.
DEMOTION_ORDER: tuple[PageRole, ...] = (
    PageRole.FILLER,
    PageRole.OTHER,
    PageRole.GENERATED,
)
