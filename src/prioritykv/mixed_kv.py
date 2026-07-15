"""Per-position BF16/INT4 plan for a real mixed-precision KV forward (W6 systems).

The keep experiments answered *which tokens matter*. This module answers the
serving question: at a fixed byte budget (a target INT4 fraction), *which token
positions* do we store in INT4 vs BF16?

Two matched-budget policies produce the SAME number of INT4 positions so the
comparison is byte-fair:

- ``structure``: protect sink + recent window + structural roles (tool / system /
  constraint / short state), demote the lowest-risk non-structure positions to
  INT4 first (spilling into structure by ascending risk only if the budget forces
  it). This is the PriorityKV decision.
- ``uniform``: demote positions evenly across the demotable range regardless of
  role (sink + recent still BF16, StreamingLLM-style). This is the control that
  quantizes structural state alongside filler.

Both return a bool mask where ``True`` means "store this position's KV in INT4".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from prioritykv.linear_risk import LinearRiskConfig, score_page
from prioritykv.page_roles import PROTECTED_ROLES, PageRole


@dataclass(frozen=True)
class MixedPlanConfig:
    """Byte-budget + protection knobs for the per-position dtype plan."""

    int4_frac: float = 0.75  # fraction of positions to store in INT4
    sink_tokens: int = 16
    recent_window: int = 128
    risk_fit_path: Optional[str] = None


def _protected_positions(n: int, cfg: MixedPlanConfig) -> np.ndarray:
    """Sink + recent window are always BF16 (RoPE / attention-sink guardrail)."""
    mask = np.zeros(n, dtype=bool)
    mask[: min(cfg.sink_tokens, n)] = True
    mask[max(0, n - cfg.recent_window) :] = True
    return mask


def _pos_risk(role: PageRole, risk_cfg: LinearRiskConfig) -> float:
    return score_page({"roles": [role.value], "n_tokens": 1}, risk_cfg)


def _int4_budget(n: int, cfg: MixedPlanConfig) -> int:
    forced_bf16 = int(_protected_positions(n, cfg).sum())
    max_int4 = max(0, n - forced_bf16)
    return int(min(max_int4, round(n * cfg.int4_frac)))


def plan_structure(
    roles: Sequence[PageRole],
    cfg: MixedPlanConfig,
    *,
    risk_cfg: Optional[LinearRiskConfig] = None,
) -> np.ndarray:
    """INT4 mask that keeps structural roles BF16, demoting lowest-risk first."""
    risk_cfg = risk_cfg or LinearRiskConfig()
    n = len(roles)
    int4 = np.zeros(n, dtype=bool)
    if n == 0:
        return int4
    forced_bf16 = _protected_positions(n, cfg)
    budget = _int4_budget(n, cfg)
    if budget <= 0:
        return int4
    is_struct = np.array(
        [r in PROTECTED_ROLES or r == PageRole.OTHER for r in roles], dtype=bool
    )
    # Demote order: non-structure ascending risk first, then structure ascending
    # risk (only if the byte budget cannot be met from filler alone). Never touch
    # sink/recent. Stable tie-break by position keeps it deterministic.
    order = sorted(
        (i for i in range(n) if not forced_bf16[i]),
        key=lambda i: (is_struct[i], _pos_risk(roles[i], risk_cfg), i),
    )
    for i in order[:budget]:
        int4[i] = True
    return int4


def plan_uniform(roles: Sequence[PageRole], cfg: MixedPlanConfig) -> np.ndarray:
    """Byte-matched control: demote evenly across demotable positions by role-blind stride."""
    n = len(roles)
    int4 = np.zeros(n, dtype=bool)
    if n == 0:
        return int4
    forced_bf16 = _protected_positions(n, cfg)
    budget = _int4_budget(n, cfg)
    cands = [i for i in range(n) if not forced_bf16[i]]
    if budget <= 0 or not cands:
        return int4
    # Evenly spaced pick so the control is not positionally biased toward the
    # prefix (that would accidentally protect mid-context state).
    idx = np.linspace(0, len(cands) - 1, num=budget, dtype=int)
    for j in np.unique(idx):
        int4[cands[int(j)]] = True
    # np.unique may drop to < budget on collisions; top up in order to stay matched.
    if int(int4.sum()) < budget:
        for i in cands:
            if int(int4.sum()) >= budget:
                break
            int4[i] = True
    return int4


def plan_int4_mask(
    roles: Sequence[PageRole],
    cfg: MixedPlanConfig,
    *,
    policy: str,
    risk_cfg: Optional[LinearRiskConfig] = None,
) -> np.ndarray:
    """Dispatch: ``structure`` (role-aware) or ``uniform`` (byte-matched control)."""
    if policy == "structure":
        return plan_structure(roles, cfg, risk_cfg=risk_cfg)
    if policy == "uniform":
        return plan_uniform(roles, cfg)
    raise ValueError(f"unknown mixed policy {policy}")
