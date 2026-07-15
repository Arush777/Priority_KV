"""Linear page-risk score (W4) — heuristic / fit skeleton for ProtectedRole++ ties.

Shipping policy uses structural rules first; this linear score only breaks ties
among unprotected pages. Fit expects atlas / page-perturb score_delta labels.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class LinearRiskConfig:
    """Feature weights: higher → prefer keeping page in BF16."""

    w_is_tool: float = 1.2
    w_is_system: float = 1.0
    w_is_constraint: float = 1.1
    w_is_sink: float = 0.8
    w_is_recent: float = 0.6
    w_token_mass: float = 0.05
    bias: float = 0.0


def page_features(meta: Mapping[str, Any]) -> dict[str, float]:
    """Extract a fixed feature vector from page / role metadata."""
    roles = {str(r).lower() for r in (meta.get("roles") or [])}
    return {
        "is_tool": 1.0 if ("tool" in roles or "tool_schema" in roles) else 0.0,
        "is_system": 1.0 if "system" in roles else 0.0,
        "is_constraint": 1.0 if ("constraint" in roles or "instruction" in roles) else 0.0,
        "is_sink": 1.0 if "sink" in roles else 0.0,
        "is_recent": 1.0 if "recent" in roles else 0.0,
        "token_mass": float(meta.get("n_tokens", meta.get("token_mass", 0)) or 0),
    }


def score_page(meta: Mapping[str, Any], cfg: Optional[LinearRiskConfig] = None) -> float:
    cfg = cfg or LinearRiskConfig()
    f = page_features(meta)
    return (
        cfg.bias
        + cfg.w_is_tool * f["is_tool"]
        + cfg.w_is_system * f["is_system"]
        + cfg.w_is_constraint * f["is_constraint"]
        + cfg.w_is_sink * f["is_sink"]
        + cfg.w_is_recent * f["is_recent"]
        + cfg.w_token_mass * f["token_mass"]
    )


def fit_ridge(
    rows: Sequence[Mapping[str, Any]],
    *,
    l2: float = 1e-2,
) -> LinearRiskConfig:
    """Fit linear weights from rows with keys features + target ``score_delta``.

    Each row: either precomputed feature keys or ``meta`` + ``score_delta``.
    Positive score_delta = keeping the page helped (prefer BF16).
    """
    if not rows:
        return LinearRiskConfig()
    xs = []
    ys = []
    for r in rows:
        if "meta" in r:
            f = page_features(r["meta"])  # type: ignore[arg-type]
        else:
            f = {k: float(r.get(k, 0.0)) for k in (
                "is_tool", "is_system", "is_constraint", "is_sink", "is_recent", "token_mass"
            )}
        xs.append([
            1.0,
            f["is_tool"],
            f["is_system"],
            f["is_constraint"],
            f["is_sink"],
            f["is_recent"],
            f["token_mass"],
        ])
        ys.append(float(r["score_delta"]))
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    xtx = x.T @ x + l2 * np.eye(x.shape[1])
    xty = x.T @ y
    w = np.linalg.solve(xtx, xty)
    return LinearRiskConfig(
        bias=float(w[0]),
        w_is_tool=float(w[1]),
        w_is_system=float(w[2]),
        w_is_constraint=float(w[3]),
        w_is_sink=float(w[4]),
        w_is_recent=float(w[5]),
        w_token_mass=float(w[6]),
    )


def status(cfg: Optional[LinearRiskConfig] = None) -> dict[str, Any]:
    cfg = cfg or LinearRiskConfig()
    return {
        "name": "linear_page_risk",
        "config": asdict(cfg),
        "role": "tie-break among unprotected pages after structural rules",
    }
