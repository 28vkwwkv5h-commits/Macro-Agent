"""Confidence score and the deployment gate (spec 2.4, 18.4, 20.2).

    confidence = agreement * 0.5 + persistence * 0.3 + stability * 0.2

In the MVP there is no turbulence/absorption measure, so stability defaults to
1.0 (no fragility information, no penalty). Section 18.3's composite plugs in
through the `stability` argument later.
"""
from __future__ import annotations

import pandas as pd

from ..config import (
    CONF_W_AGREEMENT,
    CONF_W_PERSISTENCE,
    CONF_W_STABILITY,
    CONFIDENCE_FULL,
    CONFIDENCE_PARTIAL,
    FULL_POSITIONS,
    PARTIAL_POSITIONS,
    PERSISTENCE_CAP_DAYS,
)


def persistence(regime: pd.Series, cap: int = PERSISTENCE_CAP_DAYS) -> pd.Series:
    """Consecutive sessions in the current regime, capped and scaled to 0..1.

    Day one of a new regime scores 1/cap. No regime scores 0.
    """
    run = []
    prev, count = object(), 0
    for r in regime:
        if r is None or (isinstance(r, float) and pd.isna(r)):
            prev, count = object(), 0
            run.append(0)
            continue
        count = count + 1 if r == prev else 1
        prev = r
        run.append(min(count, cap))
    return pd.Series(run, index=regime.index, dtype=float) / cap


def confidence(classified: pd.DataFrame, stability: pd.Series | float = 1.0) -> pd.Series:
    score = (
        classified["agreement"] * CONF_W_AGREEMENT
        + persistence(classified["regime"]) * CONF_W_PERSISTENCE
        + stability * CONF_W_STABILITY
    )
    # No regime means no deployment, whatever the stability term says.
    return score.where(classified["regime"].notna(), 0.0)


def position_budget(conf: float) -> int:
    """How many positions the confidence allows: 5, 3 or 0 (cash, no exceptions)."""
    if conf >= CONFIDENCE_FULL:
        return FULL_POSITIONS
    if conf >= CONFIDENCE_PARTIAL:
        return PARTIAL_POSITIONS
    return 0
