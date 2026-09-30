"""Confidence score and the deployment gate (spec 2.4, 18.4, 20.2).

    confidence = agreement * 0.5 + persistence * 0.3 + stability * 0.2

stability is 1 minus the turbulence/absorption composite (regime/stability.py).
It is what lets the system sit in cash without shame.
"""
from __future__ import annotations

import pandas as pd

from ..config import DEFAULT_PARAMS, Params


def persistence(regime: pd.Series, cap: int = 20) -> pd.Series:
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


def confidence(
    classified: pd.DataFrame, stability: pd.Series | float = 1.0, params: Params = DEFAULT_PARAMS
) -> pd.Series:
    score = (
        classified["agreement"] * params.conf_w_agreement
        + persistence(classified["regime"], params.persistence_cap_days) * params.conf_w_persistence
        + stability * params.conf_w_stability
    )
    # No regime means no deployment, whatever the stability term says.
    return score.where(classified["regime"].notna(), 0.0)


def position_budget(conf: float, params: Params = DEFAULT_PARAMS) -> int:
    """How many positions the confidence allows: 5, 3 or 0 (cash, no exceptions)."""
    if conf >= params.confidence_full:
        return params.full_positions
    if conf >= params.confidence_partial:
        return params.partial_positions
    return 0
