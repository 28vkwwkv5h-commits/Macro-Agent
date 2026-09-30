"""Four-regime classifier from the six macro series (spec 2.3, audit-corrected).

Each input votes; the regime with the most votes wins and its vote share is the
agreement term of the confidence score. Derived from market inputs only, not
economic data.
"""
from __future__ import annotations

import pandas as pd

from ..config import (
    DEFLATION,
    DEFAULT_PARAMS,
    GOLDILOCKS,
    MACRO_SERIES,
    REFLATION,
    REGIME_ORDER,
    TIGHTENING,
    Params,
)
from .trend import DOWN, FLAT, UP, trend_state

# Special spread state for Deflation: steepening while the curve was inverted
# within the lookback.
STEEPENING_FROM_INVERSION = "steepening_from_inversion"

# Each cell is the set of trend states that count as a vote for that regime.
# Deflation follows the 28 September audit: the real rate RISES (expectations
# collapse faster than nominal yields), and gold reads down then up, so either
# direction votes and only a flat gold does not.
RULES = {
    REFLATION: {
        "oil": {UP}, "gold": {UP}, "usd": {DOWN},
        "ust10y": {UP}, "spread_2s10s": {UP}, "real10y": {DOWN},
    },
    GOLDILOCKS: {
        "oil": {FLAT, DOWN}, "gold": {FLAT}, "usd": {DOWN},
        "ust10y": {FLAT, DOWN}, "spread_2s10s": {FLAT}, "real10y": {DOWN},
    },
    TIGHTENING: {
        "oil": {UP}, "gold": {DOWN}, "usd": {UP},
        "ust10y": {UP}, "spread_2s10s": {DOWN}, "real10y": {UP},
    },
    DEFLATION: {
        "oil": {DOWN}, "gold": {DOWN, UP}, "usd": {UP},
        "ust10y": {DOWN}, "spread_2s10s": {STEEPENING_FROM_INVERSION}, "real10y": {UP},
    },
}

INPUTS = tuple(MACRO_SERIES)


def regime_states(macro: pd.DataFrame, params: Params = DEFAULT_PARAMS) -> pd.DataFrame:
    """Trend state per macro input, plus whether the 2/10 was recently inverted."""
    states = pd.DataFrame(index=macro.index)
    for name in INPUTS:
        kind = MACRO_SERIES[name][2]
        states[name] = trend_state(
            macro[name], kind=kind,
            roc_window=params.trend_roc_window, ma_window=params.trend_ma_window,
        )
    spread = macro["spread_2s10s"]
    states["spread_was_inverted"] = (
        spread.rolling(params.inversion_lookback, min_periods=1).min() < 0
    )
    return states


def votes(states: pd.DataFrame) -> pd.DataFrame:
    """Number of inputs voting for each regime, per day."""
    out = {}
    for regime in REGIME_ORDER:
        count = pd.Series(0, index=states.index)
        for name, allowed in RULES[regime].items():
            if STEEPENING_FROM_INVERSION in allowed:
                hit = (states[name] == UP) & states["spread_was_inverted"]
            else:
                hit = states[name].isin(sorted(allowed))
            count = count + hit.astype(int)
        out[regime] = count
    return pd.DataFrame(out)


def classify(macro: pd.DataFrame, params: Params = DEFAULT_PARAMS) -> pd.DataFrame:
    """Daily regime and agreement (fraction of the six inputs voting for it).

    Rows where any input lacks history get regime None and agreement 0, which
    the confidence gate turns into cash. Ties go to the more defensive regime
    (REGIME_ORDER).
    """
    states = regime_states(macro, params)
    v = votes(states)
    # idxmax returns the first maximum, and columns are in REGIME_ORDER.
    best = v.idxmax(axis=1)
    agreement = v.max(axis=1) / len(INPUTS)
    complete = states[list(INPUTS)].notna().all(axis=1)
    regime = best.astype(object).where(complete, None)
    return pd.DataFrame(
        {"regime": regime, "agreement": agreement.where(complete, 0.0)},
        index=macro.index,
    )
