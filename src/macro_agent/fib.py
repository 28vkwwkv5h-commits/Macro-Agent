"""Fibonacci ladders hung from confirmed swings (spec 2.2, audit, 14.1, 15).

On the six macro series the ladder describes where each input sits (spec 2.2).
On instruments it measures asymmetry: upside to the next resistance, downside
to the support the stop hangs from (spec 14.3).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .indicators import anchored_swings

RETRACEMENTS = (0.236, 0.382, 0.5, 0.618, 0.786)
EXTENSIONS = (1.272, 1.382, 1.5, 1.618)

APPROACHING, TESTING, REJECTED, BROKEN = "approaching", "testing", "rejected", "broken"


def ladder(swing_low: float, swing_high: float) -> np.ndarray:
    """All levels for a low-to-high swing: the low, retracements, the high, extensions."""
    rng = swing_high - swing_low
    ratios = (0.0,) + RETRACEMENTS + (1.0,) + EXTENSIONS
    return np.array([swing_low + r * rng for r in ratios])


def support_below(levels: np.ndarray, price: float) -> float | None:
    below = levels[levels < price]
    return float(below.max()) if below.size else None


def resistance_above(levels: np.ndarray, price: float) -> float | None:
    above = levels[levels > price]
    return float(above.min()) if above.size else None


def macro_fib_scores(series: pd.Series, bars: int, lookback: int, testing_atr: float = 0.5) -> pd.DataFrame:
    """Per-session fib_position, fib_proximity and fib_state for a macro series.

    The swing is the one from confirmed pivots, in whichever direction it runs.
    ATR for a single series is the mean absolute daily change over 14 sessions.
    """
    s = series.astype(float)
    swings = anchored_swings(s, s, bars, lookback)
    atr = s.diff().abs().rolling(14, min_periods=14).mean()
    pos = np.full(len(s), np.nan)
    prox = np.full(len(s), np.nan)
    state = np.array([None] * len(s), dtype=object)
    values = s.to_numpy()
    # The level last tested, and the side price approached it from.
    tested_level, approach_side, sessions_beyond = None, 0, 0
    for t in range(len(s)):
        hi, lo, a, p = swings.iat[t, 0], swings.iat[t, 1], atr.iat[t], values[t]
        if np.isnan(hi) or np.isnan(lo) or np.isnan(a) or a == 0 or np.isnan(p) or hi == lo:
            continue
        levels = ladder(lo, hi)
        pos[t] = (p - lo) / (hi - lo)
        nearest = levels[np.argmin(np.abs(levels - p))]
        prox[t] = abs(p - nearest) / a
        if prox[t] <= testing_atr:
            # Testing: within half an ATR of a level.
            if tested_level != nearest:
                prev = values[t - 1] if t else p
                tested_level, approach_side = nearest, int(np.sign(prev - nearest)) or 1
            sessions_beyond = 0
            state[t] = TESTING
            continue
        if tested_level is None:
            state[t] = APPROACHING
            continue
        side = int(np.sign(p - tested_level))
        if side != approach_side:
            # Closed beyond the level; two consecutive sessions is a break.
            sessions_beyond += 1
            state[t] = BROKEN if sessions_beyond >= 2 else APPROACHING
        else:
            # Touched and closed back on the side it came from.
            sessions_beyond = 0
            state[t] = REJECTED if abs(p - tested_level) <= 2 * testing_atr * a else APPROACHING
            if state[t] == APPROACHING:
                tested_level = None
    return pd.DataFrame({"fib_position": pos, "fib_proximity": prox, "fib_state": state}, index=s.index)
