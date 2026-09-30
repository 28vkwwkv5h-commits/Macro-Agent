"""MVP selector (spec 20.5, stages per section 12).

Each stage can only narrow what the previous one allowed:

    whitelist (regime) -> has data -> confirmation -> rank by drawdown -> top N

Ranking is on drawdown from the 12-month high, deepest first (Choi 2021): the
washed-out asset has the most room back to its high. Confirmation, a close above
the close one month earlier, is mandatory and is what separates this from
averaging down. Ties break alphabetically by ticker.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from ..config import CONFIRM_WINDOW, DRAWDOWN_WINDOW, WHITELIST


def drawdown_from_high(prices: pd.DataFrame, window: int = DRAWDOWN_WINDOW) -> pd.DataFrame:
    """1 - price / rolling high. 0 at a new high, 0.6 when 60% below it."""
    high = prices.rolling(window, min_periods=window).max()
    return 1 - prices / high


def confirmed(prices: pd.DataFrame, window: int = CONFIRM_WINDOW) -> pd.DataFrame:
    """True where the close is above the close `window` sessions earlier."""
    return prices > prices.shift(window)


@dataclass
class Selection:
    selected: list[str]
    scores: dict[str, float]
    stages: dict[str, list[str]] = field(default_factory=dict)


def select(prices: pd.DataFrame, regime: str | None, n_positions: int) -> Selection:
    """Choose up to `n_positions` tickers using prices up to and including the
    last row of `prices`. The caller is responsible for truncating to the
    decision date.
    """
    stages: dict[str, list[str]] = {}
    if regime is None or n_positions <= 0:
        stages["whitelist"] = []
        return Selection([], {}, stages)

    allowed = sorted(WHITELIST[regime])
    stages["whitelist"] = allowed

    if len(prices) <= max(DRAWDOWN_WINDOW, CONFIRM_WINDOW):
        stages["has_data"] = []
        return Selection([], {}, stages)
    window = prices.iloc[-(DRAWDOWN_WINDOW + 1):]
    dd = drawdown_from_high(window).iloc[-1]
    conf = confirmed(window).iloc[-1]

    with_data = [t for t in allowed if t in window.columns and pd.notna(dd.get(t))]
    stages["has_data"] = with_data

    passed = [t for t in with_data if bool(conf[t])]
    stages["confirmed"] = passed

    scores = {t: float(dd[t]) for t in passed}
    ranked = sorted(passed, key=lambda t: (-scores[t], t))
    stages["ranked"] = ranked

    chosen = ranked[:n_positions]
    stages["selected"] = chosen
    return Selection(chosen, scores, stages)
