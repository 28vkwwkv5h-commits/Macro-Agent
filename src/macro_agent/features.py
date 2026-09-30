"""Every signal input, computed once for the whole history.

All features are causal: the row for date t depends only on data up to t. The
strategy reads rows; it never recomputes. `tests/test_features.py` proves
causality by rebuilding on truncated data and comparing.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import DEFAULT_PARAMS, MACRO_SERIES, STABILITY_BASKET, Params
from .data.market import Market
from .fib import macro_fib_scores
from .flows import flow_score
from .indicators import (
    anchored_swings,
    atr,
    confirmed_pivots,
    max_drawdown_window,
    realized_vol,
    rsi,
    week_end_mask,
    weekly_bars,
)
from .regime import classify, confidence, stability_score


@dataclass
class Features:
    params: Params
    dates: pd.DatetimeIndex
    regime: pd.DataFrame  # regime, agreement, stability, confidence
    macro_fib: dict[str, pd.DataFrame]
    atr: pd.DataFrame
    vol: pd.DataFrame
    high_52w: pd.DataFrame
    mdd: pd.DataFrame
    returns: pd.DataFrame
    flow: pd.DataFrame
    swing_high: pd.DataFrame
    swing_low: pd.DataFrame
    is_week_end: pd.Series
    # Weekly bars, indexed by each week's last session.
    w_high: pd.DataFrame
    w_low: pd.DataFrame
    w_close: pd.DataFrame
    # Recovery confirmation (spec 14.1, 17.3), known from each week's close and
    # carried forward to the following sessions.
    confirmed: pd.DataFrame
    confirm_low: pd.DataFrame


def _weekly_confirmation(w: dict[str, pd.DataFrame], params: Params):
    """Confirmation on weekly bars: a reclaim of the 10-week MA, a confirmed
    higher low, or a positive RSI divergence at the last two pivot lows.

    confirm_low is the lowest weekly low of the recent base: the level whose
    loss on a weekly close invalidates the setup (spec 14.4).
    """
    close, low = w["close"], w["low"]
    ma = close.rolling(params.ten_week_ma, min_periods=params.ten_week_ma).mean()
    above = close > ma
    lookback = params.reclaim_lookback_weeks
    was_below = (~above).astype(float).rolling(lookback, min_periods=1).max().shift(1) > 0
    reclaim = above & was_below

    w_rsi = rsi(close, 14)
    higher_low = pd.DataFrame(False, index=close.index, columns=close.columns)
    divergence = pd.DataFrame(False, index=close.index, columns=close.columns)
    recent = 8  # a pivot low older than eight weeks no longer describes the turn
    for col in close.columns:
        _, lows = confirmed_pivots(low[col], low[col], 2)
        for k in range(1, len(lows)):
            (p0, _, l0), (p1, c1, l1) = lows[k - 1], lows[k]
            end = min(c1 + recent, len(close))
            if l1 > l0:
                higher_low.iloc[c1:end, higher_low.columns.get_loc(col)] = True
            r0, r1 = w_rsi[col].iat[p0], w_rsi[col].iat[p1]
            if l1 < l0 and r1 > r0:
                divergence.iloc[c1:end, divergence.columns.get_loc(col)] = True
    higher_low &= close > low.rolling(recent, min_periods=1).min()
    confirmed = (reclaim | higher_low | divergence) & close.notna()
    confirm_low = low.rolling(recent, min_periods=1).min()
    return confirmed, confirm_low


def build_features(market: Market, params: Params = DEFAULT_PARAMS) -> Features:
    market.validate()
    p = params
    close, high, low = market.close, market.high, market.low

    classified = classify(market.macro, p)
    stab = (
        stability_score(close, STABILITY_BASKET, p.stability_window, p.stability_rank_window,
                        p.absorption_fraction, p.absorption_short)
        if p.use_stability else pd.Series(1.0, index=close.index)
    )
    regime = classified.assign(stability=stab, confidence=confidence(classified, stab, p))

    macro_fib = {
        name: macro_fib_scores(market.macro[name], p.pivot_bars, p.swing_lookback, p.fib_testing_atr)
        for name in MACRO_SERIES if name in market.macro and market.macro[name].notna().any()
    }

    swings = {c: anchored_swings(high[c], low[c], p.pivot_bars, p.swing_lookback) for c in close.columns}
    swing_high = pd.DataFrame({c: s["swing_high"] for c, s in swings.items()}, index=close.index)
    swing_low = pd.DataFrame({c: s["swing_low"] for c, s in swings.items()}, index=close.index)

    w = weekly_bars(market.open, high, low, close)
    w_conf, w_conf_low = _weekly_confirmation(w, p)
    carry = lambda f: f.astype(float).reindex(close.index).ffill()  # noqa: E731

    flow = flow_score(market, p) if p.use_flows else pd.DataFrame(np.nan, index=close.index, columns=close.columns)

    return Features(
        params=p,
        dates=close.index,
        regime=regime,
        macro_fib=macro_fib,
        atr=atr(high, low, close, p.atr_window),
        vol=realized_vol(close, p.vol_window),
        high_52w=high.rolling(252, min_periods=252).max(),
        mdd=max_drawdown_window(close, p.drawdown_window),
        returns=np.log(close).diff(),
        flow=flow,
        swing_high=swing_high,
        swing_low=swing_low,
        is_week_end=week_end_mask(close.index),
        w_high=w["high"],
        w_low=w["low"],
        w_close=w["close"],
        confirmed=carry(w_conf).fillna(0.0) > 0,
        confirm_low=carry(w_conf_low),
    )
