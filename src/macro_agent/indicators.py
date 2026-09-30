"""Causal technical indicators. Every value at date t uses data up to t only."""
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

TRADING_DAYS = 252


def true_range(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
    prev = close.shift(1)
    return np.maximum(high - low, np.maximum((high - prev).abs(), (low - prev).abs()))


def atr(high, low, close, window: int = 14) -> pd.DataFrame:
    return true_range(high, low, close).rolling(window, min_periods=window).mean()


def rsi(close: pd.DataFrame | pd.Series, window: int = 14):
    """Wilder's RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def realized_vol(close: pd.DataFrame, window: int = 63) -> pd.DataFrame:
    return np.log(close).diff().rolling(window, min_periods=window).std() * np.sqrt(TRADING_DAYS)


def rolling_pct_rank(frame, window: int):
    """Percentile of today's value within the trailing window, in (0, 1]."""
    return frame.rolling(window, min_periods=max(20, window // 4)).rank(pct=True)


def rolling_slope(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    """OLS slope of each column over the trailing window, per session."""
    # sum_k (k - kbar) y_k, via rolling sums of y and of (global index * y).
    k = np.arange(len(frame), dtype=float)[:, None]
    y = frame.to_numpy(dtype=float)
    s_y = pd.DataFrame(y, index=frame.index).rolling(window, min_periods=window).sum().to_numpy()
    s_ky = pd.DataFrame(k * y, index=frame.index).rolling(window, min_periods=window).sum().to_numpy()
    start = k - window + 1
    kbar = (window - 1) / 2.0
    num = s_ky - start * s_y - kbar * s_y
    denom = window * (window**2 - 1) / 12.0
    return pd.DataFrame(num / denom, index=frame.index, columns=frame.columns)


def obv(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    direction = np.sign(close.diff()).fillna(0.0)
    return (direction * volume.fillna(0.0)).cumsum()


def max_drawdown_window(close: pd.DataFrame, window: int) -> pd.DataFrame:
    """Largest peak-to-trough fall inside the trailing window, as a positive fraction."""
    arr = close.to_numpy(dtype=float)
    out = np.full_like(arr, np.nan)
    if len(arr) >= window:
        for j in range(arr.shape[1]):
            win = sliding_window_view(arr[:, j], window)
            dd = 1.0 - (win / np.maximum.accumulate(win, axis=1)).min(axis=1)
            out[window - 1:, j] = dd  # NaN anywhere in a window propagates to NaN
    return pd.DataFrame(out, index=close.index, columns=close.columns)


# --- Weekly bars -----------------------------------------------------------


def week_end_mask(index: pd.DatetimeIndex) -> pd.Series:
    """True on the last session of each COMPLETED calendar week in `index`.

    The final session in the data only counts as a week end if it is a Friday:
    on a Wednesday the week is not over, and treating it as over would let a
    backtest see a weekly close live trading could not. (A Thursday before a
    Friday holiday is only recognised once the next week's data arrives.)
    """
    period = index.to_period("W-FRI")
    s = pd.Series(period, index=index)
    mask = s != s.shift(-1)
    if len(index) and index[-1].weekday() != 4:
        mask.iloc[-1] = False
    return mask


def weekly_bars(open_, high, low, close) -> dict[str, pd.DataFrame]:
    """Weekly OHLC for completed weeks, indexed by each week's last session."""
    mask = week_end_mask(close.index)
    key = close.index.to_period("W-FRI")
    ends = pd.Series(close.index, index=close.index).groupby(key).max()
    complete = set(close.index[mask])
    agg = {
        "open": open_.groupby(key).first(),
        "high": high.groupby(key).max(),
        "low": low.groupby(key).min(),
        "close": close.groupby(key).last(),
    }
    for k, v in agg.items():
        v.index = pd.DatetimeIndex(ends.loc[v.index].to_numpy())
        agg[k] = v[v.index.isin(complete)]
    return agg


# --- Pivots (audit: anchor swings to confirmed pivots) ---------------------


def confirmed_pivots(high: pd.Series, low: pd.Series, bars: int):
    """Pivot highs and lows, each reported on the session it becomes CONFIRMED.

    A pivot high at bar i is a high with `bars` bars either side that do not
    exceed it. It is only knowable at bar i + bars, so that is where it appears.
    Returns two lists of (pivot_pos, confirm_pos, price).
    """
    h = high.to_numpy(dtype=float)
    lo = low.to_numpy(dtype=float)
    n = len(h)
    highs, lows = [], []
    for i in range(bars, n - bars):
        wh = h[i - bars: i + bars + 1]
        if not np.isnan(wh).any() and h[i] == wh.max() and (wh[:bars] < h[i]).all():
            highs.append((i, i + bars, h[i]))
        wl = lo[i - bars: i + bars + 1]
        if not np.isnan(wl).any() and lo[i] == wl.min() and (wl[:bars] > lo[i]).all():
            lows.append((i, i + bars, lo[i]))
    return highs, lows


def anchored_swings(high: pd.Series, low: pd.Series, bars: int, lookback: int):
    """For every session, the swing the Fibonacci ladder hangs from.

    swing_high: the highest confirmed pivot high whose pivot lies within the
    trailing `lookback` sessions. swing_low: the lowest confirmed pivot low after
    that high. Levels only move when a new pivot confirms, so they cannot be
    refitted to recent price by construction.

    Returns a DataFrame with swing_high, swing_low, swing_high_pos, swing_low_pos
    (positions), NaN where no swing is formed.
    """
    n = len(high)
    highs, lows = confirmed_pivots(high, low, bars)
    out = np.full((n, 4), np.nan)
    hi_events: list[tuple[int, float]] = []
    lo_events: list[tuple[int, float]] = []
    hp = lp = 0
    for t in range(n):
        while hp < len(highs) and highs[hp][1] <= t:
            hi_events.append((highs[hp][0], highs[hp][2]))
            hp += 1
        while lp < len(lows) and lows[lp][1] <= t:
            lo_events.append((lows[lp][0], lows[lp][2]))
            lp += 1
        window_start = t - lookback
        best_h = None
        for pos, price in reversed(hi_events):
            if pos < window_start:
                break
            if best_h is None or price > best_h[1]:
                best_h = (pos, price)
        if best_h is None:
            continue
        best_l = None
        for pos, price in reversed(lo_events):
            if pos <= best_h[0]:
                break
            if best_l is None or price < best_l[1]:
                best_l = (pos, price)
        out[t, 0], out[t, 2] = best_h[1], best_h[0]
        if best_l is not None:
            out[t, 1], out[t, 3] = best_l[1], best_l[0]
    return pd.DataFrame(out, index=high.index, columns=["swing_high", "swing_low", "swing_high_pos", "swing_low_pos"])
