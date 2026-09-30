import numpy as np
import pandas as pd
import pytest

from macro_agent.fib import BROKEN, TESTING, ladder, macro_fib_scores, resistance_above, support_below
from macro_agent.indicators import (
    anchored_swings,
    confirmed_pivots,
    max_drawdown_window,
    rolling_slope,
    week_end_mask,
    weekly_bars,
)


def test_rolling_slope_matches_polyfit():
    rng = np.random.default_rng(0)
    y = pd.DataFrame({"a": rng.normal(size=100).cumsum()})
    got = rolling_slope(y, 20)["a"]
    for t in (19, 50, 99):
        expected = np.polyfit(np.arange(20), y["a"].iloc[t - 19: t + 1], 1)[0]
        assert got.iloc[t] == pytest.approx(expected)
    assert np.isnan(got.iloc[18])


def test_max_drawdown_window():
    close = pd.DataFrame({"a": [10, 12, 9, 11, 6, 8.0]})
    dd = max_drawdown_window(close, 4)["a"]
    assert np.isnan(dd.iloc[2])
    assert dd.iloc[3] == pytest.approx(1 - 9 / 12)
    assert dd.iloc[5] == pytest.approx(1 - 6 / 11)  # window [9, 11, 6, 8]


def test_final_session_is_a_week_end_only_on_friday():
    wed = pd.bdate_range("2024-01-01", "2024-01-10")  # ends Wednesday
    assert not week_end_mask(wed).iloc[-1]
    assert week_end_mask(wed).loc["2024-01-05"]  # the Friday inside is complete
    fri = pd.bdate_range("2024-01-01", "2024-01-12")
    assert week_end_mask(fri).iloc[-1]


def test_weekly_bars_drop_the_incomplete_week():
    idx = pd.bdate_range("2024-01-01", "2024-01-10")
    close = pd.DataFrame({"a": np.arange(len(idx), dtype=float) + 1}, index=idx)
    w = weekly_bars(close, close + 1, close - 1, close)
    assert list(w["close"].index) == [pd.Timestamp("2024-01-05")]
    assert w["high"]["a"].iloc[0] == 6.0


def test_pivots_are_reported_when_confirmed():
    high = pd.Series([1, 2, 3, 10, 3, 2, 1, 1, 1.0])
    highs, lows = confirmed_pivots(high, high, 2)
    assert highs == [(3, 5, 10.0)]  # pivot at bar 3, knowable at bar 5


def test_anchored_swing_is_causal_and_sticky():
    # Rise to a peak, fall to a trough, then drift: the swing appears only once
    # both pivots confirm, and does not move while price drifts.
    v = np.concatenate([np.linspace(50, 100, 30), np.linspace(100, 60, 20), np.linspace(60, 70, 30)])
    s = pd.Series(v)
    sw = anchored_swings(s, s, bars=3, lookback=252)
    assert np.isnan(sw["swing_high"].iloc[31])  # peak at 29, confirmed at 32
    assert sw["swing_high"].iloc[40] == 100
    assert np.isnan(sw["swing_low"].iloc[50])  # trough at 49, confirmed at 52
    assert sw["swing_low"].iloc[60] == 60
    assert sw["swing_low"].iloc[79] == 60
    full = anchored_swings(s, s, 3, 252)
    part = anchored_swings(s.iloc[:55], s.iloc[:55], 3, 252)
    pd.testing.assert_frame_equal(full.iloc[:55], part)


def test_ladder_and_levels():
    lv = ladder(100, 200)
    assert list(lv[:7]) == pytest.approx([100, 123.6, 138.2, 150, 161.8, 178.6, 200])
    assert lv[-1] == pytest.approx(261.8)
    assert support_below(lv, 155) == pytest.approx(150)
    assert resistance_above(lv, 155) == pytest.approx(161.8)
    assert support_below(lv, 90) is None


def test_macro_fib_states():
    # Rise, fall to form a swing, then come back up to a level and through it.
    v = np.concatenate([np.linspace(1, 3, 40), np.linspace(3, 1, 40), np.linspace(1, 2.5, 60)])
    scores = macro_fib_scores(pd.Series(v), bars=5, lookback=252)
    assert scores["fib_position"].dropna().between(-0.5, 2.0).all()
    states = set(scores["fib_state"].dropna())
    assert TESTING in states
    assert states <= {TESTING, BROKEN, "approaching", "rejected"}
