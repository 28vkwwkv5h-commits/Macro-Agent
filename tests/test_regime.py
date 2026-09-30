import numpy as np
import pandas as pd
import pytest

from macro_agent.config import DEFLATION, GOLDILOCKS, REFLATION, TIGHTENING
from macro_agent.regime import (
    DOWN,
    FLAT,
    UP,
    classify,
    confidence,
    persistence,
    position_budget,
    trend_state,
    votes,
)

INPUTS = ["oil", "gold", "usd", "ust10y", "spread_2s10s", "real10y"]


def states_row(inverted=False, **states):
    row = {k: states.get(k, FLAT) for k in INPUTS}
    row["spread_was_inverted"] = inverted
    return pd.DataFrame([row], index=[pd.Timestamp("2020-01-01")])


def test_trend_up_down_and_unconfirmed():
    idx = pd.bdate_range("2020-01-01", periods=300)
    rising = pd.Series(np.linspace(100, 200, 300), index=idx)
    assert trend_state(rising).iloc[-1] == UP
    assert trend_state(rising[::-1].set_axis(idx)).iloc[-1] == DOWN
    # Up over 63 days but still below the 200-day average: not confirmed.
    v = np.concatenate([np.linspace(200, 100, 250), np.linspace(100, 120, 50)])
    assert trend_state(pd.Series(v, index=idx)).iloc[-1] == FLAT
    assert np.isnan(trend_state(rising).iloc[150])  # not enough history


def test_rate_series_trend_on_absolute_change():
    idx = pd.bdate_range("2020-01-01", periods=300)
    spread = pd.Series(np.linspace(-1.0, 0.5, 300), index=idx)  # crosses zero
    assert trend_state(spread, kind="rate").iloc[-1] == UP


def test_deflation_uses_rising_real_rate_per_audit():
    s = states_row(
        inverted=True, oil=DOWN, gold=DOWN, usd=UP, ust10y=DOWN, spread_2s10s=UP, real10y=UP
    )
    v = votes(s).iloc[0]
    assert v[DEFLATION] == 6
    # Gold recovering in the second leg still votes deflation.
    s2 = states_row(
        inverted=True, oil=DOWN, gold=UP, usd=UP, ust10y=DOWN, spread_2s10s=UP, real10y=UP
    )
    assert votes(s2).iloc[0][DEFLATION] == 6


def test_deflation_spread_vote_needs_prior_inversion():
    s = states_row(
        inverted=False, oil=DOWN, gold=DOWN, usd=UP, ust10y=DOWN, spread_2s10s=UP, real10y=UP
    )
    assert votes(s).iloc[0][DEFLATION] == 5


@pytest.mark.parametrize(
    "states,expected",
    [
        (dict(oil=UP, gold=UP, usd=DOWN, ust10y=UP, spread_2s10s=UP, real10y=DOWN), REFLATION),
        (dict(oil=FLAT, gold=FLAT, usd=DOWN, ust10y=FLAT, spread_2s10s=FLAT, real10y=DOWN), GOLDILOCKS),
        (dict(oil=UP, gold=DOWN, usd=UP, ust10y=UP, spread_2s10s=DOWN, real10y=UP), TIGHTENING),
    ],
)
def test_each_regime_row_scores_six(states, expected):
    v = votes(states_row(**states)).iloc[0]
    assert v[expected] == 6
    assert v.idxmax() == expected


def test_tie_goes_to_more_defensive_regime():
    # 3 votes each for tightening (usd, ust10y... ) and reflation.
    s = states_row(oil=UP, gold=UP, usd=UP, ust10y=UP, spread_2s10s=FLAT, real10y=FLAT)
    v = votes(s).iloc[0]
    assert v[TIGHTENING] == v[REFLATION] == 3
    assert v.idxmax() == TIGHTENING


def test_persistence_counts_from_one_and_caps():
    regimes = pd.Series([None, "a", "a", "b"] + ["b"] * 30)
    p = persistence(regimes, cap=20)
    assert p.iloc[0] == 0
    assert p.iloc[1] == pytest.approx(1 / 20)
    assert p.iloc[2] == pytest.approx(2 / 20)
    assert p.iloc[3] == pytest.approx(1 / 20)  # new regime resets
    assert p.iloc[-1] == 1.0


def test_confidence_matches_audit_arithmetic():
    # Audit: agreement 0.83, clarity 0.8 -> day one 0.59, day five 0.65.
    classified = pd.DataFrame({"regime": ["x"] * 5, "agreement": [5 / 6] * 5})
    c = confidence(classified, stability=0.8)
    assert round(c.iloc[0], 2) == 0.59
    assert round(c.iloc[4], 2) == 0.65


def test_position_budget_tiers():
    assert position_budget(0.60) == 5
    assert position_budget(0.59) == 3
    assert position_budget(0.40) == 3
    assert position_budget(0.3999) == 0


def test_classify_without_history_is_no_regime(market):
    _, macro = market
    classified = classify(macro)
    assert classified["regime"].iloc[0] is None
    assert confidence(classified).iloc[0] == 0.0
    assert classified["regime"].iloc[-1] in {REFLATION, GOLDILOCKS, TIGHTENING, DEFLATION}
