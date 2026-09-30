import numpy as np
import pandas as pd

from macro_agent.ban import select
from macro_agent.config import DEFLATION, GOLDILOCKS
from macro_agent.pipeline import decide
from macro_agent.sizing import equal_weights, target_shares

from .conftest import flat_prices


def shaped(prices, ticker, drawdown, bounce):
    """Give `ticker` a peak, a fall of `drawdown`, then a one-month move of `bounce`."""
    n = len(prices)
    path = np.full(n, 100.0)
    path[n - 200:] = 100 * (1 - drawdown)
    path[-21:] = 100 * (1 - drawdown) * (1 + bounce)
    prices[ticker] = path
    return prices


def test_whitelist_excludes_off_regime_assets():
    p = flat_prices(["TLT", "IEF", "SPY"])
    p = shaped(p, "SPY", 0.5, 0.05)  # best score, wrong regime
    p = shaped(p, "TLT", 0.1, 0.05)
    sel = select(p, DEFLATION, 5)
    assert "SPY" not in sel.selected
    assert sel.selected == ["TLT"]


def test_confirmation_is_mandatory():
    p = flat_prices(["TLT", "IEF"])
    p = shaped(p, "TLT", 0.6, -0.02)  # deepest drawdown, still falling
    p = shaped(p, "IEF", 0.1, 0.02)
    sel = select(p, DEFLATION, 5)
    assert sel.stages["confirmed"] == ["IEF"]
    assert sel.selected == ["IEF"]


def test_rank_deepest_drawdown_first_ties_alphabetical():
    p = flat_prices(["SPY", "QQQ", "IWM", "MDY", "XLK", "SMH"])
    for t in ["SPY", "QQQ", "IWM"]:
        p = shaped(p, t, 0.2, 0.05)
    p = shaped(p, "SMH", 0.4, 0.05)
    sel = select(p, GOLDILOCKS, 3)
    assert sel.stages["ranked"][:4] == ["SMH", "IWM", "QQQ", "SPY"]
    assert sel.selected == ["SMH", "IWM", "QQQ"]


def test_zero_budget_selects_nothing():
    p = shaped(flat_prices(["TLT"]), "TLT", 0.3, 0.05)
    assert select(p, DEFLATION, 0).selected == []
    assert select(p, None, 5).selected == []


def test_equal_weight_leaves_residual_cash():
    w = equal_weights(["B", "A", "C"])
    assert w == {"A": 0.2, "B": 0.2, "C": 0.2}
    assert round(1 - sum(w.values()), 10) == 0.4


def test_target_shares_rounds_down_once():
    shares = target_shares({"A": 0.2, "B": 0.2}, 10_000, {"A": 300.0, "B": 3000.0})
    assert shares == {"A": 6}  # 2000/3000 rounds to 0 and is dropped
    frac = target_shares({"A": 0.2}, 10_000, {"A": 300.0}, fractional_increment=0.001)
    assert frac == {"A": 6.666}


def test_decide_is_deterministic(market):
    prices, macro = market
    a = decide(prices, macro, "2011-06-30").to_dict()
    b = decide(prices, macro, "2011-06-30").to_dict()
    assert a == b


def test_decide_ignores_data_after_as_of(market):
    prices, macro = market
    before = decide(prices, macro, "2010-03-31").to_dict()
    p2, m2 = prices.copy(), macro.copy()
    p2.loc["2010-04-01":] *= 3.0
    m2.loc["2010-04-01":] = -m2.loc["2010-04-01":]
    assert decide(p2, m2, "2010-03-31").to_dict() == before


def test_decision_weights_and_cash_sum_to_one(market):
    prices, macro = market
    for as_of in ["2008-12-31", "2010-06-30", "2012-11-30"]:
        d = decide(prices, macro, as_of)
        assert len(d.selected) <= d.n_positions
        assert abs(sum(d.weights.values()) + d.cash - 1.0) < 1e-9
        assert all(w <= 0.25 for w in d.weights.values())
