import numpy as np
import pandas as pd
import pytest

from macro_agent.backtest import (
    BacktestConfig,
    TrialLedger,
    deflated_sharpe,
    max_drawdown,
    month_end_dates,
    run_backtest,
    simulate,
)

from .conftest import flat_prices


def test_zero_delay_is_refused():
    with pytest.raises(ValueError, match="look-ahead"):
        BacktestConfig("2008-01-01", "2010-01-01", execution_delay=0).validate()


def test_holdout_is_locked_by_default():
    with pytest.raises(ValueError, match="holdout"):
        BacktestConfig("2022-06-01", "2024-01-01").validate()
    cfg = BacktestConfig("2010-01-01", "2024-01-01")
    assert cfg.effective_end() == pd.Timestamp("2021-12-31")
    unlocked = BacktestConfig("2010-01-01", "2024-01-01", use_holdout=True)
    assert unlocked.effective_end() == pd.Timestamp("2024-01-01")


def test_costs_are_charged_on_turnover():
    p = flat_prices(["A"], n=5)
    equity, turnover = simulate(p, {p.index[1]: {"A": 1.0}}, cost_bps=15, cash_ticker=None)
    assert turnover.iloc[1] == 1.0
    assert equity.iloc[-1] == pytest.approx(1 - 0.0015)


def test_returns_accrue_only_after_execution():
    p = flat_prices(["A"], n=5)
    p.iloc[2:, 0] = 110.0  # +10% on the execution day itself
    equity, _ = simulate(p, {p.index[2]: {"A": 1.0}}, cost_bps=0, cash_ticker=None)
    assert equity.iloc[-1] == pytest.approx(1.0)  # bought at the close after the jump
    equity, _ = simulate(p, {p.index[1]: {"A": 1.0}}, cost_bps=0, cash_ticker=None)
    assert equity.iloc[-1] == pytest.approx(1.1)


def test_weights_drift_between_rebalances():
    p = flat_prices(["A", "B"], n=3)
    p.iloc[2, 0] = 200.0
    equity, _ = simulate(p, {p.index[0]: {"A": 0.5, "B": 0.5}}, cost_bps=0, cash_ticker=None)
    assert equity.iloc[-1] == pytest.approx(1.5)


def test_month_end_dates():
    idx = pd.bdate_range("2021-01-01", "2021-03-31")
    assert [d.day for d in month_end_dates(idx)] == [29, 26, 31]


def test_backtest_equity_unaffected_by_future_data(market):
    prices, macro = market
    cfg = BacktestConfig("2009-01-01", "2012-12-31")
    base = run_backtest(prices, macro, cfg)
    p2, m2 = prices.copy(), macro.copy()
    p2.loc["2011-07-01":] *= np.linspace(1, 2, len(p2.loc["2011-07-01":]))[:, None]
    m2.loc["2011-07-01":] += 5.0
    moved = run_backtest(p2, m2, cfg)
    cutoff = "2011-06-30"
    pd.testing.assert_series_equal(base.equity.loc[:cutoff], moved.equity.loc[:cutoff])


def test_backtest_is_reproducible(market):
    prices, macro = market
    cfg = BacktestConfig("2009-01-01", "2010-12-31")
    a = run_backtest(prices, macro, cfg)
    b = run_backtest(prices, macro, cfg)
    pd.testing.assert_series_equal(a.equity, b.equity)
    assert [d.to_dict() for d in a.decisions] == [d.to_dict() for d in b.decisions]


def test_max_drawdown():
    eq = pd.Series([1.0, 1.2, 0.9, 1.3])
    assert max_drawdown(eq) == pytest.approx(0.9 / 1.2 - 1)


def test_more_trials_deflate_the_sharpe():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.0005, 0.01, 2000))
    one = deflated_sharpe(r, n_trials=1)
    many = deflated_sharpe(r, n_trials=100, trial_sharpes=list(rng.normal(0, 0.03, 100)))
    assert one["dsr"] > 0.5
    assert many["sr0"] > 0
    assert many["dsr"] < one["dsr"]


def test_ledger_is_append_only(tmp_path):
    ledger = TrialLedger(tmp_path / "ledger.jsonl")
    for i in range(3):
        ledger.record("v", {"i": i}, 0.01 * i, 0.16 * i, ("2008-01-01", "2021-12-31"), False)
    assert ledger.n_trials == 3
    assert ledger.sharpes() == [0.0, 0.01, 0.02]
    assert TrialLedger(tmp_path / "ledger.jsonl").n_trials == 3
