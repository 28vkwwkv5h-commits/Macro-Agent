import numpy as np
import pandas as pd
import pytest

from macro_agent.backtest import (
    BacktestConfig,
    TrialLedger,
    deflated_sharpe,
    fixed_weight_benchmark,
    max_drawdown,
    run_backtest,
    summarize,
)
from macro_agent.backtest.metrics import deploy_verdict, trade_stats
from macro_agent.backtest.validation import (
    bootstrap_ci,
    evaporation_curve,
    min_backtest_length,
    paired_test,
    pbo_cscv,
)
from macro_agent.backtest.walkforward import walk_forward
from macro_agent.config import DEFAULT_PARAMS, cost_bps
from macro_agent.montecarlo import MCConfig, monte_carlo, summarize_mc

CFG = BacktestConfig("2008-01-01", "2012-12-31")


@pytest.fixture(scope="module")
def result(market, features):
    return run_backtest(market, CFG, features=features)


def test_zero_delay_is_refused():
    with pytest.raises(ValueError, match="look-ahead"):
        BacktestConfig("2008-01-01", "2010-01-01", execution_delay=0).validate()


def test_holdout_is_locked_by_default():
    with pytest.raises(ValueError, match="holdout"):
        BacktestConfig("2022-06-01", "2024-01-01").validate()
    assert BacktestConfig("2010-01-01", "2024-01-01").effective_end() == pd.Timestamp("2021-12-31")
    assert BacktestConfig("2010-01-01", "2024-01-01", use_holdout=True).effective_end() == pd.Timestamp("2024-01-01")


def test_fills_happen_at_next_open_with_costs(market, result):
    assert not result.trades.empty
    dates = market.dates
    for _, f in result.trades.head(30).iterrows():
        t = dates.get_loc(f["date"])
        raw = market.open[f["ticker"]].iat[t]
        cost = cost_bps(f["ticker"]) / 1e4
        expected = raw * (1 + cost) if f["side"] == "buy" else raw * (1 - cost)
        assert f["price"] == pytest.approx(expected)
    decision_dates = {pd.Timestamp(d["date"]) for d in result.decisions}
    fill_dates = set(result.trades["date"])
    assert all(dates[dates.get_loc(d) - 1] in decision_dates for d in fill_dates)


def test_backtest_is_reproducible(market, features, result):
    again = run_backtest(market, CFG, features=features)
    pd.testing.assert_series_equal(result.equity, again.equity)
    pd.testing.assert_frame_equal(result.trades, again.trades)


def test_equity_unaffected_by_future_data(market, result):
    cut = "2010-06-30"
    moved = market.truncate("2012-12-31")
    for frame in (moved.open, moved.high, moved.low, moved.close):
        frame.loc["2010-07-01":] *= 1.7
    moved.macro.loc["2010-07-01":] += 3.0
    other = run_backtest(moved, CFG)
    pd.testing.assert_series_equal(result.equity.loc[:cut], other.equity.loc[:cut])


def test_round_trips_and_trade_stats(result):
    rt = result.round_trips
    assert not rt.empty
    assert (rt["exit_date"] >= rt["entry_date"]).all()
    stats = trade_stats(rt)
    assert stats["trades"] == len(rt)
    assert 0 <= stats["win_rate"] <= 1
    assert sum(stats["exits_by_reason"].values()) == len(rt)


def test_invested_share_bounded(result):
    assert result.invested.between(-1e-9, 1 + 1e-9).all()


def test_drawdown_hard_stop_flattens_and_stays_flat(market, features):
    res = run_backtest(market, BacktestConfig("2008-01-01", "2012-12-31", drawdown_halt=0.001), features=features)
    assert res.halted_on is not None
    after = res.invested.loc[res.halted_on:].iloc[2:]
    assert (after < 1e-9).all()


def test_benchmark_buy_and_hold(market):
    eq = fixed_weight_benchmark(market, {"SPY": 1.0}, CFG, rebalance_monthly=False)
    spy = market.close["SPY"].loc[eq.index]
    growth = spy.iloc[-1] / spy.iloc[1]
    assert eq.iloc[-1] / eq.iloc[1] == pytest.approx(growth, rel=1e-9)


def test_summary_and_verdict():
    eq = pd.Series([1.0, 1.2, 0.9, 1.3], index=pd.bdate_range("2020-01-01", periods=4))
    assert max_drawdown(eq) == pytest.approx(0.9 / 1.2 - 1)
    s = summarize(eq)
    verdict = deploy_verdict(s, 0.8, 0.20)
    assert not verdict["deploy"] and any("drawdown" in r for r in verdict["reasons"])


def test_more_trials_deflate_the_sharpe():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.0005, 0.01, 2000))
    one = deflated_sharpe(r, n_trials=1)
    many = deflated_sharpe(r, n_trials=100, trial_sharpes=list(rng.normal(0, 0.03, 100)))
    assert one["dsr"] > 0.5 and many["dsr"] < one["dsr"]


def test_pbo_near_half_for_pure_noise_and_low_for_a_real_edge():
    rng = np.random.default_rng(3)
    noise = pd.DataFrame(rng.normal(0, 0.01, (1600, 8)))
    assert 0.2 < pbo_cscv(noise, n_splits=8)["pbo"] < 0.8
    edge = noise.copy()
    edge[0] += 0.003
    assert pbo_cscv(edge, n_splits=8)["pbo"] < 0.1


def test_bootstrap_paired_and_min_length():
    rng = np.random.default_rng(4)
    r = pd.Series(rng.normal(0.001, 0.01, 1500))
    ci = bootstrap_ci(r, n_boot=200)
    assert ci["sharpe_ci"][0] < 0.001 / 0.01 * np.sqrt(252) < ci["sharpe_ci"][1]
    assert paired_test(r, r * 0, n_boot=200)["p_value"] < 0.05
    assert min_backtest_length(100, 1.0) > min_backtest_length(10, 1.0) > 0


def test_ledger_and_evaporation_curve(tmp_path):
    ledger = TrialLedger(tmp_path / "ledger.jsonl")
    for i, sr in enumerate([0.5, 0.9, 0.7, 1.1]):
        ledger.record(f"v{i}", {"i": i}, sr / np.sqrt(252), sr, ("a", "b"), False)
    assert TrialLedger(tmp_path / "ledger.jsonl").n_trials == 4
    curve = evaporation_curve(ledger.entries())
    assert list(curve["best_sharpe_annual"]) == [0.5, 0.9, 0.9, 1.1]
    assert curve["deflation_threshold_annual"].iloc[-1] > 0


def test_walk_forward_selects_per_window(market):
    variants = {"rr3": DEFAULT_PARAMS, "rr2": DEFAULT_PARAMS.with_(min_reward_risk=2.0)}
    wf = walk_forward(market, variants, BacktestConfig("2007-01-01", "2012-12-31"), train_years=3)
    assert [w["chosen"] in variants for w in wf["windows"]] == [True] * len(wf["windows"])
    assert len(wf["windows"]) >= 2


def test_monte_carlo_is_reproducible():
    mc = MCConfig(n_paths=2, years=4, warmup_years=2, seed=5, processes=1)
    a, b = monte_carlo(DEFAULT_PARAMS, mc), monte_carlo(DEFAULT_PARAMS, mc)
    pd.testing.assert_frame_equal(a, b)
    s = summarize_mc(a)
    assert s["paths"] == 2 and s["cagr_p10"] <= s["cagr_median"] <= s["cagr_p90"]
