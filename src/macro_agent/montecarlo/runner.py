"""Run the real strategy across many simulated markets (spec 20).

- `monte_carlo`: one parameter set, many paths -> distribution of outcomes.
- `stress_ladder`: the spec 20.1 table: Gaussian and frictionless, then fat
  tails, then costs, then extra execution lag.
- `ablation`: remove one component at a time on the same paths (spec 20.2).
- `sweep`: a parameter grid on the same paths, every variant logged.

Paths are generated from seeds, so every run is reproducible and every variant
in a comparison sees exactly the same markets.
"""
from __future__ import annotations

import itertools
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from ..backtest.engine import BacktestConfig, run_backtest
from ..backtest.metrics import summarize, trade_stats
from ..config import DEFAULT_PARAMS, Params
from ..data.market import Market
from .generator import RegimeModel, bootstrap_market, synthetic_market


@dataclass(frozen=True)
class MCConfig:
    n_paths: int = 100
    years: int = 10
    warmup_years: int = 3  # history before trading starts, for indicators
    seed: int = 0
    model: RegimeModel = field(default_factory=RegimeModel)
    cost_multiplier: float = 1.0
    execution_delay: int = 1
    processes: int | None = None
    source: str = "regime"  # "regime" (synthetic model) or "bootstrap" (resampled real market)


_BASE_MARKET: Market | None = None  # set in workers for bootstrap mode


def _path(args) -> dict:
    i, params, mc = args
    seed = mc.seed + i
    start = pd.Timestamp("2000-01-03")
    trade_start = start + pd.DateOffset(years=mc.warmup_years)
    end = start + pd.DateOffset(years=mc.warmup_years + mc.years)
    if mc.source == "bootstrap":
        if _BASE_MARKET is None:
            raise RuntimeError("bootstrap mode needs a base market")
        market = bootstrap_market(_BASE_MARKET, seed)
        trade_start = market.dates[0] + pd.DateOffset(years=mc.warmup_years)
        end = market.dates[-1]
    else:
        market, _ = synthetic_market(str(start.date()), str(end.date()), seed=seed, model=mc.model)
    cfg = BacktestConfig(
        start=str(trade_start.date()), end=str(end.date()), use_holdout=True,
        cost_multiplier=mc.cost_multiplier, execution_delay=mc.execution_delay,
    )
    res = run_backtest(market, cfg, params, keep_decisions=False)
    stats = summarize(res.equity, res.invested)
    trades = trade_stats(res.round_trips)
    years = max((res.equity.index[-1] - res.equity.index[0]).days / 365.25, 1e-9)
    return {
        "path": i, "seed": seed,
        **{k: v for k, v in stats.items() if k not in ("start", "end")},
        "trades": trades.get("trades", 0),
        "trades_per_5y": trades.get("trades", 0) / years * 5,
        "win_rate": trades.get("win_rate", np.nan),
        "avg_r": trades.get("avg_r", np.nan),
        "mean_daily_return": float(res.returns.mean()),
        "halted": res.halted_on is not None,
    }


def _init_worker(base: Market | None):
    global _BASE_MARKET
    _BASE_MARKET = base


def monte_carlo(params: Params = DEFAULT_PARAMS, mc: MCConfig = MCConfig(), base_market: Market | None = None) -> pd.DataFrame:
    jobs = [(i, params, mc) for i in range(mc.n_paths)]
    procs = mc.processes or os.cpu_count() or 1
    if procs == 1 or mc.n_paths == 1:
        _init_worker(base_market)
        rows = [_path(j) for j in jobs]
    else:
        with ProcessPoolExecutor(procs, initializer=_init_worker, initargs=(base_market,)) as pool:
            rows = list(pool.map(_path, jobs))
    return pd.DataFrame(rows).sort_values("path").reset_index(drop=True)


def summarize_mc(df: pd.DataFrame) -> dict:
    """The spec 20.1 / 20.3 view of a distribution of paths."""
    q = lambda col, p: float(np.nanquantile(df[col], p))  # noqa: E731
    return {
        "paths": int(len(df)),
        "cagr_p10": q("cagr", 0.10),
        "cagr_median": q("cagr", 0.50),
        "cagr_p90": q("cagr", 0.90),
        "sharpe_median": q("sharpe", 0.50),
        "max_dd_median": q("max_drawdown", 0.50),
        "max_dd_worst_2pct": q("max_drawdown", 0.02),
        "share_of_days_in_cash_median": q("share_of_days_in_cash", 0.50),
        "trades_per_5y_median": q("trades_per_5y", 0.50),
        "win_rate_median": q("win_rate", 0.50),
        "avg_r_median": q("avg_r", 0.50),
        "share_of_paths_losing": float((df["cagr"] < 0).mean()),
        "share_of_paths_hard_stopped": float(df["halted"].mean()),
    }


def stress_ladder(params: Params = DEFAULT_PARAMS, mc: MCConfig = MCConfig()) -> pd.DataFrame:
    rows = {
        "Gaussian returns, no costs": replace(mc, model=replace(mc.model, tail_df=None), cost_multiplier=0.0),
        "Fat tails added": replace(mc, cost_multiplier=0.0),
        "Plus spread and slippage": mc,
        "Plus double costs": replace(mc, cost_multiplier=2.0 * mc.cost_multiplier),
        "Plus 5-session execution lag": replace(mc, execution_delay=5),
    }
    return pd.DataFrame({name: summarize_mc(monte_carlo(params, cfg)) for name, cfg in rows.items()}).T


ABLATIONS = {
    "As specified": {},
    "Confirmation removed": {"require_confirmation": False},
    "Regime whitelist removed": {"use_regime_whitelist": False},
    "Stops removed": {"use_stops": False},
    "Tranching removed": {"use_tranching": False},
    "Flows removed": {"use_flows": False},
    "Correlation cap removed": {"use_correlation_cap": False},
    "Stability term removed": {"use_stability": False},
    "Reward-to-risk floor at 1.5": {"min_reward_risk": 1.5},
    "Confidence gates 0.50/0.30": {"confidence_full": 0.50, "confidence_partial": 0.30,
                                   "emergency_confidence": 0.30},
}


def ablation(params: Params = DEFAULT_PARAMS, mc: MCConfig = MCConfig(), which: dict | None = None) -> pd.DataFrame:
    which = which or ABLATIONS
    return pd.DataFrame({name: summarize_mc(monte_carlo(params.with_(**ch), mc)) for name, ch in which.items()}).T


def sweep(
    grid: dict[str, list],
    params: Params = DEFAULT_PARAMS,
    mc: MCConfig = MCConfig(),
    ledger=None,
    variant_prefix: str = "mc_sweep",
) -> pd.DataFrame:
    """Every combination in `grid` on the same paths. Each is recorded in the
    ledger, because on synthetic data too, the more you search the more the
    winner is luck."""
    keys = sorted(grid)
    rows = []
    for values in itertools.product(*(grid[k] for k in keys)):
        changes = dict(zip(keys, values))
        df = monte_carlo(params.with_(**changes), mc)
        s = summarize_mc(df)
        rows.append({**changes, **s})
        if ledger is not None:
            per_period = float(df["mean_daily_return"].mean() / max(df["ann_vol"].median() / np.sqrt(252), 1e-12))
            ledger.record(
                variant=f"{variant_prefix}:{changes}", params={**params.with_(**changes).to_dict(), "mc": repr(mc)},
                sharpe_per_period=per_period, sharpe_annual=s["sharpe_median"],
                data_range=(f"synthetic seed {mc.seed}", f"{mc.n_paths} paths x {mc.years}y"),
                holdout_used=False,
            )
    return pd.DataFrame(rows)
