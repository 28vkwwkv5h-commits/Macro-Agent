"""Daily portfolio simulation with lagged execution and costs.

Timing, which is where backtests lie (spec 20.1):

- A decision on signal date s uses closes up to and including s.
- It executes `execution_delay` sessions later, at that session's close.
- The new weights earn returns only from the session after execution.

So no signal ever shares a return with the data that produced it. A delay
below one session is refused.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from ..config import CASH_TICKER, COST_BPS, HOLDOUT_START
from ..pipeline import Decision, decide


@dataclass
class BacktestConfig:
    start: str
    end: str
    cost_bps: float = COST_BPS
    execution_delay: int = 1
    cash_ticker: str | None = CASH_TICKER
    holdout_start: str = HOLDOUT_START
    use_holdout: bool = False

    def validate(self) -> None:
        if self.execution_delay < 1:
            raise ValueError("execution_delay must be at least one session (no look-ahead)")
        if self.cost_bps < 0:
            raise ValueError("cost_bps cannot be negative")
        if not self.use_holdout and pd.Timestamp(self.start) >= pd.Timestamp(self.holdout_start):
            raise ValueError(
                f"start {self.start} is inside the holdout (from {self.holdout_start}); "
                "set use_holdout=True only for the final out-of-sample evaluation"
            )

    def effective_end(self) -> pd.Timestamp:
        end = pd.Timestamp(self.end)
        if not self.use_holdout:
            end = min(end, pd.Timestamp(self.holdout_start) - pd.Timedelta(days=1))
        return end


@dataclass
class BacktestResult:
    equity: pd.Series
    turnover: pd.Series
    weights: pd.DataFrame  # target weights at each execution date
    decisions: list[Decision] = field(default_factory=list)
    config: BacktestConfig | None = None


def month_end_dates(index: pd.DatetimeIndex) -> list[pd.Timestamp]:
    """Last trading session of each calendar month present in `index`."""
    s = index.to_series()
    return list(s.groupby(index.to_period("M")).max())


def simulate(
    prices: pd.DataFrame,
    targets: dict[pd.Timestamp, dict[str, float]],
    cost_bps: float,
    cash_ticker: str | None = CASH_TICKER,
) -> tuple[pd.Series, pd.Series]:
    """Equity curve (starting at 1.0) and turnover for target weights applied at
    the close of each key date in `targets`. Weights drift with prices between
    rebalances. Cash is the residual and earns `cash_ticker`'s return.
    """
    tickers = list(prices.columns)
    col = {t: i for i, t in enumerate(tickers)}
    rets = prices.pct_change(fill_method=None).fillna(0.0).to_numpy()
    if cash_ticker and cash_ticker in col:
        cash_rets = rets[:, col[cash_ticker]]
    else:
        cash_rets = np.zeros(len(prices))

    w = np.zeros(len(tickers))
    w_cash = 1.0
    value = 1.0
    values, turns = [], []
    for i, day in enumerate(prices.index):
        if i > 0:
            gross = 1.0 + w @ rets[i] + w_cash * cash_rets[i]
            value *= gross
            w = w * (1.0 + rets[i]) / gross
            w_cash = w_cash * (1.0 + cash_rets[i]) / gross
        turnover = 0.0
        if day in targets:
            new = np.zeros(len(tickers))
            for t, wt in targets[day].items():
                new[col[t]] = wt
            turnover = float(np.abs(new - w).sum())
            value *= 1.0 - turnover * cost_bps / 1e4
            w = new
            w_cash = 1.0 - new.sum()
        values.append(value)
        turns.append(turnover)
    return (
        pd.Series(values, index=prices.index, name="equity"),
        pd.Series(turns, index=prices.index, name="turnover"),
    )


def run_backtest(
    prices: pd.DataFrame,
    macro: pd.DataFrame,
    config: BacktestConfig,
    decide_fn: Callable[[pd.DataFrame, pd.DataFrame, pd.Timestamp], Decision] = decide,
) -> BacktestResult:
    config.validate()
    end = config.effective_end()
    # Nothing on or after the holdout reaches the strategy unless unlocked.
    prices = prices.loc[:end]
    macro = macro.loc[:end]
    sim = prices.loc[config.start:]
    if len(sim) < 2:
        raise ValueError("not enough sessions in the backtest window")

    targets: dict[pd.Timestamp, dict[str, float]] = {}
    decisions: list[Decision] = []
    positions = {day: i for i, day in enumerate(sim.index)}
    for signal_date in month_end_dates(sim.index):
        exec_pos = positions[signal_date] + config.execution_delay
        if exec_pos >= len(sim.index):
            break
        decision = decide_fn(prices, macro, signal_date)
        decisions.append(decision)
        targets[sim.index[exec_pos]] = decision.weights

    equity, turnover = simulate(sim, targets, config.cost_bps, config.cash_ticker)
    weights = (
        pd.DataFrame.from_dict(targets, orient="index")
        .reindex(columns=sorted(sim.columns))
        .fillna(0.0)
    )
    return BacktestResult(equity, turnover, weights, decisions, config)


def fixed_weight_benchmark(
    prices: pd.DataFrame, weights: dict[str, float], config: BacktestConfig, rebalance: bool
) -> pd.Series:
    """Buy-and-hold (rebalance=False) or monthly-rebalanced fixed weights, with
    the same timing and costs as the strategy."""
    config.validate()
    sim = prices.loc[config.start:config.effective_end()]
    dates = month_end_dates(sim.index) if rebalance else [sim.index[0]]
    positions = {day: i for i, day in enumerate(sim.index)}
    targets = {}
    for d in dates:
        pos = positions[d] + (config.execution_delay if rebalance else 0)
        if pos < len(sim.index):
            targets[sim.index[pos]] = weights
    equity, _ = simulate(sim, targets, config.cost_bps, config.cash_ticker)
    return equity
