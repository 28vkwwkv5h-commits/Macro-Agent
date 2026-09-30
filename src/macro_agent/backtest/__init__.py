"""Backtest harness: lagged execution, costs, holdout lock, trial ledger."""
from .engine import (
    BacktestConfig,
    BacktestResult,
    fixed_weight_benchmark,
    month_end_dates,
    run_backtest,
    simulate,
)
from .ledger import TrialLedger
from .metrics import deflated_sharpe, max_drawdown, summarize

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "TrialLedger",
    "deflated_sharpe",
    "fixed_weight_benchmark",
    "max_drawdown",
    "month_end_dates",
    "run_backtest",
    "simulate",
    "summarize",
]
