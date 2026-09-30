"""Backtest harness: lagged execution, per-instrument costs, holdout lock,
trial ledger, and the statistics that judge a result (spec 8, 19)."""
from .engine import BacktestConfig, BacktestResult, fixed_weight_benchmark, run_backtest
from .ledger import TrialLedger
from .metrics import deflated_sharpe, max_drawdown, summarize

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "TrialLedger",
    "deflated_sharpe",
    "fixed_weight_benchmark",
    "max_drawdown",
    "run_backtest",
    "summarize",
]
