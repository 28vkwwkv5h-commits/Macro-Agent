"""Walk-forward selection (spec 8): fit on rolling five-year windows, test on the
following year, roll.

Each variant is backtested once over the whole period (the strategy is causal,
so a variant's returns in a year do not depend on later data). For each test
year the variant with the best Sharpe over the preceding training window is
chosen, and its returns for that year are stitched into one out-of-sample
series. Positions are not carried between variants at a switch, which slightly
flatters the stitched curve; treat it as an estimate.
"""
from __future__ import annotations

import pandas as pd

from ..config import Params
from ..data.market import Market
from .engine import BacktestConfig, run_backtest
from .metrics import sharpe, summarize


def walk_forward(
    market: Market,
    variants: dict[str, Params],
    config: BacktestConfig,
    train_years: int = 5,
    test_years: int = 1,
) -> dict:
    returns = {}
    for name, params in variants.items():
        res = run_backtest(market, config, params, keep_decisions=False)
        returns[name] = res.returns
    rets = pd.DataFrame(returns)
    first = rets.index[0]
    last = rets.index[-1]
    test_start = first + pd.DateOffset(years=train_years)
    pieces, choices = [], []
    while test_start < last:
        test_end = test_start + pd.DateOffset(years=test_years)
        train = rets.loc[test_start - pd.DateOffset(years=train_years): test_start - pd.Timedelta(days=1)]
        scores = {name: sharpe(train[name]) for name in rets.columns}
        best = max(sorted(scores), key=lambda k: scores[k])
        test = rets.loc[test_start: test_end - pd.Timedelta(days=1), best]
        pieces.append(test)
        choices.append({"test_start": str(test_start.date()), "chosen": best,
                        "train_sharpe": scores[best], "test_sharpe": sharpe(test)})
        test_start = test_end
    if not pieces:
        raise ValueError("period too short for one train/test window")
    oos = pd.concat(pieces)
    equity = (1 + oos).cumprod()
    return {"windows": choices, "oos_summary": summarize(equity), "oos_returns": oos, "variant_returns": rets}
