"""Performance statistics, including the deflated Sharpe ratio.

Deflated Sharpe: Bailey and Lopez de Prado, "The Deflated Sharpe Ratio",
Journal of Portfolio Management 40 (2014). It corrects an observed Sharpe for
sample length, non-normal returns and the number of variants tried.
"""
from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
import pandas as pd

TRADING_DAYS = 252
EULER_GAMMA = 0.5772156649015329
_N = NormalDist()


def max_drawdown(equity: pd.Series) -> float:
    """Worst peak-to-trough decline, as a negative fraction."""
    return float((equity / equity.cummax() - 1).min())


def sharpe(returns: pd.Series, periods: int = TRADING_DAYS) -> float:
    sd = returns.std(ddof=1)
    if not sd or np.isnan(sd):
        return 0.0
    return float(returns.mean() / sd * math.sqrt(periods))


def summarize(equity: pd.Series) -> dict:
    rets = equity.pct_change().dropna()
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-9)
    total = float(equity.iloc[-1] / equity.iloc[0] - 1)
    return {
        "start": str(equity.index[0].date()),
        "end": str(equity.index[-1].date()),
        "total_return": total,
        "cagr": float((1 + total) ** (1 / years) - 1),
        "ann_vol": float(rets.std(ddof=1) * math.sqrt(TRADING_DAYS)),
        "sharpe": sharpe(rets),
        "max_drawdown": max_drawdown(equity),
    }


def expected_max_sharpe(n_trials: int, sharpe_variance: float) -> float:
    """Expected maximum of n_trials Sharpe ratios drawn under the null of no
    skill: the bar the best variant must clear."""
    if n_trials <= 1 or sharpe_variance <= 0:
        return 0.0
    return math.sqrt(sharpe_variance) * (
        (1 - EULER_GAMMA) * _N.inv_cdf(1 - 1 / n_trials)
        + EULER_GAMMA * _N.inv_cdf(1 - 1 / (n_trials * math.e))
    )


def deflated_sharpe(
    returns: pd.Series, n_trials: int = 1, trial_sharpes: list[float] | None = None
) -> dict:
    """Probability that the true (per-period) Sharpe exceeds the level a lucky
    best-of-n_trials would reach by chance.

    `trial_sharpes` are the per-period Sharpes of every variant tried (from the
    trial ledger); their variance sets the deflation threshold. With one trial
    this reduces to the probabilistic Sharpe ratio against zero.
    """
    r = returns.dropna()
    t = len(r)
    sd = r.std(ddof=1)
    if t < 3 or not sd:
        return {"sr": 0.0, "sr0": 0.0, "dsr": 0.0, "n_trials": n_trials}
    sr = float(r.mean() / sd)
    skew = float(r.skew())
    kurt = float(r.kurt()) + 3.0  # pandas reports excess kurtosis
    if trial_sharpes and len(trial_sharpes) > 1:
        var_sr = float(np.var(trial_sharpes, ddof=1))
    else:
        var_sr = 0.0
    sr0 = expected_max_sharpe(n_trials, var_sr)
    denom = 1 - skew * sr + (kurt - 1) / 4 * sr**2
    z = (sr - sr0) * math.sqrt(t - 1) / math.sqrt(max(denom, 1e-12))
    return {"sr": sr, "sr0": sr0, "dsr": _N.cdf(z), "n_trials": n_trials}
