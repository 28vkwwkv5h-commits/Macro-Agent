"""The statistics that judge a result (spec 19).

- Stationary-bootstrap confidence intervals (Politis and Romano 1994).
- Probability of backtest overfitting via combinatorially symmetric
  cross-validation (Bailey, Borwein, Lopez de Prado and Zhu 2017).
- Minimum backtest length (Bailey, Borwein, Lopez de Prado and Zhu 2014).
- The evaporation curve: best in-sample Sharpe against its rising deflation
  threshold as the trial count grows (spec 19.3).
- A paired bootstrap test against a benchmark.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd

from .metrics import EULER_GAMMA, TRADING_DAYS, _N, expected_max_sharpe


def _stationary_indices(n: int, block: float, rng: np.random.Generator) -> np.ndarray:
    idx = np.empty(n, dtype=int)
    i = int(rng.integers(0, n))
    for k in range(n):
        idx[k] = i
        i = int(rng.integers(0, n)) if rng.random() < 1.0 / block else (i + 1) % n
    return idx


def _ann_sharpe(x: np.ndarray) -> float:
    sd = x.std(ddof=1)
    return float(x.mean() / sd * math.sqrt(TRADING_DAYS)) if sd > 0 else 0.0


def _col_sharpes(m: np.ndarray) -> np.ndarray:
    sd = m.std(axis=0, ddof=1)
    return np.where(sd > 0, m.mean(axis=0) / np.where(sd > 0, sd, 1.0), 0.0)


def bootstrap_ci(
    returns: pd.Series, n_boot: int = 1000, block: float = 21, alpha: float = 0.05, seed: int = 0
) -> dict:
    """Confidence intervals for annualised Sharpe and CAGR."""
    r = returns.dropna().to_numpy()
    rng = np.random.default_rng(seed)
    sharpes, cagrs = [], []
    for _ in range(n_boot):
        x = r[_stationary_indices(len(r), block, rng)]
        sharpes.append(_ann_sharpe(x))
        cagrs.append(float(np.prod(1 + x) ** (TRADING_DAYS / len(x)) - 1))
    q = [alpha / 2, 1 - alpha / 2]
    return {
        "sharpe_ci": [float(v) for v in np.quantile(sharpes, q)],
        "cagr_ci": [float(v) for v in np.quantile(cagrs, q)],
        "n_boot": n_boot,
    }


def paired_test(strategy: pd.Series, benchmark: pd.Series, n_boot: int = 1000, seed: int = 0) -> dict:
    """Bootstrap p-value that the strategy's mean daily return does not beat the benchmark's."""
    d = (strategy - benchmark).dropna().to_numpy()
    rng = np.random.default_rng(seed)
    means = [d[_stationary_indices(len(d), 21, rng)].mean() for _ in range(n_boot)]
    return {"mean_excess_ann": float(d.mean() * TRADING_DAYS), "p_value": float(np.mean(np.array(means) <= 0))}


def pbo_cscv(returns: pd.DataFrame, n_splits: int = 16) -> dict:
    """Probability of backtest overfitting.

    `returns` is sessions x variants. The sample is cut into `n_splits` blocks;
    for every way of choosing half the blocks as in-sample, the best in-sample
    variant is found and its out-of-sample rank recorded. PBO is the share of
    combinations where that winner lands in the bottom half out of sample.
    """
    if returns.shape[1] < 2:
        raise ValueError("PBO needs at least two variants")
    m = returns.dropna().to_numpy()
    blocks = np.array_split(np.arange(len(m)), n_splits)
    logits = []
    for combo in itertools.combinations(range(n_splits), n_splits // 2):
        is_idx = np.concatenate([blocks[i] for i in combo])
        oos_idx = np.concatenate([blocks[i] for i in range(n_splits) if i not in combo])
        is_sr, oos_sr = _col_sharpes(m[is_idx]), _col_sharpes(m[oos_idx])
        best = int(np.argmax(is_sr))
        rank = (oos_sr < oos_sr[best]).sum() + 0.5 * ((oos_sr == oos_sr[best]).sum() - 1)
        w = (rank + 1) / (m.shape[1] + 1)
        logits.append(math.log(w / (1 - w)))
    logits = np.array(logits)
    return {"pbo": float((logits <= 0).mean()), "n_combinations": len(logits), "median_logit": float(np.median(logits))}


def min_backtest_length(n_trials: int, sharpe_annual: float) -> float:
    """Years of history needed before the best of `n_trials` variants with this
    annualised Sharpe could not be explained by selection alone."""
    if n_trials <= 1:
        return 0.0
    if sharpe_annual <= 0:
        return float("inf")
    z = (1 - EULER_GAMMA) * _N.inv_cdf(1 - 1 / n_trials) + EULER_GAMMA * _N.inv_cdf(1 - 1 / (n_trials * math.e))
    return float((z / sharpe_annual) ** 2)


def evaporation_curve(ledger_entries: list[dict]) -> pd.DataFrame:
    """Best annualised Sharpe so far against the deflation threshold, per trial.

    If the best keeps rising but the threshold rises faster, the edge is
    evaporating and the search is mining noise (spec 19.3).
    """
    rows, sharpes = [], []
    best = -np.inf
    for i, e in enumerate(ledger_entries, start=1):
        sharpes.append(e["sharpe_per_period"])
        best = max(best, e["sharpe_annual"])
        var = float(np.var(sharpes, ddof=1)) if len(sharpes) > 1 else 0.0
        threshold = expected_max_sharpe(i, var) * math.sqrt(TRADING_DAYS)
        rows.append({"trial": i, "variant": e["variant"], "sharpe_annual": e["sharpe_annual"],
                     "best_sharpe_annual": best, "deflation_threshold_annual": threshold,
                     "margin": best - threshold})
    return pd.DataFrame(rows)
