"""Deterministic synthetic market for offline tests and dry runs.

This is not market data and says nothing about whether the strategy has an
edge. It exists so every module can be exercised with no network access.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import MACRO_SERIES, SLEEVES


def synthetic_market(
    start: str = "2005-01-03", end: str = "2024-12-31", seed: int = 7
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (prices, fred) with the full universe and the FRED series.

    Prices follow a regime-switching random walk so that trends, drawdowns and
    recoveries all occur. Same seed, same output.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, end)
    n = len(idx)

    # A slowly switching latent state drives drift signs across the universe.
    state = np.zeros(n, dtype=int)
    for i in range(1, n):
        state[i] = state[i - 1] if rng.random() > 1 / 120 else rng.integers(0, 4)

    tickers = sorted({t for group in SLEEVES.values() for t in group})
    prices = {}
    for t in tickers:
        loadings = rng.normal(0, 1, 4) * 0.0006
        vol = 0.004 if t in SLEEVES["cash"] else rng.uniform(0.008, 0.025)
        drift = loadings[state]
        if t in SLEEVES["cash"]:
            drift = np.full(n, 0.00008)
        rets = drift + rng.normal(0, vol, n)
        prices[t] = 50 * np.exp(np.cumsum(rets))
    prices = pd.DataFrame(prices, index=idx)

    fred = {}
    for _, (src, ident, _) in MACRO_SERIES.items():
        if src != "fred":
            continue
        loadings = rng.normal(0, 1, 4) * 0.004
        steps = loadings[state] + rng.normal(0, 0.04, n)
        fred[ident] = 2.0 + np.cumsum(steps) * 0.3
    fred = pd.DataFrame(fred, index=idx)
    return prices, fred
