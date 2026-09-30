"""How fragile the market is, from returns alone (spec 18.3, 18.4).

Turbulence: Kritzman and Li (2010). The Mahalanobis distance of today's
cross-asset return vector from its trailing mean, under the trailing covariance.
It fires on unusually large moves and on assets decoupling from their usual
correlation structure.

Absorption ratio: Kritzman, Li, Page and Rigobon (2011). The share of variance
explained by the top fifth of principal components. A rising ratio means
markets are coupled and shocks propagate. The signal is the standardised shift
of its 15-day average against its one-year average.

stability = 1 - mean(percentile(turbulence), percentile(absorption shift)),
both percentiles taken against each measure's own trailing history.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def turbulence_and_absorption(
    returns: pd.DataFrame, window: int, fraction: float = 0.2
) -> pd.DataFrame:
    r = returns.to_numpy(dtype=float)
    n, k = r.shape
    n_pc = max(1, int(round(k * fraction)))
    turb = np.full(n, np.nan)
    absorb = np.full(n, np.nan)
    for t in range(window, n):
        hist = r[t - window: t]
        if np.isnan(hist).any() or np.isnan(r[t]).any():
            continue
        mu = hist.mean(axis=0)
        cov = np.cov(hist, rowvar=False)
        d = r[t] - mu
        turb[t] = float(d @ np.linalg.pinv(cov) @ d)
        eig = np.linalg.eigvalsh(cov)[::-1]
        absorb[t] = float(eig[:n_pc].sum() / eig.sum())
    return pd.DataFrame({"turbulence": turb, "absorption": absorb}, index=returns.index)


def stability_score(
    close: pd.DataFrame,
    basket,
    window: int = 252,
    rank_window: int = 756,
    fraction: float = 0.2,
    short: int = 15,
) -> pd.Series:
    cols = [c for c in basket if c in close.columns]
    if len(cols) < 3:
        return pd.Series(1.0, index=close.index)
    rets = np.log(close[cols]).diff()
    raw = turbulence_and_absorption(rets, window, fraction)
    ar = raw["absorption"]
    shift = (ar.rolling(short).mean() - ar.rolling(window).mean()) / ar.rolling(window).std()
    min_p = max(20, rank_window // 4)
    turb_pct = raw["turbulence"].rolling(rank_window, min_periods=min_p).rank(pct=True)
    shift_pct = shift.rolling(rank_window, min_periods=min_p).rank(pct=True)
    stability = 1.0 - (turb_pct + shift_pct) / 2.0
    # Before either measure has history there is no fragility information:
    # stay neutral (0.5) rather than claiming calm.
    return stability.fillna(0.5).rename("stability")
