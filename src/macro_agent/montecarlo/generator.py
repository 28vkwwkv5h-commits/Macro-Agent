"""Synthetic markets from a four-regime switching model (spec 20).

This is a structural test bench, not market data. It checks whether the decision
logic behaves sensibly when regimes exist and persist, how it degrades with fat
tails, costs and lag, and which components carry the result. The regime
parameters are assumptions; change them and the results change (spec 20.6).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import (
    DEFLATION,
    GOLDILOCKS,
    MACRO_SERIES,
    REFLATION,
    REGIME_ORDER,
    SLEEVES,
    TICKER_SLEEVE,
    TIGHTENING,
    WHITELIST,
)
from ..data.market import Market

TRADING_DAYS = 252

# Direction of each macro input in each regime (+1 up, 0 flat, -1 down), the
# canonical row of the spec 2.3 table, audit-corrected for Deflation.
MACRO_DIRECTION = {
    REFLATION: {"oil": 1, "gold": 1, "usd": -1, "ust10y": 1, "spread_2s10s": 1, "real10y": -1},
    GOLDILOCKS: {"oil": 0, "gold": 0, "usd": -1, "ust10y": 0, "spread_2s10s": 0, "real10y": -1},
    TIGHTENING: {"oil": 1, "gold": -1, "usd": 1, "ust10y": 1, "spread_2s10s": -1, "real10y": 1},
    DEFLATION: {"oil": -1, "gold": -1, "usd": 1, "ust10y": -1, "spread_2s10s": 1, "real10y": 1},
}

SLEEVE_VOL = {
    "us_equity_broad": 0.17, "us_equity_sector": 0.21, "international": 0.21,
    "fixed_income": 0.08, "commodities": 0.26, "miners": 0.40, "currency": 0.08,
    "inverse": 0.17, "cash": 0.005, "individual_names": 0.35,
}
VOL_OVERRIDES = {"TLT": 0.15, "SHY": 0.02, "HYG": 0.09, "UNG": 0.45, "GDXJ": 0.48, "SILJ": 0.52}
EQUITY_SLEEVES = {"us_equity_broad", "us_equity_sector", "international", "individual_names"}


@dataclass(frozen=True)
class RegimeModel:
    """Assumptions behind a simulated market. Every field is a knob."""

    mean_regime_days: float = 160.0
    favoured_drift: float = 0.18  # annual drift of whitelisted assets in their regime
    other_drift: float = -0.04  # annual drift of everything else
    macro_price_drift: float = 0.20  # annual drift of oil/gold/usd along the regime
    macro_rate_drift: float = 1.2  # percentage points per year for yields/spreads
    rate_noise: float = 0.06  # daily sd of yields/spreads, percentage points
    cash_rate: float = 0.02
    tail_df: float | None = 4.0  # Student-t degrees of freedom; None is Gaussian
    equity_factor_loading: float = 0.7
    other_factor_loading: float = 0.3
    vol_scale: float = 1.0


def _shocks(rng: np.random.Generator, shape, df: float | None) -> np.ndarray:
    if df is None:
        return rng.standard_normal(shape)
    t = rng.standard_t(df, shape)
    return t / np.sqrt(df / (df - 2))  # unit variance


def simulate_regimes(n: int, model: RegimeModel, rng: np.random.Generator) -> np.ndarray:
    p_switch = 1.0 / model.mean_regime_days
    states = np.empty(n, dtype=int)
    states[0] = rng.integers(0, 4)
    switches = rng.random(n) < p_switch
    jumps = rng.integers(1, 4, n)
    for i in range(1, n):
        states[i] = (states[i - 1] + jumps[i]) % 4 if switches[i] else states[i - 1]
    return states


def synthetic_market(
    start: str = "2005-01-03",
    end: str = "2024-12-31",
    seed: int = 7,
    model: RegimeModel | None = None,
    tickers: list[str] | None = None,
    with_cot: bool = True,
) -> tuple[Market, pd.Series]:
    """Return (market, true regime per day). Same seed, same output."""
    model = model or RegimeModel()
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, end)
    n = len(idx)
    regimes = simulate_regimes(n, model, rng)
    names = np.array(REGIME_ORDER)
    tickers = tickers or sorted(t for s, g in SLEEVES.items() for t in g)
    dt = 1.0 / TRADING_DAYS

    # Macro-driven price series: oil (USO), gold (GLD), dollar (UUP).
    macro_ticker = {MACRO_SERIES[k][1]: k for k in ("oil", "gold", "usd")}

    market_factor = _shocks(rng, n, model.tail_df)
    close, open_, high, low, volume = {}, {}, {}, {}, {}
    for t in tickers:
        sleeve = TICKER_SLEEVE.get(t, "individual_names")
        vol = VOL_OVERRIDES.get(t, SLEEVE_VOL[sleeve]) * model.vol_scale
        if t in macro_ticker:
            direction = np.array([MACRO_DIRECTION[r][macro_ticker[t]] for r in REGIME_ORDER])
            drift = direction[regimes] * model.macro_price_drift
        elif sleeve == "cash":
            drift = np.full(n, model.cash_rate)
        else:
            fav = np.array([t in WHITELIST[r] for r in REGIME_ORDER])
            drift = np.where(fav[regimes], model.favoured_drift, model.other_drift)
        rho = model.equity_factor_loading if sleeve in EQUITY_SLEEVES else model.other_factor_loading
        if sleeve == "inverse":
            rho = -model.equity_factor_loading
        eps = rho * market_factor + np.sqrt(1 - rho**2) * _shocks(rng, n, model.tail_df)
        daily_vol = vol * np.sqrt(dt)
        rets = (drift - 0.5 * vol**2) * dt + daily_vol * eps
        c = 50.0 * np.exp(np.cumsum(rets))
        gap = np.exp(daily_vol * 0.3 * rng.standard_normal(n))
        o = np.concatenate([[c[0]], c[:-1]]) * gap
        wick = np.abs(rng.standard_normal((2, n))) * daily_vol * 0.5
        h = np.maximum(o, c) * np.exp(wick[0])
        lo = np.minimum(o, c) * np.exp(-wick[1])
        v = 1e6 * np.exp(0.3 * rng.standard_normal(n) + 2.0 * np.abs(eps) - 1.0)
        close[t], open_[t], high[t], low[t], volume[t] = c, o, h, lo, v

    frame = lambda d: pd.DataFrame(d, index=idx)  # noqa: E731
    close_df = frame(close)

    macro = {}
    for name, (src, ident, _) in MACRO_SERIES.items():
        if src == "price":
            macro[name] = close_df[ident] if ident in close_df else pd.Series(np.nan, index=idx)
            continue
        direction = np.array([MACRO_DIRECTION[r][name] for r in REGIME_ORDER])[regimes]
        steps = direction * model.macro_rate_drift * dt + model.rate_noise * rng.standard_normal(n)
        level0 = {"ust10y": 3.0, "spread_2s10s": 0.5, "real10y": 1.0}[name]
        macro[name] = pd.Series(level0 + np.cumsum(steps), index=idx)

    cot = None
    if with_cot:
        # Uninformative positioning noise: an AR(1) in [-0.6, 0.6], weekly.
        cot_cols = {}
        weekly = idx[idx.weekday == 1]
        for t in tickers:
            if TICKER_SLEEVE.get(t) in ("commodities", "miners"):
                x = np.zeros(len(weekly))
                for i in range(1, len(weekly)):
                    x[i] = 0.95 * x[i - 1] + 0.05 * rng.standard_normal()
                s = pd.Series(np.clip(x, -0.6, 0.6), index=weekly + pd.Timedelta(days=6))
                cot_cols[t] = s.reindex(s.index.union(idx)).ffill(limit=10).reindex(idx)
        cot = pd.DataFrame(cot_cols, index=idx) if cot_cols else None

    market = Market(
        open=frame(open_), high=frame(high), low=frame(low), close=close_df,
        volume=frame(volume), macro=pd.DataFrame(macro, index=idx), cot=cot,
    )
    return market, pd.Series(names[regimes], index=idx, name="true_regime")


def bootstrap_market(market: Market, seed: int, block: int = 21) -> Market:
    """Stationary block bootstrap of a real market (Politis and Romano 1994).

    Resamples whole days across every instrument and macro series at once, so
    cross-asset correlation and fat tails survive while the historical path
    does not. Levels are rebuilt from resampled returns (yields from changes).
    """
    rng = np.random.default_rng(seed)
    n = len(market.dates)
    order = np.empty(n, dtype=int)
    i = int(rng.integers(1, n))
    for k in range(n):
        order[k] = i
        i = int(rng.integers(1, n)) if rng.random() < 1.0 / block else i + 1
        if i >= n:
            i = int(rng.integers(1, n))

    idx = market.dates

    c_ret = (market.close / market.close.shift(1)).to_numpy()[order]
    c_ret = np.where(np.isfinite(c_ret), c_ret, 1.0)
    first = market.close.bfill().iloc[0].to_numpy()
    closes = first * np.cumprod(c_ret, axis=0)
    prev = np.vstack([first, closes[:-1]])
    out = {}
    for name in ("open", "high", "low"):
        r = (getattr(market, name) / market.close.shift(1)).to_numpy()[order]
        r = np.where(np.isfinite(r), r, 1.0)
        out[name] = pd.DataFrame(prev * r, index=idx, columns=market.close.columns)
    close_df = pd.DataFrame(closes, index=idx, columns=market.close.columns)
    out["high"] = np.maximum(out["high"], np.maximum(out["open"], close_df))
    out["low"] = np.minimum(out["low"], np.minimum(out["open"], close_df))
    volume = pd.DataFrame(market.volume.to_numpy()[order], index=idx, columns=market.close.columns)

    macro = {}
    for name, (src, ident, kind) in MACRO_SERIES.items():
        if name not in market.macro:
            continue
        if src == "price" and ident in close_df:
            macro[name] = close_df[ident]
            continue
        s = market.macro[name]
        d = s.diff().to_numpy()[order]
        d = np.where(np.isfinite(d), d, 0.0)
        macro[name] = pd.Series(s.dropna().iloc[0] + np.cumsum(d), index=idx)
    return Market(
        open=out["open"], high=out["high"], low=out["low"], close=close_df, volume=volume,
        macro=pd.DataFrame(macro, index=idx), cot=None, flows=None,
    )
