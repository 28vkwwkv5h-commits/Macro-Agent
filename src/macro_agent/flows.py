"""Flow score per instrument, 0 to 1 (spec 18.5).

| Component                                   | Weight | Sign   | Source                  |
| Hedging pressure, commercials (commodities) | 0.40   | Follow | Basu and Miffre 2013    |
| On-balance volume slope                     | 0.20   | Follow | Practitioner            |
| Relative strength versus SPY                | 0.20   | Follow | Practitioner            |
| Retail ETF fund flow                        | 0.20   | Fade   | Frazzini and Lamont 2008|

For instruments without hedging pressure its weight is spread across the rest.
Any component without data drops out and the remaining weights renormalise.
Each component is a percentile against the instrument's own trailing history.

The score is never a signal on its own. It becomes a multiplier of 0.5 to 1.0
on the BAN score, so it can halve a candidate but never veto one.
"""
from __future__ import annotations

import pandas as pd

from .config import BENCHMARK, DEFAULT_PARAMS, Params
from .data.market import Market
from .indicators import obv, rolling_pct_rank, rolling_slope

WEIGHTS = {"hedging": 0.40, "obv": 0.20, "rel_strength": 0.20, "retail": 0.20}


def flow_components(market: Market, params: Params = DEFAULT_PARAMS) -> dict[str, pd.DataFrame]:
    close, volume = market.close, market.volume
    w, rank_w = params.flow_slope_window, params.flow_rank_window
    comps: dict[str, pd.DataFrame] = {}

    if (volume > 0).any().any():
        avg_vol = volume.rolling(w, min_periods=w).mean().replace(0, float("nan"))
        slope = rolling_slope(obv(close, volume), w) / avg_vol
        comps["obv"] = rolling_pct_rank(slope, rank_w)

    if BENCHMARK in close:
        bench = close[BENCHMARK]
        rel = (close / close.shift(w)).div(bench / bench.shift(w), axis=0) - 1
        comps["rel_strength"] = rolling_pct_rank(rel, rank_w)

    if market.cot is not None and not market.cot.empty:
        hp = market.cot.reindex(columns=close.columns)
        comps["hedging"] = rolling_pct_rank(hp, params.cot_rank_weeks * 5)

    if market.flows is not None and not market.flows.empty:
        net = market.flows.reindex(columns=close.columns).rolling(21, min_periods=21).sum()
        comps["retail"] = 1.0 - rolling_pct_rank(net, rank_w)  # fade retail flow
    return comps


def flow_score(market: Market, params: Params = DEFAULT_PARAMS) -> pd.DataFrame:
    comps = flow_components(market, params)
    close = market.close
    total = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    weight = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    for name, frame in comps.items():
        frame = frame.reindex_like(close)
        present = frame.notna()
        total += frame.fillna(0.0) * WEIGHTS[name]
        weight += present * WEIGHTS[name]
    return (total / weight.where(weight > 0)).rename_axis(columns=None)
