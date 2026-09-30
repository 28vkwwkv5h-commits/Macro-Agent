"""Remote data sources (spec 11.5).

Prices: Tiingo adjusted closes when an API key is available, Stooq otherwise.
Not yfinance. Yields and real rates: FRED. The 2/10 spread is FRED T10Y2Y
directly, never computed from two other series.
"""
from __future__ import annotations

import io

import pandas as pd
import requests

TIMEOUT = 30


def _get(url: str, params: dict | None = None) -> requests.Response:
    resp = requests.get(url, params=params, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp


class StooqSource:
    """Daily closes from stooq.com (split and dividend adjusted)."""

    url = "https://stooq.com/q/d/l/"

    def fetch(self, ticker: str) -> pd.Series:
        resp = _get(self.url, {"s": f"{ticker.lower()}.us", "i": "d"})
        df = pd.read_csv(io.StringIO(resp.text))
        if "Close" not in df.columns:
            raise ValueError(f"Stooq returned no data for {ticker}")
        s = pd.Series(df["Close"].to_numpy(dtype=float), index=pd.to_datetime(df["Date"]))
        return s.sort_index().rename(ticker)


class TiingoSource:
    """Daily adjusted closes from api.tiingo.com. Needs an API key."""

    url = "https://api.tiingo.com/tiingo/daily/{ticker}/prices"

    def __init__(self, api_key: str, start: str = "1990-01-01"):
        self.api_key = api_key
        self.start = start

    def fetch(self, ticker: str) -> pd.Series:
        resp = _get(
            self.url.format(ticker=ticker),
            {"startDate": self.start, "token": self.api_key, "format": "json"},
        )
        rows = resp.json()
        if not rows:
            raise ValueError(f"Tiingo returned no data for {ticker}")
        idx = pd.to_datetime([r["date"][:10] for r in rows])
        s = pd.Series([float(r["adjClose"]) for r in rows], index=idx)
        return s.sort_index().rename(ticker)


class FredSource:
    """Series from FRED's public CSV endpoint (no key required)."""

    url = "https://fred.stlouisfed.org/graph/fredgraph.csv"

    def fetch(self, series_id: str) -> pd.Series:
        resp = _get(self.url, {"id": series_id})
        df = pd.read_csv(io.StringIO(resp.text))
        date_col = df.columns[0]
        values = pd.to_numeric(df[series_id], errors="coerce")  # FRED uses "." for missing
        s = pd.Series(values.to_numpy(dtype=float), index=pd.to_datetime(df[date_col]))
        return s.dropna().sort_index().rename(series_id)
