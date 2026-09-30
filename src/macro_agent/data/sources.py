"""Remote data sources (spec 11.5).

Prices: Tiingo adjusted OHLCV when an API key is available, Stooq otherwise.
Not yfinance. Yields and real rates: FRED. The 2/10 spread is FRED T10Y2Y
directly, never computed from two other series. Hedging pressure: the CFTC
Commitments of Traders (legacy, futures only), free and weekly.
"""
from __future__ import annotations

import io

import pandas as pd
import requests

TIMEOUT = 30
OHLCV = ["open", "high", "low", "close", "volume"]


def _get(url: str, params: dict | None = None) -> requests.Response:
    resp = requests.get(url, params=params, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp


class StooqSource:
    """Daily OHLCV from stooq.com (split and dividend adjusted)."""

    url = "https://stooq.com/q/d/l/"

    def fetch(self, ticker: str) -> pd.DataFrame:
        resp = _get(self.url, {"s": f"{ticker.lower()}.us", "i": "d"})
        df = pd.read_csv(io.StringIO(resp.text))
        if "Close" not in df.columns:
            raise ValueError(f"Stooq returned no data for {ticker}")
        df = df.rename(columns=str.lower).set_index("date")
        df.index = pd.to_datetime(df.index)
        if "volume" not in df.columns:
            df["volume"] = 0.0
        return df[OHLCV].astype(float).sort_index()


class TiingoSource:
    """Daily adjusted OHLCV from api.tiingo.com. Needs an API key."""

    url = "https://api.tiingo.com/tiingo/daily/{ticker}/prices"

    def __init__(self, api_key: str, start: str = "1990-01-01"):
        self.api_key = api_key
        self.start = start

    def fetch(self, ticker: str) -> pd.DataFrame:
        resp = _get(
            self.url.format(ticker=ticker),
            {"startDate": self.start, "token": self.api_key, "format": "json"},
        )
        rows = resp.json()
        if not rows:
            raise ValueError(f"Tiingo returned no data for {ticker}")
        df = pd.DataFrame(rows)
        out = pd.DataFrame(
            {
                "open": df["adjOpen"], "high": df["adjHigh"], "low": df["adjLow"],
                "close": df["adjClose"], "volume": df["adjVolume"],
            }
        ).astype(float)
        out.index = pd.to_datetime(df["date"].str[:10])
        return out.sort_index()


class FredSource:
    """Series from FRED's public CSV endpoint (no key required)."""

    url = "https://fred.stlouisfed.org/graph/fredgraph.csv"

    def fetch(self, series_id: str) -> pd.Series:
        resp = _get(self.url, {"id": series_id})
        df = pd.read_csv(io.StringIO(resp.text))
        values = pd.to_numeric(df[series_id], errors="coerce")  # FRED uses "." for missing
        s = pd.Series(values.to_numpy(dtype=float), index=pd.to_datetime(df[df.columns[0]]))
        return s.dropna().sort_index().rename(series_id)


class CftcSource:
    """Legacy futures-only Commitments of Traders from the CFTC's public API.

    Returns commercial hedging pressure: (commercial long - commercial short) /
    open interest, indexed by the Tuesday report date.
    """

    url = "https://publicreporting.cftc.gov/resource/6dca-aqww.json"

    def fetch(self, contract_code: str) -> pd.Series:
        resp = _get(
            self.url,
            {
                "cftc_contract_market_code": contract_code,
                "$order": "report_date_as_yyyy_mm_dd",
                "$limit": 50000,
            },
        )
        rows = resp.json()
        if not rows:
            raise ValueError(f"CFTC returned no data for {contract_code}")
        df = pd.DataFrame(rows)
        longs = pd.to_numeric(df["comm_positions_long_all"])
        shorts = pd.to_numeric(df["comm_positions_short_all"])
        oi = pd.to_numeric(df["open_interest_all"])
        s = pd.Series(((longs - shorts) / oi).to_numpy(), index=pd.to_datetime(df["report_date_as_yyyy_mm_dd"]))
        return s.groupby(level=0).last().sort_index().rename(contract_code)
