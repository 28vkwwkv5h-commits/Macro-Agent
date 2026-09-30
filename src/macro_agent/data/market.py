"""The single container every module reads market data from."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class Market:
    """Daily OHLCV per ticker, the six macro series, and optional positioning data.

    All frames share one trading calendar (`close.index`).

    - `cot`: commercial hedging pressure per ticker (net commercial position as a
      fraction of open interest), already aligned to the *release* date so it is
      never visible before the public could see it.
    - `flows`: retail ETF net creations per ticker, in dollars.
    """

    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame
    macro: pd.DataFrame
    cot: pd.DataFrame | None = None
    flows: pd.DataFrame | None = None

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.close.index

    @property
    def tickers(self) -> list[str]:
        return list(self.close.columns)

    def truncate(self, end) -> "Market":
        """Everything up to and including `end`. Used to prove causality."""
        end = pd.Timestamp(end)
        cut = lambda f: None if f is None else f.loc[:end]  # noqa: E731
        return Market(
            open=cut(self.open), high=cut(self.high), low=cut(self.low),
            close=cut(self.close), volume=cut(self.volume), macro=cut(self.macro),
            cot=cut(self.cot), flows=cut(self.flows),
        )

    def validate(self) -> None:
        idx = self.close.index
        for name in ("open", "high", "low", "volume", "macro"):
            frame = getattr(self, name)
            if not frame.index.equals(idx):
                raise ValueError(f"{name} is not on the close calendar")
        if not idx.is_monotonic_increasing or idx.has_duplicates:
            raise ValueError("calendar must be strictly increasing")

