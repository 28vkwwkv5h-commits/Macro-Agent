"""On-disk cache and assembly into a `Market`. Runs read from here, never the network."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import COT_CONTRACTS, MACRO_SERIES
from .market import Market
from .sources import OHLCV

# COT reports are as of Tuesday and published Friday afternoon. Treat them as
# visible from the following Monday so no backtest trades on them early.
COT_RELEASE_LAG_DAYS = 6
# FRED skips bond-market holidays; forward-fill at most this many sessions.
MAX_FILL_SESSIONS = 5


class DataStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def _path(self, kind: str, name: str) -> Path:
        return self.root / kind / f"{name}.csv"

    def has(self, kind: str, name: str) -> bool:
        return self._path(kind, name).exists()

    def save_frame(self, kind: str, name: str, frame: pd.DataFrame) -> None:
        path = self._path(kind, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index_label="date")

    def save_series(self, kind: str, name: str, series: pd.Series) -> None:
        self.save_frame(kind, name, series.rename("value").to_frame())

    def load_frame(self, kind: str, name: str) -> pd.DataFrame:
        return pd.read_csv(self._path(kind, name), index_col="date", parse_dates=True)

    def load_series(self, kind: str, name: str) -> pd.Series:
        return self.load_frame(kind, name)["value"].astype(float).rename(name)

    def load_market(self, tickers, flows_csv: str | Path | None = None) -> Market:
        bars = {t: self.load_frame("prices", t) for t in tickers if self.has("prices", t)}
        if not bars:
            raise FileNotFoundError(f"no cached prices under {self.root}")
        fred = {
            ident: self.load_series("fred", ident)
            for src, ident, _ in MACRO_SERIES.values()
            if src == "fred" and self.has("fred", ident)
        }
        cot = {
            code: self.load_series("cot", code)
            for code in set(COT_CONTRACTS.values())
            if self.has("cot", code)
        }
        flows = pd.read_csv(flows_csv, index_col=0, parse_dates=True) if flows_csv else None
        return assemble_market(bars, fred, cot, flows)


def assemble_market(
    bars: dict[str, pd.DataFrame],
    fred: dict[str, pd.Series],
    cot: dict[str, pd.Series] | None = None,
    flows: pd.DataFrame | None = None,
) -> Market:
    """Put per-ticker OHLCV, FRED series and COT series on one trading calendar.

    The calendar is the union of all price dates. A ticker's bars are NaN before
    it listed; nothing is back-filled.
    """
    calendar = pd.DatetimeIndex(sorted(set().union(*(b.index for b in bars.values()))))
    fields = {f: pd.DataFrame({t: b[f] for t, b in bars.items()}).reindex(calendar) for f in OHLCV}

    def on_calendar(s: pd.Series) -> pd.Series:
        return s.reindex(s.index.union(calendar)).sort_index().ffill(limit=MAX_FILL_SESSIONS).reindex(calendar)

    macro = {}
    for name, (src, ident, _) in MACRO_SERIES.items():
        if src == "price" and ident in fields["close"]:
            macro[name] = fields["close"][ident]
        elif src == "fred" and ident in fred:
            macro[name] = on_calendar(fred[ident])
    cot_frame = None
    if cot:
        cols = {}
        for ticker, code in COT_CONTRACTS.items():
            if code in cot and ticker in fields["close"]:
                s = cot[code].copy()
                s.index = s.index + pd.Timedelta(days=COT_RELEASE_LAG_DAYS)
                # Weekly data: carry each release until the next one (about 5 sessions).
                cols[ticker] = s.reindex(s.index.union(calendar)).sort_index().ffill(limit=10).reindex(calendar)
        cot_frame = pd.DataFrame(cols, index=calendar) if cols else None
    if flows is not None:
        flows = flows.reindex(calendar)
    market = Market(
        open=fields["open"], high=fields["high"], low=fields["low"],
        close=fields["close"], volume=fields["volume"].fillna(0.0),
        macro=pd.DataFrame(macro, index=calendar), cot=cot_frame, flows=flows,
    )
    market.validate()
    return market
