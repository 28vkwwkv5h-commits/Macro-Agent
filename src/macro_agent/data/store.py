"""On-disk cache. Every run reads from here, never from the network directly."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import MACRO_SERIES


class DataStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def _path(self, kind: str, name: str) -> Path:
        return self.root / kind / f"{name}.csv"

    def save(self, kind: str, name: str, series: pd.Series) -> None:
        path = self._path(kind, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        series.rename("value").to_frame().to_csv(path, index_label="date")

    def load(self, kind: str, name: str) -> pd.Series:
        df = pd.read_csv(self._path(kind, name), index_col="date", parse_dates=True)
        return df["value"].astype(float).rename(name)

    def has(self, kind: str, name: str) -> bool:
        return self._path(kind, name).exists()

    def load_prices(self, tickers) -> pd.DataFrame:
        """Closes for every cached ticker in `tickers`, on the union calendar."""
        cols = {t: self.load("prices", t) for t in tickers if self.has("prices", t)}
        return pd.DataFrame(cols).sort_index()

    def load_fred(self) -> pd.DataFrame:
        ids = [ident for src, ident, _ in MACRO_SERIES.values() if src == "fred"]
        return pd.DataFrame({i: self.load("fred", i) for i in ids}).sort_index()


def build_macro_frame(prices: pd.DataFrame, fred: pd.DataFrame) -> pd.DataFrame:
    """The six regime inputs as named columns on the trading calendar.

    FRED series skip bond-market holidays, so they are forward-filled onto the
    price calendar for at most five sessions. Longer gaps stay NaN and the
    stale-data guard halts the run.
    """
    cols = {}
    for name, (src, ident, _) in MACRO_SERIES.items():
        frame = prices if src == "price" else fred
        if ident in frame.columns:
            cols[name] = frame[ident]
    macro = pd.DataFrame(cols)
    calendar = prices.index
    macro = macro.reindex(macro.index.union(calendar)).sort_index().ffill(limit=5)
    return macro.reindex(calendar)
