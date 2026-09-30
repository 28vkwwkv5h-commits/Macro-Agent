import pandas as pd

from macro_agent.data import DataStore, FredSource, StooqSource, build_macro_frame, synthetic_market
from macro_agent.data import sources


class FakeResponse:
    def __init__(self, text):
        self.text = text

    def raise_for_status(self):
        pass


def test_fred_parser_drops_missing(monkeypatch):
    csv = "observation_date,T10Y2Y\n2024-01-02,0.5\n2024-01-03,.\n2024-01-04,-0.1\n"
    monkeypatch.setattr(sources.requests, "get", lambda *a, **k: FakeResponse(csv))
    s = FredSource().fetch("T10Y2Y")
    assert list(s) == [0.5, -0.1]
    assert s.index[-1] == pd.Timestamp("2024-01-04")


def test_stooq_parser(monkeypatch):
    csv = "Date,Open,High,Low,Close,Volume\n2024-01-03,1,1,1,11.0,5\n2024-01-02,1,1,1,10.0,5\n"
    monkeypatch.setattr(sources.requests, "get", lambda *a, **k: FakeResponse(csv))
    s = StooqSource().fetch("SPY")
    assert list(s) == [10.0, 11.0]  # sorted ascending


def test_store_roundtrip_and_macro_frame(tmp_path):
    prices, fred = synthetic_market(start="2020-01-01", end="2020-12-31")
    store = DataStore(tmp_path)
    for t in ["USO", "GLD", "UUP", "SPY"]:
        store.save("prices", t, prices[t])
    for c in fred.columns:
        store.save("fred", c, fred[c])
    loaded = store.load_prices(["USO", "GLD", "UUP", "SPY", "NOPE"])
    assert list(loaded.columns) == ["USO", "GLD", "UUP", "SPY"]
    pd.testing.assert_series_equal(loaded["SPY"], prices["SPY"], check_names=False, check_freq=False)

    fred_gappy = store.load_fred().drop(pd.Timestamp("2020-07-03"))  # bond holiday
    macro = build_macro_frame(loaded, fred_gappy)
    assert list(macro.columns) == ["oil", "gold", "usd", "ust10y", "spread_2s10s", "real10y"]
    assert macro.loc["2020-07-03"].notna().all()  # forward-filled onto the price calendar


def test_synthetic_market_is_deterministic():
    a, _ = synthetic_market(end="2006-01-01", seed=3)
    b, _ = synthetic_market(end="2006-01-01", seed=3)
    pd.testing.assert_frame_equal(a, b)
