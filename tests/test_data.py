import pandas as pd

from macro_agent.data import CftcSource, DataStore, FredSource, StooqSource, assemble_market
from macro_agent.data import sources
from macro_agent.montecarlo import synthetic_market


class FakeResponse:
    def __init__(self, text=None, payload=None):
        self.text = text
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def test_fred_parser_drops_missing(monkeypatch):
    csv = "observation_date,T10Y2Y\n2024-01-02,0.5\n2024-01-03,.\n2024-01-04,-0.1\n"
    monkeypatch.setattr(sources.requests, "get", lambda *a, **k: FakeResponse(csv))
    s = FredSource().fetch("T10Y2Y")
    assert list(s) == [0.5, -0.1]


def test_stooq_parser_returns_sorted_ohlcv(monkeypatch):
    csv = ("Date,Open,High,Low,Close,Volume\n2024-01-03,1,2,0.5,11.0,5\n"
           "2024-01-02,1,2,0.5,10.0,5\n")
    monkeypatch.setattr(sources.requests, "get", lambda *a, **k: FakeResponse(csv))
    df = StooqSource().fetch("SPY")
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert list(df["close"]) == [10.0, 11.0]


def test_cftc_hedging_pressure(monkeypatch):
    rows = [{"report_date_as_yyyy_mm_dd": "2024-01-02T00:00:00.000", "comm_positions_long_all": "300",
             "comm_positions_short_all": "100", "open_interest_all": "1000"}]
    monkeypatch.setattr(sources.requests, "get", lambda *a, **k: FakeResponse(payload=rows))
    s = CftcSource().fetch("088691")
    assert s.iloc[0] == 0.2


def test_store_roundtrip_and_assembly(tmp_path):
    m, _ = synthetic_market(start="2020-01-01", end="2020-12-31", tickers=["USO", "GLD", "UUP", "SPY"])
    store = DataStore(tmp_path)
    for t in m.tickers:
        bars = pd.DataFrame({f: getattr(m, f)[t] for f in ("open", "high", "low", "close", "volume")})
        store.save_frame("prices", t, bars)
    fred = {"DGS10": m.macro["ust10y"], "T10Y2Y": m.macro["spread_2s10s"].drop(pd.Timestamp("2020-07-03")),
            "DFII10": m.macro["real10y"]}
    for k, v in fred.items():
        store.save_series("fred", k, v)
    # A COT report dated Tuesday 2020-06-02 must not be visible before the following Monday.
    store.save_series("cot", "088691", pd.Series([0.3], index=[pd.Timestamp("2020-06-02")]))
    market = store.load_market(["USO", "GLD", "UUP", "SPY", "NOPE"])
    assert market.tickers == ["USO", "GLD", "UUP", "SPY"]
    pd.testing.assert_series_equal(market.close["SPY"], m.close["SPY"], check_names=False, check_freq=False)
    assert market.macro.loc["2020-07-03"].notna().all()  # FRED holiday forward-filled
    assert pd.isna(market.cot["GLD"].loc["2020-06-05"])
    assert market.cot["GLD"].loc["2020-06-08"] == 0.3


def test_assemble_market_validates_calendar():
    m, _ = synthetic_market(start="2020-01-01", end="2020-03-31", tickers=["SPY"])
    bars = {"SPY": pd.DataFrame({f: getattr(m, f)["SPY"] for f in ("open", "high", "low", "close", "volume")})}
    market = assemble_market(bars, {})
    market.validate()
    assert market.macro.empty or market.macro.index.equals(market.dates)


def test_synthetic_market_is_deterministic():
    a, ra = synthetic_market(end="2006-01-01", seed=3)
    b, rb = synthetic_market(end="2006-01-01", seed=3)
    pd.testing.assert_frame_equal(a.close, b.close)
    pd.testing.assert_series_equal(ra, rb)
