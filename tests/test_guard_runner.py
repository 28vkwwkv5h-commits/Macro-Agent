import json

import pandas as pd
import pytest

from macro_agent.backtest import BacktestConfig, run_backtest
from macro_agent.config import DEFAULT_PARAMS, cost_bps
from macro_agent.execution import Order, PaperBroker
from macro_agent.guard import (
    GuardConfig,
    Halt,
    check_drawdown,
    check_kill_switch,
    check_open_orders,
    check_orders,
    check_stale,
    kill_switch_engaged,
)
from macro_agent.runner import run_once

P = DEFAULT_PARAMS


@pytest.fixture
def guard(tmp_path, monkeypatch):
    monkeypatch.delenv("MACRO_AGENT_KILL", raising=False)
    return GuardConfig(kill_file=tmp_path / "KILL")


def test_kill_switch_file_and_env(guard, monkeypatch):
    assert not kill_switch_engaged(guard)
    guard.kill_file.write_text("stop")
    with pytest.raises(Halt):
        check_kill_switch(guard)
    guard.kill_file.unlink()
    monkeypatch.setenv("MACRO_AGENT_KILL", "1")
    assert kill_switch_engaged(guard)


def test_stale_data_halts():
    idx = pd.bdate_range("2024-01-01", "2024-01-31")
    frame = pd.DataFrame({"fresh": 1.0, "old": 1.0}, index=idx)
    frame.loc["2024-01-25":, "old"] = None
    with pytest.raises(Halt, match="old"):
        check_stale(frame, "2024-01-31", 2)
    check_stale(frame.loc[:"2024-01-26", ["fresh"]], "2024-01-29", 2)  # Friday close on Monday


def test_open_orders_and_drawdown():
    with pytest.raises(Halt, match="unfilled"):
        check_open_orders([{"id": 1}])
    check_drawdown(81, 100, 0.20)
    with pytest.raises(Halt, match="hard stop"):
        check_drawdown(79, 100, 0.20)


PX = {"GLD": 100.0, "TLT": 100.0, "IEF": 100.0, "XLU": 100.0, "XLP": 100.0, "SPY": 100.0}


@pytest.mark.parametrize(
    "orders,holdings,cash,match",
    [
        ([Order("GLD", "sell", 5)], {"GLD": 3}, 0, "short"),
        ([Order("GLD", "buy", 20)], {}, 1000, "leverage"),
        ([Order("SPY", "buy", 1)], {}, 10_000, "whitelist"),
        ([Order("GLD", "buy", 30)], {}, 10_000, "above 25%"),
        ([Order("GLD", "buy", 1)] * 11, {}, 10_000, "daily maximum"),
    ],
)
def test_order_checks_catch_rogue_orders(guard, orders, holdings, cash, match):
    with pytest.raises(Halt, match=match):
        check_orders(orders, holdings, cash, PX, "deflation", P, guard)


def test_order_checks_pass_a_sane_rebalance(guard):
    orders = [Order("GLD", "sell", 20), Order("TLT", "buy", 20)]
    check_orders(orders, {"GLD": 20}, 8000, PX, "deflation", P, guard)
    too_many = {t: 1.0 for t in ["GLD", "TLT", "IEF", "XLU", "XLP"]}
    with pytest.raises(Halt, match="positions"):
        check_orders([Order("SPY", "buy", 1)], too_many, 10_000, PX, "goldilocks", P, guard)


# --- the live runner ---------------------------------------------------------


def test_dry_run_changes_nothing(market, features, guard, tmp_path):
    broker = PaperBroker(tmp_path / "acct.json", starting_cash=1e6)
    report = run_once(market, broker, tmp_path, guard, features=features)
    assert report.status == "ok" and report.dry_run
    assert broker.positions() == {}
    assert not (tmp_path / "book.json").exists()
    assert json.loads((tmp_path / "audit.jsonl").read_text().splitlines()[-1])["event"] == "run"


def _entry_day(market, features, guard, tmp_path) -> str:
    for day in market.dates[900::5]:
        r = run_once(market, PaperBroker(tmp_path / "probe.json", 1e6), tmp_path / "probe", guard,
                     as_of=day, features=features)
        if any(o["reason"] == "entry" for o in r.orders):
            return day
    pytest.fail("no entry day in fixture")


def test_live_run_persists_and_refuses_duplicates(market, features, guard, tmp_path):
    day = _entry_day(market, features, guard, tmp_path)
    broker = PaperBroker(tmp_path / "acct.json", starting_cash=1e6)
    state = tmp_path / "live"
    first = run_once(market, broker, state, guard, as_of=day, dry_run=False, features=features)
    assert first.status == "ok" and first.fills
    book = json.loads((state / "book.json").read_text())
    assert {t: p["shares"] for t, p in book["positions"].items()} == pytest.approx(broker.positions())
    # Same session again with the pre-trade book restored: the decision id repeats.
    (state / "book.json").write_text(json.dumps({"cash": 1e6, "positions": {}, "cooldown_until": {}, "peak_equity": 1e6}))
    fresh = PaperBroker(tmp_path / "acct2.json", starting_cash=1e6)
    again = run_once(market, fresh, state, guard, as_of=day, dry_run=False, features=features)
    assert again.status == "duplicate" and fresh.positions() == {}


def test_kill_switch_flattens(market, features, guard, tmp_path):
    broker = PaperBroker(tmp_path / "acct.json", starting_cash=1e6)
    broker.place(Order("SPY", "buy", 10), 50.0)
    (tmp_path / "book.json").write_text(json.dumps({"cash": broker.cash(), "positions": {
        "SPY": {"ticker": "SPY", "entry_date": "2012-01-02", "entry_level": 1.0, "stop": 0.5,
                "initial_stop": 0.5, "atr_frozen": 0.1, "confirm_low": 0.4, "swing_low": 0.3,
                "swing_high": 2.0, "target_shares": 10, "rr_at_entry": 3.0, "ban_at_entry": 1.0,
                "shares": 10.0}}}))
    guard.kill_file.write_text("stop")
    report = run_once(market, broker, tmp_path, guard, dry_run=False, features=features)
    assert report.status == "flattened"
    assert broker.positions() == {}


def test_book_broker_drift_halts_and_trips(market, features, guard, tmp_path):
    broker = PaperBroker(tmp_path / "acct.json", starting_cash=1e6)
    broker.place(Order("GLD", "buy", 5), 50.0)  # a position the book knows nothing about
    report = run_once(market, broker, tmp_path, guard, dry_run=False, features=features)
    assert report.status == "halted" and "disagree" in report.message
    assert guard.kill_file.exists()


class NextOpenBroker(PaperBroker):
    """Fills at the next session's open with the backtest's costs, so a live
    replay can be compared fill-for-fill with the backtest (spec 19.4)."""

    def __init__(self, path, cash, market):
        super().__init__(path, starting_cash=cash)
        self.market = market
        self.t = 0

    def place(self, order, price):
        px = self.market.open[order.ticker].iat[self.t + 1]
        cost = cost_bps(order.ticker) / 1e4
        return super().place(order, px * (1 + cost) if order.side == "buy" else px * (1 - cost))


def test_live_replay_reconciles_with_backtest(market, features, guard, tmp_path):
    """Spec 19.4: replay live trading through the runner and confirm the same fills."""
    start, end = "2009-03-02", "2009-09-30"
    cfg = BacktestConfig(start, end, cash_ticker=None, drawdown_halt=None)
    bt = run_backtest(market, cfg, features=features)
    assert len(bt.trades) >= 4, "fixture window should contain trading"

    broker = NextOpenBroker(tmp_path / "acct.json", 1_000_000.0, market)
    dates = market.dates
    live_fills = []
    for t in range(dates.searchsorted(pd.Timestamp(start)), dates.get_loc(pd.Timestamp(end))):
        broker.t = t
        r = run_once(market, broker, tmp_path / "state", guard, as_of=dates[t], dry_run=False, features=features)
        assert r.status == "ok", r.message
        live_fills += [(dates[t + 1], f["ticker"], f["side"], round(f["filled"], 6)) for f in r.fills]
    bt_fills = [(row.date, row.ticker, row.side, round(row.quantity, 6)) for row in bt.trades.itertuples()]
    assert live_fills == bt_fills
