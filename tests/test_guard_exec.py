import pandas as pd
import pytest

from macro_agent.execution import Order, PaperBroker, diff_orders, reconcile
from macro_agent.guard import (
    AuditLog,
    GuardConfig,
    Halt,
    check_kill_switch,
    check_orders,
    check_stale,
    kill_switch_engaged,
)
from macro_agent.runner import run_once


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
    check_stale(frame[["fresh"]], "2024-01-31", 2)
    # Friday close is fine on Monday.
    check_stale(frame.loc[:"2024-01-26", ["fresh"]], "2024-01-29", 2)


def test_order_limits(guard):
    prices = {"A": 10.0}
    orders = [Order("A", "buy", 1)] * 11
    with pytest.raises(Halt, match="orders"):
        check_orders(orders, prices, 1000, guard)
    with pytest.raises(Halt, match="turnover"):
        check_orders([Order("A", "buy", 41)], prices, 1000, guard)
    assert check_orders([Order("A", "buy", 40)], prices, 1000, guard) == pytest.approx(0.4)


def test_diff_orders_sells_first_then_alphabetical():
    orders = diff_orders({"B": 10, "A": 5, "C": 3}, {"C": 8, "D": 2, "A": 5})
    assert [(o.ticker, o.side, o.quantity) for o in orders] == [
        ("C", "sell", 5), ("D", "sell", 2), ("B", "buy", 10),
    ]
    assert reconcile({"A": 5}, {"A": 5}) == {}
    assert reconcile({"A": 5}, {"A": 4}) == {"A": 1}


def test_paper_broker_persists(tmp_path):
    path = tmp_path / "acct.json"
    b = PaperBroker(path, starting_cash=1000)
    b.place(Order("A", "buy", 5), 100)
    b2 = PaperBroker(path)
    assert b2.positions() == {"A": 5}
    assert b2.cash() == 500
    b2.place(Order("A", "sell", 5), 110)
    assert PaperBroker(path).positions() == {}


def test_default_turnover_cap_blocks_initial_deployment(market, guard, tmp_path):
    # Spec section 7 caps daily turnover at 40%; deploying from cash is ~100%.
    prices, macro = market
    broker = PaperBroker(tmp_path / "acct.json")
    report = run_once(prices, macro, broker, guard, AuditLog(tmp_path / "a.jsonl"), dry_run=False)
    assert report.status == "halted" and "turnover" in report.message
    assert broker.positions() == {}


def test_dry_run_places_nothing(market, guard, tmp_path):
    prices, macro = market
    guard.max_daily_turnover = 1.0
    broker = PaperBroker(tmp_path / "acct.json")
    audit = AuditLog(tmp_path / "audit.jsonl")
    report = run_once(prices, macro, broker, guard, audit)
    assert report.status == "ok" and report.dry_run
    assert broker.positions() == {}
    assert audit.entries()[-1]["event"] == "run"


def test_kill_switch_flattens(market, guard, tmp_path):
    prices, macro = market
    broker = PaperBroker(tmp_path / "acct.json")
    broker.place(Order("SPY", "buy", 10), 50.0)
    guard.kill_file.write_text("stop")
    report = run_once(prices, macro, broker, guard, AuditLog(tmp_path / "a.jsonl"), dry_run=False)
    assert report.status == "flattened"
    assert broker.positions() == {}


def test_stale_input_halts_run(market, guard, tmp_path):
    prices, macro = market
    macro = macro.copy()
    macro.loc["2012-12-20":, "real10y"] = None
    alerts = []
    report = run_once(
        prices, macro, PaperBroker(tmp_path / "acct.json"), guard,
        AuditLog(tmp_path / "a.jsonl"), alert=alerts.append,
    )
    assert report.status == "halted" and "real10y" in report.message
    assert alerts


class StuckBroker(PaperBroker):
    def place(self, order, price):  # accepts orders, never fills
        pass


def test_unreconciled_trips_kill_switch(market, guard, tmp_path):
    prices, macro = market
    broker = StuckBroker(tmp_path / "acct.json", starting_cash=1_000_000)
    guard.max_daily_turnover = 1.0
    report = run_once(
        prices, macro, broker, guard, AuditLog(tmp_path / "a.jsonl"), dry_run=False
    )
    assert report.orders
    assert report.status == "halted"
    assert guard.kill_file.exists()
