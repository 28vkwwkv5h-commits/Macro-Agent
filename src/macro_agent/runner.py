"""One scheduled run: guard, decide, (maybe) trade, reconcile, record.

This is the only code the orchestrating agent invokes. It runs the same
`Strategy.step` the backtester runs and reports what happened. It never forms
a view (spec 0, 6b).

State kept between runs, under `state_dir`:
- book.json: positions with their stop/state machine, cooldowns, peak equity
- executed.json: decision ids already sent, so a retry never doubles a trade
- audit.jsonl: append-only record of every run
- KILL: the kill switch file
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd

from .config import DEFAULT_PARAMS, MACRO_SERIES, WHITELIST, Params
from .data.market import Market
from .execution import Broker, Order, reconcile
from .features import Features, build_features
from .guard import (
    AuditLog,
    GuardConfig,
    Halt,
    check_drawdown,
    check_open_orders,
    check_orders,
    check_stale,
    kill_switch_engaged,
    trip_kill_switch,
)
from .positions import EXIT_KILL, Book
from .strategy import Strategy, apply_fill

RECONCILE_ATTEMPTS = 2


@dataclass
class RunReport:
    as_of: str
    dry_run: bool
    status: str  # "ok", "halted", "flattened", "duplicate"
    message: str = ""
    decision: dict | None = None
    orders: list[dict] = field(default_factory=list)
    fills: list[dict] = field(default_factory=list)
    unreconciled: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class RunState:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.book_path = self.root / "book.json"
        self.executed_path = self.root / "executed.json"

    def load_book(self, cash: float) -> Book:
        if self.book_path.exists():
            return Book.from_dict(json.loads(self.book_path.read_text(encoding="utf-8")))
        return Book(cash=cash, peak_equity=cash)

    def save_book(self, book: Book) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.book_path.write_text(json.dumps(book.to_dict(), indent=2, default=float), encoding="utf-8")

    def executed(self) -> set[str]:
        if not self.executed_path.exists():
            return set()
        return set(json.loads(self.executed_path.read_text(encoding="utf-8")))

    def mark_executed(self, decision_id: str) -> None:
        ids = sorted(self.executed() | {decision_id})
        self.root.mkdir(parents=True, exist_ok=True)
        self.executed_path.write_text(json.dumps(ids), encoding="utf-8")


def _place_all(broker: Broker, book: Book, orders: list[Order], prices: dict, report: RunReport):
    for order in orders:
        filled, fill_px = broker.place(order, prices[order.ticker])
        if filled > 0:
            apply_fill(book, Order(order.ticker, order.side, filled, order.reason, order.decision_id), fill_px)
            report.fills.append({**order.to_dict(), "filled": filled, "price": fill_px})


def run_once(
    market: Market,
    broker: Broker,
    state_dir: str | Path,
    guard: GuardConfig,
    params: Params = DEFAULT_PARAMS,
    as_of=None,
    dry_run: bool = True,
    alert: Callable[[str], None] = lambda msg: None,
    features: Features | None = None,
) -> RunReport:
    """`features` may be passed in when already built for a longer history
    starting on the same date; features are causal, so rows up to `as_of` are
    identical either way (tests/test_features.py)."""
    as_of = pd.Timestamp(as_of) if as_of is not None else market.dates[-1]
    market = market.truncate(as_of)
    t = len(market.dates) - 1
    report = RunReport(as_of=str(market.dates[t].date()), dry_run=dry_run, status="ok")
    state = RunState(state_dir)
    audit = AuditLog(Path(state_dir) / "audit.jsonl")
    last_px = market.close.ffill().iloc[t].dropna().to_dict()

    def halt(message: str, trip: bool = False) -> RunReport:
        report.status, report.message = "halted", message
        if trip:
            trip_kill_switch(guard, message)
        audit.append("halt", **report.to_dict())
        alert(f"macro-agent halted: {message}")
        return report

    # The broker is the source of truth for cash and shares.
    book = state.load_book(broker.cash())
    book.cash = broker.cash()
    broker_pos = broker.positions()
    book_pos = {tk: p.shares for tk, p in book.positions.items() if p.shares > 0}
    drift = reconcile(book_pos, broker_pos)
    if drift:
        return halt(f"book and broker disagree before trading: {drift}", trip=True)

    # 1. Kill switch: flatten to cash and stop.
    if kill_switch_engaged(guard):
        orders = [Order(tk, "sell", q, EXIT_KILL) for tk, q in sorted(broker_pos.items()) if q > 0]
        report.orders = [o.to_dict() for o in orders]
        if not dry_run:
            for tk in list(book.positions):
                book.positions[tk].exiting = EXIT_KILL
            _place_all(broker, book, orders, last_px, report)
            state.save_book(book)
        report.status, report.message = "flattened", "kill switch engaged"
        audit.append("kill_switch", **report.to_dict())
        alert("macro-agent kill switch engaged: flattened to cash")
        return report

    # 2. Broker hygiene and data freshness.
    try:
        check_open_orders(broker.open_orders())
        universe = sorted((set().union(*WHITELIST.values()) | set(book.positions)) & set(market.tickers))
        missing = sorted(set().union(*WHITELIST.values()) - set(market.tickers))
        if missing:
            report.warnings.append(f"whitelisted tickers with no data (silent holes): {missing}")
        check_stale(market.macro[[c for c in MACRO_SERIES if c in market.macro]], as_of, guard.stale_business_days)
        check_stale(market.close[universe], as_of, guard.stale_business_days)
    except Halt as exc:
        return halt(str(exc))

    # 3. Portfolio hard stop (spec 11.3).
    equity = book.equity(last_px)
    book.peak_equity = max(book.peak_equity, equity)
    try:
        check_drawdown(equity, book.peak_equity, guard.max_drawdown)
    except Halt as exc:
        if not dry_run:
            orders = [Order(tk, "sell", q, "drawdown_hard_stop") for tk, q in sorted(broker_pos.items()) if q > 0]
            _place_all(broker, book, orders, last_px, report)
            state.save_book(book)
        return halt(str(exc), trip=True)

    # 4. Decide. Same function as the backtest.
    features = features or build_features(market, params)
    strategy = Strategy(market, features, params)
    decision = strategy.step(t, book)
    report.decision = decision.to_dict()
    report.orders = [o.to_dict() for o in decision.orders]
    if decision.decision_id in state.executed():
        report.status, report.message = "duplicate", "decision already executed; nothing sent"
        audit.append("duplicate", **report.to_dict())
        return report

    # 5. Do the orders make sense?
    try:
        check_orders(decision.orders, broker_pos, broker.cash(), last_px, decision.regime, params, guard)
    except Halt as exc:
        return halt(str(exc))

    if dry_run:
        # The step updated the plan in memory only; a dry run never saves it.
        report.message = "dry run: no orders placed"
        audit.append("run", **report.to_dict())
        return report

    # 6. Trade, then reconcile against the broker.
    state.mark_executed(decision.decision_id)
    _place_all(broker, book, decision.orders, last_px, report)
    # Re-read the broker up to RECONCILE_ATTEMPTS times (fills can land late on
    # a real venue); anything still different trips the kill switch.
    for _ in range(RECONCILE_ATTEMPTS):
        target = {tk: p.shares for tk, p in book.positions.items() if p.shares > 0}
        report.unreconciled = reconcile(target, broker.positions())
        if not report.unreconciled:
            break
    state.save_book(book)
    if report.unreconciled:
        return halt(f"unreconciled after {RECONCILE_ATTEMPTS} attempts: {report.unreconciled}", trip=True)
    report.message = "orders placed and reconciled"
    audit.append("run", **report.to_dict())
    return report
