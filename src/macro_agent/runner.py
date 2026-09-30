"""One scheduled run: guard, decide, size, diff, (maybe) trade, reconcile.

This is the only code the orchestrating agent invokes. It runs the deterministic
modules and reports what happened. It never forms a view.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Callable

import pandas as pd

from .config import WHITELIST
from .execution import Broker, Order, diff_orders, reconcile
from .guard import (
    AuditLog,
    GuardConfig,
    Halt,
    check_orders,
    check_stale,
    kill_switch_engaged,
    trip_kill_switch,
)
from .pipeline import decide
from .sizing import target_shares

RECONCILE_ATTEMPTS = 2


@dataclass
class RunReport:
    as_of: str
    dry_run: bool
    status: str  # "ok", "halted", "flattened"
    message: str = ""
    decision: dict | None = None
    orders: list[dict] = field(default_factory=list)
    unreconciled: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _equity(broker: Broker, prices: dict[str, float]) -> float:
    return broker.cash() + sum(q * prices[t] for t, q in broker.positions().items())


def _execute(broker: Broker, orders: list[Order], prices: dict[str, float], target):
    """Place orders, then re-diff and retry. Returns what is still unreconciled."""
    for _ in range(RECONCILE_ATTEMPTS):
        for order in orders:
            broker.place(order, prices[order.ticker])
        diff = reconcile(target, broker.positions())
        if not diff:
            return {}
        orders = diff_orders(target, broker.positions())
    return reconcile(target, broker.positions())


def run_once(
    prices: pd.DataFrame,
    macro: pd.DataFrame,
    broker: Broker,
    guard: GuardConfig,
    audit: AuditLog,
    as_of=None,
    dry_run: bool = True,
    fractional_increment: float | None = None,
    alert: Callable[[str], None] = lambda msg: None,
) -> RunReport:
    as_of = pd.Timestamp(as_of) if as_of is not None else prices.index[-1]
    last_px = prices.loc[:as_of].ffill().iloc[-1].dropna().to_dict()
    report = RunReport(as_of=str(as_of.date()), dry_run=dry_run, status="ok")

    def halt(message: str) -> RunReport:
        report.status, report.message = "halted", message
        audit.append("halt", **report.to_dict())
        alert(f"macro-agent halted: {message}")
        return report

    # 1. Kill switch: flatten to cash and stop.
    if kill_switch_engaged(guard):
        orders = diff_orders({}, broker.positions())
        report.orders = [o.to_dict() for o in orders]
        if not dry_run:
            report.unreconciled = _execute(broker, orders, last_px, {})
        report.status, report.message = "flattened", "kill switch engaged"
        audit.append("kill_switch", **report.to_dict())
        alert("macro-agent kill switch engaged: flattened to cash")
        return report

    # 2. Stale data: the six macro inputs and every whitelisted instrument we hold data for.
    universe = sorted(set().union(*WHITELIST.values()) & set(prices.columns))
    missing = sorted(set().union(*WHITELIST.values()) - set(prices.columns))
    if missing:
        report.warnings.append(f"whitelisted tickers with no data (silent holes): {missing}")
    try:
        check_stale(macro.loc[:as_of], as_of, guard.stale_business_days)
        check_stale(prices.loc[:as_of, universe], as_of, guard.stale_business_days)
    except Halt as exc:
        return halt(str(exc))

    # 3. Decide. Deterministic, data only, no portfolio input.
    decision = decide(prices, macro, as_of)
    report.decision = decision.to_dict()

    # 4. Size (the one place rounding happens) and diff against the broker.
    equity = _equity(broker, last_px)
    target = target_shares(decision.weights, equity, last_px, fractional_increment)
    orders = diff_orders(target, broker.positions())
    report.orders = [o.to_dict() for o in orders]

    # 5. Order-count and turnover limits.
    try:
        check_orders(orders, last_px, equity, guard)
    except Halt as exc:
        return halt(str(exc))

    if dry_run:
        report.message = "dry run: no orders placed"
        audit.append("run", **report.to_dict())
        return report

    # 6. Trade and reconcile. Anything unreconciled after two attempts trips the kill switch.
    report.unreconciled = _execute(broker, orders, last_px, target)
    if report.unreconciled:
        trip_kill_switch(guard, f"unreconciled after {RECONCILE_ATTEMPTS} attempts")
        return halt(f"unreconciled positions: {report.unreconciled}")
    report.message = "orders placed and reconciled"
    audit.append("run", **report.to_dict())
    return report
