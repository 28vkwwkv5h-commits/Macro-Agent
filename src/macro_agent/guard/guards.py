"""Guardrails (spec 7, 11.3). Every check either passes silently or raises Halt.

There is no turnover cap. Rogue trading is caught by checking whether orders
make sense, not by limiting how much the system may trade:

- the portfolio the orders would produce must obey the spec's own limits
  (position size, position count, sleeve and names caps, the regime whitelist,
  no leverage, no shorts);
- no run starts while the broker shows unfilled orders, and no decision is sent
  twice (so a retry cannot double a trade);
- a 20% peak-to-trough drawdown flattens the book and stops (spec 11.3);
- the kill switch, the stale-data halt and dry run by default.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import (
    KILL_ENV_VAR,
    MAX_DRAWDOWN_HALT,
    MAX_ORDERS_PER_DAY,
    STALE_BUSINESS_DAYS,
    TICKER_SLEEVE,
    WHITELIST,
    Params,
)

TOLERANCE = 0.02  # prices move between sizing and the check


class Halt(RuntimeError):
    """A guard fired. The run stops; nothing further is traded."""


@dataclass
class GuardConfig:
    kill_file: Path = field(default_factory=lambda: Path("state/KILL"))
    kill_env_var: str = KILL_ENV_VAR
    max_orders_per_day: int = MAX_ORDERS_PER_DAY
    stale_business_days: int = STALE_BUSINESS_DAYS
    max_drawdown: float = MAX_DRAWDOWN_HALT


def kill_switch_engaged(cfg: GuardConfig) -> bool:
    """A file or an env flag. Either one is enough. Trippable from a phone."""
    flag = os.environ.get(cfg.kill_env_var, "").strip().lower()
    return flag in {"1", "true", "yes", "on"} or Path(cfg.kill_file).exists()


def check_kill_switch(cfg: GuardConfig) -> None:
    if kill_switch_engaged(cfg):
        raise Halt("kill switch engaged")


def trip_kill_switch(cfg: GuardConfig, reason: str) -> None:
    path = Path(cfg.kill_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{datetime.now(timezone.utc).isoformat()} {reason}\n", encoding="utf-8")


def check_stale(frame: pd.DataFrame, as_of, max_business_days: int) -> None:
    """Halt if any column's last valid observation is more than
    `max_business_days` sessions before `as_of`. Never trade on stale data."""
    as_of = pd.Timestamp(as_of).normalize()
    stale = {}
    for col in frame.columns:
        last = frame[col].last_valid_index()
        if last is None:
            stale[col] = "no data"
            continue
        gap = int(np.busday_count(pd.Timestamp(last).date(), as_of.date()))
        if gap > max_business_days:
            stale[col] = str(pd.Timestamp(last).date())
    if stale:
        raise Halt(f"stale data as of {as_of.date()}: {stale}")


def check_open_orders(open_orders: list) -> None:
    if open_orders:
        raise Halt(f"broker shows {len(open_orders)} unfilled order(s); refusing to trade on top of them")


def check_drawdown(equity: float, peak: float, max_drawdown: float) -> None:
    if peak > 0 and equity < (1 - max_drawdown) * peak:
        raise Halt(f"drawdown {1 - equity / peak:.1%} breaches the {max_drawdown:.0%} hard stop")


def check_orders(
    orders,
    holdings: dict[str, float],
    cash: float,
    prices: dict[str, float],
    regime: str | None,
    params: Params,
    cfg: GuardConfig,
) -> None:
    """Halt unless the post-trade portfolio obeys the spec's limits."""
    if len(orders) > cfg.max_orders_per_day:
        raise Halt(f"{len(orders)} orders exceeds the daily maximum of {cfg.max_orders_per_day}")
    after = dict(holdings)
    cash_after = cash
    buys = set()
    for o in orders:
        if o.quantity <= 0:
            raise Halt(f"non-positive quantity in {o}")
        if o.side == "sell":
            if o.quantity > after.get(o.ticker, 0.0) + 1e-6:
                raise Halt(f"sell of {o.quantity} {o.ticker} exceeds holding: would be a short")
            after[o.ticker] = after.get(o.ticker, 0.0) - o.quantity
            cash_after += o.quantity * prices[o.ticker]
        else:
            buys.add(o.ticker)
            after[o.ticker] = after.get(o.ticker, 0.0) + o.quantity
            cash_after -= o.quantity * prices[o.ticker]
    if cash_after < -TOLERANCE * max(cash, 1.0):
        raise Halt(f"orders need {-cash_after:,.2f} more cash than the account holds: no leverage")
    equity = cash_after + sum(q * prices[t] for t, q in after.items() if q > 0)
    if equity <= 0:
        raise Halt("non-positive post-trade equity")
    open_positions = [t for t, q in after.items() if q > 1e-9]
    if len(open_positions) > params.full_positions:
        raise Halt(f"{len(open_positions)} positions exceeds the maximum of {params.full_positions}")
    allowed = WHITELIST.get(regime, frozenset()) if params.use_regime_whitelist else None
    for t in sorted(buys):
        if allowed is not None and t not in allowed:
            raise Halt(f"buy of {t} is outside the {regime} whitelist")
        weight = after[t] * prices[t] / equity
        if weight > params.max_position + TOLERANCE:
            raise Halt(f"{t} would be {weight:.1%} of the account, above {params.max_position:.0%}")
        sleeve = TICKER_SLEEVE.get(t, "individual_names")
        sleeve_w = sum(after[x] * prices[x] for x in open_positions if TICKER_SLEEVE.get(x, "individual_names") == sleeve) / equity
        cap = params.max_names if sleeve == "individual_names" else params.max_sleeve
        if sleeve_w > cap + TOLERANCE:
            raise Halt(f"{sleeve} would be {sleeve_w:.1%} of the account, above {cap:.0%}")


class AuditLog:
    """Append-only JSONL: inputs, scores, regime, confidence, orders, outcomes."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def append(self, event: str, **payload) -> dict:
        entry = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "event": event}
        entry.update(payload)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, sort_keys=True, default=str) + "\n")
        return entry

    def entries(self) -> list[dict]:
        if not self.path.exists():
            return []
        with self.path.open(encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]
