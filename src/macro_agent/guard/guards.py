"""Guardrails. Every check either passes silently or raises Halt."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import KILL_ENV_VAR, MAX_DAILY_TURNOVER, MAX_ORDERS_PER_DAY, STALE_BUSINESS_DAYS


class Halt(RuntimeError):
    """A guard fired. The run stops; nothing further is traded."""


@dataclass
class GuardConfig:
    kill_file: Path = field(default_factory=lambda: Path("state/KILL"))
    kill_env_var: str = KILL_ENV_VAR
    max_orders_per_day: int = MAX_ORDERS_PER_DAY
    max_daily_turnover: float = MAX_DAILY_TURNOVER
    stale_business_days: int = STALE_BUSINESS_DAYS


def kill_switch_engaged(cfg: GuardConfig) -> bool:
    """A file or an env flag. Either one is enough."""
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


def check_orders(orders, prices: dict[str, float], equity: float, cfg: GuardConfig) -> float:
    """Halt on too many orders or too much turnover. Returns the turnover."""
    if len(orders) > cfg.max_orders_per_day:
        raise Halt(f"{len(orders)} orders exceeds the daily maximum of {cfg.max_orders_per_day}")
    traded = sum(o.quantity * prices[o.ticker] for o in orders)
    turnover = traded / equity if equity > 0 else 0.0
    if turnover > cfg.max_daily_turnover + 1e-12:
        raise Halt(
            f"turnover {turnover:.1%} exceeds the daily maximum of {cfg.max_daily_turnover:.0%}"
        )
    return turnover


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
