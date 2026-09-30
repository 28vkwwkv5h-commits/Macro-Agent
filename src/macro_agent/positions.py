"""Positions, the book, and the per-position state machine (spec 15, 16).

A position sits in one of four states and the stop rule differs in each
(spec 16.6). The state is kept explicitly rather than inferred from price.

| State         | Entered when                                   | Stop                                       |
| ACCUMULATING  | Confirmation fires, tranches running           | Fib level minus 1.0 ATR, frozen            |
| IMPULSE       | Weekly close above the prior swing high        | Unchanged                                  |
| CONSOLIDATING | Weekly range < 60% of the impulse's avg range  | max(stop, 0.618 of impulse - 0.5 imp. ATR) |
| RESOLVED      | Weekly close above the flag high               | Ratchet up the ladder, new impulse begins  |

RESOLVED is transient: the position re-enters IMPULSE the same week. Stops only
ever move up. Every stop and exit is judged on a weekly close (spec 15.5).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from .config import Params
from .fib import ladder

ACCUMULATING = "ACCUMULATING"
IMPULSE = "IMPULSE"
CONSOLIDATING = "CONSOLIDATING"

# Exit reasons
EXIT_STOP = "fib_stop"
EXIT_INVALIDATION = "structural_invalidation"
EXIT_FLAG_TIME = "flag_time_limit"
EXIT_REGIME = "regime_exit"
EXIT_CONFIDENCE = "confidence_exit"
EXIT_KILL = "kill_switch"
COOLDOWN_REASONS = {EXIT_STOP, EXIT_INVALIDATION}


@dataclass
class Position:
    ticker: str
    entry_date: str
    entry_level: float  # the fib support the stop hangs from
    stop: float
    initial_stop: float
    atr_frozen: float
    confirm_low: float
    swing_low: float
    swing_high: float
    target_shares: float  # full size once every tranche is in
    rr_at_entry: float
    ban_at_entry: float
    tranches_total: int = 4
    shares: float = 0.0
    cost_basis: float = 0.0  # total cash paid, including costs
    state: str = ACCUMULATING
    tranches_done: int = 0
    next_tranche_date: str | None = None  # a calendar date, stable across data reloads
    lowest_low: float = np.inf
    impulse_start: float | None = None
    impulse_high: float | None = None
    impulse_atr: float | None = None
    impulse_ranges: list = field(default_factory=list)
    flag_high: float | None = None
    flag_low: float | None = None
    flag_bars: int = 0
    off_whitelist_days: int = 0
    exiting: str | None = None  # exit reason once an exit order is issued

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Position":
        d = dict(d)
        if d.get("lowest_low") is None:
            d["lowest_low"] = np.inf
        return cls(**d)


@dataclass
class Book:
    cash: float
    positions: dict[str, Position] = field(default_factory=dict)
    cooldown_until: dict[str, str] = field(default_factory=dict)  # ticker -> first date re-entry is allowed
    peak_equity: float = 0.0

    def equity(self, prices: dict[str, float]) -> float:
        return self.cash + sum(p.shares * prices[t] for t, p in self.positions.items() if p.shares)

    def reserved_cash(self, prices: dict[str, float]) -> float:
        """Cash earmarked for tranches not yet bought."""
        return sum(
            max(p.target_shares - p.shares, 0.0) * prices[t]
            for t, p in self.positions.items()
            if not p.exiting and p.tranches_done < p.tranches_total
        )

    def to_dict(self) -> dict:
        return {
            "cash": self.cash,
            "peak_equity": self.peak_equity,
            "cooldown_until": dict(sorted(self.cooldown_until.items())),
            "positions": {t: p.to_dict() for t, p in sorted(self.positions.items())},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Book":
        return cls(
            cash=float(d["cash"]),
            positions={t: Position.from_dict(p) for t, p in d.get("positions", {}).items()},
            cooldown_until=dict(d.get("cooldown_until", {})),
            peak_equity=float(d.get("peak_equity", 0.0)),
        )


def weekly_update(
    pos: Position,
    w_high: float,
    w_low: float,
    w_close: float,
    prior_swing_high: float,
    daily_atr: float,
    params: Params,
) -> str | None:
    """Advance one completed weekly bar. Returns an exit reason, or None.

    Exits are judged against the stop as it stood coming into the week; the
    state and stop are then updated for the next week.
    """
    pos.lowest_low = min(pos.lowest_low, w_low)

    if params.use_stops and w_close < pos.stop:
        return EXIT_STOP
    if w_close < pos.confirm_low:
        return EXIT_INVALIDATION

    if pos.state == ACCUMULATING:
        if np.isfinite(prior_swing_high) and w_close > prior_swing_high:
            pos.state = IMPULSE
            pos.impulse_start = pos.lowest_low
            pos.impulse_high = w_high
            pos.impulse_ranges = [w_high - w_low]
        return None

    if pos.state == IMPULSE:
        prior = pos.impulse_ranges[:]
        pos.impulse_high = max(pos.impulse_high, w_high)
        pos.impulse_ranges.append(w_high - w_low)
        if prior and (w_high - w_low) < params.consolidation_range * float(np.mean(prior)):
            pos.state = CONSOLIDATING
            pos.impulse_atr = daily_atr  # frozen for the life of this flag
            leg = pos.impulse_high - pos.impulse_start
            pattern = pos.impulse_high - params.pattern_retrace * leg - params.pattern_atr * pos.impulse_atr
            pos.stop = max(pos.stop, pattern)
            pos.flag_high, pos.flag_low, pos.flag_bars = w_high, w_low, 1
        return None

    # CONSOLIDATING
    if w_close > pos.flag_high:
        # RESOLVED: ratchet up the ladder, then start a new impulse leg.
        levels = ladder(pos.swing_low, pos.swing_high)
        cleared = levels[levels < w_close]
        if cleared.size:
            buffer = params.stop_atr * (pos.impulse_atr or pos.atr_frozen)
            pos.stop = max(pos.stop, float(cleared.max()) - buffer)
        pos.state = IMPULSE
        pos.impulse_start = pos.flag_low
        pos.impulse_high = w_high
        pos.impulse_ranges = [w_high - w_low]
        pos.flag_high = pos.flag_low = None
        pos.flag_bars = 0
        return None
    pos.flag_bars += 1
    pos.flag_high = max(pos.flag_high, w_high)
    pos.flag_low = min(pos.flag_low, w_low)
    if pos.flag_bars > params.max_flag_weeks:
        return EXIT_FLAG_TIME
    return None
