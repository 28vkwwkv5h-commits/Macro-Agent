"""Event-driven daily backtest.

Timing, which is where backtests lie (spec 20.1):

- The strategy decides at the close of session t using data up to t.
- Its orders fill at the OPEN of session t + execution_delay (default 1), so no
  signal ever shares a price with the data that produced it. A delay below one
  session is refused.
- Every fill pays half the spread plus slippage for that instrument (spec 8).
- Idle cash earns the T-bill ETF's return.

The engine calls exactly the same `Strategy.step` the live runner calls.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import CASH_TICKER, DEFAULT_PARAMS, HOLDOUT_START, MAX_DRAWDOWN_HALT, Params, cost_bps
from ..data.market import Market
from ..features import Features, build_features
from ..positions import Book
from ..execution.orders import Order
from ..strategy import Strategy, apply_fill


@dataclass
class BacktestConfig:
    start: str
    end: str
    initial_capital: float = 1_000_000.0
    execution_delay: int = 1
    cost_multiplier: float = 1.0  # 0 = frictionless, for structural tests only
    cash_ticker: str | None = CASH_TICKER
    drawdown_halt: float | None = MAX_DRAWDOWN_HALT  # spec 11.3: flatten and stop
    holdout_start: str = HOLDOUT_START
    use_holdout: bool = False

    def validate(self) -> None:
        if self.execution_delay < 1:
            raise ValueError("execution_delay must be at least one session (no look-ahead)")
        if self.cost_multiplier < 0:
            raise ValueError("cost_multiplier cannot be negative")
        if not self.use_holdout and pd.Timestamp(self.start) >= pd.Timestamp(self.holdout_start):
            raise ValueError(
                f"start {self.start} is inside the holdout (from {self.holdout_start}); "
                "set use_holdout=True only for the final out-of-sample evaluation"
            )

    def effective_end(self) -> pd.Timestamp:
        end = pd.Timestamp(self.end)
        if not self.use_holdout:
            end = min(end, pd.Timestamp(self.holdout_start) - pd.Timedelta(days=1))
        return end


@dataclass
class BacktestResult:
    equity: pd.Series
    invested: pd.Series  # fraction of equity in positions
    trades: pd.DataFrame  # every fill
    round_trips: pd.DataFrame  # every closed position
    decisions: list[dict] = field(default_factory=list)  # sessions that produced orders
    halted_on: str | None = None  # date the drawdown hard stop fired, if it did
    config: BacktestConfig | None = None
    params: Params | None = None

    @property
    def returns(self) -> pd.Series:
        return self.equity.pct_change().dropna()


def run_backtest(
    market: Market,
    config: BacktestConfig,
    params: Params = DEFAULT_PARAMS,
    features: Features | None = None,
    keep_decisions: bool = True,
) -> BacktestResult:
    config.validate()
    # Nothing on or after the holdout reaches the strategy unless unlocked.
    market = market.truncate(config.effective_end())
    if features is None or len(features.dates) != len(market.dates):
        features = build_features(market, params)
    strat = Strategy(market, features, params)
    dates = market.dates
    start = int(dates.searchsorted(pd.Timestamp(config.start)))
    if start >= len(dates) - 1:
        raise ValueError("not enough sessions in the backtest window")

    opens = market.open.to_numpy(dtype=float)
    col = strat.col
    cash_ret = np.zeros(len(dates))
    if config.cash_ticker in col:
        c = strat.mark[:, col[config.cash_ticker]]
        with np.errstate(invalid="ignore", divide="ignore"):
            r = c[1:] / c[:-1] - 1
        cash_ret[1:] = np.where(np.isfinite(r), r, 0.0)

    book = Book(cash=config.initial_capital, peak_equity=config.initial_capital)
    pending: dict[int, list] = {}
    fills, trips, decisions = [], [], []
    meta: dict[str, dict] = {}  # per open position: entry facts and running cash flows
    equity, invested = [], []
    halted_on = None

    for t in range(start, len(dates)):
        day = dates[t]
        if t > start and book.cash > 0:
            book.cash *= 1 + cash_ret[t]
        for order in pending.pop(t, []):
            j = col[order.ticker]
            px = opens[t, j] if np.isfinite(opens[t, j]) else strat.mark[t - 1, j]
            if not np.isfinite(px):
                continue
            cost = cost_bps(order.ticker) * config.cost_multiplier / 1e4
            fill_px = px * (1 + cost) if order.side == "buy" else px * (1 - cost)
            qty = apply_fill(book, order, fill_px)
            m = meta.get(order.ticker)
            if qty > 0 and m is not None:
                fills.append({
                    "date": day, "ticker": order.ticker, "side": order.side, "quantity": qty,
                    "price": fill_px, "cost_bps": cost * 1e4, "reason": order.reason,
                })
                if order.side == "buy":
                    m["cost"] += qty * fill_px
                    m["bought"] += qty
                    m["first_fill"] = m["first_fill"] or day
                else:
                    m["proceeds"] += qty * fill_px
            if order.ticker not in book.positions and m is not None:
                if m["bought"] > 0:
                    trips.append(_round_trip(order.ticker, m, day, dates))
                del meta[order.ticker]

        prices = strat.prices(t)
        eq = book.equity(prices)
        book.peak_equity = max(book.peak_equity, eq)
        equity.append(eq)
        invested.append(1 - book.cash / eq if eq > 0 else 0.0)

        if (halted_on is None and config.drawdown_halt is not None
                and eq < (1 - config.drawdown_halt) * book.peak_equity):
            # Hard stop: flatten, then stay in cash for the rest of the run,
            # as the live runner does until a human reviews it.
            halted_on = str(day.date())
            orders = []
            for tk, pos in sorted(book.positions.items()):
                pos.exiting = "drawdown_hard_stop"
                if tk in meta:
                    meta[tk]["reason"] = "drawdown_hard_stop"
                if pos.shares > 0:
                    orders.append(Order(tk, "sell", pos.shares, "drawdown_hard_stop"))
            book.positions = {tk: p for tk, p in book.positions.items() if p.shares > 0}
            pending = {t + config.execution_delay: orders} if t + config.execution_delay < len(dates) else {}
            continue
        if halted_on is None and t + config.execution_delay < len(dates):
            decision = strat.step(t, book)
            if decision.orders:
                pending.setdefault(t + config.execution_delay, []).extend(decision.orders)
                if keep_decisions:
                    decisions.append(decision.to_dict())
            for tk, pos in book.positions.items():
                if tk not in meta:
                    meta[tk] = {"entry_date": pos.entry_date, "initial_stop": pos.initial_stop,
                                "rr": pos.rr_at_entry, "reason": None, "cost": 0.0,
                                "proceeds": 0.0, "bought": 0.0, "first_fill": None}
            for tk, reason in decision.exits.items():
                if tk in meta:
                    meta[tk]["reason"] = reason
                    if tk not in book.positions and meta[tk]["bought"] == 0:
                        del meta[tk]  # plan dropped before any fill

    idx = dates[start:]
    return BacktestResult(
        equity=pd.Series(equity, index=idx, name="equity") / config.initial_capital,
        invested=pd.Series(invested, index=idx, name="invested"),
        trades=pd.DataFrame(fills),
        round_trips=pd.DataFrame(trips),
        decisions=decisions,
        halted_on=halted_on,
        config=config,
        params=params,
    )


def _round_trip(ticker: str, m: dict, exit_day, dates) -> dict:
    avg_entry = m["cost"] / m["bought"]
    risk = (avg_entry - m["initial_stop"]) * m["bought"]
    pnl = m["proceeds"] - m["cost"]
    return {
        "ticker": ticker, "entry_date": pd.Timestamp(m["entry_date"]), "exit_date": exit_day,
        "reason": m["reason"], "cost": m["cost"], "proceeds": m["proceeds"], "pnl": pnl,
        "return": pnl / m["cost"], "r_multiple": pnl / risk if risk > 0 else np.nan,
        "rr_at_entry": m["rr"],
        "sessions": int(dates.get_loc(exit_day) - dates.get_loc(m["first_fill"])),
    }


def fixed_weight_benchmark(
    market: Market, weights: dict[str, float], config: BacktestConfig, rebalance_monthly: bool
) -> pd.Series:
    """Buy-and-hold or monthly-rebalanced fixed weights, same timing and costs."""
    config.validate()
    m = market.truncate(config.effective_end())
    close = m.close.ffill()
    opens = m.open
    dates = m.dates
    start = int(dates.searchsorted(pd.Timestamp(config.start)))
    period = dates.to_period("M")
    rebal = {start + config.execution_delay}
    if rebalance_monthly:
        for t in range(start + 1, len(dates) - 1):
            if period[t] != period[t + 1]:
                rebal.add(t + config.execution_delay)
    cash, shares = config.initial_capital, {tk: 0.0 for tk in weights}
    out = []
    for t in range(start, len(dates)):
        if t in rebal:
            eq = cash + sum(q * close[tk].iat[t - 1] for tk, q in shares.items())
            for tk, w in weights.items():
                px = opens[tk].iat[t]
                cost = cost_bps(tk) * config.cost_multiplier / 1e4
                target = w * eq / (px * (1 + cost))  # costs come out of the weight, not on margin
                delta = target - shares[tk]
                cash -= delta * px + abs(delta) * px * cost
                shares[tk] = target
        out.append(cash + sum(q * close[tk].iat[t] for tk, q in shares.items()))
    return pd.Series(out, index=dates[start:], name="equity") / config.initial_capital
