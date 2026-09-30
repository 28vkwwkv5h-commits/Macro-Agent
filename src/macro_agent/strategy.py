"""The decision function. One call per session, at the close.

    step(t, book) -> Decision(orders for the next session, audit record)

The backtester and the live runner both call this, so the code that is tested is
the code that trades (spec 19.4). It reads precomputed causal features and the
book; it has no model call and no randomness (spec 0).

Order of operations each session (spec 12: each stage can only narrow):

1. Regime and confidence. Below 0.40, everything exits and nothing enters.
2. Exits for held positions, fastest first: fib stop, structural invalidation,
   flag time limit (weekly close only), then regime exit.
3. Remaining tranches for positions still accumulating.
4. Entries into free slots, where slots come from confidence:
   whitelist -> data -> confirmation -> reward-to-risk >= minimum ->
   anchors agree -> BAN score rank -> correlation prune -> vol-targeted size.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from .config import (
    TICKER_SLEEVE,
    WHITELIST,
    Params,
)
from .data.market import Market
from .execution.orders import Order
from .features import Features
from .fib import ladder, resistance_above, support_below
from .positions import (
    COOLDOWN_REASONS,
    EXIT_CONFIDENCE,
    EXIT_REGIME,
    Book,
    Position,
    weekly_update,
)
from .regime import position_budget

SHARE_INCREMENT = 1e-6  # fractional shares (spec 12.5: rounding happens once, here)


@dataclass
class Candidate:
    ticker: str
    rr: float
    upside: float
    downside: float
    entry_level: float
    stop: float
    mdd: float
    flow: float
    ban: float = 0.0
    mdd_rank: float = 0.0


@dataclass
class Decision:
    date: str
    decision_id: str
    regime: str | None
    confidence: float
    budget: int
    orders: list[Order]
    candidates: list[dict] = field(default_factory=list)
    rejections: dict[str, int] = field(default_factory=dict)
    exits: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["orders"] = [o.to_dict() for o in self.orders]
        return d


def _sessions_later(day, n: int) -> str:
    return str((pd.Timestamp(day) + pd.offsets.BDay(n)).date())


def _floor(q: float) -> float:
    return math.floor(q / SHARE_INCREMENT + 1e-9) * SHARE_INCREMENT


class Strategy:
    def __init__(self, market: Market, features: Features, params: Params):
        self.p = params
        self.f = features
        self.dates = market.dates
        self.tickers = list(market.close.columns)
        self.col = {t: i for i, t in enumerate(self.tickers)}
        self.close = market.close.to_numpy(dtype=float)
        # Valuation uses the last known close, so a halted or delisted holding
        # keeps its last price rather than vanishing from equity.
        self.mark = market.close.ffill().to_numpy(dtype=float)
        a = lambda frame: frame.reindex(columns=self.tickers).to_numpy(dtype=float)  # noqa: E731
        self.atr = a(features.atr)
        self.vol = a(features.vol)
        self.high_52w = a(features.high_52w)
        self.mdd = a(features.mdd)
        self.rets = a(features.returns)
        self.flow = a(features.flow)
        self.swing_high = a(features.swing_high)
        self.swing_low = a(features.swing_low)
        self.confirmed = features.confirmed.reindex(columns=self.tickers).to_numpy(dtype=bool)
        self.confirm_low = a(features.confirm_low)
        self.week_end = features.is_week_end.to_numpy(dtype=bool)
        self.regime = features.regime["regime"].to_numpy(dtype=object)
        self.conf = features.regime["confidence"].to_numpy(dtype=float)
        # Consecutive sessions with confidence below the emergency line.
        below = ~(np.nan_to_num(self.conf) >= params.emergency_confidence)
        self.below_run = np.zeros(len(below), dtype=int)
        for i, b in enumerate(below):
            self.below_run[i] = (self.below_run[i - 1] + 1 if i else 1) if b else 0
        # Weekly bars, looked up by the session they complete on.
        self.w_pos = {d: i for i, d in enumerate(features.w_close.index)}
        self.w_high = features.w_high.reindex(columns=self.tickers).to_numpy(dtype=float)
        self.w_low = features.w_low.reindex(columns=self.tickers).to_numpy(dtype=float)
        self.w_close = features.w_close.reindex(columns=self.tickers).to_numpy(dtype=float)

    # --- helpers -----------------------------------------------------------

    def prices(self, t: int) -> dict[str, float]:
        row = self.mark[t]
        return {tk: float(row[i]) for tk, i in self.col.items() if np.isfinite(row[i])}

    def _decision_id(self, t: int, book: Book) -> str:
        blob = json.dumps({"date": str(self.dates[t].date()), "book": book.to_dict()}, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def _prior_swing_high(self, t_week: int, j: int) -> float:
        lo = max(0, t_week - self.p.swing_high_weeks)
        window = self.w_high[lo:t_week, j]
        window = window[np.isfinite(window)]
        return float(window.max()) if window.size else np.inf

    def evaluate(self, t: int, ticker: str) -> Candidate | str:
        """Score one candidate at session t, or return why it was rejected."""
        p, j = self.p, self.col[ticker]
        c = self.close[t, j]
        a = self.atr[t, j]
        hi, lo = self.swing_high[t, j], self.swing_low[t, j]
        h52 = self.high_52w[t, j]
        if not all(np.isfinite(x) for x in (c, a, hi, lo, h52, self.vol[t, j], self.mdd[t, j])):
            return "no_data"
        if p.require_confirmation and not self.confirmed[t, j]:
            return "unconfirmed"
        if c <= lo:
            return "below_swing_low"
        levels = ladder(lo, hi)
        support = support_below(levels, c)
        conf_low = self.confirm_low[t, j]
        # Downside hangs from the closest level below price: the nearest fib
        # support or the structural low of the base, whichever is higher.
        candidates = [x for x in (support, conf_low) if x is not None and np.isfinite(x) and x < c]
        if not candidates:
            return "no_support"
        entry_level = max(candidates)
        stop = entry_level - p.stop_atr * a
        downside = (c - stop) / c  # the TRUE risk: to the stop, not to the level (15.3)

        if c >= (1 - p.breakout_band) * h52:
            # At or near the 52-week high: the anchor is the next fib extension.
            target = resistance_above(levels[levels > hi], c)
            if target is None:
                return "no_target"
            upside = target / c - 1
        else:
            upside = h52 / c - 1  # primary anchor, George and Hwang 2004
            fib_target = hi if c < hi else resistance_above(levels, c)
            if fib_target is None:
                return "no_target"
            fib_up = fib_target / c - 1
            ratio = max(upside, fib_up) / max(min(upside, fib_up), 1e-9)
            if ratio > p.anchor_disagreement:
                return "anchors_disagree"
        rr = upside / downside
        if rr < p.min_reward_risk:
            return "rr_below_min"
        flow = self.flow[t, j]
        flow = 0.5 if not np.isfinite(flow) else flow
        return Candidate(ticker, rr, upside, downside, entry_level, stop, self.mdd[t, j], flow)

    def _correlated(self, t: int, ticker: str, others: list[str]) -> bool:
        if not self.p.use_correlation_cap or not others:
            return False
        w = self.p.corr_window
        if t < w:
            return False
        x = self.rets[t - w + 1: t + 1, self.col[ticker]]
        for o in others:
            y = self.rets[t - w + 1: t + 1, self.col[o]]
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() < w // 2:
                continue
            if np.corrcoef(x[ok], y[ok])[0, 1] > self.p.max_correlation:
                return True
        return False

    def _size(self, t: int, cand: Candidate, book: Book, prices: dict, budget: int) -> float:
        """Target shares for a new position (spec 5)."""
        p, j = self.p, self.col[cand.ticker]
        equity = book.equity(prices)
        if equity <= 0:
            return 0.0
        vol = max(self.vol[t, j], 1e-4)
        weight = min(p.max_position, p.vol_target / (vol * math.sqrt(max(budget, 1))))

        def held_weight(pred) -> float:
            return sum(
                max(pos.target_shares, pos.shares) * prices[tk]
                for tk, pos in book.positions.items()
                if not pos.exiting and pred(tk)
            ) / equity

        sleeve = TICKER_SLEEVE.get(cand.ticker, "individual_names")
        weight = min(weight, p.max_sleeve - held_weight(lambda tk: TICKER_SLEEVE.get(tk) == sleeve))
        if sleeve == "individual_names":
            weight = min(weight, p.max_names - held_weight(lambda tk: TICKER_SLEEVE.get(tk) == sleeve))
        available = book.cash - book.reserved_cash(prices)
        value = min(weight * equity, available)
        if value < p.min_position * equity:
            return 0.0
        return _floor(value / prices[cand.ticker])

    # --- the step -----------------------------------------------------------

    def step(self, t: int, book: Book) -> Decision:
        p = self.p
        date = str(self.dates[t].date())
        prices = self.prices(t)
        regime = self.regime[t] if isinstance(self.regime[t], str) else None
        conf = float(self.conf[t]) if np.isfinite(self.conf[t]) else 0.0
        budget = position_budget(conf, p)
        decision_id = self._decision_id(t, book)
        orders: list[Order] = []
        exits: dict[str, str] = {}

        def exit_(tk: str, reason: str):
            pos = book.positions[tk]
            exits[tk] = reason
            if reason in COOLDOWN_REASONS:
                # Dated from the decision, so backtest and live agree (spec 15.5).
                book.cooldown_until[tk] = _sessions_later(self.dates[t], p.reentry_cooldown)
            if pos.shares > 0:
                pos.exiting = reason
                orders.append(Order(tk, "sell", pos.shares, reason, decision_id))
            else:
                del book.positions[tk]  # nothing filled yet: just drop the plan

        live = sorted(tk for tk, pos in book.positions.items() if not pos.exiting)

        # 1. Confidence below the emergency line: cash, no exceptions.
        if self.below_run[t] >= p.confidence_exit_days:
            for tk in live:
                exit_(tk, EXIT_CONFIDENCE)
            return Decision(date, decision_id, regime, conf, 0, orders, exits=exits)

        # 2. Exits, and 3. tranches, for each live position.
        if p.use_regime_whitelist:
            allowed = WHITELIST.get(regime, frozenset())
        else:
            allowed = frozenset(x for x in self.tickers if TICKER_SLEEVE.get(x) != "cash")
        tw = self.w_pos.get(self.dates[t]) if self.week_end[t] else None
        for tk in live:
            pos, j = book.positions[tk], self.col[tk]
            if tw is not None and np.isfinite(self.w_close[tw, j]):
                reason = weekly_update(
                    pos, self.w_high[tw, j], self.w_low[tw, j], self.w_close[tw, j],
                    self._prior_swing_high(tw, j), self.atr[t, j], p,
                )
                if reason:
                    exit_(tk, reason)
                    continue
            pos.off_whitelist_days = 0 if tk in allowed else pos.off_whitelist_days + 1
            if pos.off_whitelist_days >= p.regime_exit_days:
                exit_(tk, EXIT_REGIME)
                continue
            # Tranches pause while the regime disallows the asset: the whitelist
            # outranks the schedule (spec 12.1).
            if (tk in allowed and pos.tranches_done < pos.tranches_total
                    and pos.next_tranche_date and date >= pos.next_tranche_date):
                qty = _floor(pos.target_shares / pos.tranches_total)
                if pos.tranches_done == pos.tranches_total - 1:
                    qty = _floor(pos.target_shares - pos.shares)
                # A position that has already grown never tops up past the cap.
                headroom = p.max_position * book.equity(prices) / prices[tk] - pos.shares
                qty = min(qty, _floor(max(headroom, 0.0)))
                if qty > 0:
                    orders.append(Order(tk, "buy", qty, f"tranche_{pos.tranches_done + 1}", decision_id))
                pos.tranches_done += 1
                pos.next_tranche_date = _sessions_later(self.dates[t], p.tranche_spacing)

        # 4. Entries into free slots.
        held = sorted(tk for tk, pos in book.positions.items() if not pos.exiting)
        slots = budget - len(held)
        rejections: dict[str, int] = {}
        chosen: list[Candidate] = []
        if slots > 0 and regime is not None:
            pool = []
            for tk in sorted(allowed):
                if tk not in self.col or tk in book.positions or book.cooldown_until.get(tk, "") > date:
                    continue
                res = self.evaluate(t, tk)
                if isinstance(res, str):
                    rejections[res] = rejections.get(res, 0) + 1
                else:
                    pool.append(res)
            # Choi 2021: rank by 252-day maximum drawdown, deepest highest.
            if pool:
                ordered = sorted(pool, key=lambda c: (c.mdd, c.ticker))
                for rank, c in enumerate(ordered, start=1):
                    c.mdd_rank = rank / len(ordered)
                for c in pool:
                    flow_mult = 0.5 + 0.5 * c.flow if p.use_flows else 1.0
                    c.ban = c.rr * c.mdd_rank * flow_mult
            for c in sorted(pool, key=lambda c: (-c.ban, c.ticker)):
                if len(chosen) >= slots:
                    break
                if self._correlated(t, c.ticker, held + [x.ticker for x in chosen]):
                    rejections["correlated"] = rejections.get("correlated", 0) + 1
                    continue
                target = self._size(t, c, book, prices, budget)
                if target <= 0:
                    rejections["no_capacity"] = rejections.get("no_capacity", 0) + 1
                    continue
                tranches = p.tranches if p.use_tranching else 1
                first = _floor(target / tranches) if tranches > 1 else target
                if first <= 0:
                    rejections["no_capacity"] = rejections.get("no_capacity", 0) + 1
                    continue
                book.positions[c.ticker] = Position(
                    ticker=c.ticker, entry_date=date, entry_level=c.entry_level,
                    stop=c.stop, initial_stop=c.stop, atr_frozen=float(self.atr[t, self.col[c.ticker]]),
                    confirm_low=float(self.confirm_low[t, self.col[c.ticker]]),
                    swing_low=float(self.swing_low[t, self.col[c.ticker]]),
                    swing_high=float(self.swing_high[t, self.col[c.ticker]]),
                    target_shares=target, rr_at_entry=c.rr, ban_at_entry=c.ban,
                    tranches_total=tranches, tranches_done=1,
                    next_tranche_date=_sessions_later(self.dates[t], p.tranche_spacing),
                )
                orders.append(Order(c.ticker, "buy", first, "entry", decision_id))
                chosen.append(c)

        return Decision(
            date, decision_id, regime, conf, budget, orders,
            candidates=[asdict(c) for c in chosen], rejections=rejections, exits=exits,
        )


def apply_fill(book: Book, order: Order, price: float) -> float:
    """Apply a fill to the book. `price` already includes costs. Returns the
    quantity actually filled (buys are capped by available cash)."""
    pos = book.positions.get(order.ticker)
    if order.side == "buy":
        qty = min(order.quantity, _floor(max(book.cash, 0.0) / price)) if price > 0 else 0.0
        if pos is None or qty <= 0:
            if pos is not None and pos.shares <= 0:
                del book.positions[order.ticker]
            return 0.0
        pos.shares += qty
        pos.cost_basis += qty * price
        book.cash -= qty * price
        return qty
    if pos is None:
        return 0.0
    qty = min(order.quantity, pos.shares)
    book.cash += qty * price
    pos.shares -= qty
    if pos.shares <= SHARE_INCREMENT:
        del book.positions[order.ticker]
    return qty
