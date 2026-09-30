import copy

import numpy as np
import pytest

from macro_agent.config import DEFAULT_PARAMS, WHITELIST
from macro_agent.execution import Order
from macro_agent.positions import (
    ACCUMULATING,
    CONSOLIDATING,
    EXIT_CONFIDENCE,
    EXIT_FLAG_TIME,
    EXIT_INVALIDATION,
    EXIT_STOP,
    IMPULSE,
    Book,
    Position,
    weekly_update,
)
from macro_agent.strategy import Candidate, Strategy, apply_fill

P = DEFAULT_PARAMS


def position(**kw) -> Position:
    base = dict(
        ticker="SILJ", entry_date="2024-01-05", entry_level=5.20, stop=5.00, initial_stop=5.00,
        atr_frozen=0.20, confirm_low=4.90, swing_low=4.00, swing_high=8.00, target_shares=100,
        rr_at_entry=4.0, ban_at_entry=2.0,
    )
    base.update(kw)
    return Position(**base)


# --- state machine (spec 15, 16) --------------------------------------------


def test_stop_is_judged_on_weekly_close():
    pos = position()
    assert weekly_update(pos, 5.4, 4.95, 5.10, np.inf, 0.2, P) is None  # wick below, close above
    assert weekly_update(pos, 5.2, 4.80, 4.95, np.inf, 0.2, P) == EXIT_STOP


def test_structural_invalidation_below_confirmation_low():
    pos = position(stop=4.50, confirm_low=4.90)
    assert weekly_update(pos, 5.0, 4.6, 4.85, np.inf, 0.2, P) == EXIT_INVALIDATION


def test_stops_can_be_disabled_for_ablation():
    pos = position(confirm_low=0.0)
    assert weekly_update(pos, 5.0, 4.5, 4.6, np.inf, 0.2, P.with_(use_stops=False)) is None


def test_spec_16_2_worked_example():
    # Entry at a fib support of $5.20, impulse high $6.50, leg $1.30. Half the
    # impulse ATR is $0.14, so the stop sits just under the 0.618 level, ~$5.56.
    pos = position(stop=5.00)
    weekly_update(pos, 5.60, 5.20, 5.55, 5.50, 0.20, P)  # breaks the prior swing high
    assert pos.state == IMPULSE and pos.impulse_start == 5.20
    weekly_update(pos, 6.50, 5.60, 6.40, 5.60, 0.20, P)  # impulse continues, wide range
    assert pos.stop == 5.00  # unchanged during the impulse
    weekly_update(pos, 6.45, 6.10, 6.20, 6.50, 0.28, P)  # range contracts: a flag
    assert pos.state == CONSOLIDATING
    assert pos.stop == pytest.approx(6.50 - 0.618 * 1.30 - 0.5 * 0.28, abs=1e-9)
    assert pos.stop == pytest.approx(5.56, abs=0.01)


def test_resolution_ratchets_up_and_stop_never_falls():
    pos = position(stop=5.00)
    weekly_update(pos, 5.60, 5.20, 5.55, 5.50, 0.20, P)
    weekly_update(pos, 6.50, 5.60, 6.40, 5.60, 0.20, P)
    weekly_update(pos, 6.45, 6.10, 6.20, 6.50, 0.28, P)
    flag_stop = pos.stop
    weekly_update(pos, 6.80, 6.30, 6.70, 6.50, 0.28, P)  # close above the flag high
    assert pos.state == IMPULSE
    # Ladder 4 -> 8: the highest level below 6.70 is 0.618 (6.472); minus 1 impulse ATR.
    assert pos.stop == pytest.approx(max(flag_stop, 4.0 + 0.618 * 4.0 - 0.28))
    assert pos.stop >= flag_stop


def test_flag_time_limit():
    pos = position(stop=5.00, confirm_low=0.0)
    weekly_update(pos, 5.60, 5.20, 5.55, 5.50, 0.20, P)
    weekly_update(pos, 6.50, 5.60, 6.40, 5.60, 0.20, P)
    weekly_update(pos, 6.45, 6.10, 6.20, 6.50, 0.28, P)
    reasons = [weekly_update(pos, 6.40, 6.10, 6.25, 6.50, 0.28, P) for _ in range(12)]
    assert reasons[:-1] == [None] * 11
    assert reasons[-1] == EXIT_FLAG_TIME


# --- fills and the book -------------------------------------------------------


def test_apply_fill_caps_buys_at_cash_and_removes_closed_positions():
    book = Book(cash=1000.0)
    book.positions["A"] = position(ticker="A")
    assert apply_fill(book, Order("A", "buy", 50), 25.0) == pytest.approx(40)
    assert book.cash == pytest.approx(0.0)
    assert apply_fill(book, Order("A", "sell", 40), 30.0) == 40
    assert "A" not in book.positions and book.cash == pytest.approx(1200.0)


def test_book_roundtrip():
    book = Book(cash=10.0, positions={"A": position(ticker="A", shares=3)}, cooldown_until={"B": "2024-02-01"})
    again = Book.from_dict(book.to_dict())
    assert again.to_dict() == book.to_dict()


# --- the strategy on a full synthetic market ------------------------------------


@pytest.fixture(scope="module")
def strat(market, features):
    return Strategy(market, features, P)


def _entry_day(strat) -> int:
    for t in range(600, len(strat.dates)):
        d = strat.step(t, Book(cash=1e6, peak_equity=1e6))
        if any(o.reason == "entry" for o in d.orders):
            return t
    pytest.fail("no entries found on the fixture")


def test_evaluate_scores_true_risk(strat):
    t = _entry_day(strat)
    d = strat.step(t, Book(cash=1e6, peak_equity=1e6))
    c = Candidate(**d.candidates[0])
    j = strat.col[c.ticker]
    close = strat.close[t, j]
    assert c.stop < c.entry_level < close
    assert c.downside == pytest.approx((close - c.stop) / close)  # to the stop, not the level
    assert c.rr >= P.min_reward_risk
    assert c.ticker in WHITELIST[d.regime]


def test_step_is_deterministic_and_respects_limits(strat):
    t = _entry_day(strat)
    b1, b2 = Book(cash=1e6, peak_equity=1e6), Book(cash=1e6, peak_equity=1e6)
    d1, d2 = strat.step(t, b1), strat.step(t, b2)
    assert d1.to_dict() == d2.to_dict()
    assert b1.to_dict() == b2.to_dict()
    assert len(b1.positions) <= d1.budget
    prices = strat.prices(t)
    for tk, pos in b1.positions.items():
        assert pos.target_shares * prices[tk] <= P.max_position * 1e6 + 1e-6
        assert tk in WHITELIST[d1.regime]
    entries = [o for o in d1.orders if o.reason == "entry"]
    for o in entries:
        assert o.quantity == pytest.approx(b1.positions[o.ticker].target_shares / P.tranches, rel=1e-4)


def test_confidence_exit_flattens(strat, market):
    t = int(np.nanargmin(np.where(np.arange(len(strat.conf)) > 600, strat.conf, np.inf)))
    assert strat.conf[t] < P.emergency_confidence
    book = Book(cash=0.0, peak_equity=1.0)
    book.positions["SPY"] = position(ticker="SPY", shares=10.0)
    d = strat.step(t, copy.deepcopy(book))
    assert [(o.ticker, o.side, o.reason) for o in d.orders] == [("SPY", "sell", EXIT_CONFIDENCE)]


def test_correlation_cap(strat, market, features):
    assert strat._correlated(800, "SPY", ["SPY"])
    assert not strat._correlated(800, "SPY", ["SH"])  # inverse: negatively correlated
    off = Strategy(market, features, P.with_(use_correlation_cap=False))
    assert not off._correlated(800, "SPY", ["SPY"])


def test_stop_sets_cooldown_from_the_decision_date(strat):
    t = 900
    book = Book(cash=0.0, peak_equity=1.0)
    # A stop far above any price: the next weekly close must exit.
    while not strat.week_end[t] or strat.below_run[t] >= P.confidence_exit_days:
        t += 1
    tk = sorted(WHITELIST[strat.regime[t]])[0]
    book.positions[tk] = position(ticker=tk, shares=5.0, stop=1e9, confirm_low=0.0)
    d = strat.step(t, book)
    assert d.exits[tk] == EXIT_STOP
    assert book.cooldown_until[tk] > str(strat.dates[t].date())


def test_state_starts_accumulating(strat):
    t = _entry_day(strat)
    book = Book(cash=1e6, peak_equity=1e6)
    strat.step(t, book)
    assert all(p.state == ACCUMULATING and p.tranches_done == 1 for p in book.positions.values())
