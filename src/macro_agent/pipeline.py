"""The decision pipeline: one function, same inputs, same output.

No stage reads the current portfolio (spec 12.5), so a decision depends only on
market data up to `as_of`. Every stage's input and output set is recorded so a
run can be replayed from the audit log.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import pandas as pd

from .ban import select
from .regime import classify, confidence, position_budget
from .sizing import equal_weights


@dataclass
class Decision:
    as_of: str
    regime: str | None
    agreement: float
    confidence: float
    n_positions: int
    selected: list[str]
    weights: dict[str, float]
    cash: float
    scores: dict[str, float] = field(default_factory=dict)
    stages: dict[str, list[str]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def decide(prices: pd.DataFrame, macro: pd.DataFrame, as_of) -> Decision:
    """Run every stage on data up to and including `as_of`.

    Both frames are truncated here, so nothing after `as_of` can influence the
    result regardless of what the caller passes in.
    """
    as_of = pd.Timestamp(as_of)
    prices = prices.loc[:as_of]
    macro = macro.loc[:as_of]
    if macro.empty or prices.empty:
        raise ValueError(f"no data on or before {as_of.date()}")

    classified = classify(macro)
    conf_series = confidence(classified)
    last = classified.iloc[-1]
    regime = last["regime"] if isinstance(last["regime"], str) else None
    conf = round(float(conf_series.iloc[-1]), 10)
    budget = position_budget(conf)

    sel = select(prices, regime, budget)
    weights = equal_weights(sel.selected)
    cash = round(1.0 - sum(weights.values()), 10)
    return Decision(
        as_of=str(as_of.date()),
        regime=regime,
        agreement=float(last["agreement"]),
        confidence=conf,
        n_positions=budget,
        selected=sel.selected,
        weights=weights,
        cash=cash,
        scores=sel.scores,
        stages=sel.stages,
    )
