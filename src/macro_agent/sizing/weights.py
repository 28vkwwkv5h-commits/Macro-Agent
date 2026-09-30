"""Sizing (spec 5, MVP equal weight per 20.5).

Each selected position gets 1 / FULL_POSITIONS of the book, so a full slate is
fully invested, three positions are 60% invested, and every unfilled slot is
cash. Cash is a residual and is never sized.

All rounding happens here and nowhere else (spec 12.5).
"""
from __future__ import annotations

import math

from ..config import FULL_POSITIONS, MAX_SINGLE_POSITION


def equal_weights(selected: list[str], slots: int = FULL_POSITIONS) -> dict[str, float]:
    if len(selected) > slots:
        raise ValueError(f"{len(selected)} selections exceed {slots} slots")
    w = min(1.0 / slots, MAX_SINGLE_POSITION)
    return {t: w for t in sorted(selected)}


def target_shares(
    weights: dict[str, float],
    equity: float,
    prices: dict[str, float],
    fractional_increment: float | None = None,
) -> dict[str, float]:
    """Round target weights down to whole shares, or to the venue's fractional
    increment. Rounding down means the book never exceeds its target weight;
    the remainder stays in cash.
    """
    out = {}
    for t in sorted(weights):
        raw = weights[t] * equity / prices[t]
        step = fractional_increment or 1.0
        qty = math.floor(raw / step + 1e-9) * step
        if fractional_increment:
            qty = round(qty, 10)
        if qty > 0:
            out[t] = qty
    return out
