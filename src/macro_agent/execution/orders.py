"""Target minus actual equals the order list."""
from __future__ import annotations

from dataclasses import asdict, dataclass

EPS = 1e-9


@dataclass(frozen=True)
class Order:
    ticker: str
    side: str  # "buy" or "sell"
    quantity: float
    order_type: str = "market"  # market-on-open on rebalance day (spec 6)
    limit_price: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def diff_orders(
    target: dict[str, float], current: dict[str, float], order_type: str = "market"
) -> list[Order]:
    """Sells first, so they fund the buys, then buys. Alphabetical within each
    side, so the same inputs always produce the same order list."""
    sells, buys = [], []
    for t in sorted(set(target) | set(current)):
        delta = target.get(t, 0.0) - current.get(t, 0.0)
        if delta < -EPS:
            sells.append(Order(t, "sell", round(-delta, 10), order_type))
        elif delta > EPS:
            buys.append(Order(t, "buy", round(delta, 10), order_type))
    return sells + buys


def reconcile(target: dict[str, float], actual: dict[str, float]) -> dict[str, float]:
    """Intended minus actual holdings, for every ticker that differs."""
    diff = {}
    for t in sorted(set(target) | set(actual)):
        d = target.get(t, 0.0) - actual.get(t, 0.0)
        if abs(d) > EPS:
            diff[t] = round(d, 10)
    return diff
