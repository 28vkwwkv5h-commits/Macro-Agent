"""Orders, and target-versus-actual reconciliation."""
from __future__ import annotations

from dataclasses import asdict, dataclass

EPS = 1e-9


@dataclass(frozen=True)
class Order:
    ticker: str
    side: str  # "buy" or "sell"
    quantity: float
    reason: str = ""
    decision_id: str = ""
    order_type: str = "market"

    def to_dict(self) -> dict:
        return asdict(self)


def reconcile(target: dict[str, float], actual: dict[str, float]) -> dict[str, float]:
    """Intended minus actual holdings, for every ticker that differs."""
    diff = {}
    for t in sorted(set(target) | set(actual)):
        d = target.get(t, 0.0) - actual.get(t, 0.0)
        if abs(d) > 1e-6:
            diff[t] = round(d, 10)
    return diff
