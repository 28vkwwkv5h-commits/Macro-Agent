"""Venue adapters. Anything that implements `Broker` can run the book (spec 6)."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Protocol

from .orders import Order


class Broker(Protocol):
    def positions(self) -> dict[str, float]: ...

    def cash(self) -> float: ...

    def open_orders(self) -> list[dict]: ...

    def place(self, order: Order, price: float) -> tuple[float, float]:
        """Submit an order at about `price`; return (quantity filled, fill price)."""
        ...


class PaperBroker:
    """Fills every order in full at the given price. State lives in a JSON file
    so a paper account persists across runs."""

    def __init__(self, path: str | Path, starting_cash: float = 100_000.0):
        self.path = Path(path)
        if self.path.exists():
            state = json.loads(self.path.read_text(encoding="utf-8"))
        else:
            state = {"cash": starting_cash, "positions": {}}
        self._cash = float(state["cash"])
        self._positions = {k: float(v) for k, v in state["positions"].items()}

    def positions(self) -> dict[str, float]:
        return dict(self._positions)

    def cash(self) -> float:
        return self._cash

    def open_orders(self) -> list[dict]:
        return []  # paper fills are immediate

    def place(self, order: Order, price: float) -> tuple[float, float]:
        sign = 1 if order.side == "buy" else -1
        qty = order.quantity
        if order.side == "sell":
            qty = min(qty, self._positions.get(order.ticker, 0.0))
        else:
            qty = min(qty, math.floor(max(self._cash, 0.0) / price * 1e6) / 1e6) if price > 0 else 0.0
        new = self._positions.get(order.ticker, 0.0) + sign * qty
        self._cash -= sign * qty * price
        if abs(new) < 1e-9:
            self._positions.pop(order.ticker, None)
        else:
            self._positions[order.ticker] = round(new, 10)
        self._save()
        return qty, price

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        state = {"cash": self._cash, "positions": dict(sorted(self._positions.items()))}
        self.path.write_text(json.dumps(state, indent=2), encoding="utf-8")


class RobinhoodMCPBroker:
    """Adapter for the Robinhood agentic trading MCP (spec 6).

    Not implemented: the MCP's tool names and schemas need to be confirmed
    against the live server before any order code is written. Use a dedicated
    agentic account with no margin and no options permissions, and keep the
    main account out of reach (spec 6, 11.2).
    """

    endpoint = "https://agent.robinhood.com/mcp/trading"

    def positions(self) -> dict[str, float]:
        raise NotImplementedError("Robinhood MCP adapter not yet implemented")

    def cash(self) -> float:
        raise NotImplementedError("Robinhood MCP adapter not yet implemented")

    def open_orders(self) -> list[dict]:
        raise NotImplementedError("Robinhood MCP adapter not yet implemented")

    def place(self, order: Order, price: float) -> tuple[float, float]:
        raise NotImplementedError("Robinhood MCP adapter not yet implemented")
