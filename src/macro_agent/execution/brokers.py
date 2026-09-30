"""Venue adapters. Anything that implements `Broker` can run the book."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from .orders import Order


class Broker(Protocol):
    def positions(self) -> dict[str, float]: ...

    def cash(self) -> float: ...

    def place(self, order: Order, price: float) -> None: ...


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

    def equity(self, prices: dict[str, float]) -> float:
        return self._cash + sum(q * prices[t] for t, q in self._positions.items())

    def place(self, order: Order, price: float) -> None:
        sign = 1 if order.side == "buy" else -1
        qty = self._positions.get(order.ticker, 0.0) + sign * order.quantity
        self._cash -= sign * order.quantity * price
        if abs(qty) < 1e-9:
            self._positions.pop(order.ticker, None)
        else:
            self._positions[order.ticker] = round(qty, 10)
        self._save()

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        state = {"cash": self._cash, "positions": dict(sorted(self._positions.items()))}
        self.path.write_text(json.dumps(state, indent=2), encoding="utf-8")


class RobinhoodMCPBroker:
    """Adapter for the Robinhood agentic trading MCP (spec 6).

    Not implemented: the MCP's tool names and schemas need to be confirmed
    against the live server before any order code is written. Use a dedicated
    agentic account and keep the main account out of reach.
    """

    endpoint = "https://agent.robinhood.com/mcp/trading"

    def positions(self) -> dict[str, float]:
        raise NotImplementedError("Robinhood MCP adapter not yet implemented")

    def cash(self) -> float:
        raise NotImplementedError("Robinhood MCP adapter not yet implemented")

    def place(self, order: Order, price: float) -> None:
        raise NotImplementedError("Robinhood MCP adapter not yet implemented")
