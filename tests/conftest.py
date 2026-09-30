import numpy as np
import pandas as pd
import pytest

from macro_agent.data.market import Market
from macro_agent.features import build_features
from macro_agent.montecarlo import synthetic_market


@pytest.fixture(scope="session")
def market() -> Market:
    m, _ = synthetic_market(start="2005-01-03", end="2012-12-31", seed=11)
    return m


@pytest.fixture(scope="session")
def features(market):
    return build_features(market)


def make_market(close: pd.DataFrame, macro: pd.DataFrame | None = None, spread: float = 0.01) -> Market:
    """A Market from closes: open = previous close, high/low a fixed band around."""
    open_ = close.shift(1).fillna(close)
    high = np.maximum(open_, close) * (1 + spread)
    low = np.minimum(open_, close) * (1 - spread)
    volume = close * 0 + 1e6
    macro = macro if macro is not None else pd.DataFrame(index=close.index)
    return Market(open_, high, low, close, volume, macro)
