import pandas as pd
import pytest

from macro_agent.data import build_macro_frame, synthetic_market


@pytest.fixture(scope="session")
def market():
    prices, fred = synthetic_market(start="2005-01-03", end="2012-12-31", seed=11)
    return prices, build_macro_frame(prices, fred)


def flat_prices(tickers, n=400, start="2020-01-01", value=100.0) -> pd.DataFrame:
    idx = pd.bdate_range(start, periods=n)
    return pd.DataFrame({t: [value] * n for t in tickers}, index=idx, dtype=float)
