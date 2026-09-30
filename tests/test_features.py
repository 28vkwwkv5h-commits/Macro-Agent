"""Causality: features built on truncated data must equal the full build up to
the cut. This is the test that makes every backtest number believable."""
import numpy as np
import pandas as pd
import pytest

from macro_agent.features import build_features

FRAMES = ["atr", "vol", "high_52w", "mdd", "returns", "flow", "swing_high", "swing_low",
          "confirmed", "confirm_low"]


@pytest.mark.parametrize("cut", ["2010-06-11", "2010-06-16"])  # a Friday and a Wednesday
def test_features_are_causal(market, features, cut):
    part = build_features(market.truncate(cut))
    for name in FRAMES:
        full = getattr(features, name).loc[:cut]
        got = getattr(part, name)
        pd.testing.assert_frame_equal(full, got, check_freq=False, obj=name)
    pd.testing.assert_frame_equal(features.regime.loc[:cut], part.regime, check_freq=False)
    pd.testing.assert_series_equal(features.is_week_end.loc[:cut].iloc[:-1], part.is_week_end.iloc[:-1],
                                   check_freq=False)
    for k, frame in features.macro_fib.items():
        pd.testing.assert_frame_equal(frame.loc[:cut], part.macro_fib[k], check_freq=False)


def test_features_shapes(market, features):
    assert features.atr.shape == market.close.shape
    assert features.regime["confidence"].between(0, 1).all()
    flow = features.flow.stack().dropna()
    assert flow.between(0, 1).all()
    assert features.confirmed.dtypes.eq(bool).all()
    assert np.isfinite(features.swing_high.iloc[-1]).any()
