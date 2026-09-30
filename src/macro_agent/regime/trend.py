"""Per-series trend: sign of a 63-day change, confirmed against the 200-day MA."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import TREND_MA_WINDOW, TREND_ROC_WINDOW

UP = 1
FLAT = 0
DOWN = -1


def trend_state(
    series: pd.Series,
    kind: str = "price",
    roc_window: int = TREND_ROC_WINDOW,
    ma_window: int = TREND_MA_WINDOW,
) -> pd.Series:
    """UP if the change is positive and the level is above its moving average,
    DOWN if negative and below, FLAT when the two disagree. NaN until there is
    enough history.
    """
    if kind == "price":
        change = series / series.shift(roc_window) - 1
    elif kind == "rate":
        change = series - series.shift(roc_window)
    else:
        raise ValueError(f"unknown series kind {kind!r}")
    ma = series.rolling(ma_window, min_periods=ma_window).mean()

    out = pd.Series(float(FLAT), index=series.index)
    out[(change > 0) & (series > ma)] = UP
    out[(change < 0) & (series < ma)] = DOWN
    out[change.isna() | ma.isna() | series.isna()] = np.nan
    return out
