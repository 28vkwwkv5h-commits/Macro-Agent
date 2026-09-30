"""Fixed parameters for the MVP (spec section 20.5).

Every number here is a knob, and every knob turned while looking at results is a
step toward a curve fit (spec section 10). Change them only at a scheduled review,
and record the change in the trial ledger.
"""
from __future__ import annotations

# --- Regimes --------------------------------------------------------------

REFLATION = "reflation"
GOLDILOCKS = "goldilocks"
TIGHTENING = "tightening"
DEFLATION = "deflation"

# Tie-break order when two regimes collect the same number of votes: the more
# defensive regime wins. Fixed so the classifier is deterministic.
REGIME_ORDER = (DEFLATION, TIGHTENING, REFLATION, GOLDILOCKS)

# --- Macro inputs (spec 2.1, sources per 11.5) ----------------------------

# name -> (source, identifier, kind). "price" series trend on percentage change,
# "rate" series (yields, spreads) trend on absolute change because they can be
# zero or negative.
MACRO_SERIES = {
    "oil": ("price", "USO", "price"),
    "gold": ("price", "GLD", "price"),
    "usd": ("price", "UUP", "price"),
    "ust10y": ("fred", "DGS10", "rate"),
    "spread_2s10s": ("fred", "T10Y2Y", "rate"),
    "real10y": ("fred", "DFII10", "rate"),
}

TREND_ROC_WINDOW = 63  # spec 2.3: sign of a 63-day rate of change
TREND_MA_WINDOW = 200  # confirmed against the 200-day moving average
INVERSION_LOOKBACK = 252  # "steepening from inversion": spread was < 0 within a year

# --- Confidence (spec 2.4 / 18.4, gates per 20.2) -------------------------

CONF_W_AGREEMENT = 0.5
CONF_W_PERSISTENCE = 0.3
CONF_W_STABILITY = 0.2
PERSISTENCE_CAP_DAYS = 20

CONFIDENCE_FULL = 0.60  # at or above: up to FULL_POSITIONS
CONFIDENCE_PARTIAL = 0.40  # at or above: up to PARTIAL_POSITIONS; below: cash
FULL_POSITIONS = 5
PARTIAL_POSITIONS = 3

# --- Universe (spec 4.1) --------------------------------------------------

SLEEVES = {
    "us_equity_broad": ("SPY", "QQQ", "IWM", "MDY"),
    "us_equity_sector": (
        "XLK", "XLE", "XLF", "XLI", "XLV", "XLU", "XLB", "XLY", "XLP", "XLRE", "XLC", "SMH",
    ),
    "international": ("EFA", "EEM", "EWJ", "FXI"),
    "fixed_income": ("TLT", "IEF", "SHY", "LQD", "HYG", "TIP"),
    "commodities": ("DBC", "GLD", "SLV", "USO", "UNG", "COPX", "DBA"),
    "miners": ("GDX", "GDXJ", "SIL", "SILJ", "XME"),
    "currency": ("UUP", "USDU", "FXE", "FXY", "FXF"),
    "inverse": ("SH", "PSQ", "RWM", "TBF", "DOG"),
    "cash": ("BIL", "USFR", "SGOV"),
}

CASH_TICKER = "BIL"  # residual cash earns this instrument's return in backtests

# --- Regime whitelist (spec 4.2) ------------------------------------------
# Instruments outside the whitelist are ineligible regardless of score. Cash is
# never "selected": it is whatever is left over.

WHITELIST = {
    REFLATION: frozenset(
        SLEEVES["commodities"] + SLEEVES["miners"] + SLEEVES["international"]
        + ("XLE", "XLB", "TIP")
    ),
    GOLDILOCKS: frozenset(
        SLEEVES["us_equity_broad"] + ("XLK", "SMH", "XLY", "LQD", "HYG")
    ),
    TIGHTENING: frozenset(("UUP", "USDU", "SH", "PSQ", "RWM", "DOG", "SHY")),
    DEFLATION: frozenset(("TLT", "IEF", "GLD", "XLU", "XLP")),
}

# --- Selection (spec 20.5) ------------------------------------------------

DRAWDOWN_WINDOW = 252  # drawdown from the 12-month high, per Choi 2021
CONFIRM_WINDOW = 21  # one-month confirmation: close above the close a month ago

# --- Sizing (spec 5) ------------------------------------------------------

MAX_SINGLE_POSITION = 0.25

# --- Backtest (spec 8, 20.1, 20.5) ----------------------------------------

COST_BPS = 15.0
HOLDOUT_START = "2022-01-01"

# --- Guardrails (spec 7) --------------------------------------------------

MAX_ORDERS_PER_DAY = 10
MAX_DAILY_TURNOVER = 0.40
STALE_BUSINESS_DAYS = 2  # "older than 48 hours", counted in sessions
KILL_ENV_VAR = "MACRO_AGENT_KILL"
