"""Every parameter in the system, in one place, with the spec section it comes from.

Every number here is a knob, and every knob turned while looking at results is a
step toward a curve fit (spec section 10). Change them through `Params` so each
variant is recorded in the trial ledger, never by editing this file mid-study.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace

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
# "rate" series (yields, spreads) on absolute change because they can be <= 0.
MACRO_SERIES = {
    "oil": ("price", "USO", "price"),
    "gold": ("price", "GLD", "price"),
    "usd": ("price", "UUP", "price"),
    "ust10y": ("fred", "DGS10", "rate"),
    "spread_2s10s": ("fred", "T10Y2Y", "rate"),
    "real10y": ("fred", "DFII10", "rate"),
}

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
    # Screened watchlist of individual names, maximum 20 (spec 4.1). Empty by default.
    "individual_names": (),
}

TICKER_SLEEVE = {t: s for s, group in SLEEVES.items() for t in group}
CASH_TICKER = "BIL"  # idle cash earns this instrument's return in backtests
BENCHMARK = "SPY"  # relative strength reference (spec 3)

# Commodity-linked instruments use CFTC hedging pressure (spec 18.2). Value is the
# CFTC contract market code of the underlying future.
COT_CONTRACTS = {
    "GLD": "088691", "GDX": "088691", "GDXJ": "088691",  # gold
    "SLV": "084691", "SIL": "084691", "SILJ": "084691",  # silver
    "USO": "067651",  # WTI crude
    "UNG": "023651",  # natural gas
    "COPX": "085692",  # copper
}

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

# Basket for turbulence and the absorption ratio (spec 18.3): broad, liquid,
# cross-asset. Computed from returns alone.
STABILITY_BASKET = ("SPY", "EFA", "EEM", "TLT", "IEF", "HYG", "GLD", "USO", "UUP", "DBC")

# --- Trading costs (spec 8) -----------------------------------------------
# Commission-free does not mean cost-free. Every fill pays half the bid-ask
# spread plus slippage. Half-spreads are estimates by sleeve, in basis points.

SLIPPAGE_BPS = 5.0
HALF_SPREAD_BPS = {
    "us_equity_broad": 1.0,
    "us_equity_sector": 2.0,
    "international": 3.0,
    "fixed_income": 2.0,
    "commodities": 5.0,
    "miners": 8.0,
    "currency": 8.0,
    "inverse": 4.0,
    "cash": 1.0,
    "individual_names": 5.0,
}
HALF_SPREAD_OVERRIDES = {"SILJ": 15.0, "SIL": 12.0, "GDXJ": 10.0, "COPX": 12.0, "FXF": 15.0,
                         "USDU": 12.0, "DBA": 10.0, "XLC": 3.0, "XLRE": 3.0}


def cost_bps(ticker: str) -> float:
    half = HALF_SPREAD_OVERRIDES.get(ticker, HALF_SPREAD_BPS.get(TICKER_SLEEVE.get(ticker, ""), 5.0))
    return half + SLIPPAGE_BPS


# --- Tunable parameters ---------------------------------------------------


@dataclass(frozen=True)
class Params:
    """Every strategy knob. Variants are made with `with_(...)` and logged."""

    # Regime trend (spec 2.3)
    trend_roc_window: int = 63
    trend_ma_window: int = 200
    inversion_lookback: int = 252

    # Confidence (spec 2.4, 18.4) and gates (spec 20.2)
    conf_w_agreement: float = 0.5
    conf_w_persistence: float = 0.3
    conf_w_stability: float = 0.2
    persistence_cap_days: int = 20
    confidence_full: float = 0.60
    confidence_partial: float = 0.40
    full_positions: int = 5
    partial_positions: int = 3

    # Stability: turbulence and absorption ratio (spec 18.3)
    stability_window: int = 252
    stability_rank_window: int = 756
    absorption_fraction: float = 0.2  # top fifth of eigenvectors
    absorption_short: int = 15
    use_stability: bool = True

    # Fibonacci (spec 2.2, audit, 14.1)
    pivot_bars: int = 10  # a pivot needs N lower highs (higher lows) either side
    swing_lookback: int = 252
    fib_testing_atr: float = 0.5  # "testing" is within 0.5 ATR of a level

    # BAN score (spec 17.6)
    drawdown_window: int = 252
    breakout_band: float = 0.02  # within 2% of the 52-week high: use fib extension
    anchor_disagreement: float = 2.0  # skip if the two upside anchors differ by > 2x
    min_reward_risk: float = 3.0  # high-conviction filter on TRUE reward-to-risk
    require_confirmation: bool = True  # spec 20.2: the component that carries the result
    use_regime_whitelist: bool = True
    ten_week_ma: int = 10
    reclaim_lookback_weeks: int = 4
    use_flows: bool = True

    # Flows (spec 18.5)
    flow_slope_window: int = 63
    flow_rank_window: int = 252
    cot_rank_weeks: int = 156

    # Correlation cap (audit)
    corr_window: int = 63
    max_correlation: float = 0.7
    use_correlation_cap: bool = True

    # Sizing (spec 5)
    vol_target: float = 0.12
    vol_window: int = 63
    max_position: float = 0.25
    max_sleeve: float = 0.50
    max_names: float = 0.20
    min_position: float = 0.01  # skip entries smaller than 1% of equity: no dust

    # Entry (spec 13.4, 14.4)
    tranches: int = 4
    tranche_spacing: int = 10  # sessions: one every two weeks
    use_tranching: bool = True
    reentry_cooldown: int = 21  # no re-entry on a stopped setup for a cycle

    # Stops and position state (spec 15, 16)
    atr_window: int = 14
    stop_atr: float = 1.0  # entry stop: fib level minus 1.0 ATR
    pattern_retrace: float = 0.618
    pattern_atr: float = 0.5
    consolidation_range: float = 0.60
    max_flag_weeks: int = 12
    swing_high_weeks: int = 4
    use_stops: bool = True

    # Exits (spec 12.4)
    emergency_confidence: float = 0.40
    confidence_exit_days: int = 1  # consecutive sessions below the line before exiting
    regime_exit_days: int = 5  # off-whitelist this many sessions before exiting

    def with_(self, **changes) -> "Params":
        return replace(self, **changes)

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT_PARAMS = Params()

# --- Backtest (spec 8, 20.1) ----------------------------------------------

HOLDOUT_START = "2022-01-01"
DEPLOY_MIN_SHARPE = 0.8
DEPLOY_MAX_DRAWDOWN = 0.20

# --- Guardrails (spec 7, 11.3) --------------------------------------------

MAX_ORDERS_PER_DAY = 10
STALE_BUSINESS_DAYS = 2  # "older than 48 hours", counted in sessions
MAX_DRAWDOWN_HALT = 0.20  # hard stop, flatten and review (spec 11.3)
KILL_ENV_VAR = "MACRO_AGENT_KILL"


@dataclass(frozen=True)
class GuardLimits:
    max_orders_per_day: int = MAX_ORDERS_PER_DAY
    stale_business_days: int = STALE_BUSINESS_DAYS
    max_drawdown: float = MAX_DRAWDOWN_HALT
