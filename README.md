# Macro-Agent

A regime-driven trading agent, built from the *Regime-Driven Trading Agent — Build Spec*.
This repository is the **minimum viable version** (spec §20.5). The spec's own verdict is to
build the four things that carry the result, prove them on real data with the lag in place,
and only then add components one at a time.

**The one rule:** the LLM never makes a trading decision. Every signal is a deterministic
Python function of market data. The orchestrating agent runs `macro-agent run` and reports
what happened. It does not form a view.

## What is built

| Module | What it does | Spec |
|---|---|---|
| `data/` | Stooq/Tiingo adjusted closes, FRED yields (DGS10, T10Y2Y, DFII10), CSV cache, deterministic synthetic fixture market | §9, §11.5 |
| `regime/` | Four-regime classifier from six macro series (63-day change confirmed by 200-day MA), audit-corrected Deflation row, confidence score, 0.60/0.40 gates | §2.3, §2.4, audit, §20.2 |
| `ban/` | Regime whitelist → one-month confirmation → rank by drawdown from 12-month high (Choi 2021) → top N, ties alphabetical | §4.2, §12, §20.5 |
| `sizing/` | Equal weight, 20% per slot, 25% cap, cash residual; the only place rounding to shares happens | §5, §12.5 |
| `backtest/` | Daily simulation, execution one session after the signal (zero lag refused), 15bp costs, holdout locked from 2022, SPY and 60/40 benchmarks, deflated Sharpe, append-only trial ledger | §8, §19, §20.1 |
| `execution/` | Broker adapter interface, paper broker, order diff (sells first), reconciliation with two attempts | §6 |
| `guard/` | Kill switch (file or env), stale-data halt, 10 orders/day, 40% turnover/day, append-only audit log, dry run by default | §7 |
| `pipeline.py`, `runner.py` | One `decide()` function; one scheduled run: guard → decide → size → diff → trade → reconcile | §12 |

**Deliberately not built** (per §20.5: "Nothing else"): Fibonacci scoring, flows engine,
turbulence/absorption stability term, correlation cap, tranched entry, position state machine,
fib-anchored stops, Sleeve B, walk-forward, PBO/CSCV, intramonth emergency exit. Add each only
after the MVP works on real history, and keep it only if it survives deflation.

## Usage

```bash
pip install -e '.[dev]'
pytest

macro-agent fetch                                   # needs network; set TIINGO_API_KEY to prefer Tiingo
macro-agent decide                                  # today's decision from cached data
macro-agent backtest --start 2008-01-01 --end 2021-12-31
macro-agent run                                     # dry run against the paper broker
macro-agent run --live                              # paper broker, places orders

# Anything except fetch also takes --synthetic to run offline on the fixture market.
macro-agent backtest --synthetic --start 2008-01-01 --end 2021-12-31
```

Kill switch: `touch state/KILL` or `export MACRO_AGENT_KILL=1`. The next run flattens to cash and stops.

## Interpretations where the spec was ambiguous

These are choices, recorded so they can be reviewed rather than discovered.

- **Deflation votes.** Real rate RISING (audit). Gold votes Deflation when trending either up or
  down ("down then up"), not when flat. "Steepening from inversion" means the 2/10 is trending up
  and was below zero within the past 252 sessions. The stray "falling hard" cell is ignored.
- **Trend.** UP if the 63-day change is positive *and* the level is above its 200-day MA; DOWN if
  both negative; otherwise FLAT. Yields and spreads use absolute change, prices use percentage.
- **Regime ties** go to the more defensive regime: Deflation, Tightening, Reflation, Goldilocks.
- **Stability term.** The MVP has no turbulence measure, so `stability = 1.0`. The §18.4 composite
  plugs into `regime.confidence(..., stability=...)`.
- **Confirmation (MVP).** Close above the close 21 sessions earlier.
- **Ranking.** Deeper drawdown from the 252-session high ranks higher (washed-out plus confirmed).
- **Sizing.** Each position is 1/5 of the book, so three positions are 60% invested.
- **Cash** is never selected. It is the residual and earns BIL's return in backtests.
- **Tightening's "short duration"** is SHY only. TBF is not whitelisted.
- **"The one-period lag."** A decision on month-end close *s* executes at the close of the next
  session and earns returns only after that. `execution_delay` below 1 is refused.
- **Stale data.** "Older than 48 hours" is counted as more than two business days, so Friday's
  close is still valid on Monday.
- **`exec/`** is named `execution/` because `exec` is a Python builtin.

## Open issues needing a decision

1. **The 40% daily turnover cap blocks deployment.** Deploying from cash is ~100% turnover, and
   a full monthly rotation can reach 200%. As specified, the guard halts the first live run
   (a test covers this). `macro-agent run --max-turnover 1.0` overrides it. The spec needs to say
   whether the cap applies to scheduled rebalances or only to off-cycle runs.
2. **Robinhood MCP adapter is a stub.** Its tool names and schemas need to be confirmed against
   the live server before any order code is written.
3. **No real data has been run.** The build environment could not reach Stooq or FRED. The
   fetchers are tested against mocked responses only. Every backtest number so far comes from
   synthetic data and says nothing about edge.
4. **Universe availability.** Validate every ticker against what the broker supports (§11.6).
   `run` warns about whitelisted tickers with no data.
