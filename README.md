# Macro-Agent

A regime-driven trading agent, built from the *Regime-Driven Trading Agent — Build Spec*
(all twenty sections, including the 28 September audit corrections), with a backtest harness
and a Monte Carlo lab for refining it.

**The one rule:** the LLM never makes a trading decision. Every signal is a deterministic
Python function of market data. The orchestrating agent runs `macro-agent run` and reports
what happened. It does not form a view.

## How it trades

The system decides **every session** and trades only when a high-conviction setup appears.
There is no calendar rebalance. Positions leave only for a defined reason.

Each session, at the close (spec §12: every stage can only narrow what the one before allowed):

1. **Regime and confidence.** Six macro series vote for one of four regimes. Confidence is
   `0.5·agreement + 0.3·persistence + 0.2·stability`, where stability comes from turbulence
   and the absorption ratio. At ≥ 0.60 up to 5 positions are allowed, at ≥ 0.40 up to 3, and
   below 0.40 everything exits to cash.
2. **Exits**, judged on the weekly close:
   - the Fibonacci stop;
   - structural invalidation (a weekly close below the confirmation low);
   - the 12-week flag time limit;
   - a regime exit once the asset has been off the whitelist for 5 sessions.
3. **Tranches.** Positions still accumulating buy their next quarter every two weeks. This
   pauses while the regime disallows the asset.
4. **Entries** into free slots:
   `whitelist → confirmation → reward-to-risk ≥ 3 (to the real stop) → upside anchors agree →
   BAN score rank → correlation cap → volatility-targeted size`.

Orders fill at the **next session's open**, paying half the bid–ask spread plus 5bp of slippage
for that instrument.

**BAN score (§17.6):** `reward-to-risk × drawdown rank × confirmation × (0.5 + 0.5·flow)`.
- Upside is measured to the 52-week high, cross-checked against the Fibonacci target, or to the
  next extension for a breakout.
- Downside is measured to the stop (the Fibonacci support minus 1 ATR), not to the level itself.

**Position state machine (§16.6):**
- **ACCUMULATING:** the stop is the Fibonacci level minus 1 ATR, frozen.
- **IMPULSE:** the stop is unchanged.
- **CONSOLIDATING:** the stop is `max(stop, impulse high − 0.618·leg − 0.5·impulse ATR)`.
- **RESOLVED:** the stop ratchets up the ladder and a new impulse begins.

Stops only ever move up.

## Modules

| Module | What it does | Spec |
|---|---|---|
| `data/` | Stooq/Tiingo OHLCV, FRED (DGS10, T10Y2Y, DFII10), CFTC Commitments of Traders, cache, `Market` | §9, §11.5 |
| `indicators.py`, `fib.py` | ATR, RSI, weekly bars, OBV, **confirmed pivots**, anchored swings, Fibonacci ladders, macro fib position/proximity/state | §2.2, audit |
| `regime/` | Trend votes, audit-corrected classifier, turbulence + absorption stability, confidence | §2.3–2.4, §18.3–18.4 |
| `flows.py` | Hedging pressure 0.40 (follow), OBV 0.20, relative strength 0.20, retail ETF flow 0.20 (**fade**); 13F removed | §18.5 |
| `features.py` | Every signal input, computed once and causally | — |
| `positions.py` | Position, Book, and the stop state machine | §15, §16 |
| `strategy.py` | `Strategy.step()`: the one decision function, shared by backtest and live | §12, §14, §17 |
| `backtest/` | Event-driven engine, metrics, deflated Sharpe, trial ledger, PBO, minimum backtest length, bootstrap intervals, evaporation curve, walk-forward | §8, §19 |
| `montecarlo/` | Regime-switching market generator, block bootstrap of real data, stress ladder, ablation, sweeps | §20 |
| `guard/`, `execution/`, `runner.py` | Kill switch, stale data, order sanity checks, drawdown hard stop, audit log; paper broker; live run | §6, §7, §11.3 |

## Usage

```bash
pip install -e '.[dev]'
pytest                                   # 67 tests, all offline

macro-agent fetch                        # needs network; set TIINGO_API_KEY to prefer Tiingo
macro-agent backtest --start 2008-01-01 --end 2021-12-31 --out results/base
macro-agent backtest ... --param min_reward_risk=2.5 --variant rr2.5
macro-agent walkforward --start 2008-01-01 --end 2021-12-31 --grid min_reward_risk=2,3,4
macro-agent evaporation                  # is the search finding edge or mining noise?

macro-agent mc run --paths 200 --years 10           # distribution of outcomes
macro-agent mc stress --paths 100                   # §20.1 table: tails, costs, lag
macro-agent mc ablation --paths 100                 # §20.2: remove one component at a time
macro-agent mc sweep --grid min_reward_risk=2,3,4 --grid confidence_exit_days=1,3,5
macro-agent mc run --source bootstrap               # resample the cached real market

macro-agent decide                       # what it would do today
macro-agent run                          # one scheduled run, dry run, paper broker
macro-agent run --live                   # paper broker, places orders
```

`--synthetic` runs any command except `fetch` and `mc` on a simulated market.
Every strategy knob is a `Params` field in `config.py` and can be overridden with
`--param name=value`.

Kill switch: `touch state/KILL` or `export MACRO_AGENT_KILL=1`. The next run flattens to
cash and stops.

## Refining without fooling yourself

- **The holdout (2022 onward) is locked.** Backtests refuse to see it without `--use-holdout`.
  Use it once, at the end.
- **Every backtest and sweep is written to an append-only trial ledger.** The deflated Sharpe
  and the evaporation curve use the ledger's trial count, so the more variants you try, the
  higher the bar the winner must clear (§19).
- **`walkforward` reports the probability of backtest overfitting** across the grid it searched.
- **Monte Carlo sweeps are also logged.** A setting that only wins on one seed is noise.
- **The Monte Carlo generator's regime parameters are assumptions** (`RegimeModel`). It tests
  whether the logic behaves sensibly, not whether it has an edge (§20.6). Edge can only be
  shown on real history.

## Where later sections override earlier ones

The spec evolved as it went. Where sections conflict, the latest one wins:
- **Selection:** §14 and §17.6 replace the §4.3 momentum ranking.
- **Energy term:** §17.3's maximum-drawdown rank replaces the §14.1 energy composite.
- **Clarity term:** §18.4's stability measure replaces §2.4's clarity.
- **Flows:** the §18.5 flows engine replaces §3 (retail flow is faded, 13F removed).
- **200-day gate and calendar:** §14.3 removes the 200-day gate. The monthly calendar is
  dropped for daily decisions; the §12.4 emergency exit still applies.
- **Stops:** §16 refines §15. The stop ratchets up the ladder only when a flag resolves,
  otherwise the §16 flag tolerance would never apply.
- **Sleeve B:** folded into the core per §14.3. Tranching, confirmation-before-entry and the
  graduation idea remain.
- **Confidence gates:** 0.60 / 0.40, per §20.2.

## Interpretations where the spec was ambiguous

- **Deflation votes.** Real rate rising (audit). Gold votes Deflation when trending up or down;
  only flat gold doesn't. "Steepening from inversion" means the 2/10 is trending up and was
  below zero within the past year.
- **Ties** between regimes go to the more defensive one.
- **Pivots** need 10 lower bars on each side and count only from the session they are
  confirmed. The swing is the highest confirmed pivot high in 252 sessions, and the lowest
  confirmed pivot low after it.
- **Confirmation**, on weekly bars, is any one of:
  - a reclaim of the 10-week average within the last 4 weeks;
  - a confirmed higher low;
  - a positive RSI divergence.

  The confirmation low is the lowest weekly low of the last 8 weeks.
- **Downside** hangs from the nearest Fibonacci support or the confirmation low, whichever is
  closer to price.
- **Upside anchors** "disagree materially" when one is more than twice the other.
- **Breakout.** Within 2% of the 52-week high, the target is the next Fibonacci extension.
- **Sizing:** `min(25%, 12% / (vol × √slots))`, capped by the sleeve (50%) and names (20%)
  limits and by cash not already reserved for pending tranches. Entries below 1% of equity
  are skipped.
- **Minimum reward-to-risk is 3.0.** The spec gives examples (4:1 beats 1.5:1) but no number.
- **Stops** are judged on the weekly close and exit at the next session's open (§15.5).
- **Re-entry cooldown:** 21 sessions after a stop or invalidation, counted from the exit
  decision.
- **Retail ETF flows** are optional input (`flows.csv`), because no free source exists (§11.5).
  Without them the other flow components are reweighted.
- **COT reports** count as visible from the Monday after each Tuesday report date.
- **Hard stop.** A 20% drawdown flattens the book and stops trading, in the backtest as well
  as live, until reviewed.

## Guardrails

There is no turnover cap. Instead, every run checks that its orders make sense:
- the resulting portfolio must respect position size, position count, sleeve and names caps
  and the regime whitelist;
- no leverage and no shorts;
- the broker must show no unfilled orders;
- each decision can be sent only once;
- book and broker must agree;
- the 20% drawdown hard stop (§11.3);
- the kill switch, the stale-data halt, dry run by default, and the append-only audit log.

## Open issues

1. **Nothing has run on real data yet.** The build environment couldn't reach Stooq, FRED or
   the CFTC. Run `macro-agent fetch` and then a backtest on your machine.
2. **The emergency exit and regime exits churn on synthetic markets.** The daily regime read
   wobbles, confidence dips under 0.40, and the book is flattened. Tune it with
   `confidence_exit_days`, `regime_exit_days` and `emergency_confidence` via `mc sweep`, then
   confirm on real data.
3. **The Robinhood MCP adapter is a stub.** Confirm the tool schemas before writing order code.
4. **Universe availability** at the broker still needs validating (§11.6).
5. **A Thursday before a Friday holiday** is recognised as a week end one session late in live
   trading.
6. **Walk-forward** stitches each variant's returns and doesn't carry positions across a
   switch, so treat it as an estimate.
