"""Command line entry point.

    macro-agent fetch                     download prices and FRED series into the cache
    macro-agent decide [--as-of DATE]     print the pipeline's decision
    macro-agent backtest --start --end    run the harness, record the trial in the ledger
    macro-agent run [--live]              one scheduled run (dry run unless --live)

Add --synthetic to any command except fetch to use the offline fixture market.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

from .backtest import (
    BacktestConfig,
    TrialLedger,
    deflated_sharpe,
    fixed_weight_benchmark,
    run_backtest,
    summarize,
)
from .backtest.metrics import TRADING_DAYS
from .config import MACRO_SERIES, SLEEVES
from .data import (
    DataStore,
    FredSource,
    StooqSource,
    TiingoSource,
    build_macro_frame,
    synthetic_market,
)
from .execution import PaperBroker, RobinhoodMCPBroker
from .guard import AuditLog, GuardConfig
from .pipeline import decide
from .runner import run_once


def _all_tickers() -> list[str]:
    return sorted({t for group in SLEEVES.values() for t in group})


def _load(args) -> tuple[pd.DataFrame, pd.DataFrame]:
    if args.synthetic:
        prices, fred = synthetic_market()
    else:
        store = DataStore(args.cache)
        prices = store.load_prices(_all_tickers())
        if prices.empty:
            sys.exit(f"no cached prices in {args.cache}; run `macro-agent fetch` first")
        fred = store.load_fred()
    return prices, build_macro_frame(prices, fred)


def cmd_fetch(args) -> int:
    store = DataStore(args.cache)
    key = os.environ.get("TIINGO_API_KEY")
    prices_src = TiingoSource(key) if key else StooqSource()
    failures = []
    for t in _all_tickers():
        try:
            store.save("prices", t, prices_src.fetch(t))
            print(f"prices  {t}")
        except Exception as exc:  # keep going; report every failure at the end
            failures.append(f"{t}: {exc}")
    fred = FredSource()
    for src, ident, _ in MACRO_SERIES.values():
        if src != "fred":
            continue
        try:
            store.save("fred", ident, fred.fetch(ident))
            print(f"fred    {ident}")
        except Exception as exc:
            failures.append(f"{ident}: {exc}")
    for f in failures:
        print(f"FAILED  {f}", file=sys.stderr)
    return 1 if failures else 0


def cmd_decide(args) -> int:
    prices, macro = _load(args)
    as_of = args.as_of or prices.index[-1]
    print(json.dumps(decide(prices, macro, as_of).to_dict(), indent=2))
    return 0


def cmd_backtest(args) -> int:
    prices, macro = _load(args)
    config = BacktestConfig(
        start=args.start,
        end=args.end,
        cost_bps=args.cost_bps,
        execution_delay=args.delay,
        use_holdout=args.use_holdout,
    )
    result = run_backtest(prices, macro, config)
    stats = summarize(result.equity)
    rets = result.equity.pct_change().dropna()

    ledger = TrialLedger(args.ledger)
    ledger.record(
        variant=args.variant,
        params={"cost_bps": args.cost_bps, "delay": args.delay, "synthetic": args.synthetic},
        sharpe_per_period=stats["sharpe"] / TRADING_DAYS**0.5,
        sharpe_annual=stats["sharpe"],
        data_range=(stats["start"], stats["end"]),
        holdout_used=args.use_holdout,
    )
    dsr = deflated_sharpe(rets, ledger.n_trials, ledger.sharpes())

    bench = {}
    if "SPY" in prices.columns:
        bench["buy_hold_spy"] = summarize(
            fixed_weight_benchmark(prices, {"SPY": 1.0}, config, rebalance=False)
        )
    if {"SPY", "IEF"} <= set(prices.columns):
        bench["sixty_forty"] = summarize(
            fixed_weight_benchmark(prices, {"SPY": 0.6, "IEF": 0.4}, config, rebalance=True)
        )

    in_cash = sum(1 for d in result.decisions if not d.selected)
    out = {
        "strategy": stats,
        "deflated_sharpe": dsr,
        "months": len(result.decisions),
        "share_of_months_in_cash": in_cash / max(len(result.decisions), 1),
        "benchmarks": bench,
        "trials_in_ledger": ledger.n_trials,
        "holdout_used": args.use_holdout,
    }
    if args.synthetic:
        out["warning"] = "synthetic data: says nothing about real-world edge"
    print(json.dumps(out, indent=2))
    return 0


def cmd_run(args) -> int:
    prices, macro = _load(args)
    guard = GuardConfig(kill_file=Path(args.state) / "KILL")
    if args.max_turnover is not None:
        guard.max_daily_turnover = args.max_turnover
    audit = AuditLog(Path(args.state) / "audit.jsonl")
    if args.broker == "paper":
        broker = PaperBroker(Path(args.state) / "paper_account.json")
    else:
        broker = RobinhoodMCPBroker()
    report = run_once(
        prices,
        macro,
        broker,
        guard,
        audit,
        as_of=args.as_of,
        dry_run=not args.live,
        alert=lambda msg: print(f"ALERT: {msg}", file=sys.stderr),
    )
    print(json.dumps(report.to_dict(), indent=2))
    return 0 if report.status == "ok" else 2


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="macro-agent", description=__doc__.splitlines()[0])
    p.add_argument("--cache", default="cache", help="data cache directory")
    sub = p.add_subparsers(dest="command", required=True)

    f = sub.add_parser("fetch", help="download data into the cache")
    f.set_defaults(func=cmd_fetch)

    def data_opts(sp):
        sp.add_argument("--synthetic", action="store_true", help="use the offline fixture market")

    d = sub.add_parser("decide", help="print the pipeline decision")
    data_opts(d)
    d.add_argument("--as-of")
    d.set_defaults(func=cmd_decide)

    b = sub.add_parser("backtest", help="run the backtest harness")
    data_opts(b)
    b.add_argument("--start", required=True)
    b.add_argument("--end", required=True)
    b.add_argument("--cost-bps", type=float, default=15.0)
    b.add_argument("--delay", type=int, default=1, help="sessions between signal and execution")
    b.add_argument("--variant", default="mvp", help="name recorded in the trial ledger")
    b.add_argument("--ledger", default="state/trial_ledger.jsonl")
    b.add_argument(
        "--use-holdout",
        action="store_true",
        help="unlock 2022 onward. Final out-of-sample evaluation only.",
    )
    b.set_defaults(func=cmd_backtest)

    r = sub.add_parser("run", help="one scheduled run")
    data_opts(r)
    r.add_argument("--as-of")
    r.add_argument("--state", default="state")
    r.add_argument("--broker", choices=["paper", "robinhood"], default="paper")
    r.add_argument("--live", action="store_true", help="place orders. Default is dry run.")
    r.add_argument(
        "--max-turnover",
        type=float,
        help="override the 40%% daily turnover halt (e.g. 1.0 for an initial deployment)",
    )
    r.set_defaults(func=cmd_run)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
