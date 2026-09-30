"""Command line entry point.

    macro-agent fetch                               download OHLCV, FRED and CFTC data into the cache
    macro-agent decide [--as-of DATE]               what the strategy would do today, from a flat book
    macro-agent backtest --start --end              full backtest; records the trial in the ledger
    macro-agent walkforward --start --end --grid    rolling 5y fit / 1y test selection + PBO
    macro-agent mc run|stress|ablation|sweep        Monte Carlo on simulated or resampled markets
    macro-agent evaporation                         best Sharpe vs its deflation threshold, per trial
    macro-agent run [--live]                        one scheduled run (dry run unless --live)

Any strategy parameter can be overridden with --param name=value (repeatable);
every override is recorded in the trial ledger. Add --synthetic to decide,
backtest, walkforward or run to use a simulated market instead of the cache.
"""
from __future__ import annotations

import argparse
import dataclasses
import itertools
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
from .backtest.metrics import TRADING_DAYS, deploy_verdict, trade_stats
from .backtest.validation import bootstrap_ci, evaporation_curve, min_backtest_length, paired_test, pbo_cscv
from .backtest.walkforward import walk_forward
from .config import (
    COT_CONTRACTS,
    DEFAULT_PARAMS,
    DEPLOY_MAX_DRAWDOWN,
    DEPLOY_MIN_SHARPE,
    MACRO_SERIES,
    SLEEVES,
    Params,
)
from .data import CftcSource, DataStore, FredSource, StooqSource, TiingoSource
from .execution import PaperBroker, RobinhoodMCPBroker
from .features import build_features
from .guard import GuardConfig
from .montecarlo import MCConfig, ablation, monte_carlo, stress_ladder, summarize_mc, sweep, synthetic_market
from .positions import Book
from .runner import run_once
from .strategy import Strategy


def _all_tickers() -> list[str]:
    return sorted({t for group in SLEEVES.values() for t in group})


def _coerce(field: dataclasses.Field, raw: str):
    kind = field.type if isinstance(field.type, str) else field.type.__name__
    if "bool" in kind:
        return raw.lower() in {"1", "true", "yes", "on"}
    if "int" in kind:
        return int(raw)
    if "float" in kind:
        return float(raw)
    return raw


def _params(pairs: list[str] | None) -> Params:
    fields = {f.name: f for f in dataclasses.fields(Params)}
    changes = {}
    for pair in pairs or []:
        name, _, raw = pair.partition("=")
        if name not in fields:
            sys.exit(f"unknown parameter {name!r}; see macro_agent/config.py Params")
        changes[name] = _coerce(fields[name], raw)
    return DEFAULT_PARAMS.with_(**changes)


def _grid(pairs: list[str] | None) -> dict[str, list]:
    fields = {f.name: f for f in dataclasses.fields(Params)}
    grid = {}
    for pair in pairs or []:
        name, _, raw = pair.partition("=")
        if name not in fields:
            sys.exit(f"unknown parameter {name!r}")
        grid[name] = [_coerce(fields[name], v) for v in raw.split(",")]
    return grid


def _load(args):
    if getattr(args, "synthetic", False):
        market, _ = synthetic_market(seed=args.seed)
        return market
    store = DataStore(args.cache)
    try:
        return store.load_market(_all_tickers(), getattr(args, "flows", None))
    except FileNotFoundError as exc:
        sys.exit(f"{exc}; run `macro-agent fetch` first")


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, default=str))


# --- commands ---------------------------------------------------------------


def cmd_fetch(args) -> int:
    store = DataStore(args.cache)
    key = os.environ.get("TIINGO_API_KEY")
    prices_src = TiingoSource(key) if key else StooqSource()
    failures = []
    for t in _all_tickers():
        try:
            store.save_frame("prices", t, prices_src.fetch(t))
            print(f"prices  {t}")
        except Exception as exc:  # keep going; report every failure at the end
            failures.append(f"{t}: {exc}")
    fred = FredSource()
    for src, ident, _ in MACRO_SERIES.values():
        if src == "fred":
            try:
                store.save_series("fred", ident, fred.fetch(ident))
                print(f"fred    {ident}")
            except Exception as exc:
                failures.append(f"{ident}: {exc}")
    cftc = CftcSource()
    for code in sorted(set(COT_CONTRACTS.values())):
        try:
            store.save_series("cot", code, cftc.fetch(code))
            print(f"cot     {code}")
        except Exception as exc:
            failures.append(f"COT {code}: {exc}")
    for f in failures:
        print(f"FAILED  {f}", file=sys.stderr)
    return 1 if failures else 0


def cmd_decide(args) -> int:
    market = _load(args)
    if args.as_of:
        market = market.truncate(args.as_of)
    params = _params(args.param)
    strat = Strategy(market, build_features(market, params), params)
    book = Book(cash=1_000_000.0, peak_equity=1_000_000.0)
    decision = strat.step(len(market.dates) - 1, book)
    _print(decision.to_dict())
    return 0


def _benchmarks(market, config) -> dict:
    out = {}
    if "SPY" in market.tickers:
        out["buy_hold_spy"] = fixed_weight_benchmark(market, {"SPY": 1.0}, config, rebalance_monthly=False)
    if {"SPY", "IEF"} <= set(market.tickers):
        out["sixty_forty"] = fixed_weight_benchmark(market, {"SPY": 0.6, "IEF": 0.4}, config, rebalance_monthly=True)
    return out


def cmd_backtest(args) -> int:
    market = _load(args)
    params = _params(args.param)
    config = BacktestConfig(
        start=args.start, end=args.end, execution_delay=args.delay,
        cost_multiplier=args.cost_multiplier, use_holdout=args.use_holdout,
    )
    res = run_backtest(market, config, params)
    stats = summarize(res.equity, res.invested)
    ledger = TrialLedger(args.ledger)
    ledger.record(
        variant=args.variant, params={**params.to_dict(), "delay": args.delay,
                                      "cost_multiplier": args.cost_multiplier, "synthetic": args.synthetic},
        sharpe_per_period=stats["sharpe"] / TRADING_DAYS**0.5, sharpe_annual=stats["sharpe"],
        data_range=(stats["start"], stats["end"]), holdout_used=args.use_holdout,
    )
    bench = _benchmarks(market, config)
    out = {
        "strategy": stats,
        "trades": trade_stats(res.round_trips),
        "deflated_sharpe": deflated_sharpe(res.returns, ledger.n_trials, ledger.sharpes()),
        "min_backtest_years_for_this_sharpe": min_backtest_length(ledger.n_trials, stats["sharpe"]),
        "bootstrap": bootstrap_ci(res.returns),
        "benchmarks": {k: summarize(v) for k, v in bench.items()},
        "vs_spy": paired_test(res.returns, bench["buy_hold_spy"].pct_change().dropna()) if "buy_hold_spy" in bench else None,
        "deploy_verdict": deploy_verdict(stats, DEPLOY_MIN_SHARPE, DEPLOY_MAX_DRAWDOWN),
        "trials_in_ledger": ledger.n_trials,
        "holdout_used": args.use_holdout,
    }
    if args.synthetic:
        out["warning"] = "synthetic data: says nothing about real-world edge"
    if args.out:
        outdir = Path(args.out)
        outdir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"strategy": res.equity, **bench}).to_csv(outdir / "equity.csv")
        res.trades.to_csv(outdir / "fills.csv", index=False)
        res.round_trips.to_csv(outdir / "round_trips.csv", index=False)
        (outdir / "decisions.json").write_text(json.dumps(res.decisions, default=str, indent=1))
        (outdir / "summary.json").write_text(json.dumps(out, default=str, indent=2))
    _print(out)
    return 0


def cmd_walkforward(args) -> int:
    market = _load(args)
    base = _params(args.param)
    grid = _grid(args.grid)
    keys = sorted(grid)
    variants = {
        json.dumps(dict(zip(keys, vals))): base.with_(**dict(zip(keys, vals)))
        for vals in itertools.product(*(grid[k] for k in keys))
    } or {"base": base}
    config = BacktestConfig(start=args.start, end=args.end, use_holdout=args.use_holdout)
    wf = walk_forward(market, variants, config, args.train_years, 1)
    ledger = TrialLedger(args.ledger)
    for name, r in wf["variant_returns"].items():
        s = summarize((1 + r.dropna()).cumprod())
        ledger.record(f"walkforward:{name}", variants[name].to_dict(), s["sharpe"] / TRADING_DAYS**0.5,
                      s["sharpe"], (s["start"], s["end"]), args.use_holdout)
    out = {"windows": wf["windows"], "oos_summary": wf["oos_summary"], "trials_in_ledger": ledger.n_trials}
    if len(variants) > 1:
        out["pbo"] = pbo_cscv(wf["variant_returns"], n_splits=args.pbo_splits)
    _print(out)
    return 0


def cmd_mc(args) -> int:
    params = _params(args.param)
    mc = MCConfig(n_paths=args.paths, years=args.years, seed=args.seed, source=args.source,
                  processes=args.processes)
    base = None
    if args.source == "bootstrap":
        base = DataStore(args.cache).load_market(_all_tickers())
    if args.action == "run":
        df = monte_carlo(params, mc, base)
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(args.out, index=False)
        _print(summarize_mc(df))
    elif args.action == "stress":
        print(stress_ladder(params, mc).to_string(float_format=lambda v: f"{v:.3f}"))
    elif args.action == "ablation":
        print(ablation(params, mc).to_string(float_format=lambda v: f"{v:.3f}"))
    elif args.action == "sweep":
        table = sweep(_grid(args.grid), params, mc, TrialLedger(args.ledger))
        print(table.to_string(float_format=lambda v: f"{v:.3f}"))
    return 0


def cmd_evaporation(args) -> int:
    curve = evaporation_curve(TrialLedger(args.ledger).entries())
    if curve.empty:
        print("ledger is empty")
        return 0
    if args.out:
        curve.to_csv(args.out, index=False)
    print(curve.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    return 0


def cmd_run(args) -> int:
    market = _load(args)
    guard = GuardConfig(kill_file=Path(args.state) / "KILL")
    broker = PaperBroker(Path(args.state) / "paper_account.json") if args.broker == "paper" else RobinhoodMCPBroker()
    report = run_once(
        market, broker, args.state, guard, _params(args.param), as_of=args.as_of,
        dry_run=not args.live, alert=lambda msg: print(f"ALERT: {msg}", file=sys.stderr),
    )
    _print(report.to_dict())
    return 0 if report.status in ("ok", "duplicate") else 2


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="macro-agent", description=__doc__.splitlines()[0])
    p.add_argument("--cache", default="cache", help="data cache directory")
    p.add_argument("--ledger", default="state/trial_ledger.jsonl")
    sub = p.add_subparsers(dest="command", required=True)

    def data_opts(sp):
        sp.add_argument("--synthetic", action="store_true", help="use a simulated market")
        sp.add_argument("--seed", type=int, default=7, help="seed for --synthetic")
        sp.add_argument("--param", action="append", metavar="NAME=VALUE")

    sub.add_parser("fetch", help="download data into the cache").set_defaults(func=cmd_fetch)

    d = sub.add_parser("decide", help="what the strategy would do today")
    data_opts(d)
    d.add_argument("--as-of")
    d.set_defaults(func=cmd_decide)

    b = sub.add_parser("backtest", help="run the backtest harness")
    data_opts(b)
    b.add_argument("--start", required=True)
    b.add_argument("--end", required=True)
    b.add_argument("--delay", type=int, default=1, help="sessions between decision and fill")
    b.add_argument("--cost-multiplier", type=float, default=1.0)
    b.add_argument("--variant", default="base", help="name recorded in the trial ledger")
    b.add_argument("--out", help="directory for equity, fills, round trips and decisions")
    b.add_argument("--use-holdout", action="store_true",
                   help="unlock 2022 onward. Final out-of-sample evaluation only.")
    b.set_defaults(func=cmd_backtest)

    w = sub.add_parser("walkforward", help="rolling fit/test selection over a parameter grid")
    data_opts(w)
    w.add_argument("--start", required=True)
    w.add_argument("--end", required=True)
    w.add_argument("--grid", action="append", metavar="NAME=V1,V2,...")
    w.add_argument("--train-years", type=int, default=5)
    w.add_argument("--pbo-splits", type=int, default=16)
    w.add_argument("--use-holdout", action="store_true")
    w.set_defaults(func=cmd_walkforward)

    m = sub.add_parser("mc", help="Monte Carlo")
    m.add_argument("action", choices=["run", "stress", "ablation", "sweep"])
    m.add_argument("--paths", type=int, default=100)
    m.add_argument("--years", type=int, default=10)
    m.add_argument("--seed", type=int, default=0)
    m.add_argument("--source", choices=["regime", "bootstrap"], default="regime",
                   help="simulated regime model, or block-bootstrap of the cached real market")
    m.add_argument("--processes", type=int)
    m.add_argument("--param", action="append", metavar="NAME=VALUE")
    m.add_argument("--grid", action="append", metavar="NAME=V1,V2,...")
    m.add_argument("--out", help="CSV of per-path results (run only)")
    m.set_defaults(func=cmd_mc)

    e = sub.add_parser("evaporation", help="evaporation curve from the trial ledger")
    e.add_argument("--out")
    e.set_defaults(func=cmd_evaporation)

    r = sub.add_parser("run", help="one scheduled run")
    data_opts(r)
    r.add_argument("--as-of")
    r.add_argument("--state", default="state")
    r.add_argument("--broker", choices=["paper", "robinhood"], default="paper")
    r.add_argument("--live", action="store_true", help="place orders. Default is dry run.")
    r.set_defaults(func=cmd_run)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
