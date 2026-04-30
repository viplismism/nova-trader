"""Minimal CLI for the rebuilt Nova Trader core."""

from __future__ import annotations

import argparse
import json
from datetime import datetime

from dateutil.relativedelta import relativedelta

from src.core import CoreConfig, PortfolioState, TradingService


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nova", description="Lean trading research, backtesting, and execution.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser("analyze", help="Run analysis for one or more tickers.")
    analyze.add_argument("tickers", nargs="+", help="Ticker symbols, e.g. AAPL MSFT NVDA")
    analyze.add_argument("--start-date", type=str, help="Analysis start date (YYYY-MM-DD)")
    analyze.add_argument("--end-date", type=str, default=datetime.now().strftime("%Y-%m-%d"), help="Analysis end date (YYYY-MM-DD)")
    analyze.add_argument("--months-back", type=int, default=12, help="Lookback window when --start-date is not provided")
    analyze.add_argument("--cash", type=float, default=100_000.0, help="Portfolio cash")
    analyze.add_argument("--mode", choices=("dry_run", "paper"), default="dry_run", help="Execution mode")
    analyze.add_argument("--execute", action="store_true", help="Execute the resulting decisions")
    analyze.add_argument("--json", action="store_true", help="Print raw JSON output")

    backtest = subparsers.add_parser("backtest", help="Run a backtest.")
    backtest.add_argument("tickers", nargs="+", help="Ticker symbols, e.g. AAPL MSFT NVDA")
    backtest.add_argument("--start-date", type=str, required=True, help="Backtest start date (YYYY-MM-DD)")
    backtest.add_argument("--end-date", type=str, required=True, help="Backtest end date (YYYY-MM-DD)")
    backtest.add_argument("--cash", type=float, default=100_000.0, help="Initial cash")
    backtest.add_argument("--months-back", type=int, default=12, help="Analysis lookback window per rebalance")
    backtest.add_argument("--json", action="store_true", help="Print raw JSON output")

    return parser


def _resolve_start_date(start_date: str | None, end_date: str, months_back: int) -> str:
    if start_date:
        return start_date
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    return (end_dt - relativedelta(months=months_back)).strftime("%Y-%m-%d")


def _print_analysis(result) -> None:
    print(f"\nAnalysis window: {result.start_date} -> {result.end_date}")
    print(f"Execution mode: {result.mode}")
    print("")
    for ticker, report in result.reports.items():
        decision = report.decision
        print(
            f"{ticker}: {decision['action'].upper()} {decision['quantity']} "
            f"(confidence={decision['confidence']:.1f}, price=${report.current_price:,.2f})"
        )
        for model_name, signal in report.signals.items():
            print(
                f"  - {model_name}: {signal.get('signal', 'neutral')} "
                f"[{signal.get('confidence', 0):.1f}]"
            )
    if result.execution is not None:
        print("")
        print(result.execution.summary())


def _print_backtest(result) -> None:
    print(f"\nBacktest window: {result.start_date} -> {result.end_date}")
    print(f"Trades executed: {result.trade_count}")
    print(f"Final value: ${result.final_value:,.2f}")
    metrics = result.metrics
    print(
        "Metrics: "
        f"Sharpe={metrics.get('sharpe_ratio')}, "
        f"Sortino={metrics.get('sortino_ratio')}, "
        f"MaxDD={metrics.get('max_drawdown')}"
    )


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "analyze":
        start_date = _resolve_start_date(args.start_date, args.end_date, args.months_back)
        service = TradingService(
            CoreConfig(
                initial_cash=args.cash,
                execution_mode=args.mode,
                analysis_lookback_months=args.months_back,
            )
        )
        result = service.analyze(
            tickers=args.tickers,
            start_date=start_date,
            end_date=args.end_date,
            portfolio=PortfolioState(cash=args.cash),
            execute=args.execute,
        )
        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            _print_analysis(result)
        return

    if args.command == "backtest":
        service = TradingService(
            CoreConfig(
                initial_cash=args.cash,
                analysis_lookback_months=args.months_back,
            )
        )
        result = service.backtest(
            tickers=args.tickers,
            start_date=args.start_date,
            end_date=args.end_date,
            initial_cash=args.cash,
        )
        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            _print_backtest(result)
        return


if __name__ == "__main__":
    main()
