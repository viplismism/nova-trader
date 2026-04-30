"""Service layer for analysis, execution, and backtesting."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
from dateutil.relativedelta import relativedelta

from src.backtesting.costs import CostModel
from src.backtesting.metrics import PerformanceMetricsCalculator
from src.backtesting.portfolio import Portfolio
from src.backtesting.trader import TradeExecutor
from src.backtesting.valuation import calculate_portfolio_value, compute_exposures
from src.core.config import CoreConfig
from src.core.models import AnalysisResult, BacktestResult, PortfolioState, PositionState
from src.core.pipeline import TradingPipeline
from src.data.api import get_price_data
from src.execution.bridge import ExecutionBridge
from src.execution.circuit_breaker import PortfolioSnapshot


class TradingService:
    """Thin façade over the new pipeline."""

    def __init__(self, config: CoreConfig | None = None, api_key: str | None = None) -> None:
        self.config = config or CoreConfig()
        self.pipeline = TradingPipeline(config=self.config, api_key=api_key)

    def analyze(
        self,
        *,
        tickers: list[str],
        start_date: str,
        end_date: str,
        portfolio: PortfolioState | None = None,
        execute: bool = False,
    ) -> AnalysisResult:
        portfolio = portfolio or PortfolioState(cash=self.config.initial_cash)
        result = self.pipeline.analyze(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            portfolio=portfolio,
        )
        if execute:
            bridge = ExecutionBridge.paper() if self.config.execution_mode == "paper" else ExecutionBridge.dry_run()
            snapshot = self._execution_snapshot(result)
            result.execution = bridge.execute(result.decisions(), snapshot)
        return result

    def backtest(
        self,
        *,
        tickers: list[str],
        start_date: str,
        end_date: str,
        initial_cash: float | None = None,
    ) -> BacktestResult:
        initial_cash = float(initial_cash or self.config.initial_cash)
        portfolio = Portfolio(
            tickers=tickers,
            initial_cash=initial_cash,
            margin_requirement=self.config.margin_requirement,
        )
        executor = TradeExecutor(CostModel())
        metrics_calculator = PerformanceMetricsCalculator()

        values: list[dict] = []
        trade_count = 0
        dates = pd.date_range(start_date, end_date, freq="B")
        if len(dates) == 0:
            return BacktestResult(start_date, end_date, {}, [], initial_cash, 0)

        for current_date in dates:
            current_date_str = current_date.strftime("%Y-%m-%d")
            previous_date_str = (current_date - relativedelta(days=1)).strftime("%Y-%m-%d")
            lookback_start = (current_date - relativedelta(months=self.config.analysis_lookback_months)).strftime("%Y-%m-%d")
            current_prices = self._execution_prices(tickers, previous_date_str, current_date_str)
            if len(current_prices) != len(tickers):
                continue

            analysis = self.analyze(
                tickers=tickers,
                start_date=lookback_start,
                end_date=previous_date_str,
                portfolio=self._portfolio_state_from_backtest(
                    portfolio=portfolio,
                    current_prices=current_prices,
                    trade_date=current_date,
                ),
                execute=False,
            )

            for ticker, report in analysis.reports.items():
                decision = report.decision
                executed = executor.execute_trade(
                    ticker=ticker,
                    action=decision["action"],
                    quantity=decision["quantity"],
                    current_price=current_prices[ticker],
                    portfolio=portfolio,
                )
                if executed > 0:
                    trade_count += 1

            total_value = calculate_portfolio_value(portfolio, current_prices)
            exposures = compute_exposures(portfolio, current_prices)
            values.append(
                {
                    "Date": current_date,
                    "Portfolio Value": total_value,
                    "Long Exposure": exposures["Long Exposure"],
                    "Short Exposure": exposures["Short Exposure"],
                    "Gross Exposure": exposures["Gross Exposure"],
                    "Net Exposure": exposures["Net Exposure"],
                    "Long/Short Ratio": exposures["Long/Short Ratio"],
                }
            )

        metrics = metrics_calculator.compute_metrics(values)
        final_value = values[-1]["Portfolio Value"] if values else initial_cash
        curve = [
            {
                "date": point["Date"].strftime("%Y-%m-%d"),
                "portfolio_value": point["Portfolio Value"],
            }
            for point in values
        ]
        return BacktestResult(
            start_date=start_date,
            end_date=end_date,
            metrics=metrics,
            equity_curve=curve,
            final_value=final_value,
            trade_count=trade_count,
        )

    def _execution_snapshot(self, result: AnalysisResult) -> PortfolioSnapshot:
        positions: dict[str, dict] = {}
        nav = result.portfolio.cash
        for ticker, position in result.portfolio.positions.items():
            price = result.reports.get(ticker).current_price if ticker in result.reports else 0.0
            long_value = position.long * price
            short_value = position.short * price
            nav += long_value - short_value
            positions[ticker] = {
                "long": position.long,
                "short": position.short,
                "value": abs(long_value - short_value),
                "price": price,
            }
        return PortfolioSnapshot(
            nav=nav,
            cash=result.portfolio.cash,
            positions=positions,
            day_start_nav=result.portfolio.day_start_nav or nav,
            peak_nav=result.portfolio.peak_nav or nav,
        )

    def _execution_prices(self, tickers: list[str], start_date: str, end_date: str) -> dict[str, float]:
        prices: dict[str, float] = {}
        for ticker in tickers:
            frame = get_price_data(ticker, start_date, end_date)
            if frame.empty:
                continue
            bar = frame.iloc[-1]
            execution_price = bar.get("open")
            if execution_price is None or pd.isna(execution_price):
                execution_price = bar.get("close")
            prices[ticker] = float(execution_price)
        return prices

    def _portfolio_state_from_backtest(
        self,
        *,
        portfolio: Portfolio,
        current_prices: dict[str, float],
        trade_date: datetime,
    ) -> PortfolioState:
        snapshot = portfolio.get_snapshot()
        state = PortfolioState(cash=float(snapshot["cash"]))
        for ticker, position in snapshot["positions"].items():
            state.positions[ticker] = PositionState(
                long=int(position["long"]),
                short=int(position["short"]),
            )
        nav = calculate_portfolio_value(portfolio, current_prices)
        state.day_start_nav = nav
        state.peak_nav = nav
        return state
