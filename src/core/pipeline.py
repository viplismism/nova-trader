"""Minimal trading pipeline: signals -> risk -> decisions."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd

from src.alpha.features import score_factor_investing, score_fundamentals, score_sentiment
from src.core.config import CoreConfig
from src.core.models import AnalysisResult, PortfolioState, TickerReport
from src.data.api import get_prices, prices_to_df
from src.portfolio.construction import VotingConfig, vote_on_ticker
from src.risk import calculate_volatility_metrics, compute_position_limit

_SIGNAL_MODELS = {
    "factor": score_factor_investing,
    "fundamentals": score_fundamentals,
    "sentiment": score_sentiment,
}


class TradingPipeline:
    """Pure analysis pipeline for the rebuilt core."""

    def __init__(self, config: CoreConfig | None = None, api_key: str | None = None) -> None:
        self.config = config or CoreConfig()
        self.api_key = api_key

    def analyze(
        self,
        *,
        tickers: list[str],
        start_date: str,
        end_date: str,
        portfolio: PortfolioState,
    ) -> AnalysisResult:
        tickers = [ticker.upper() for ticker in tickers]
        for ticker in tickers:
            portfolio.ensure_ticker(ticker)

        price_frames = self._load_price_frames(tickers, start_date, end_date, portfolio)
        current_prices = {
            ticker: float(frame["close"].iloc[-1])
            for ticker, frame in price_frames.items()
            if not frame.empty
        }
        nav, gross_exposure_pct = self._portfolio_stats(portfolio, current_prices)
        correlation_matrix = self._correlation_matrix(price_frames)
        drawdown_multiplier = self._drawdown_multiplier(
            nav=nav,
            peak_nav=portfolio.peak_nav or nav,
        )

        reports: dict[str, TickerReport] = {}
        vote_config = self._vote_config()

        for ticker in tickers:
            frame = price_frames.get(ticker, pd.DataFrame())
            current_price = float(frame["close"].iloc[-1]) if not frame.empty else 0.0

            signals = self._score_ticker(ticker=ticker, end_date=end_date)
            risk = self._risk_for_ticker(
                ticker=ticker,
                portfolio=portfolio,
                current_prices=current_prices,
                price_frames=price_frames,
                correlation_matrix=correlation_matrix,
                nav=nav,
                gross_exposure_pct=gross_exposure_pct,
                drawdown_multiplier=drawdown_multiplier,
            )
            max_shares = int(risk["remaining_position_limit"] // current_price) if current_price > 0 else 0
            decision = vote_on_ticker(
                ticker=ticker,
                ticker_signals=signals,
                max_shares=max_shares,
                current_price=current_price,
                position=portfolio.get_position_dict(ticker),
                config=vote_config,
            ).to_dict()

            reports[ticker] = TickerReport(
                ticker=ticker,
                current_price=current_price,
                signals=signals,
                risk=risk,
                decision=decision,
            )

        return AnalysisResult(
            start_date=start_date,
            end_date=end_date,
            mode=self.config.execution_mode,
            portfolio=portfolio,
            reports=reports,
        )

    def _load_price_frames(
        self,
        tickers: list[str],
        start_date: str,
        end_date: str,
        portfolio: PortfolioState,
    ) -> dict[str, pd.DataFrame]:
        frames: dict[str, pd.DataFrame] = {}
        all_tickers = sorted(set(tickers) | set(portfolio.positions))
        for ticker in all_tickers:
            prices = get_prices(ticker=ticker, start_date=start_date, end_date=end_date, api_key=self.api_key)
            if not prices:
                frames[ticker] = pd.DataFrame()
                continue
            frames[ticker] = prices_to_df(prices)
        return frames

    def _score_ticker(self, *, ticker: str, end_date: str) -> dict[str, dict[str, Any]]:
        signals: dict[str, dict[str, Any]] = {}
        for model_name in self.config.signal_models:
            scorer = _SIGNAL_MODELS[model_name]
            signals[model_name] = scorer(ticker, end_date, self.api_key)
        return signals

    def _portfolio_stats(self, portfolio: PortfolioState, current_prices: dict[str, float]) -> tuple[float, float]:
        nav = float(portfolio.cash)
        gross = 0.0
        for ticker, position in portfolio.positions.items():
            price = current_prices.get(ticker, 0.0)
            nav += position.long * price
            nav -= position.short * price
            gross += abs(position.long) * price
            gross += abs(position.short) * price
        nav = max(nav, 1.0)
        return nav, gross / nav if nav > 0 else 0.0

    def _correlation_matrix(self, price_frames: dict[str, pd.DataFrame]) -> pd.DataFrame | None:
        returns: dict[str, pd.Series] = {}
        for ticker, frame in price_frames.items():
            if frame.empty or len(frame) < self.config.risk.correlation_min_data_points + 1:
                continue
            series = frame["close"].pct_change().dropna()
            if not series.empty:
                returns[ticker] = series
        if len(returns) < 2:
            return None
        returns_df = pd.DataFrame(returns).dropna(how="any")
        if returns_df.shape[0] < self.config.risk.correlation_min_data_points:
            return None
        return returns_df.corr()

    def _drawdown_multiplier(self, *, nav: float, peak_nav: float) -> float:
        if peak_nav <= 0:
            return 1.0
        drawdown = (nav - peak_nav) / peak_nav
        if drawdown >= -self.config.risk.drawdown_reduction_threshold:
            return 1.0
        excess = abs(drawdown) - self.config.risk.drawdown_reduction_threshold
        reduction = 1 - (excess / self.config.risk.drawdown_reduction_threshold) * self.config.risk.drawdown_max_reduction
        return max(1 - self.config.risk.drawdown_max_reduction, reduction)

    def _risk_for_ticker(
        self,
        *,
        ticker: str,
        portfolio: PortfolioState,
        current_prices: dict[str, float],
        price_frames: dict[str, pd.DataFrame],
        correlation_matrix: pd.DataFrame | None,
        nav: float,
        gross_exposure_pct: float,
        drawdown_multiplier: float,
    ) -> dict[str, Any]:
        frame = price_frames.get(ticker, pd.DataFrame())
        current_price = current_prices.get(ticker, 0.0)
        if frame.empty or current_price <= 0:
            return {
                "remaining_position_limit": 0.0,
                "current_price": current_price,
                "reasoning": {"error": "Missing price history"},
            }

        vol_metrics = calculate_volatility_metrics(frame, self.config.risk.volatility_lookback_days)
        position = portfolio.get_position_dict(ticker)
        current_position_value = abs((position.get("long", 0) - position.get("short", 0)) * current_price)
        avg_correlation = self._average_correlation(ticker, portfolio, correlation_matrix)
        limit = compute_position_limit(
            portfolio_value=nav,
            current_position_value=current_position_value,
            available_cash=portfolio.cash,
            gross_exposure_pct=gross_exposure_pct,
            annualized_volatility=vol_metrics["annualized_volatility"],
            avg_correlation=avg_correlation,
            drawdown_multiplier=drawdown_multiplier,
            config=self.config.risk,
        )
        return {
            "remaining_position_limit": float(limit["remaining_position_limit"]),
            "current_price": float(current_price),
            "volatility_metrics": vol_metrics,
            "correlation_metrics": {"avg_correlation": avg_correlation},
            "reasoning": {
                "portfolio_value": float(nav),
                "current_position_value": float(current_position_value),
                "gross_exposure_pct": float(gross_exposure_pct),
                "drawdown_multiplier": float(drawdown_multiplier),
                "position_limit": float(limit["position_limit"]),
                "combined_limit_pct": float(limit["combined_limit_pct"]),
            },
        }

    def _average_correlation(
        self,
        ticker: str,
        portfolio: PortfolioState,
        correlation_matrix: pd.DataFrame | None,
    ) -> float | None:
        if correlation_matrix is None or ticker not in correlation_matrix.columns:
            return None
        active = [
            name
            for name, position in portfolio.positions.items()
            if name != ticker and abs(position.long - position.short) > 0 and name in correlation_matrix.columns
        ]
        if not active:
            return None
        correlations = correlation_matrix.loc[ticker, active].dropna()
        if correlations.empty:
            return None
        return float(correlations.mean())

    def _vote_config(self) -> VotingConfig:
        agent_weights = {
            model_name: self.config.signal_weights.get(model_name, 1.0)
            for model_name in self.config.signal_models
        }
        base = self.config.voting
        return VotingConfig(
            strong_signal_threshold=base.strong_signal_threshold,
            weak_signal_threshold=base.weak_signal_threshold,
            min_consensus_pct=base.min_consensus_pct,
            min_confidence_to_count=base.min_confidence_to_count,
            position_sizing_mode=base.position_sizing_mode,
            agent_weights=agent_weights,
        )
