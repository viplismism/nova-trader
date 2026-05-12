"""Typed query routing contracts.

The router is the product boundary between natural-language user questions
and the deterministic ETL/agent pipeline. Keep these schemas model-agnostic:
a rule-based router, BERT classifier, or small local LLM should all return
the same QueryRoute shape.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, field_validator


class QueryIntent(str, Enum):
    SINGLE_STOCK_RECOMMENDATION = "single_stock_recommendation"
    PORTFOLIO_REVIEW = "portfolio_review"
    RISK_EXPOSURE = "risk_exposure"
    BACKTEST_REQUEST = "backtest_request"
    NEWS_SENTIMENT = "news_sentiment"
    VALUATION = "valuation"
    TECHNICAL_ANALYSIS = "technical_analysis"
    COMPARE_ASSETS = "compare_assets"
    EXPLAIN_RECOMMENDATION = "explain_recommendation"
    EXECUTION_READINESS = "execution_readiness"
    GENERAL_RESEARCH = "general_research"


class TimeHorizon(str, Enum):
    INTRADAY = "intraday"
    SHORT_TERM = "short_term"
    MEDIUM_TERM = "medium_term"
    LONG_TERM = "long_term"
    UNSPECIFIED = "unspecified"


class DataModule(str, Enum):
    PRICES = "prices"
    TECHNICALS = "technicals"
    FUNDAMENTALS = "fundamentals"
    VALUATION = "valuation"
    NEWS = "news"
    SENTIMENT = "sentiment"
    INSIDER_TRADES = "insider_trades"
    PORTFOLIO = "portfolio"
    RISK = "risk"
    BACKTEST = "backtest"
    EXECUTION = "execution"


class QueryRoute(BaseModel):
    """Structured interpretation of a user query."""

    raw_query: str
    intent: QueryIntent
    tickers: list[str] = Field(default_factory=list)
    horizon: TimeHorizon = TimeHorizon.UNSPECIFIED
    required_modules: list[DataModule] = Field(default_factory=list)
    needs_portfolio: bool = False
    answer_style: str = "recommendation"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    rationale: str = ""

    @field_validator("tickers")
    @classmethod
    def normalize_tickers(cls, tickers: list[str]) -> list[str]:
        seen: set[str] = set()
        normalized: list[str] = []
        for ticker in tickers:
            clean = ticker.strip().upper()
            if not clean or clean in seen:
                continue
            seen.add(clean)
            normalized.append(clean)
        return normalized

    @field_validator("required_modules")
    @classmethod
    def dedupe_modules(cls, modules: list[DataModule]) -> list[DataModule]:
        seen: set[DataModule] = set()
        result: list[DataModule] = []
        for module in modules:
            if module in seen:
                continue
            seen.add(module)
            result.append(module)
        return result
