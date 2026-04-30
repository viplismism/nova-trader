"""Sentiment signal scoring."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import BaseModel as PydanticBaseModel, Field as PydanticField

from src.data.api import get_company_news, get_insider_trades

INSIDER_WEIGHT = 0.30
NEWS_WEIGHT = 0.70


def score_sentiment(ticker: str, end_date: str, api_key: str | None = None) -> dict:
    trades = get_insider_trades(ticker=ticker, end_date=end_date, limit=1000, api_key=api_key) or []
    insider = {"bullish": 0, "bearish": 0, "total": 0, "net_value": 0}
    if trades:
        shares = pd.Series([t.transaction_shares for t in trades]).dropna()
        if not shares.empty:
            sigs = np.where(shares < 0, "bearish", "bullish").tolist()
            insider = {
                "bullish": sigs.count("bullish"),
                "bearish": sigs.count("bearish"),
                "total": len(sigs),
                "net_value": round(sum(t.transaction_value or 0 for t in trades), 2),
            }

    news = get_company_news(ticker=ticker, end_date=end_date, limit=100, api_key=api_key) or []
    news_counts = {"bullish": 0, "bearish": 0, "neutral": 0, "total": 0}
    if news:
        sentiments = pd.Series([n.sentiment for n in news]).dropna()
        if not sentiments.empty:
            mapped = np.where(
                sentiments == "negative",
                "bearish",
                np.where(sentiments == "positive", "bullish", "neutral"),
            ).tolist()
            news_counts = {
                "bullish": mapped.count("bullish"),
                "bearish": mapped.count("bearish"),
                "neutral": mapped.count("neutral"),
                "total": len(mapped),
            }

    bull_w = insider["bullish"] * INSIDER_WEIGHT + news_counts["bullish"] * NEWS_WEIGHT
    bear_w = insider["bearish"] * INSIDER_WEIGHT + news_counts["bearish"] * NEWS_WEIGHT
    overall = "bullish" if bull_w > bear_w else "bearish" if bear_w > bull_w else "neutral"
    total_w = insider["total"] * INSIDER_WEIGHT + news_counts["total"] * NEWS_WEIGHT
    confidence = round(max(bull_w, bear_w) / total_w * 100, 2) if total_w > 0 else 0

    insider_sig = (
        "bullish"
        if insider["bullish"] > insider["bearish"]
        else "bearish"
        if insider["bearish"] > insider["bullish"]
        else "neutral"
    )
    news_sig = (
        "bullish"
        if news_counts["bullish"] > news_counts["bearish"]
        else "bearish"
        if news_counts["bearish"] > news_counts["bullish"]
        else "neutral"
    )

    reasoning = {
        "insider_trading": {
            "signal": insider_sig,
            "confidence": round(max(insider["bullish"], insider["bearish"]) / max(insider["total"], 1) * 100),
            "metrics": {
                "total_trades": insider["total"],
                "bullish": insider["bullish"],
                "bearish": insider["bearish"],
                "net_value": insider["net_value"],
                "weight": INSIDER_WEIGHT,
            },
        },
        "news_sentiment": {
            "signal": news_sig,
            "confidence": round(max(news_counts["bullish"], news_counts["bearish"]) / max(news_counts["total"], 1) * 100),
            "metrics": {
                "total_articles": news_counts["total"],
                "bullish": news_counts["bullish"],
                "bearish": news_counts["bearish"],
                "neutral": news_counts["neutral"],
                "weight": NEWS_WEIGHT,
            },
        },
        "combined_analysis": {
            "total_weighted_bullish": round(bull_w, 1),
            "total_weighted_bearish": round(bear_w, 1),
        },
    }
    status = "success" if (insider["total"] > 0 or news_counts["total"] > 0) else "degraded"
    return {
        "signal": overall,
        "confidence": confidence if status == "success" else 0,
        "reasoning": reasoning,
        "status": status,
        "identity": "sentiment_analyst",
    }


class NewsSentimentSchema(PydanticBaseModel):
    sentiment: str = PydanticField(description="positive, negative, or neutral")
    confidence: int = PydanticField(description="Confidence 0-100")


def score_news_sentiment(
    ticker: str,
    end_date: str,
    call_llm,
    *,
    api_key: str | None = None,
    agent_id: str,
    state=None,
) -> dict:
    company_news = get_company_news(ticker=ticker, end_date=end_date, limit=100, api_key=api_key) or []
    news_signals: list[str] = []
    sentiment_confidences: dict[int, int] = {}
    llm_classified = 0

    if company_news:
        recent = company_news[:10]
        needs_classification = [n for n in recent if n.sentiment is None][:5]
        for article in needs_classification:
            prompt = (
                f"You are a senior news sentiment analyst at a hedge fund. "
                f"Classify this headline's impact on {ticker} stock.\n"
                f"Positive = revenue beats, upgrades, product wins. "
                f"Negative = misses, lawsuits, downgrades, departures. "
                f"Neutral = routine filings, minor updates.\n\n"
                f"Headline: {article.title}\n\n"
                f"Respond with sentiment (positive/negative/neutral) and confidence (0-100)."
            )
            response = call_llm(prompt, NewsSentimentSchema, agent_name=agent_id, state=state)
            if response:
                article.sentiment = response.sentiment.lower()
                sentiment_confidences[id(article)] = response.confidence
            else:
                article.sentiment = "neutral"
                sentiment_confidences[id(article)] = 0
            llm_classified += 1

        sentiments = pd.Series([n.sentiment for n in company_news]).dropna()
        news_signals = np.where(
            sentiments == "negative",
            "bearish",
            np.where(sentiments == "positive", "bullish", "neutral"),
        ).tolist()

    bull = news_signals.count("bullish") if news_signals else 0
    bear = news_signals.count("bearish") if news_signals else 0
    neutral = news_signals.count("neutral") if news_signals else 0
    total = len(news_signals)
    if total == 0:
        return {
            "signal": "neutral",
            "confidence": 0,
            "reasoning": {"news_sentiment": {"signal": "neutral", "confidence": 0, "metrics": {"total_articles": 0}}},
            "status": "degraded",
        }

    overall = "bullish" if bull > bear else "bearish" if bear > bull else "neutral"
    proportion_conf = max(bull, bear) / total * 100
    if sentiment_confidences:
        matching = [
            article
            for article in company_news
            if id(article) in sentiment_confidences
            and (
                (overall == "bullish" and article.sentiment == "positive")
                or (overall == "bearish" and article.sentiment == "negative")
                or (overall == "neutral" and article.sentiment == "neutral")
            )
        ]
        llm_confs = [sentiment_confidences[id(article)] for article in matching]
        confidence = round(0.7 * (sum(llm_confs) / len(llm_confs)) + 0.3 * proportion_conf, 2) if llm_confs else round(proportion_conf, 2)
    else:
        confidence = round(proportion_conf, 2)

    return {
        "signal": overall,
        "confidence": confidence,
        "reasoning": {
            "news_sentiment": {
                "signal": overall,
                "confidence": confidence,
                "metrics": {
                    "total_articles": total,
                    "bullish": bull,
                    "bearish": bear,
                    "neutral": neutral,
                    "llm_classified": llm_classified,
                },
            }
        },
    }
