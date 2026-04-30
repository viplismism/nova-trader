"""End-to-end checks for the rebuilt core pipeline.

These tests use small synthetic price series and patched data access to verify
that the new analysis service can turn deterministic signals into an actionable
decision without depending on live market data.
"""

from datetime import date, timedelta

from src.core import CoreConfig, PortfolioState, TradingService
from src.data.models import Price


def _price_series(closes: list[float]) -> list[Price]:
    start = date(2025, 1, 1)
    result: list[Price] = []
    for idx, close in enumerate(closes):
        day = (start + timedelta(days=idx)).isoformat()
        result.append(
            Price(
                open=close,
                close=close,
                high=close * 1.01,
                low=close * 0.99,
                volume=1_000_000,
                time=f"{day}T00:00:00Z",
            )
        )
    return result


def test_analyze_returns_buy_for_bullish_signal(monkeypatch):
    closes = [100 + i for i in range(140)]

    monkeypatch.setattr(
        "src.core.pipeline.get_prices",
        lambda *args, **kwargs: _price_series(closes),
    )
    monkeypatch.setattr(
        "src.core.pipeline._SIGNAL_MODELS",
        {
            "factor": lambda *args, **kwargs: {"signal": "bullish", "confidence": 90, "status": "success"},
            "fundamentals": lambda *args, **kwargs: {"signal": "bullish", "confidence": 80, "status": "success"},
            "sentiment": lambda *args, **kwargs: {"signal": "neutral", "confidence": 40, "status": "success"},
        },
    )

    service = TradingService(CoreConfig())
    result = service.analyze(
        tickers=["AAPL"],
        start_date="2025-01-01",
        end_date="2025-05-31",
        portfolio=PortfolioState(cash=100_000),
        execute=False,
    )

    decision = result.reports["AAPL"].decision
    assert decision["action"] == "buy"
    assert decision["quantity"] > 0
    assert result.reports["AAPL"].risk["current_price"] > 0
