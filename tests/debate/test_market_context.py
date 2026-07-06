"""Tests for the debate market-context paragraph. No network: get_prices /
get_market_cap are monkeypatched at the src.tools.api module the builder calls."""

import pytest

from src.data.models import Price
from src.debate import market_context
from src.debate.market_context import build_market_context


def _price(close: float, high: float, low: float, day: str) -> Price:
    return Price(open=close, close=close, high=high, low=low, volume=1_000, time=day)


@pytest.fixture(autouse=True)
def _ctx_enabled(monkeypatch):
    monkeypatch.delenv("NOVA_DEBATE_MARKET_CTX", raising=False)


def test_build_market_context_prose(monkeypatch):
    prices = [
        _price(100.0, 105.0, 95.0, "2026-06-15"),
        _price(110.0, 112.0, 99.0, "2026-06-16"),
    ]
    monkeypatch.setattr(market_context._api, "get_prices", lambda t, s, e: prices)
    monkeypatch.setattr(market_context._api, "get_market_cap", lambda t, e: 3.21e12)

    out = build_market_context("NVDA")
    assert "Live market data for NVDA" in out
    assert "last close $110.00" in out
    assert "+10.00%" in out                      # (110-100)/100
    assert "30-day range $95.00–$112.00" in out
    assert "market cap $3.21T" in out


def test_market_cap_failure_keeps_price_context(monkeypatch):
    prices = [_price(50.0, 51.0, 49.0, "2026-06-16")]
    monkeypatch.setattr(market_context._api, "get_prices", lambda t, s, e: prices)
    monkeypatch.setattr(market_context._api, "get_market_cap",
                        lambda t, e: (_ for _ in ()).throw(RuntimeError("api down")))

    out = build_market_context("AAPL")
    assert "last close $50.00" in out
    assert "market cap" not in out


def test_prices_failure_returns_empty(monkeypatch):
    monkeypatch.setattr(market_context._api, "get_prices",
                        lambda t, s, e: (_ for _ in ()).throw(RuntimeError("network")))
    assert build_market_context("NVDA") == ""


def test_no_prices_returns_empty(monkeypatch):
    monkeypatch.setattr(market_context._api, "get_prices", lambda t, s, e: [])
    assert build_market_context("NVDA") == ""


def test_env_kill_switch(monkeypatch):
    monkeypatch.setenv("NOVA_DEBATE_MARKET_CTX", "0")
    # If the kill-switch works, the API is never touched — make sure it would blow up.
    monkeypatch.setattr(market_context._api, "get_prices",
                        lambda t, s, e: (_ for _ in ()).throw(AssertionError("should not fetch")))
    assert build_market_context("NVDA") == ""
