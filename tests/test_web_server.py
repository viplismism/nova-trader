from fastapi import HTTPException
from fastapi.testclient import TestClient

from src.web.server import _parse_tickers, app


def test_web_app_imports():
    assert app.title == "AlphaDesk"


def test_methodology_page_is_served():
    client = TestClient(app)
    simple = client.get("/methodology")
    audit = client.get("/methodology/audit")
    trust = client.get("/methodology/trust-audit")
    assert simple.status_code == audit.status_code == trust.status_code == 200
    assert "the simple version" in simple.text
    assert "AlphaDesk analyst numbers and formulas" in audit.text
    assert "AlphaDesk analyst trust audit" in trust.text


def test_parse_tickers_deduplicates_and_validates():
    assert _parse_tickers("aapl, nvda AAPL") == ["AAPL", "NVDA"]

    try:
        _parse_tickers("AAPL, not/a/ticker")
    except HTTPException as exc:
        assert exc.status_code == 400
    else:
        raise AssertionError("invalid ticker should raise")
