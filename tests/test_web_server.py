from fastapi import HTTPException
from fastapi.testclient import TestClient

from src.web.server import _parse_tickers, app


def test_web_app_imports():
    assert app.title == "AlphaDesk"


def test_methodology_page_is_served():
    response = TestClient(app).get("/methodology")
    assert response.status_code == 200
    assert "AlphaDesk analyst numbers and formulas" in response.text
    assert "Consensus confidence" in response.text


def test_parse_tickers_deduplicates_and_validates():
    assert _parse_tickers("aapl, nvda AAPL") == ["AAPL", "NVDA"]

    try:
        _parse_tickers("AAPL, not/a/ticker")
    except HTTPException as exc:
        assert exc.status_code == 400
    else:
        raise AssertionError("invalid ticker should raise")
