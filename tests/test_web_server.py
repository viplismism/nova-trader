from fastapi import HTTPException

from src.web.server import _parse_tickers, app


def test_web_app_imports():
    assert app.title == "AlphaDesk"


def test_parse_tickers_deduplicates_and_validates():
    assert _parse_tickers("aapl, nvda AAPL") == ["AAPL", "NVDA"]

    try:
        _parse_tickers("AAPL, not/a/ticker")
    except HTTPException as exc:
        assert exc.status_code == 400
    else:
        raise AssertionError("invalid ticker should raise")
