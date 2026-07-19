from fastapi import HTTPException
from fastapi.testclient import TestClient

from src.web.server import _parse_tickers, app


def test_web_app_imports():
    assert app.title == "AlphaDesk"


def test_internal_methodology_is_not_publicly_served():
    client = TestClient(app)
    for path in (
        "/methodology",
        "/methodology/technical",
        "/methodology/audit",
        "/methodology/trust-audit",
    ):
        assert client.get(path).status_code == 404


def test_parse_tickers_deduplicates_and_validates():
    assert _parse_tickers("aapl, nvda AAPL") == ["AAPL", "NVDA"]

    try:
        _parse_tickers("AAPL, not/a/ticker")
    except HTTPException as exc:
        assert exc.status_code == 400
    else:
        raise AssertionError("invalid ticker should raise")
