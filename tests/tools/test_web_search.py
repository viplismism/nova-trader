from src.tools.web_search import get_web_research


class _FakeResponse:
    text = """
    <html>
      <body>
        <a class="result__a" href="https://duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fapple">Apple stock news</a>
        <div class="result__snippet">Apple reports strong demand and margin expansion.</div>
      </body>
    </html>
    """

    def raise_for_status(self) -> None:
        return None


def test_get_web_research_uses_duckduckgo_fallback(monkeypatch):
    calls = []

    def fake_get(url, params, headers, timeout):
        calls.append((url, params, headers, timeout))
        return _FakeResponse()

    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.setattr("src.tools.web_search.requests.get", fake_get)

    results = get_web_research("aapl", limit=3)

    assert calls
    assert results[0].ticker == "AAPL"
    assert results[0].title == "Apple stock news"
    assert results[0].url == "https://example.com/apple"
    assert results[0].source == "duckduckgo"


def test_get_web_research_falls_back_when_tavily_errors(monkeypatch):
    def fake_post(*args, **kwargs):
        raise RuntimeError("bad search key")

    def fake_get(url, params, headers, timeout):
        return _FakeResponse()

    monkeypatch.setenv("TAVILY_API_KEY", "configured-but-bad")
    monkeypatch.setattr("src.tools.web_search.requests.post", fake_post)
    monkeypatch.setattr("src.tools.web_search.requests.get", fake_get)

    results = get_web_research("AAPL", limit=3)

    assert results
    assert results[0].source == "duckduckgo"
