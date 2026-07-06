from src.tools.community import CommunityPost, get_community_posts


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self._payload


def test_get_community_posts_parses_feed_and_filters_junk(monkeypatch):
    calls = []
    payload = {
        "code": 0,
        "data": [
            {"title": "NVDA earnings", "desc": "Data center revenue keeps compounding quarter over quarter."},
            {"title": "", "desc": "lol"},  # too short
            {"title": "", "desc": "!!! $$$ 123 ???"},  # too few letters
            {"title": "to the moon", "desc": ""},  # exact meme phrase
            {"title": "Open an account on moomoo today", "desc": "referral bonus inside"},  # brand spam
            {"title": "<b>Margins</b> look &amp; feel strong", "desc": "guidance raised again this quarter"},
        ],
    }

    def fake_get(url, params, headers, timeout):
        calls.append((url, params, headers, timeout))
        return _FakeResponse(payload)

    monkeypatch.delenv("COMMUNITY_FEED_URL", raising=False)
    monkeypatch.setattr("src.tools.community.requests.get", fake_get)

    posts = get_community_posts("nvda", limit=30)

    url, params, _headers, timeout = calls[0]
    assert url == "https://ai-news-search.moomoo.com/stock_feed"
    assert params == {"keyword": "NVDA", "size": 30}
    assert timeout == 15

    assert [type(p) for p in posts] == [CommunityPost, CommunityPost]
    assert posts[0].ticker == "NVDA"
    assert posts[0].source == "moomoo-community"
    assert "Data center revenue" in posts[0].text
    # HTML tags stripped and entities unescaped
    assert posts[1].text == "Margins look & feel strong guidance raised again this quarter"


def test_get_community_posts_returns_empty_on_http_error(monkeypatch):
    def fake_get(*args, **kwargs):
        raise RuntimeError("connection refused")

    monkeypatch.setattr("src.tools.community.requests.get", fake_get)

    assert get_community_posts("AAPL") == []


def test_get_community_posts_returns_empty_on_bad_payload(monkeypatch):
    monkeypatch.setattr(
        "src.tools.community.requests.get",
        lambda *a, **k: _FakeResponse({"code": 1, "data": []}),
    )

    assert get_community_posts("AAPL") == []


def test_get_community_posts_returns_empty_for_blank_ticker():
    assert get_community_posts("  ") == []
