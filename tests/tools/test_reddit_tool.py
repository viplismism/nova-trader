"""Tests for the three-tier Reddit fetcher: official API -> Bright Data -> cache.

All network calls are monkeypatched; nothing here touches the internet.
"""

import json

import pytest

from src.data.models import RedditPost
from src.tools import reddit


class _Resp:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch, tmp_path):
    for var in ("REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET", "BRIGHTDATA_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("NOVA_SOCIAL_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(reddit.time, "sleep", lambda *_: None)
    return tmp_path


def test_official_path_maps_fields(monkeypatch):
    monkeypatch.setenv("REDDIT_CLIENT_ID", "cid")
    monkeypatch.setenv("REDDIT_CLIENT_SECRET", "sec")

    def fake_post(url, **kwargs):
        assert url == reddit._TOKEN_URL
        return _Resp({"access_token": "tok"})

    def fake_get(url, params=None, headers=None, timeout=None):
        if url.endswith("/search"):
            return _Resp({"data": {"children": [{"data": {
                "title": "AAPL to the moon",
                "selftext": "calls printed",
                "score": 42,
                "num_comments": 7,
                "created_utc": 1700000000,
                "permalink": "/r/wallstreetbets/comments/abc/aapl/",
            }}]}})
        # comment fetch
        return _Resp([{}, {"data": {"children": [{"data": {"body": "nice DD"}}]}}])

    monkeypatch.setattr(reddit.requests, "post", fake_post)
    monkeypatch.setattr(reddit.requests, "get", fake_get)

    posts = reddit.get_reddit_posts("aapl", per_subreddit=1)

    assert len(posts) == len(reddit._SUBREDDITS)  # one hit per subreddit
    p = posts[0]
    assert p.ticker == "AAPL"
    assert p.subreddit == "wallstreetbets"
    assert p.title == "AAPL to the moon"
    assert p.body == "calls printed"
    assert p.score == 42
    assert p.num_comments == 7
    assert p.created_utc == 1700000000.0
    assert p.permalink == "https://www.reddit.com/r/wallstreetbets/comments/abc/aapl/"
    assert p.top_comments == ["nice DD"]


def test_official_fails_falls_back_to_brightdata(monkeypatch, _isolated_env):
    monkeypatch.setenv("REDDIT_CLIENT_ID", "cid")
    monkeypatch.setenv("REDDIT_CLIENT_SECRET", "sec")
    monkeypatch.setenv("BRIGHTDATA_API_KEY", "bd-key")

    poll_calls = []

    def fake_post(url, **kwargs):
        if url == reddit._TOKEN_URL:
            return _Resp({}, status_code=500)  # official auth is down
        assert url.endswith("/trigger")
        assert kwargs["params"]["discover_by"] == "keyword"
        assert kwargs["json"][0]["keyword"] == "TSLA"
        return _Resp({"snapshot_id": "snap-1"})

    def fake_get(url, params=None, headers=None, timeout=None):
        assert "snapshot/snap-1" in url
        poll_calls.append(url)
        if len(poll_calls) == 1:
            return _Resp(None, status_code=202)  # still building on first poll
        return _Resp([{
            "subreddit": "r/stocks",
            "title": "TSLA earnings",
            "description": "margins improved",
            "num_upvotes": 10,
            "num_comments": 3,
            "date_posted": "2026-06-01T12:00:00Z",
            "url": "https://www.reddit.com/r/stocks/comments/xyz/",
        }])

    monkeypatch.setattr(reddit.requests, "post", fake_post)
    monkeypatch.setattr(reddit.requests, "get", fake_get)

    posts = reddit.get_reddit_posts("TSLA")

    assert len(poll_calls) == 2
    assert len(posts) == 1
    p = posts[0]
    assert p.subreddit == "stocks"
    assert p.title == "TSLA earnings"
    assert p.body == "margins improved"
    assert p.score == 10
    assert p.num_comments == 3
    assert p.created_utc > 0
    assert p.permalink == "https://www.reddit.com/r/stocks/comments/xyz/"
    assert p.top_comments == []
    # A successful Bright Data fetch also seeds the snapshot cache.
    assert (_isolated_env / "reddit-TSLA.json").exists()


def test_both_sources_fail_reads_snapshot_cache(monkeypatch, _isolated_env):
    monkeypatch.setenv("REDDIT_CLIENT_ID", "cid")
    monkeypatch.setenv("REDDIT_CLIENT_SECRET", "sec")
    monkeypatch.setenv("BRIGHTDATA_API_KEY", "bd-key")

    cached = RedditPost(ticker="NVDA", subreddit="stocks", title="cached post", score=5)
    (_isolated_env / "reddit-NVDA.json").write_text(json.dumps([cached.model_dump()]))

    def fake_post(url, **kwargs):
        return _Resp({}, status_code=500)  # everything live is down

    def fake_get(url, **kwargs):
        return _Resp({}, status_code=500)

    monkeypatch.setattr(reddit.requests, "post", fake_post)
    monkeypatch.setattr(reddit.requests, "get", fake_get)

    posts = reddit.get_reddit_posts("nvda")

    assert len(posts) == 1
    assert posts[0].title == "cached post"
    assert posts[0].score == 5


def test_brightdata_poll_times_out_returns_empty(monkeypatch):
    monkeypatch.setenv("BRIGHTDATA_API_KEY", "bd-key")
    monkeypatch.setenv("NOVA_BRIGHTDATA_MAX_WAIT", "0")  # deadline already passed

    def fake_post(url, **kwargs):
        return _Resp({"snapshot_id": "snap-slow"})

    def fake_get(url, **kwargs):  # pragma: no cover - must never be reached
        raise AssertionError("poll should not run past the deadline")

    monkeypatch.setattr(reddit.requests, "post", fake_post)
    monkeypatch.setattr(reddit.requests, "get", fake_get)

    assert reddit.get_reddit_posts("AMD") == []


def test_no_creds_and_no_cache_returns_empty(monkeypatch):
    def boom(*args, **kwargs):  # pragma: no cover - must never be reached
        raise AssertionError("no network calls expected without credentials")

    monkeypatch.setattr(reddit.requests, "post", boom)
    monkeypatch.setattr(reddit.requests, "get", boom)

    assert reddit.get_reddit_posts("MSFT") == []
