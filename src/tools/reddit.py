"""Reddit fetcher — the retail-sentiment / social-buzz lens.

Uses Reddit's official OAuth API (application-only / read-only), so it works
reliably from datacenters (unlike scraping the public JSON, which gets IP-blocked
the same way DuckDuckGo does). Needs free app credentials:

    REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET  (from reddit.com/prefs/apps, "script" app)
    REDDIT_USER_AGENT                        (optional; a descriptive string)

With no credentials it returns [] -> the snapshot stores an empty list -> the agent
abstains. Failures never break a run.
"""

from __future__ import annotations

import logging
import os

import requests

from src.data.models import RedditPost
from src.utils.progress import current_fetch_owner, progress

logger = logging.getLogger(__name__)

# Subreddits where tickers actually get discussed by retail traders.
_SUBREDDITS = ["wallstreetbets", "stocks", "investing", "StockMarket"]
_TOKEN_URL = "https://www.reddit.com/api/v1/access_token"
_OAUTH = "https://oauth.reddit.com"


def _user_agent() -> str:
    return os.getenv("REDDIT_USER_AGENT", "nova-trader/0.1 (retail-sentiment lens)")


def _get_token() -> str | None:
    cid = os.getenv("REDDIT_CLIENT_ID")
    secret = os.getenv("REDDIT_CLIENT_SECRET")
    if not cid or not secret:
        return None
    resp = requests.post(
        _TOKEN_URL,
        auth=(cid, secret),
        data={"grant_type": "client_credentials"},
        headers={"User-Agent": _user_agent()},
        timeout=20,
    )
    resp.raise_for_status()
    return resp.json().get("access_token")


def _top_comments(token: str, permalink_path: str, limit: int = 3) -> list[str]:
    """Pull the top few comments on a post (best-effort; empty on any hiccup)."""
    try:
        resp = requests.get(
            f"{_OAUTH}{permalink_path}.json",
            params={"sort": "top", "limit": limit},
            headers={"Authorization": f"bearer {token}", "User-Agent": _user_agent()},
            timeout=20,
        )
        resp.raise_for_status()
        listings = resp.json()
        # listings[1] is the comment tree; each child has data.body
        comments = listings[1]["data"]["children"] if len(listings) > 1 else []
        out = []
        for c in comments[:limit]:
            body = (c.get("data", {}) or {}).get("body", "")
            if body and body not in ("[deleted]", "[removed]"):
                out.append(body)
        return out
    except Exception:  # comments are a nice-to-have, never fatal
        return []


def get_reddit_posts(ticker: str, *, per_subreddit: int = 5, with_comments: bool = True) -> list[RedditPost]:
    """Search the investing subreddits for recent posts mentioning the ticker."""
    progress.record_fetch(current_fetch_owner.get(), "reddit")
    ticker = ticker.upper()
    try:
        token = _get_token()
    except Exception as exc:
        logger.warning("Reddit auth failed for %s: %s", ticker, exc)
        return []
    if not token:
        return []  # no credentials configured

    posts: list[RedditPost] = []
    headers = {"Authorization": f"bearer {token}", "User-Agent": _user_agent()}
    for sub in _SUBREDDITS:
        try:
            resp = requests.get(
                f"{_OAUTH}/r/{sub}/search",
                params={"q": ticker, "restrict_sr": 1, "sort": "new", "limit": per_subreddit, "t": "month"},
                headers=headers,
                timeout=20,
            )
            resp.raise_for_status()
            for child in resp.json().get("data", {}).get("children", []):
                d = child.get("data", {}) or {}
                permalink = f"https://www.reddit.com{d.get('permalink', '')}" if d.get("permalink") else ""
                comments = _top_comments(token, d.get("permalink", ""), 3) if with_comments and permalink else []
                posts.append(RedditPost(
                    ticker=ticker,
                    subreddit=sub,
                    title=d.get("title", ""),
                    body=d.get("selftext", "") or "",
                    score=int(d.get("score", 0) or 0),
                    num_comments=int(d.get("num_comments", 0) or 0),
                    created_utc=float(d.get("created_utc", 0) or 0),
                    permalink=permalink,
                    top_comments=comments,
                ))
        except Exception as exc:
            logger.warning("Reddit search failed for %s in r/%s: %s", ticker, sub, exc)
            continue
    return posts
