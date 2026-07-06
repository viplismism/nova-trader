"""Community-feed fetcher for retail chatter on a ticker.

Pulls recent stock-discussion posts from the moomoo community feed (override
with COMMUNITY_FEED_URL) and applies a quality filter so downstream scoring
only sees substantive posts. Fetching and filtering only — sentiment scoring
lives elsewhere. Any failure degrades to an empty list, never an exception.
"""

from __future__ import annotations

import html
import logging
import os
import re

import requests
from pydantic import BaseModel

logger = logging.getLogger(__name__)

_DEFAULT_FEED_URL = "https://ai-news-search.moomoo.com/stock_feed"
_UA = "nova-trader-community/0.1"
_TIMEOUT_SECONDS = 15

_TAG_RE = re.compile(r"<[^>]+>")
# Brand-name / referral spam that dominates the raw feed.
_BRAND_RE = re.compile(
    r"(https?://\S*(?:brightdata|moomoo|futunn)\S*|bright\s*data|moomoo|futunn|futu|opend)",
    re.I,
)
# Exact throwaway phrases that carry no analyzable signal.
_MEME_PHRASES = ("to the moon", "nice", "buy", "sell", "lol", "moon")


class CommunityPost(BaseModel):
    ticker: str
    text: str
    source: str = "moomoo-community"


def _clean(text: str) -> str:
    if not text:
        return ""
    text = _TAG_RE.sub(" ", text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _low_quality(text: str) -> bool:
    if len(text) < 15:
        return True
    if len(re.sub(r"[^a-zA-Z]", "", text)) < 8:
        return True
    if text.lower() in _MEME_PHRASES:
        return True
    if _BRAND_RE.search(text):
        return True
    return False


def get_community_posts(ticker: str, *, limit: int = 30) -> list[CommunityPost]:
    """Fetch quality-filtered community posts for a ticker; empty list on any failure."""
    ticker = (ticker or "").strip().upper()
    if not ticker:
        return []

    feed_url = os.environ.get("COMMUNITY_FEED_URL", _DEFAULT_FEED_URL)
    try:
        response = requests.get(
            feed_url,
            params={"keyword": ticker, "size": limit},
            headers={"User-Agent": _UA},
            timeout=_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:  # noqa: BLE001 - degrade to empty, never raise into the run
        logger.warning("Community feed request failed for %s: %s", ticker, exc)
        return []

    # The feed signals success with code == 0; anything else means no data.
    if not isinstance(payload, dict) or payload.get("code") != 0:
        logger.warning("Community feed returned no data for %s", ticker)
        return []

    posts: list[CommunityPost] = []
    for item in payload.get("data") or []:
        if not isinstance(item, dict):
            continue
        text = _clean(f"{item.get('title', '')} {item.get('desc', '')}")
        if _low_quality(text):
            continue
        posts.append(CommunityPost(ticker=ticker, text=text[:240]))
    return posts
