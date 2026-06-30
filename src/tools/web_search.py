"""Lightweight web research fetcher for the demo council.

Uses Tavily when a key is configured. Otherwise falls back to DuckDuckGo's
HTML endpoint so the MVP can still perform real web lookups without adding a
paid search dependency. Failures bubble to snapshot, where they become an empty
view and an explicit agent abstention.
"""

from __future__ import annotations

import html
import logging
import os
import re
from html.parser import HTMLParser
from urllib.parse import parse_qs, unquote, urlparse

import requests

from src.data.models import WebSearchResult
from src.utils.progress import current_fetch_owner, progress

logger = logging.getLogger(__name__)


class _DDGParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.results: list[dict[str, str]] = []
        self._in_link = False
        self._in_snippet = False
        self._current: dict[str, str] | None = None
        self._text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_d = {k: v or "" for k, v in attrs}
        klass = attrs_d.get("class", "")
        if tag == "a" and "result__a" in klass:
            self._in_link = True
            self._text = []
            self._current = {"url": _normalize_ddg_url(attrs_d.get("href", ""))}
        elif "result__snippet" in klass:
            self._in_snippet = True
            self._text = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._in_link:
            if self._current is not None:
                self._current["title"] = _clean_text(" ".join(self._text))
            self._in_link = False
            self._text = []
        elif self._in_snippet and tag in {"a", "div"}:
            if self._current is not None:
                self._current["snippet"] = _clean_text(" ".join(self._text))
                if self._current.get("title") and self._current.get("url"):
                    self.results.append(self._current)
            self._current = None
            self._in_snippet = False
            self._text = []

    def handle_data(self, data: str) -> None:
        if self._in_link or self._in_snippet:
            self._text.append(data)


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(text or "")).strip()


def _normalize_ddg_url(url: str) -> str:
    if not url:
        return ""
    parsed = urlparse(url)
    if "duckduckgo.com" in parsed.netloc and parsed.query:
        uddg = parse_qs(parsed.query).get("uddg")
        if uddg:
            return unquote(uddg[0])
    return url


def _search_tavily(query: str, ticker: str, limit: int) -> list[WebSearchResult]:
    key = os.getenv("TAVILY_API_KEY")
    if not key:
        return []
    response = requests.post(
        "https://api.tavily.com/search",
        json={
            "api_key": key,
            "query": query,
            "search_depth": "basic",
            "max_results": limit,
            "include_answer": False,
            "include_raw_content": False,
        },
        timeout=20,
    )
    response.raise_for_status()
    payload = response.json()
    out: list[WebSearchResult] = []
    for item in payload.get("results", [])[:limit]:
        out.append(
            WebSearchResult(
                ticker=ticker,
                title=_clean_text(item.get("title", "")),
                url=str(item.get("url", "")),
                snippet=_clean_text(item.get("content", "")),
                source="tavily",
            )
        )
    return [item for item in out if item.title and item.url]


def _extract_json_array(text: str) -> list[dict]:
    """Pull a JSON array out of model text (tolerates ```json fences / surrounding prose)."""
    import json
    s = (text or "").strip()
    start, end = s.find("["), s.rfind("]")
    if start == -1 or end <= start:
        return []
    try:
        data = json.loads(s[start : end + 1])
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def _search_claude_native(query: str, ticker: str, limit: int) -> list[WebSearchResult]:
    """Use Claude's own server-side web_search tool — reliable from datacenters (no
    third-party key, unlike the DuckDuckGo scrape which gets IP-blocked on cloud hosts).
    Claude searches, then returns the results as JSON we parse into WebSearchResult."""
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        return []
    from anthropic import Anthropic

    client = Anthropic(api_key=key)
    model = os.getenv("NOVA_WEB_SEARCH_MODEL", "claude-haiku-4-5")
    # allowed_callers=["direct"] lets non-programmatic-tool models (e.g. Haiku) use the tool
    tool = {"type": "web_search_20260209", "name": "web_search", "max_uses": 3, "allowed_callers": ["direct"]}
    prompt = (
        f"Search the web for recent, relevant information about: {query}\n\n"
        f"Then return ONLY a JSON array of up to {limit} objects, each exactly "
        '{"title": "...", "url": "https://...", "snippet": "1-2 sentence summary"} '
        "using the real titles and URLs you found. Output nothing except the JSON array."
    )
    messages = [{"role": "user", "content": prompt}]
    resp = None
    for _ in range(4):  # web_search can return pause_turn; continue until it settles
        resp = client.messages.create(model=model, max_tokens=2000, tools=[tool], messages=messages)
        if resp.stop_reason == "pause_turn":
            messages.append({"role": "assistant", "content": resp.content})
            continue
        break
    text = "".join(getattr(b, "text", "") for b in (resp.content if resp else []) if getattr(b, "type", "") == "text")
    out: list[WebSearchResult] = []
    seen: set[str] = set()
    for item in _extract_json_array(text)[:limit]:
        url = str(item.get("url", "")) if isinstance(item, dict) else ""
        if not url or url in seen:
            continue
        seen.add(url)
        out.append(WebSearchResult(
            ticker=ticker,
            title=_clean_text(item.get("title", "")),
            url=url,
            snippet=_clean_text(item.get("snippet", "")),
            source="claude",
        ))
    return [item for item in out if item.title and item.url]


def _search_duckduckgo(query: str, ticker: str, limit: int) -> list[WebSearchResult]:
    response = requests.get(
        "https://duckduckgo.com/html/",
        params={"q": query},
        headers={"User-Agent": os.getenv("NOVA_WEB_USER_AGENT", "nova-trader/0.1")},
        timeout=20,
    )
    response.raise_for_status()
    parser = _DDGParser()
    parser.feed(response.text)
    out: list[WebSearchResult] = []
    seen: set[str] = set()
    for item in parser.results:
        url = item.get("url", "")
        if not url or url in seen:
            continue
        seen.add(url)
        out.append(
            WebSearchResult(
                ticker=ticker,
                title=item.get("title", ""),
                url=url,
                snippet=item.get("snippet", ""),
                source="duckduckgo",
            )
        )
        if len(out) >= limit:
            break
    return out


def get_web_research(ticker: str, *, question: str = "", limit: int = 8) -> list[WebSearchResult]:
    progress.record_fetch(current_fetch_owner.get(), "web_research")
    ticker = ticker.upper()
    query = " ".join(
        part
        for part in (
            ticker,
            "stock latest earnings guidance analyst rating risk news",
            question,
        )
        if part
    )
    try:
        tavily = _search_tavily(query, ticker, limit)
    except Exception as exc:
        logger.warning("Tavily web search failed for %s: %s", ticker, exc)
        tavily = []
    if tavily:
        return tavily
    # Tavily (if keyed) → Claude native search (reliable on cloud) → DuckDuckGo (last resort).
    try:
        native = _search_claude_native(query, ticker, limit)
    except Exception as exc:
        logger.warning("Claude web search failed for %s: %s", ticker, exc)
        native = []
    if native:
        return native
    try:
        return _search_duckduckgo(query, ticker, limit)
    except Exception as exc:
        logger.warning("DuckDuckGo web search failed for %s: %s", ticker, exc)
        return []
