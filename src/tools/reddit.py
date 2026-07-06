"""Reddit fetcher — the retail-sentiment / social-buzz lens.

Three-tier acquisition, tried in order until one yields posts:

1. Official Reddit OAuth API (application-only / read-only). Fast (seconds) and
   works reliably from datacenters. Needs free app credentials:
       REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET  (reddit.com/prefs/apps, "script" app)
       REDDIT_USER_AGENT                        (optional; a descriptive string)
2. Bright Data managed Reddit scraper (discover-by-keyword) when
   BRIGHTDATA_API_KEY is set. It is an async snapshot API, so the poll is
   hard-bounded (NOVA_BRIGHTDATA_MAX_WAIT seconds, default 90) — this runs
   inside a live analysis and cannot stall the run.
3. A local snapshot cache: every successful fetch (either source) is written to
   <NOVA_SOCIAL_CACHE_DIR or ~/.nova-trader/social-cache>/reddit-<TICKER>.json,
   and read back when both live sources fail. Stale beats nothing.

With no credentials and no cache it returns [] -> the snapshot stores an empty
list -> the agent abstains. Failures never break a run.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import requests

from src.data.models import RedditPost
from src.utils.progress import current_fetch_owner, progress

logger = logging.getLogger(__name__)

# Subreddits where tickers actually get discussed by retail traders.
_SUBREDDITS = ["wallstreetbets", "stocks", "investing", "StockMarket"]
_TOKEN_URL = "https://www.reddit.com/api/v1/access_token"
_OAUTH = "https://oauth.reddit.com"

_BRIGHTDATA_BASE = "https://api.brightdata.com/datasets/v3"
_BRIGHTDATA_POSTS_DATASET = "gd_lvz8ah06191smkebj4"  # Bright Data's Reddit posts dataset


def _user_agent() -> str:
    return os.getenv("REDDIT_USER_AGENT", "nova-trader/0.1 (retail-sentiment lens)")


# --- Tier 1: official Reddit OAuth API -------------------------------------


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


def _fetch_official(ticker: str, per_subreddit: int, with_comments: bool) -> list[RedditPost]:
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


# --- Tier 2: Bright Data managed scraper ------------------------------------


def _brightdata_headers(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def _brightdata_pick(record: dict, *keys: str, default=None):
    for k in keys:
        if k in record and record[k] not in (None, ""):
            return record[k]
    return default


def _brightdata_to_post(ticker: str, record: dict) -> RedditPost:
    created = _brightdata_pick(record, "created_utc")
    if not isinstance(created, (int, float)) or isinstance(created, bool):
        # Bright Data ships ISO timestamps in date_posted; fall back to 0 on junk.
        from datetime import datetime

        try:
            raw = str(_brightdata_pick(record, "date_posted", "date", default="")).replace("Z", "+00:00")
            created = datetime.fromisoformat(raw).timestamp() if raw else 0.0
        except ValueError:
            created = 0.0
    url = str(_brightdata_pick(record, "url", "post_url", "permalink", default="") or "")
    return RedditPost(
        ticker=ticker,
        subreddit=str(_brightdata_pick(record, "subreddit", "community_name", default="") or "").removeprefix("r/"),
        title=str(_brightdata_pick(record, "title", default="") or ""),
        body=str(_brightdata_pick(record, "description", "selftext", "post_text", "body", default="") or ""),
        score=int(_brightdata_pick(record, "num_upvotes", "upvotes", "score", default=0) or 0),
        num_comments=int(_brightdata_pick(record, "num_comments", "comments_count", default=0) or 0),
        created_utc=float(created or 0),
        permalink=url,
        top_comments=[],
    )


def _pending_path(ticker: str) -> Path:
    return _cache_path(ticker).with_name(f"brightdata-pending-{ticker.upper()}.json")


def _load_pending_snapshot(ticker: str) -> str | None:
    """Snapshot id from a previous run still worth harvesting (< 30 min old)."""
    path = _pending_path(ticker)
    try:
        data = json.loads(path.read_text())
        if time.time() - float(data.get("ts", 0)) < 1800:
            return str(data["snapshot_id"])
        path.unlink(missing_ok=True)  # too old — a fresh trigger is better
    except FileNotFoundError:
        pass
    except Exception:
        path.unlink(missing_ok=True)
    return None


def _fetch_brightdata(ticker: str, num_of_posts: int) -> list[RedditPost]:
    key = os.getenv("BRIGHTDATA_API_KEY")
    if not key:
        return []
    max_wait = float(os.getenv("NOVA_BRIGHTDATA_MAX_WAIT", "90"))
    headers = _brightdata_headers(key)

    # Bright Data discovery jobs routinely take minutes — longer than a live run
    # can wait. So: resume a pending snapshot from an earlier run if one exists
    # (that run seeded it; this run harvests it), else trigger a fresh job and
    # persist its id so the NEXT run can harvest even if we time out below.
    snapshot_id = _load_pending_snapshot(ticker)
    if snapshot_id is None:
        try:
            resp = requests.post(
                f"{_BRIGHTDATA_BASE}/trigger",
                params={
                    "dataset_id": _BRIGHTDATA_POSTS_DATASET,
                    "format": "json",
                    "type": "discover_new",
                    "discover_by": "keyword",
                },
                headers=headers,
                json=[{"keyword": ticker, "date": "Past week", "num_of_posts": num_of_posts}],
                timeout=60,
            )
            resp.raise_for_status()
            snapshot_id = resp.json()["snapshot_id"]
            try:
                path = _pending_path(ticker)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({"snapshot_id": snapshot_id, "ts": time.time()}))
            except Exception:  # marker is best-effort; the poll below still runs
                pass
        except Exception as exc:
            logger.warning("Bright Data trigger failed for %s: %s", ticker, exc)
            return []

    poll_every = 5.0
    deadline = time.monotonic() + max_wait
    records: list[dict] = []
    while time.monotonic() < deadline:
        # Bright Data's poll endpoint throws transient 502s and read-timeouts
        # while a snapshot builds — those are retryable, not fatal. Only the
        # deadline ends the loop.
        try:
            out = requests.get(
                f"{_BRIGHTDATA_BASE}/snapshot/{snapshot_id}",
                params={"format": "json"},
                headers=headers,
                timeout=30,
            )
            # 202 = snapshot still building; keep polling until the deadline.
            if out.status_code == 202:
                time.sleep(poll_every)
                continue
            out.raise_for_status()
            data = out.json()
        except Exception as exc:
            logger.warning("Bright Data poll retry for %s: %s", ticker, exc)
            time.sleep(poll_every)
            continue
        if isinstance(data, list):
            records = data
            break
        if data.get("status") in ("running", "building", "collecting"):
            time.sleep(poll_every)
            continue
        records = data.get("data", []) or []
        break
    else:
        # Still building — keep the pending marker so the next run harvests it.
        logger.warning("Bright Data snapshot %s not ready after %ss; will resume next run", snapshot_id, max_wait)

    if records:
        _pending_path(ticker).unlink(missing_ok=True)

    posts = []
    for r in records:
        try:
            posts.append(_brightdata_to_post(ticker, r or {}))
        except Exception:  # one malformed record must not sink the batch
            continue
    return posts


# --- Tier 3: local snapshot cache -------------------------------------------


def _cache_path(ticker: str) -> Path:
    root = os.getenv("NOVA_SOCIAL_CACHE_DIR") or str(Path.home() / ".nova-trader" / "social-cache")
    return Path(root) / f"reddit-{ticker}.json"


def _cache_write(ticker: str, posts: list[RedditPost]) -> None:
    try:
        path = _cache_path(ticker)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps([p.model_dump() for p in posts]), encoding="utf-8")
    except Exception as exc:  # cache is best-effort
        logger.warning("Reddit cache write failed for %s: %s", ticker, exc)


def _cache_read(ticker: str) -> list[RedditPost]:
    try:
        path = _cache_path(ticker)
        if not path.exists():
            return []
        return [RedditPost(**item) for item in json.loads(path.read_text(encoding="utf-8"))]
    except Exception as exc:
        logger.warning("Reddit cache read failed for %s: %s", ticker, exc)
        return []


# --- Public entry point ------------------------------------------------------


def get_reddit_posts(ticker: str, *, per_subreddit: int = 5, with_comments: bool = True) -> list[RedditPost]:
    """Fetch recent Reddit posts mentioning the ticker (official API -> Bright Data -> cache)."""
    progress.record_fetch(current_fetch_owner.get(), "reddit")
    ticker = ticker.upper()

    posts = _fetch_official(ticker, per_subreddit, with_comments)
    if posts:
        _cache_write(ticker, posts)
        return posts

    posts = _fetch_brightdata(ticker, num_of_posts=per_subreddit * len(_SUBREDDITS))
    if posts:
        _cache_write(ticker, posts)
        return posts

    return _cache_read(ticker)  # stale beats nothing; caller can't tell and doesn't need to
