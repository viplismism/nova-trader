"""Reddit / retail-sentiment analyst.

Reads recent posts (and their top comments) about the ticker from the investing
subreddits and turns them into a directional signal. Two ingredients:
  - sentiment: bullish-vs-bearish term balance, weighted by each post's upvotes
    (a heavily-upvoted post reflects more of the crowd than a 1-vote one)
  - buzz:      how much the ticker is being talked about (mention count) — a spike
    is itself informative, and it scales confidence

Deterministic and keyword-based, mirroring news_sentiment. An LLM upgrade can be
layered on later. Cites the top threads as evidence.
"""

from __future__ import annotations

import logging
import math

from src.agents._text import clip as _clip, term_count as _term_count
from src.schemas.context import RunContext
from src.schemas.signals import Signal, WebSourceCitation
from src.schemas.views import RedditView
from src.utils.progress import progress

logger = logging.getLogger(__name__)

AGENT_ID = "reddit_sentiment"

# Retail vocabulary — includes WSB slang, which carries real directional meaning there.
_BULLISH = {
    "buy", "bull", "bullish", "long", "calls", "call", "moon", "rocket", "squeeze",
    "breakout", "rip", "ripping", "tendies", "yolo", "undervalued", "oversold",
    "beat", "beats", "upgrade", "strong", "growth", "rally", "pump", "green",
    "hold", "holding", "diamond", "bagstrong", "accumulate", "support",
}
_BEARISH = {
    "sell", "bear", "bearish", "short", "puts", "put", "dump", "dumping", "crash",
    "overvalued", "overbought", "miss", "misses", "downgrade", "weak", "drop",
    "tank", "tanking", "red", "bagholder", "bagholding", "rugpull", "fraud",
    "lawsuit", "bankrupt", "dilution", "resistance", "topped", "dead",
}


def run_reddit_sentiment_agent(ctx: RunContext, view: RedditView, recorder=None) -> Signal:  # noqa: ARG001
    if not view.posts:
        return Signal.abstained(
            agent_id=AGENT_ID, ticker=view.ticker,
            reason="No Reddit posts found (no credentials, or no recent mentions)",
        )

    progress.update_status(AGENT_ID, view.ticker, f"Reading {len(view.posts)} Reddit posts")
    try:
        bull = 0.0
        bear = 0.0
        total_score = 0
        sources: list[WebSourceCitation] = []
        for post in view.posts:
            text = " ".join([post.title, post.body, *post.top_comments])
            # upvote weight: a post with more net upvotes speaks for more of the crowd.
            weight = 1.0 + math.log1p(max(post.score, 0))
            bull += _term_count(text, _BULLISH) * weight
            bear += _term_count(text, _BEARISH) * weight
            total_score += max(post.score, 0)
            if post.permalink:
                sources.append(WebSourceCitation(
                    title=f"[r/{post.subreddit}] {post.title}"[:140],
                    url=post.permalink,
                    snippet=_clip(post.body or (post.top_comments[0] if post.top_comments else ""), 300),
                ))

        total_terms = max(bull + bear, 1.0)
        score = (bull - bear) / total_terms        # [-1, 1] sentiment lean
        mentions = len(view.posts)

        if score >= 0.15:
            direction = "bullish"
        elif score <= -0.15:
            direction = "bearish"
        else:
            direction = "neutral"

        # Confidence: sentiment strength, nudged up by buzz volume (more mentions =
        # more conviction in the read), capped so a keyword scan never claims certainty.
        buzz_boost = min(mentions, 20) * 0.015
        confidence = min(0.85, 0.45 + abs(score) * 0.40 + buzz_boost)

        reasoning = (
            f"Across {mentions} Reddit posts ({total_score:,} combined upvotes) the "
            f"upvote-weighted term balance was {bull:.0f} bullish vs {bear:.0f} bearish "
            f"(net {score:+.0%}), so the retail read leans {direction}."
        )
        key_factors = [
            f"mentions={mentions}",
            f"combined_upvotes={total_score}",
            f"net_sentiment={score:+.0%}",
        ]
        progress.update_status(AGENT_ID, view.ticker, "Done")
        return Signal(
            agent_id=AGENT_ID,
            ticker=view.ticker,
            direction=direction,
            confidence=confidence,
            reasoning=reasoning,
            key_factors=key_factors,
            web_sources=sources[:8],
        )
    except Exception as exc:
        logger.exception("reddit sentiment agent failed for %s", view.ticker)
        return Signal.failed(agent_id=AGENT_ID, ticker=view.ticker, error=str(exc))
