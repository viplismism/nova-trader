"""Social sentiment analyst — the retail-crowd lens (Reddit + community feed).

Reads recent Reddit posts (and their top comments) about the ticker plus the
moomoo community feed, and turns them into a directional signal:
  - sentiment: VADER with a finance-tuned lexicon ("moon", "bagholder", "puts"...)
    — plain VADER scores WSB slang as neutral, the overlay is what makes it work
  - weighting: each Reddit post weighted by engagement (1 + upvotes + comments),
    so a 5,000-upvote post speaks for more of the crowd than a 1-vote one
  - buzz: mention volume scales confidence — a spike in chatter is itself signal

Deterministic (no LLM call). Cites the top Reddit threads as evidence.
"""

from __future__ import annotations

import logging

from src.agents._sentiment import engagement_weight, is_scoreable, mood, score_text, weighted_sentiment
from src.agents._text import clip as _clip
from src.schemas.context import RunContext
from src.schemas.signals import Signal, WebSourceCitation
from src.schemas.views import RedditView
from src.utils.progress import progress

logger = logging.getLogger(__name__)

AGENT_ID = "reddit_sentiment"


def run_reddit_sentiment_agent(ctx: RunContext, view: RedditView, recorder=None) -> Signal:  # noqa: ARG001
    if not view.posts and not view.community_texts:
        return Signal.abstained(
            agent_id=AGENT_ID, ticker=view.ticker,
            reason="No social posts found (no credentials, or no recent mentions)",
        )

    n_reddit, n_community = len(view.posts), len(view.community_texts)
    progress.update_status(AGENT_ID, view.ticker, f"Reading {n_reddit} Reddit + {n_community} community posts")
    try:
        pairs: list[tuple[float, float]] = []
        total_upvotes = 0
        sources: list[WebSourceCitation] = []

        for post in view.posts:
            text = " ".join([post.title, post.body, *post.top_comments])
            pairs.append((score_text(text), engagement_weight(post.score, post.num_comments)))
            total_upvotes += max(post.score, 0)
            if post.permalink:
                sources.append(WebSourceCitation(
                    title=f"[r/{post.subreddit}] {post.title}"[:140],
                    url=post.permalink,
                    snippet=_clip(post.body or (post.top_comments[0] if post.top_comments else ""), 300),
                ))

        # Community-feed posts carry no engagement metadata — each counts once.
        # Non-English posts are skipped: VADER reads them as 0.0, which would
        # silently drag the weighted mean toward a false neutral.
        n_scoreable = 0
        for text in view.community_texts:
            if is_scoreable(text):
                pairs.append((score_text(text), 1.0))
                n_scoreable += 1

        if not pairs:
            return Signal.abstained(
                agent_id=AGENT_ID, ticker=view.ticker,
                reason=f"{n_community} community posts found but none scoreable (non-English)",
            )

        combined = weighted_sentiment(pairs)
        label = mood(combined)
        direction = label if label in ("bullish", "bearish") else "neutral"
        net = combined or 0.0

        # Confidence: sentiment strength, nudged up by buzz volume, capped well
        # below certainty — this is a crowd read, not a valuation.
        mentions = n_reddit + n_community
        confidence = min(0.85, 0.45 + abs(net) * 0.40 + min(mentions, 20) * 0.015)

        parts = [f"{n_reddit} Reddit posts ({total_upvotes:,} combined upvotes)"]
        if n_community:
            parts.append(f"{n_community} community-feed posts")
        reasoning = (
            f"Across {' and '.join(parts)}, the engagement-weighted finance-lexicon "
            f"sentiment is {net:+.2f}, so the retail read leans {label}."
        )
        key_factors = [
            f"reddit_mentions={n_reddit}",
            f"community_posts={n_community}",
            f"combined_upvotes={total_upvotes}",
            f"weighted_sentiment={net:+.2f}",
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
        logger.exception("social sentiment agent failed for %s", view.ticker)
        return Signal.failed(agent_id=AGENT_ID, ticker=view.ticker, error=str(exc))
