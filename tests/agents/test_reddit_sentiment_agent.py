"""Social sentiment agent — VADER finance-lexicon scoring over Reddit + community feed."""

from datetime import date

from src.agents.reddit_sentiment import run_reddit_sentiment_agent
from src.data.models import RedditPost
from src.schemas.context import RunContext, RunRequest
from src.schemas.portfolio import Portfolio, Position
from src.schemas.views import RedditView


def _ctx() -> RunContext:
    return RunContext(request=RunRequest(
        tickers=["AAPL"], start_date=date(2024, 1, 1), end_date=date(2024, 3, 1),
        portfolio=Portfolio(cash=100_000, margin_requirement=0.5, positions={"AAPL": Position()}),
    ))


def _post(title, body="", score=10, sub="stocks", comments=None):
    return RedditPost(ticker="AAPL", subreddit=sub, title=title, body=body, score=score,
                      num_comments=5, permalink="https://www.reddit.com/r/x/abc", top_comments=comments or [])


def test_abstains_without_posts():
    sig = run_reddit_sentiment_agent(_ctx(), RedditView(ticker="AAPL", posts=[]))
    assert sig.status == "abstained"


def test_bullish_when_crowd_is_bullish():
    posts = [
        _post("AAPL calls printing, moon soon", "strong buy, breakout", score=500),
        _post("Bullish on AAPL, accumulating", "undervalued imo", score=120),
    ]
    sig = run_reddit_sentiment_agent(_ctx(), RedditView(ticker="AAPL", posts=posts))
    assert sig.status == "ok"
    assert sig.direction == "bullish"
    assert sig.web_sources  # cites the threads


def test_bearish_when_crowd_is_bearish():
    posts = [
        _post("AAPL puts, this is going to crash", "overvalued, dump it", score=300),
        _post("Bearish AAPL, bagholders beware", "downgrade incoming", score=80),
    ]
    sig = run_reddit_sentiment_agent(_ctx(), RedditView(ticker="AAPL", posts=posts))
    assert sig.direction == "bearish"


def test_upvotes_weight_the_read():
    # one heavily-upvoted bullish post should outweigh a low-score bearish one
    posts = [
        _post("AAPL moon rocket calls bullish breakout", score=5000),
        _post("AAPL puts crash", score=1),
    ]
    sig = run_reddit_sentiment_agent(_ctx(), RedditView(ticker="AAPL", posts=posts))
    assert sig.direction == "bullish"


def test_community_only_still_signals():
    # no Reddit posts, but the moomoo community feed alone can drive the read
    view = RedditView(ticker="AAPL", posts=[],
                      community_texts=["AAPL to the moon, loading calls", "bullish breakout coming"])
    sig = run_reddit_sentiment_agent(_ctx(), view)
    assert sig.status == "ok"
    assert sig.direction == "bullish"
    assert "community_posts=2" in sig.key_factors


def test_community_counts_into_combined_read():
    # bearish community chatter should temper a mildly bullish reddit read
    posts = [_post("AAPL calls", score=0)]
    view = RedditView(ticker="AAPL", posts=posts,
                      community_texts=["dump this bagholder trap, puts", "rug pull incoming, sell"])
    sig = run_reddit_sentiment_agent(_ctx(), view)
    assert sig.status == "ok"
    assert sig.direction in ("bearish", "neutral")  # community pulled it down from bullish
