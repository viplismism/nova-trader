"""The agent registry — the ONE place that says which agents exist
and what each one is allowed to see.

Adding an agent = adding one line here. Nothing else changes elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Type

from pydantic import BaseModel

from src.schemas.context import RunContext
from src.schemas.signals import Signal
from src.schemas.views import (
    AdaptiveResearchView,
    FilingsView,
    FinancialsView,
    InsiderView,
    NewsSentimentView,
    PersonaView,
    PriceView,
    RedditView,
    WebResearchView,
)
from src.agents.adaptive_research import run_adaptive_research_agent
from src.agents.fundamentals import run_fundamentals_agent
from src.agents.growth import run_growth_agent
from src.agents.insider_sentiment import run_insider_sentiment_agent
from src.agents.news_sentiment import run_news_sentiment_agent
from src.agents.reddit_sentiment import run_reddit_sentiment_agent
from src.agents.sec_filings import run_sec_filings_agent
from src.agents.technical import run_technical_agent
from src.agents.valuation import run_valuation_agent
from src.agents.warren_buffett import run_warren_buffett_agent
from src.agents.web_research import run_web_research_agent


@dataclass(frozen=True)
class AgentSpec:
    """How to run one analyst agent: which view it gets, which function runs it.

    Runner signature: (ctx, view, recorder=None) -> Signal. The recorder is
    optional — pure-math agents ignore it; LLM-using agents pass it to call_llm
    so every prompt/response is captured.
    """

    agent_id: str
    display_name: str
    view_class: Type[BaseModel]
    runner: Callable[..., Signal]


# ── Analyst registry ──
# Add a new agent: append one AgentSpec line. The engine reads from this.
AGENT_REGISTRY: dict[str, AgentSpec] = {
    "technical": AgentSpec(
        agent_id="technical",
        display_name="Technical Analyst",
        view_class=PriceView,
        runner=run_technical_agent,
    ),
    "fundamentals": AgentSpec(
        agent_id="fundamentals",
        display_name="Fundamentals Analyst",
        view_class=FinancialsView,
        runner=run_fundamentals_agent,
    ),
    "growth": AgentSpec(
        agent_id="growth",
        display_name="Growth Analyst",
        view_class=FinancialsView,
        runner=run_growth_agent,
    ),
    "valuation": AgentSpec(
        agent_id="valuation",
        display_name="Valuation Analyst",
        view_class=FinancialsView,
        runner=run_valuation_agent,
    ),
    "news_sentiment": AgentSpec(
        agent_id="news_sentiment",
        display_name="News Sentiment Analyst",
        view_class=NewsSentimentView,
        runner=run_news_sentiment_agent,
    ),
    "web_research": AgentSpec(
        agent_id="web_research",
        display_name="Web Research Analyst",
        view_class=WebResearchView,
        runner=run_web_research_agent,
    ),
    "reddit_sentiment": AgentSpec(
        agent_id="reddit_sentiment",
        display_name="Reddit Sentiment Analyst",
        view_class=RedditView,
        runner=run_reddit_sentiment_agent,
    ),
    "sec_filings": AgentSpec(
        agent_id="sec_filings",
        display_name="SEC Filings Analyst",
        view_class=FilingsView,
        runner=run_sec_filings_agent,
    ),
    "adaptive_research": AgentSpec(
        agent_id="adaptive_research",
        display_name="Adaptive Research Analyst",
        view_class=AdaptiveResearchView,
        runner=run_adaptive_research_agent,
    ),
    "insider_sentiment": AgentSpec(
        agent_id="insider_sentiment",
        display_name="Insider Sentiment Analyst",
        view_class=InsiderView,
        runner=run_insider_sentiment_agent,
    ),
    "warren_buffett": AgentSpec(
        agent_id="warren_buffett",
        display_name="Warren Buffett",
        view_class=PersonaView,
        runner=run_warren_buffett_agent,
    ),
}


def get_agent(agent_id: str) -> AgentSpec:
    if agent_id not in AGENT_REGISTRY:
        raise KeyError(f"Unknown agent_id: {agent_id!r}. Known: {sorted(AGENT_REGISTRY)}")
    return AGENT_REGISTRY[agent_id]


def all_agent_ids() -> list[str]:
    return list(AGENT_REGISTRY.keys())
