from datetime import date, datetime, timezone

from src.agents.web_research import run_web_research_agent
from src.data.models import WebSearchResult
from src.schemas.context import ModelConfig, RunContext, RunRequest
from src.schemas.portfolio import Portfolio
from src.schemas.views import WebResearchView


def _ctx() -> RunContext:
    return RunContext(
        request=RunRequest(
            tickers=["AAPL"],
            start_date=date(2026, 1, 1),
            end_date=date(2026, 6, 1),
            portfolio=Portfolio(cash=100_000.0),
            model=ModelConfig(provider="MiniMax", name="MiniMax-M2.7"),
        ),
        as_of=datetime(2026, 6, 15, tzinfo=timezone.utc),
    )


def test_web_research_agent_abstains_without_results():
    signal = run_web_research_agent(_ctx(), WebResearchView(ticker="AAPL", results=[]))

    assert signal.status == "abstained"
    assert signal.agent_id == "web_research"


def test_web_research_agent_returns_sourced_signal():
    view = WebResearchView(
        ticker="AAPL",
        results=[
            WebSearchResult(
                ticker="AAPL",
                title="Apple shares rise after guidance raised",
                url="https://example.com/apple-guidance",
                snippet="Analysts cite strong demand, margin expansion, and record services growth.",
                source="test",
            ),
            WebSearchResult(
                ticker="AAPL",
                title="Regulation risk remains for Apple",
                url="https://example.com/apple-risk",
                snippet="Investors still monitor competition and regulation risk.",
                source="test",
            ),
        ],
    )

    signal = run_web_research_agent(_ctx(), view)

    assert signal.status == "ok"
    assert signal.agent_id == "web_research"
    assert "Live web research" in signal.reasoning
    assert "source_count=2" in signal.key_factors
    assert any("Apple shares rise" in factor for factor in signal.key_factors)
