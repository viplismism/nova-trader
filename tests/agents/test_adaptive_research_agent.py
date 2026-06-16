from datetime import date, datetime, timezone

from src.agents.adaptive_research import run_adaptive_research_agent
from src.data.models import FilingExcerpt, WebSearchResult
from src.schemas.context import ModelConfig, RunContext, RunRequest
from src.schemas.portfolio import Portfolio
from src.schemas.views import AdaptiveResearchView


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


def _view() -> AdaptiveResearchView:
    return AdaptiveResearchView(
        ticker="AAPL",
        metrics=[],
        market_cap=3_000_000_000_000,
        news=[],
        filings=[],
        web_results=[],
    )


def test_adaptive_research_runs_planned_web_and_filing_queries(monkeypatch):
    calls = {"llm": [], "web": [], "filings": []}

    def fake_call_llm(prompt, pydantic_model, **kwargs):
        calls["llm"].append(pydantic_model.__name__)
        if pydantic_model.__name__ == "_ResearchPlan":
            return pydantic_model(
                focus="Check demand and regulatory risk.",
                web_queries=["AAPL demand latest", "AAPL regulation risk"],
                filing_queries=["services demand margin", "regulatory risk competition"],
            )
        return pydantic_model(
            signal="bullish",
            confidence=72,
            reasoning="The retrieved evidence is constructive, with demand offsetting known risk.",
            key_findings=["Demand source supports growth (https://example.com/aapl)", "Filing cites risk but not enough to flip the call (AAPL-10K-0001)"],
        )

    def fake_web(ticker, question="", limit=8):
        calls["web"].append((ticker, question, limit))
        return [
            WebSearchResult(
                ticker=ticker,
                title="Apple demand improves",
                url="https://example.com/aapl",
                snippet="Demand and margins remain strong.",
                source="test",
            )
        ]

    def fake_filings(ticker, queries=None, **kwargs):
        calls["filings"].append((ticker, queries, kwargs))
        return [
            FilingExcerpt(
                ticker=ticker,
                chunk_id="AAPL-10K-0001",
                form="10-K",
                fiscal_year="2025",
                item="Item 1A",
                url="https://sec.example/aapl",
                text="The company discusses competition and regulatory risk.",
            )
        ]

    monkeypatch.setattr("src.agents.adaptive_research.call_llm", fake_call_llm)
    monkeypatch.setattr("src.agents.adaptive_research.get_web_research", fake_web)
    monkeypatch.setattr("src.agents.adaptive_research.get_sec_filing_excerpts", fake_filings)

    signal = run_adaptive_research_agent(_ctx(), _view())

    assert signal.status == "ok"
    assert signal.agent_id == "adaptive_research"
    assert signal.direction == "bullish"
    assert signal.confidence == 0.72
    assert len(calls["web"]) == 2
    assert calls["filings"][0][1] == ["services demand margin", "regulatory risk competition"]
    assert "web_sources=1" in signal.key_factors
    assert "filing_sources=1" in signal.key_factors


def test_adaptive_research_abstains_without_evidence(monkeypatch):
    def fake_call_llm(prompt, pydantic_model, **kwargs):
        if pydantic_model.__name__ == "_ResearchPlan":
            return pydantic_model(
                focus="No evidence plan.",
                web_queries=["nothing"],
                filing_queries=["nothing"],
            )
        raise AssertionError("synthesis should not run without evidence")

    monkeypatch.setattr("src.agents.adaptive_research.call_llm", fake_call_llm)
    monkeypatch.setattr("src.agents.adaptive_research.get_web_research", lambda **kwargs: [])
    monkeypatch.setattr("src.agents.adaptive_research.get_sec_filing_excerpts", lambda **kwargs: [])

    signal = run_adaptive_research_agent(_ctx(), _view())

    assert signal.status == "abstained"
    assert signal.agent_id == "adaptive_research"
