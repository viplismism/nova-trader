from datetime import date, datetime, timezone

from src.agents.sec_filings import run_sec_filings_agent
from src.data.models import FilingExcerpt
from src.schemas.context import ModelConfig, RunContext, RunRequest
from src.schemas.portfolio import Portfolio
from src.schemas.views import FilingsView


def _ctx() -> RunContext:
    return RunContext(
        request=RunRequest(
            tickers=["AAPL"],
            start_date=date(2026, 1, 1),
            end_date=date(2026, 6, 1),
            portfolio=Portfolio(cash=100_000.0),
            model=ModelConfig(provider="OpenAI", name="gpt-4.1-mini"),
        ),
        as_of=datetime(2026, 6, 15, tzinfo=timezone.utc),
    )


def test_sec_filings_agent_abstains_without_excerpts():
    signal = run_sec_filings_agent(_ctx(), FilingsView(ticker="AAPL", excerpts=[]))

    assert signal.status == "abstained"
    assert signal.agent_id == "sec_filings"


def test_sec_filings_agent_returns_cited_signal():
    view = FilingsView(
        ticker="AAPL",
        excerpts=[
            FilingExcerpt(
                ticker="AAPL",
                chunk_id="AAPL-10K-0001",
                form="10-K",
                fiscal_year="2025",
                item="Item 1",
                url="https://www.sec.gov/example",
                text="Revenue increased with stronger demand, margin expansion, liquidity, and cash generation.",
            ),
            FilingExcerpt(
                ticker="AAPL",
                chunk_id="AAPL-10K-0042",
                form="10-K",
                fiscal_year="2025",
                item="Item 1A",
                url="https://www.sec.gov/example",
                text="Risk factors include competition, supply uncertainty, regulation, and cybersecurity risk.",
            ),
        ],
    )

    signal = run_sec_filings_agent(_ctx(), view)

    assert signal.status == "ok"
    assert signal.agent_id == "sec_filings"
    assert signal.key_factors[0].startswith("[AAPL-10K-0001]")
    assert "constructive terms" in signal.reasoning
