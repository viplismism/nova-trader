from rich.console import Console

from src.chat_cli import ChatSettings, NovaChat, _extract_tickers, _is_analysis_prompt, _recommendation_summary_text, _render_recommendation, _ticker_details_text
from src.schemas.signals import Consensus, Decisions, Recommendation, Signal, TickerDecision


def test_extract_tickers_from_natural_prompt():
    assert _extract_tickers("analyze AAPL and NVDA for me") == ["AAPL", "NVDA"]


def test_extract_tickers_deduplicates_symbols():
    assert _extract_tickers("run TSLA, tsla, MSFT") == ["TSLA", "MSFT"]


def test_explain_setup_is_not_treated_as_ticker_request():
    assert _extract_tickers("explain the setup") == []
    assert not _is_analysis_prompt("explain the setup")


def test_bare_ticker_list_is_analysis_prompt():
    assert _is_analysis_prompt("AAPL,NVDA")


def test_chat_model_command_updates_provider_and_model():
    console = Console(record=True)
    chat = NovaChat(console, ChatSettings(provider="OpenAI", model="gpt-4o-mini"))

    chat.handle("model MiniMax MiniMax-M2.7")

    assert chat.settings.provider == "MiniMax"
    assert chat.settings.model == "MiniMax-M2.7"


def test_chat_agents_command_updates_agent_list():
    console = Console(record=True)
    chat = NovaChat(console, ChatSettings(provider="OpenAI", model="gpt-4o-mini"))

    chat.handle("agents technical,valuation")

    assert chat.settings.agents == ["technical", "valuation"]


def test_show_last_without_run_is_friendly_message():
    console = Console(record=True)
    chat = NovaChat(console, ChatSettings(provider="OpenAI", model="gpt-4o-mini"))

    chat.handle("show last")

    assert "No previous run" in console.export_text()


def test_render_recommendation_uses_ticker_keyed_consensus():
    console = Console(record=True)
    recommendation = Recommendation(
        run_id="test-run",
        as_of="2026-05-31T00:00:00Z",
        tickers=["AAPL"],
        signals=[
            Signal(
                agent_id="technical",
                ticker="AAPL",
                direction="neutral",
                confidence=0.5,
            )
        ],
        consensus={
            "AAPL": Consensus(
                ticker="AAPL",
                direction="neutral",
                confidence=0.5,
                weighted_score=0.0,
            )
        },
        decisions=Decisions(
            per_ticker={
                "AAPL": TickerDecision(
                    ticker="AAPL",
                    action="hold",
                    confidence=0.5,
                    reasoning="Neutral consensus",
                )
            }
        ),
        summary="AAPL: consensus=neutral, action=hold",
    )

    _render_recommendation(console, recommendation)

    rendered = console.export_text()
    assert "AAPL" in rendered
    assert "HOLD" in rendered


def test_recommendation_summary_text_is_plain_transcript_output():
    recommendation = Recommendation(
        run_id="test-run",
        as_of="2026-05-31T00:00:00Z",
        tickers=["AAPL"],
        signals=[
            Signal(
                agent_id="technical",
                ticker="AAPL",
                direction="neutral",
                confidence=0.5,
            )
        ],
        consensus={
            "AAPL": Consensus(
                ticker="AAPL",
                direction="neutral",
                confidence=0.5,
                weighted_score=0.0,
            )
        },
        decisions=Decisions(
            per_ticker={
                "AAPL": TickerDecision(
                    ticker="AAPL",
                    action="hold",
                    confidence=0.5,
                    reasoning="Neutral consensus",
                )
            }
        ),
        summary="AAPL: consensus=neutral, action=hold",
    )

    text = _recommendation_summary_text(recommendation)

    assert "Run test-run" in text
    assert "AAPL" in text
    assert "HOLD" in text
    assert "Analyst reasoning" in text
    assert "technical" in text


def test_ticker_details_text_shows_agent_reasoning():
    recommendation = Recommendation(
        run_id="test-run",
        as_of="2026-05-31T00:00:00Z",
        tickers=["AAPL"],
        signals=[
            Signal(
                agent_id="technical",
                ticker="AAPL",
                direction="bullish",
                confidence=0.75,
                reasoning="Momentum and trend are constructive.",
            )
        ],
        consensus={
            "AAPL": Consensus(
                ticker="AAPL",
                direction="bullish",
                confidence=0.75,
                weighted_score=0.5,
            )
        },
        decisions=Decisions(
            per_ticker={
                "AAPL": TickerDecision(
                    ticker="AAPL",
                    action="hold",
                    confidence=0.75,
                    reasoning="No hedge available.",
                )
            }
        ),
        summary="AAPL: consensus=bullish, action=hold",
    )

    text = _ticker_details_text(recommendation, "AAPL")

    assert "AAPL details" in text
    assert "technical" in text
    assert "Momentum and trend" in text


def test_unknown_chat_prompt_gets_intro_instead_of_ticker_error():
    console = Console(record=True)
    chat = NovaChat(console, ChatSettings(provider="OpenAI", model="gpt-4o-mini"))

    chat.handle("hi")
    chat.handle("what's this is?")

    rendered = console.export_text()
    assert "portfolio-aware recommendation agent" in rendered
