from src.chat.signal_card import build_signal_cards, clean_reasoning_text, signal_cards_context_text
from src.schemas.signals import (
    Consensus,
    Decisions,
    Limits,
    Recommendation,
    Signal,
    TickerDecision,
    TickerLimit,
)


def _recommendation() -> Recommendation:
    return Recommendation(
        run_id="run-1",
        as_of="2026-06-15T00:00:00Z",
        tickers=["AAPL"],
        signals=[
            Signal(
                agent_id="technical",
                ticker="AAPL",
                direction="bullish",
                confidence=0.74,
                reasoning="Trend is constructive.",
                explain_reasoning='<think>private</think>{"explanation":"The trend and momentum setup is constructive."}',
                key_factors=["price above trend"],
            ),
            Signal(
                agent_id="news_sentiment",
                ticker="AAPL",
                direction="neutral",
                confidence=0.56,
                reasoning="0 positive / 1 negative / 9 neutral headlines",
            ),
        ],
        consensus={
            "AAPL": Consensus(
                ticker="AAPL",
                direction="neutral",
                confidence=0.62,
                weighted_score=0.12,
                bull_count=1,
                bear_count=0,
                neutral_count=1,
            )
        },
        limits=Limits(
            per_ticker={
                "AAPL": TickerLimit(
                    ticker="AAPL",
                    current_price=310.26,
                    max_position_dollars=19282,
                    max_shares=62,
                    annualized_volatility=0.22,
                    correlation_multiplier=1.0,
                )
            }
        ),
        decisions=Decisions(
            per_ticker={
                "AAPL": TickerDecision(
                    ticker="AAPL",
                    action="hold",
                    confidence=0.62,
                    reasoning="Neutral consensus with mixed signals.",
                )
            }
        ),
        summary="AAPL: consensus=neutral (62%), action=hold",
        risk_reasoning="**Risk manager** kept the position inside the volatility budget.",
        portfolio_reasoning="Portfolio manager accepted the hold because signals were mixed.",
    )


def test_clean_reasoning_prefers_explanation_and_removes_think_tokens():
    text = '<think>hidden chain</think>{"explanation":"Visible client-safe paragraph."}'

    cleaned = clean_reasoning_text(text)

    assert cleaned == "Visible client-safe paragraph."
    assert "<think>" not in cleaned


def test_signal_cards_preserve_decision_risk_and_clean_agent_reasoning():
    card = build_signal_cards(_recommendation())[0]

    assert card.ticker == "AAPL"
    assert card.action == "hold"
    assert card.consensus_direction == "neutral"
    assert card.vote_summary == {"bullish": 1, "bearish": 0, "neutral": 1, "abstained": 0, "failed": 0}
    assert card.risk.max_shares == 62
    assert card.agents[0].reasoning == "The trend and momentum setup is constructive."
    assert "<think>" not in card.agents[0].reasoning
    assert "Risk manager kept" in card.risk_reasoning


def test_signal_cards_context_is_grounded_and_contains_no_raw_prompt_noise():
    context = signal_cards_context_text(_recommendation())

    assert "Run: run-1" in context
    assert "AAPL: HOLD at 62%" in context
    assert "Technical Analyst: bullish 74%" in context
    assert "The trend and momentum setup is constructive" in context
    assert "<think>" not in context
    assert "raw model output" not in context.lower()
