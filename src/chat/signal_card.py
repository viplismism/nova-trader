"""Signal-card artifacts derived from a Recommendation.

The engine remains the source of truth. This module only reshapes the final
Recommendation into a compact, demo/API-friendly object that can be shown in a
browser, used as chat context, or serialized without exposing raw prompts.
"""

from __future__ import annotations

import json
import re

from pydantic import BaseModel, Field

from src.registry import AGENT_REGISTRY
from src.schemas.signals import (
    FilingCitation,
    Recommendation,
    ValuationTarget,
    WebSourceCitation,
)


_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", flags=re.IGNORECASE | re.DOTALL)
_RAW_REASONING_NOTE_RE = re.compile(
    r"raw model output;\s*no dedicated reasoning stream",
    flags=re.IGNORECASE,
)


def _json_object_from_text(text: str) -> dict | None:
    stripped = (text or "").strip()
    if not stripped:
        return None
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped, flags=re.IGNORECASE).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end <= start:
            return None
        try:
            parsed = json.loads(stripped[start : end + 1])
        except json.JSONDecodeError:
            return None
    return parsed if isinstance(parsed, dict) else None


def _clean_markdown(text: str) -> str:
    text = text.replace("**", "").replace("__", "").replace("`", "")
    text = re.sub(r"^\s*[-*]\s+", "", text, flags=re.MULTILINE)
    return text


def clean_reasoning_text(text: str | None, *, limit: int = 900) -> str:
    """Return user-facing reasoning without raw provider wrappers.

    Some providers put hidden thinking inside ``<think>`` tags before the JSON
    payload. We prefer the explicit explanation/reasoning field when present,
    otherwise we remove think blocks and lightweight markdown before display.
    """

    raw = str(text or "").strip()
    if not raw:
        return ""

    parsed = _json_object_from_text(raw)
    if parsed:
        for key in ("explanation", "reasoning", "rationale", "analysis", "summary"):
            value = parsed.get(key)
            if isinstance(value, str) and value.strip():
                raw = value.strip()
                break

    if raw.lower().startswith("<think>") and "</think>" not in raw.lower():
        return ""

    raw = _THINK_BLOCK_RE.sub("", raw)
    raw = _RAW_REASONING_NOTE_RE.sub("", raw)
    raw = raw.replace("<think>", "").replace("</think>", "")
    raw = _clean_markdown(raw)
    raw = " ".join(raw.split())
    if not raw:
        return ""
    if len(raw) > limit:
        return raw[: limit - 1].rstrip() + "..."
    return raw


def _agent_label(agent_id: str) -> str:
    spec = AGENT_REGISTRY.get(agent_id)
    if spec:
        return spec.display_name
    return agent_id.replace("_", " ").title()


class AgentSignalCard(BaseModel):
    agent_id: str
    label: str
    ticker: str
    status: str
    direction: str
    confidence: float
    reasoning: str = ""
    key_factors: list[str] = Field(default_factory=list)
    web_sources: list[WebSourceCitation] = Field(default_factory=list)
    filing_sources: list[FilingCitation] = Field(default_factory=list)
    error: str | None = None


class RiskLimitCard(BaseModel):
    current_price: float | None = None
    max_position_dollars: float | None = None
    max_shares: int | None = None
    annualized_volatility: float | None = None
    correlation_multiplier: float | None = None
    remaining_position_limit: float | None = None


class SignalCard(BaseModel):
    run_id: str
    as_of: str
    ticker: str
    action: str
    action_confidence: float
    quantity: int = 0
    consensus_direction: str
    consensus_confidence: float
    weighted_score: float = 0.0
    # S&P-STARS-style rating + the valuation agent's 12-month price target.
    stars: int = 3
    stars_label: str = "Hold"
    valuation_target: ValuationTarget | None = None
    vote_summary: dict[str, int] = Field(default_factory=dict)
    decision_reasoning: str = ""
    risk: RiskLimitCard = Field(default_factory=RiskLimitCard)
    risk_reasoning: str = ""
    portfolio_reasoning: str = ""
    agents: list[AgentSignalCard] = Field(default_factory=list)
    key_risks: list[str] = Field(default_factory=list)
    what_would_change: list[str] = Field(default_factory=list)
    hedge_pair_id: str | None = None


def _agent_card(signal) -> AgentSignalCard:
    reasoning = (
        clean_reasoning_text(getattr(signal, "explain_reasoning", ""))
        or clean_reasoning_text(getattr(signal, "reasoning", ""))
    )
    error = clean_reasoning_text(getattr(signal, "error", "") or "", limit=240) or None
    factors: list[str] = []
    for factor in getattr(signal, "key_factors", []) or []:
        cleaned = clean_reasoning_text(str(factor), limit=1400)
        if cleaned:
            factors.append(cleaned)
    return AgentSignalCard(
        agent_id=signal.agent_id,
        label=_agent_label(signal.agent_id),
        ticker=signal.ticker,
        status=signal.status,
        direction=signal.direction,
        confidence=float(signal.confidence or 0.0),
        reasoning=reasoning,
        key_factors=factors,
        web_sources=list(getattr(signal, "web_sources", []) or []),
        filing_sources=list(getattr(signal, "filing_sources", []) or []),
        error=error,
    )


def _risk_card(limit) -> RiskLimitCard:
    if limit is None:
        return RiskLimitCard()
    return RiskLimitCard(
        current_price=float(limit.current_price),
        max_position_dollars=float(limit.max_position_dollars),
        max_shares=int(limit.max_shares),
        annualized_volatility=float(limit.annualized_volatility),
        correlation_multiplier=float(limit.correlation_multiplier),
        remaining_position_limit=float(limit.remaining_position_limit),
    )


def _vote_summary(consensus, agents: list[AgentSignalCard]) -> dict[str, int]:
    failed = sum(1 for agent in agents if agent.status == "failed")
    abstained = sum(1 for agent in agents if agent.status == "abstained")
    if consensus is not None:
        return {
            "bullish": int(consensus.bull_count),
            "bearish": int(consensus.bear_count),
            "neutral": int(consensus.neutral_count),
            "abstained": len(consensus.abstained) if getattr(consensus, "abstained", None) is not None else abstained,
            "failed": len(consensus.failed) if getattr(consensus, "failed", None) is not None else failed,
        }
    return {
        "bullish": sum(1 for agent in agents if agent.status == "ok" and agent.direction == "bullish"),
        "bearish": sum(1 for agent in agents if agent.status == "ok" and agent.direction == "bearish"),
        "neutral": sum(1 for agent in agents if agent.status == "ok" and agent.direction == "neutral"),
        "abstained": abstained,
        "failed": failed,
    }


def _key_risks(card: SignalCard) -> list[str]:
    risks: list[str] = []
    failed = [agent.label for agent in card.agents if agent.status == "failed"]
    abstained = [agent.label for agent in card.agents if agent.status == "abstained"]
    bearish = [agent for agent in card.agents if agent.status == "ok" and agent.direction == "bearish"]
    bullish = [agent for agent in card.agents if agent.status == "ok" and agent.direction == "bullish"]

    if failed:
        risks.append(f"{', '.join(failed)} did not complete, so the council had a thinner signal set.")
    if abstained:
        risks.append(f"{', '.join(abstained)} abstained because their available inputs were not strong enough.")
    if card.risk.annualized_volatility is not None and card.risk.annualized_volatility >= 0.45:
        risks.append(f"Annualized volatility is {card.risk.annualized_volatility:.0%}, so sizing is constrained by risk limits.")
    if card.action in {"buy", "cover"} and bearish:
        top = max(bearish, key=lambda agent: agent.confidence)
        risks.append(f"{top.label} is still bearish at {top.confidence:.0%}, which limits conviction.")
    if card.action in {"sell", "short"} and bullish:
        top = max(bullish, key=lambda agent: agent.confidence)
        risks.append(f"{top.label} is still bullish at {top.confidence:.0%}, which is the main opposing read.")
    if not risks:
        risks.append("No major model exception was flagged beyond normal market and position-sizing risk.")
    return risks


def _what_would_change(card: SignalCard) -> list[str]:
    changes: list[str] = []
    action = card.action
    if action == "hold":
        changes.append("A clearer majority across technical, fundamentals, growth, sentiment, and valuation would move this from watchlist to action.")
        changes.append("Risk limits would matter more if the signal strengthens enough to require actual sizing.")
    elif action in {"buy", "cover"}:
        changes.append("The call would weaken if bearish agents overtake the bullish side or consensus confidence falls materially.")
        changes.append("A tighter risk limit, higher volatility, or weaker portfolio fit would reduce the approved size.")
    elif action in {"sell", "short"}:
        changes.append("The call would soften if bullish evidence improves enough to pull consensus back toward neutral.")
        changes.append("Borrow, liquidity, option volatility, and holding period still decide the best execution instrument.")
    else:
        changes.append("The next run should focus on whether the signal mix becomes more directional.")
    return changes


def build_signal_cards(recommendation: Recommendation) -> list[SignalCard]:
    cards: list[SignalCard] = []
    by_ticker: dict[str, list] = {}
    for signal in recommendation.signals:
        by_ticker.setdefault(signal.ticker, []).append(signal)

    for ticker in recommendation.tickers:
        decision = recommendation.decisions.per_ticker.get(ticker)
        consensus = recommendation.consensus.get(ticker)
        if decision is None:
            continue
        ticker_signals = by_ticker.get(ticker, [])
        agents = [_agent_card(signal) for signal in ticker_signals]
        valuation_target = next(
            (s.valuation_target for s in ticker_signals
             if s.agent_id == "valuation" and s.valuation_target is not None),
            None,
        )
        card = SignalCard(
            run_id=recommendation.run_id,
            as_of=recommendation.as_of,
            ticker=ticker,
            action=decision.action,
            action_confidence=float(decision.confidence or 0.0),
            quantity=int(decision.quantity or 0),
            consensus_direction=consensus.direction if consensus is not None else "neutral",
            consensus_confidence=float(consensus.confidence if consensus is not None else decision.confidence or 0.0),
            weighted_score=float(consensus.weighted_score if consensus is not None else 0.0),
            stars=int(consensus.stars if consensus is not None else 3),
            stars_label=consensus.stars_label if consensus is not None else "Hold",
            valuation_target=valuation_target,
            vote_summary=_vote_summary(consensus, agents),
            decision_reasoning=clean_reasoning_text(decision.reasoning, limit=700),
            risk=_risk_card(recommendation.limits.per_ticker.get(ticker)),
            risk_reasoning=clean_reasoning_text(recommendation.risk_reasoning, limit=1000),
            portfolio_reasoning=clean_reasoning_text(recommendation.portfolio_reasoning, limit=1000),
            agents=agents,
            hedge_pair_id=decision.hedge_pair_id,
        )
        card.key_risks = _key_risks(card)
        card.what_would_change = _what_would_change(card)
        cards.append(card)
    return cards


def signal_cards_context_text(recommendation: Recommendation) -> str:
    """Plain context for grounded follow-up Q&A."""

    lines = [
        f"Run: {recommendation.run_id}",
        f"As of: {recommendation.as_of}",
        f"Summary: {recommendation.summary}",
        "",
        "Signal cards:",
    ]
    for card in build_signal_cards(recommendation):
        votes = card.vote_summary
        lines.extend([
            "",
            f"{card.ticker}: {card.action.upper()} at {card.action_confidence:.0%}; consensus {card.consensus_direction.upper()} at {card.consensus_confidence:.0%}; weighted score {card.weighted_score:+.2f}; rating {card.stars}/5 ({card.stars_label}).",
            f"Votes: {votes.get('bullish', 0)} bullish, {votes.get('bearish', 0)} bearish, {votes.get('neutral', 0)} neutral, {votes.get('abstained', 0)} abstained, {votes.get('failed', 0)} failed.",
        ])
        if card.valuation_target is not None:
            vt = card.valuation_target
            dissent = (vt.upside < 0 and card.consensus_direction == "bullish") or \
                      (vt.upside > 0 and card.consensus_direction == "bearish")
            note = (
                " NOTE: this target comes from the valuation analyst alone (a deliberately "
                f"conservative intrinsic-value model) and DISAGREES with the {card.consensus_direction} "
                "consensus — present it as that analyst's dissent, never as the desk's own target."
                if dissent else " (valuation analyst's model)"
            )
            lines.append(
                f"Valuation analyst 12-month target: ${vt.target_price:,.2f} ({vt.upside:+.1%} vs ${vt.current_price:,.2f}); "
                f"intrinsic fair value ${vt.fair_value:,.2f}; cost of equity {vt.cost_of_equity:.1%}.{note}"
            )
        if card.quantity:
            lines.append(f"Approved size: {card.quantity} shares.")
        if card.decision_reasoning:
            lines.append(f"Decision reason: {card.decision_reasoning}")
        if card.risk.current_price is not None:
            lines.append(
                "Risk limit: "
                f"price ${card.risk.current_price:,.2f}; max {card.risk.max_shares} shares "
                f"(${card.risk.max_position_dollars:,.0f}); volatility {card.risk.annualized_volatility:.0%}; "
                f"correlation multiplier {card.risk.correlation_multiplier:.2f}."
            )
        lines.append("Analyst views:")
        for agent in card.agents:
            stance = f"{agent.direction} {agent.confidence:.0%}" if agent.status == "ok" else agent.status
            reason = agent.error or agent.reasoning or "No user-facing reasoning supplied."
            lines.append(f"- {agent.label}: {stance}. {reason}")
            if agent.key_factors:
                lines.append("  Evidence: " + " | ".join(agent.key_factors[:4]))
        if card.risk_reasoning:
            lines.append(f"Risk manager reasoning: {card.risk_reasoning}")
        if card.portfolio_reasoning:
            lines.append(f"Portfolio manager reasoning: {card.portfolio_reasoning}")
        lines.append("Key risks: " + " ".join(card.key_risks))
        lines.append("What would change: " + " ".join(card.what_would_change))
    return "\n".join(lines).strip()


def build_qa_messages(system_prompt: str, question: str, context: str | None = None) -> list[dict[str, str]]:
    """Assemble the follow-up Q&A message list shared by the web /api/ask endpoint and
    the CLI chat loop: a system persona, an optional grounded-context system block, then
    the user question. Callers supply their own persona prompt and pre-built context."""
    messages = [{"role": "system", "content": system_prompt}]
    if context:
        messages.append({"role": "system", "content": context})
    messages.append({"role": "user", "content": question})
    return messages
