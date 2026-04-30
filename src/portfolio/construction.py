"""Deterministic portfolio construction.

This package is the start of the move away from "agents decide portfolio"
toward explicit, testable portfolio logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field


_SIGNAL_MAP = {
    "bullish": 1.0,
    "bearish": -1.0,
    "neutral": 0.0,
}


@dataclass
class VotingConfig:
    """Configuration for the voting algorithm."""

    strong_signal_threshold: float = 0.4
    weak_signal_threshold: float = 0.2
    min_consensus_pct: float = 0.30
    min_confidence_to_count: float = 30.0
    position_sizing_mode: str = "proportional"
    agent_weights: dict[str, float] = field(default_factory=dict)


@dataclass
class VoteResult:
    """Result of voting for a single ticker."""

    ticker: str
    action: str
    quantity: int
    confidence: float
    reasoning: str
    weighted_score: float
    bull_count: int
    bear_count: int
    neutral_count: int
    total_voters: int
    failed_count: int
    signal_details: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "quantity": self.quantity,
            "confidence": round(self.confidence, 1),
            "reasoning": self.reasoning,
            "vote_details": {
                "weighted_score": round(self.weighted_score, 4),
                "bull": self.bull_count,
                "bear": self.bear_count,
                "neutral": self.neutral_count,
                "total": self.total_voters,
                "failed": self.failed_count,
            },
        }


def vote_on_ticker(
    ticker: str,
    ticker_signals: dict[str, dict],
    max_shares: int,
    current_price: float,
    position: dict,
    config: VotingConfig | None = None,
) -> VoteResult:
    """Run deterministic weighted voting for a single ticker."""
    cfg = config or VotingConfig()

    usable_signals = {}
    failed_count = 0
    for agent_id, sig in ticker_signals.items():
        status = sig.get("status", "success")
        if status == "failed":
            failed_count += 1
            continue
        confidence = sig.get("confidence") or 0
        if confidence < cfg.min_confidence_to_count:
            continue
        usable_signals[agent_id] = sig

    if not usable_signals:
        return VoteResult(
            ticker=ticker,
            action="hold",
            quantity=0,
            confidence=0,
            reasoning="No usable signals (all failed or below confidence threshold)",
            weighted_score=0,
            bull_count=0,
            bear_count=0,
            neutral_count=0,
            total_voters=0,
            failed_count=failed_count,
        )

    total_weight = 0.0
    weighted_score = 0.0
    bull_count = 0
    bear_count = 0
    neutral_count = 0
    signal_details = []

    for agent_id, sig in usable_signals.items():
        signal_str = (sig.get("signal") or "neutral").lower()
        confidence = sig.get("confidence") or 50
        conf_norm = confidence / 100.0 if confidence > 1 else confidence

        direction = _SIGNAL_MAP.get(signal_str, 0.0)
        agent_weight = cfg.agent_weights.get(agent_id, 1.0)
        if sig.get("status") == "degraded":
            agent_weight *= 0.7

        weight = agent_weight * conf_norm
        total_weight += agent_weight
        weighted_score += direction * weight

        if direction > 0:
            bull_count += 1
        elif direction < 0:
            bear_count += 1
        else:
            neutral_count += 1

        signal_details.append(
            {
                "agent": agent_id,
                "signal": signal_str,
                "confidence": confidence,
                "weight": round(agent_weight, 2),
                "contribution": round(direction * weight, 4),
            }
        )

    total_voters = bull_count + bear_count + neutral_count
    normalized_score = weighted_score / total_weight if total_weight > 0 else 0

    action = "hold"
    long_shares = int(position.get("long") or 0)
    short_shares = int(position.get("short") or 0)

    bull_pct = bull_count / total_voters if total_voters > 0 else 0
    bear_pct = bear_count / total_voters if total_voters > 0 else 0

    if normalized_score > cfg.strong_signal_threshold and bull_pct >= cfg.min_consensus_pct:
        action = "cover" if short_shares > 0 else "buy"
    elif normalized_score > cfg.weak_signal_threshold and bull_pct >= cfg.min_consensus_pct:
        action = "cover" if short_shares > 0 else "buy"
    elif normalized_score < -cfg.strong_signal_threshold and bear_pct >= cfg.min_consensus_pct:
        action = "sell" if long_shares > 0 else "short"
    elif normalized_score < -cfg.weak_signal_threshold and bear_pct >= cfg.min_consensus_pct:
        action = "sell" if long_shares > 0 else "short"

    quantity = 0
    if action in ("buy", "short"):
        signal_strength = abs(normalized_score)
        if cfg.position_sizing_mode == "proportional":
            quantity = max(1, int(max_shares * signal_strength))
        else:
            quantity = max_shares if max_shares > 0 else 0
        quantity = min(quantity, max_shares)
    elif action == "sell":
        quantity = long_shares
    elif action == "cover":
        quantity = short_shares

    effective_confidence = abs(normalized_score) * 100
    direction_word = "BULLISH" if normalized_score > 0 else "BEARISH" if normalized_score < 0 else "NEUTRAL"
    reasoning = (
        f"{direction_word} consensus: score={normalized_score:+.3f}, "
        f"bulls={bull_count}/{total_voters}, bears={bear_count}/{total_voters}. "
        f"Action: {action} {quantity} shares."
    )
    if failed_count > 0:
        reasoning += f" ({failed_count} agent{'s' if failed_count != 1 else ''} failed)"

    return VoteResult(
        ticker=ticker,
        action=action,
        quantity=quantity,
        confidence=effective_confidence,
        reasoning=reasoning,
        weighted_score=normalized_score,
        bull_count=bull_count,
        bear_count=bear_count,
        neutral_count=neutral_count,
        total_voters=total_voters,
        failed_count=failed_count,
        signal_details=signal_details,
    )
