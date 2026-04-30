"""Signal payload helpers for the rebuilt core."""

from dataclasses import dataclass, field
from typing import Any, Literal

SignalStatus = Literal["success", "degraded", "failed"]


@dataclass
class SignalResult:
    """Standardized signal with execution status."""

    signal: str | None
    confidence: float
    status: SignalStatus
    reasoning: dict | str | None = None
    error: str | None = None
    agent_id: str | None = None
    data_sources_used: list[str] = field(default_factory=list)

    @classmethod
    def success(
        cls,
        signal: str,
        confidence: float,
        reasoning: dict | str | None = None,
        agent_id: str | None = None,
        data_sources: list[str] | None = None,
    ) -> "SignalResult":
        return cls(
            signal=signal,
            confidence=confidence,
            status="success",
            reasoning=reasoning,
            agent_id=agent_id,
            data_sources_used=data_sources or [],
        )

    @classmethod
    def degraded(
        cls,
        signal: str,
        confidence: float,
        reasoning: dict | str | None = None,
        error: str | None = None,
        agent_id: str | None = None,
    ) -> "SignalResult":
        return cls(
            signal=signal,
            confidence=max(0, confidence * 0.7),
            status="degraded",
            reasoning=reasoning,
            error=error,
            agent_id=agent_id,
        )

    @classmethod
    def failed(
        cls,
        error: str,
        agent_id: str | None = None,
    ) -> "SignalResult":
        return cls(
            signal=None,
            confidence=0,
            status="failed",
            error=error,
            agent_id=agent_id,
        )

    def to_dict(self) -> dict[str, Any]:
        data = {
            "signal": self.signal or "neutral",
            "confidence": self.confidence,
            "status": self.status,
        }
        if self.reasoning:
            data["reasoning"] = self.reasoning
        if self.error:
            data["error"] = self.error
        if self.data_sources_used:
            data["data_sources"] = self.data_sources_used
        return data

    @property
    def is_usable(self) -> bool:
        return self.status != "failed"


def count_signal_health(signals_by_model: dict[str, dict]) -> dict[str, int]:
    counts = {"success": 0, "degraded": 0, "failed": 0, "total": 0}
    for model_name, ticker_signals in signals_by_model.items():
        if model_name.startswith("risk_management_agent"):
            continue
        for _, signal in ticker_signals.items():
            status = signal.get("status", "success")
            counts[status] = counts.get(status, 0) + 1
            counts["total"] += 1
    return counts


def signals_healthy_enough(signals_by_model: dict[str, dict], max_failure_pct: float = 0.5) -> bool:
    health = count_signal_health(signals_by_model)
    if health["total"] == 0:
        return False
    return (health["failed"] / health["total"]) <= max_failure_pct
