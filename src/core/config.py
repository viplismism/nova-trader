"""Small configuration objects for the new core."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.portfolio.construction import VotingConfig
from src.risk import RiskConfig


@dataclass(slots=True)
class CoreConfig:
    """Configuration for the minimal trading core."""

    initial_cash: float = 100_000.0
    execution_mode: str = "dry_run"
    analysis_lookback_months: int = 12
    margin_requirement: float = 0.50
    signal_models: tuple[str, ...] = ("factor", "fundamentals", "sentiment")
    signal_weights: dict[str, float] = field(
        default_factory=lambda: {
            "factor": 1.20,
            "fundamentals": 1.00,
            "sentiment": 0.80,
        }
    )
    voting: VotingConfig = field(default_factory=VotingConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
