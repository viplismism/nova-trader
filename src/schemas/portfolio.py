"""Portfolio + position schemas.

Matches the runtime shape that main.py / backtester construct,
but typed so risk/portfolio managers can rely on field names.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Position(BaseModel):
    long: int = 0
    short: int = 0
    long_cost_basis: float = 0.0
    short_cost_basis: float = 0.0
    short_margin_used: float = 0.0


class RealizedGains(BaseModel):
    long: float = 0.0
    short: float = 0.0


class Portfolio(BaseModel):
    cash: float
    margin_requirement: float = 0.0
    margin_used: float = 0.0
    positions: dict[str, Position] = Field(default_factory=dict)
    realized_gains: dict[str, RealizedGains] = Field(default_factory=dict)

    @classmethod
    def from_legacy_dict(cls, raw: dict) -> "Portfolio":
        """Bridge from the dict shape that main.py / backtester build today."""
        return cls(
            cash=raw.get("cash", 0.0),
            margin_requirement=raw.get("margin_requirement", 0.0),
            margin_used=raw.get("margin_used", 0.0),
            positions={
                ticker: Position(**pos)
                for ticker, pos in raw.get("positions", {}).items()
            },
            realized_gains={
                ticker: RealizedGains(**rg)
                for ticker, rg in raw.get("realized_gains", {}).items()
            },
        )
