"""Per-run context carried through every step of the pipeline."""

from __future__ import annotations

import uuid
from datetime import date, datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

from src.schemas.portfolio import Portfolio


class ModelConfig(BaseModel):
    """Which LLM to use for this run."""

    provider: str = "OpenAI"
    name: str = "gpt-4.1"


class RunRequest(BaseModel):
    """What the user / CLI / API asked for."""

    tickers: list[str]
    start_date: date
    end_date: date
    portfolio: Portfolio
    model: ModelConfig = Field(default_factory=ModelConfig)
    portfolio_mode: Literal["research", "long_only", "long_short"] = "research"
    show_reasoning: bool = False
    selected_agents: list[str] = Field(default_factory=list)


class RunContext(BaseModel):
    """Runtime metadata threaded through every agent.

    No mutable state lives here — agents receive views, not the context's data.
    This is for trace IDs, timestamps, config, and the determinism seed.
    """

    run_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    as_of: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    request: RunRequest
    code_sha: str | None = None
    seed: int | None = None  # Passed to OpenAI for reproducibility; defaults derived from run_id.

    def derive_seed(self) -> int:
        """Derive a stable 31-bit seed from the run_id when none was provided."""
        if self.seed is not None:
            return self.seed
        # Stable hash of run_id → 31-bit positive int (fits OpenAI's seed range).
        h = 0
        for c in self.run_id:
            h = (h * 31 + ord(c)) & 0x7FFFFFFF
        return h

    @property
    def start_date(self) -> str:
        return self.request.start_date.isoformat()

    @property
    def end_date(self) -> str:
        return self.request.end_date.isoformat()

    @property
    def tickers(self) -> list[str]:
        return self.request.tickers
