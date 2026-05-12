"""Router interfaces for model-backed query understanding."""

from __future__ import annotations

from typing import Protocol

from src.router.schemas import QueryRoute


class QueryRouter(Protocol):
    """Contract implemented by BERT, SLM, or hosted LLM routers."""

    def route(self, query: str) -> QueryRoute:
        """Classify a user query into a structured route."""
        ...
