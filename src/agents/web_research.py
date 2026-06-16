"""Live web research analyst.

Reads search snippets collected during snapshot construction and turns them into
a transparent directional signal. The vote is deterministic; the shared explain
layer can narrate the "why" afterward without changing the verdict.
"""

from __future__ import annotations

import logging
import re

from src.schemas.context import RunContext
from src.schemas.signals import Signal, WebSourceCitation
from src.schemas.views import WebResearchView
from src.utils.progress import progress

logger = logging.getLogger(__name__)

AGENT_ID = "web_research"

_CONSTRUCTIVE_TERMS = {
    "beat",
    "beats",
    "bullish",
    "demand",
    "expansion",
    "growth",
    "guidance raised",
    "margin",
    "outperform",
    "profit",
    "record",
    "raised",
    "strong",
    "upgrade",
}

_RISK_TERMS = {
    "bearish",
    "competition",
    "cut",
    "decline",
    "downgrade",
    "investigation",
    "lawsuit",
    "miss",
    "misses",
    "pressure",
    "regulation",
    "risk",
    "slowdown",
    "weak",
}


def _term_count(text: str, terms: set[str]) -> int:
    low = text.lower()
    total = 0
    for term in terms:
        if " " in term:
            total += low.count(term)
        else:
            total += len(re.findall(rf"\b{re.escape(term)}\b", low))
    return total


def _snippet(text: str, limit: int = 190) -> str:
    clean = " ".join((text or "").split())
    if len(clean) <= limit:
        return clean
    return clean[: limit - 1].rstrip() + "..."


def run_web_research_agent(ctx: RunContext, view: WebResearchView, recorder=None) -> Signal:  # noqa: ARG001
    if not view.results:
        return Signal.abstained(
            agent_id=AGENT_ID,
            ticker=view.ticker,
            reason="No live web research results available",
        )

    progress.update_status(AGENT_ID, view.ticker, f"Reading {len(view.results)} web results")
    try:
        constructive = 0
        risk = 0
        evidence: list[str] = []
        web_sources: list[WebSourceCitation] = []
        for result in view.results:
            text = f"{result.title} {result.snippet}"
            constructive += _term_count(text, _CONSTRUCTIVE_TERMS)
            risk += _term_count(text, _RISK_TERMS)
            if result.url:
                web_sources.append(
                    WebSourceCitation(
                        title=result.title or result.url,
                        url=result.url,
                        snippet=_snippet(result.snippet),
                    )
                )
            if len(evidence) < 5:
                snippet = _snippet(result.snippet)
                if snippet:
                    evidence.append(f"{result.title}: {snippet} ({result.url})")
                else:
                    evidence.append(f"{result.title} ({result.url})")

        total_terms = max(constructive + risk, 1)
        score = (constructive - risk) / total_terms
        if score >= 0.15:
            direction = "bullish"
        elif score <= -0.15:
            direction = "bearish"
        else:
            direction = "neutral"

        confidence = min(0.84, 0.50 + abs(score) * 0.38 + min(len(view.results), 8) * 0.01)
        reasoning = (
            f"Live web research showed {constructive} constructive terms vs {risk} risk terms "
            f"across {len(view.results)} sourced search results."
        )
        progress.update_status(AGENT_ID, view.ticker, "Done")
        return Signal(
            agent_id=AGENT_ID,
            ticker=view.ticker,
            direction=direction,
            confidence=confidence,
            reasoning=reasoning,
            key_factors=[
                f"constructive_terms={constructive}",
                f"risk_terms={risk}",
                f"source_count={len(view.results)}",
                *evidence,
            ],
            web_sources=web_sources[:8],
        )
    except Exception as exc:
        logger.exception("web research agent failed for %s", view.ticker)
        return Signal.failed(agent_id=AGENT_ID, ticker=view.ticker, error=str(exc))
