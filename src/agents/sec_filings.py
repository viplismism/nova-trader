"""SEC filings analyst.

Reads excerpts from the latest 10-K/10-Q and produces a cited signal. The vote is
deterministic; LLM narration can explain it later through the shared explain layer.
"""

from __future__ import annotations

import logging
import re

from src.schemas.context import RunContext
from src.schemas.signals import FilingCitation, Signal
from src.schemas.views import FilingsView
from src.utils.progress import progress


logger = logging.getLogger(__name__)

AGENT_ID = "sec_filings"

_POSITIVE_TERMS = {
    "growth",
    "increase",
    "increased",
    "demand",
    "profitability",
    "margin",
    "cash",
    "liquidity",
    "backlog",
    "recurring",
    "expansion",
    "efficiency",
    "repurchase",
}

_NEGATIVE_TERMS = {
    "risk",
    "risks",
    "decline",
    "decreased",
    "competition",
    "competitive",
    "litigation",
    "regulatory",
    "uncertain",
    "uncertainty",
    "cybersecurity",
    "depend",
    "dependency",
    "concentration",
    "supply",
    "shortage",
    "debt",
    "impairment",
    "adverse",
    "material weakness",
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
    clean = " ".join(text.split())
    if len(clean) <= limit:
        return clean
    return clean[: limit - 1].rstrip() + "..."


def run_sec_filings_agent(ctx: RunContext, view: FilingsView, recorder=None) -> Signal:  # noqa: ARG001
    if not view.excerpts:
        return Signal.abstained(
            agent_id=AGENT_ID,
            ticker=view.ticker,
            reason="No SEC filing excerpts available",
        )

    progress.update_status(AGENT_ID, view.ticker, f"Reading {len(view.excerpts)} SEC excerpts")
    try:
        pos = 0
        neg = 0
        cited: list[str] = []
        filing_sources: list[FilingCitation] = []
        for excerpt in view.excerpts:
            pos += _term_count(excerpt.text, _POSITIVE_TERMS)
            neg += _term_count(excerpt.text, _NEGATIVE_TERMS)
            filing_sources.append(
                FilingCitation(
                    chunk_id=excerpt.chunk_id,
                    form=excerpt.form,
                    fiscal_year=excerpt.fiscal_year,
                    item=excerpt.item,
                    url=excerpt.url,
                    snippet=_snippet(excerpt.text),
                )
            )
            if len(cited) < 5:
                cited.append(
                    f"[{excerpt.chunk_id}] {excerpt.form} {excerpt.fiscal_year} {excerpt.item}: "
                    f"{_snippet(excerpt.text)}"
                )
        total = max(pos + neg, 1)
        score = (pos - neg) / total
        if score >= 0.15:
            direction = "bullish"
        elif score <= -0.15:
            direction = "bearish"
        else:
            direction = "neutral"
        confidence = min(0.82, 0.50 + abs(score) * 0.45 + min(len(view.excerpts), 10) * 0.01)
        reasoning = (
            f"SEC filing excerpts showed {pos} constructive terms vs {neg} risk terms "
            f"across {len(view.excerpts)} cited passages."
        )
        progress.update_status(AGENT_ID, view.ticker, "Done")
        return Signal(
            agent_id=AGENT_ID,
            ticker=view.ticker,
            direction=direction,
            confidence=confidence,
            reasoning=reasoning,
            key_factors=cited,
            filing_sources=filing_sources[:8],
        )
    except Exception as exc:
        logger.exception("sec filings agent failed for %s", view.ticker)
        return Signal.failed(agent_id=AGENT_ID, ticker=view.ticker, error=str(exc))
