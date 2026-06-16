"""Shared text helpers for the research analysts (web_research, sec_filings,
adaptive_research) — term counting and snippet clipping. Previously duplicated
in each agent."""

from __future__ import annotations

import re


def term_count(text: str, terms: set[str]) -> int:
    """Count term occurrences in text (word-boundary for single words, substring for phrases)."""
    low = (text or "").lower()
    total = 0
    for term in terms:
        if " " in term:
            total += low.count(term)
        else:
            total += len(re.findall(rf"\b{re.escape(term)}\b", low))
    return total


def clip(text: str, limit: int = 200) -> str:
    """Collapse whitespace and truncate to `limit` chars with an ellipsis."""
    clean = " ".join(str(text or "").split())
    if len(clean) <= limit:
        return clean
    return clean[: limit - 1].rstrip() + "..."
