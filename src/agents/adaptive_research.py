"""Tool-use-style adaptive research analyst.

This agent mirrors the useful part of the hedge-fund demo without depending on
Anthropic hosted tools. MiniMax plans what to investigate; Nova executes Tavily /
DuckDuckGo and SEC filing searches locally; MiniMax then synthesizes a structured
signal from those retrieved facts.
"""

from __future__ import annotations

import logging
from typing import Literal

from pydantic import BaseModel, Field

from src.data.models import FilingExcerpt, WebSearchResult
from src.schemas.context import RunContext
from src.schemas.signals import FilingCitation, Signal, WebSourceCitation
from src.schemas.views import AdaptiveResearchView
from src.tools.sec_filings import get_sec_filing_excerpts
from src.tools.web_search import get_web_research
from src.utils.llm import call_llm
from src.utils.progress import current_fetch_owner, progress

logger = logging.getLogger(__name__)

AGENT_ID = "adaptive_research"


class _ResearchPlan(BaseModel):
    focus: str = Field(default="", description="One sentence describing the research angle.")
    web_queries: list[str] = Field(default_factory=list, description="2-3 live web search queries.")
    filing_queries: list[str] = Field(default_factory=list, description="2-3 SEC filing passage queries.")


class _ResearchMemo(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"] = "neutral"
    confidence: int = Field(default=50, description="0-100")
    reasoning: str = Field(default="", description="2-4 paragraph-free sentences explaining the signal.")
    key_findings: list[str] = Field(default_factory=list)


def _state(ctx: RunContext) -> dict:
    return {
        "metadata": {
            "model_name": ctx.request.model.name,
            "model_provider": ctx.request.model.provider,
        }
    }


def _default_plan(ticker: str) -> _ResearchPlan:
    return _ResearchPlan(
        focus=f"Check whether fresh market evidence and filing risk disclosures support the current {ticker} setup.",
        web_queries=[
            f"{ticker} latest earnings guidance analyst rating risks",
            f"{ticker} stock recent news demand margins competition",
        ],
        filing_queries=[
            "revenue growth margin demand outlook",
            "risk factors competition regulation customer concentration",
        ],
    )


def _clip(text: str, limit: int = 420) -> str:
    clean = " ".join(str(text or "").split())
    if len(clean) <= limit:
        return clean
    return clean[: limit - 1].rstrip() + "..."


def _dedupe_web(items: list[WebSearchResult], limit: int = 8) -> list[WebSearchResult]:
    out: list[WebSearchResult] = []
    seen: set[str] = set()
    for item in items:
        key = item.url or item.title
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(item)
        if len(out) >= limit:
            break
    return out


def _dedupe_filings(items: list[FilingExcerpt], limit: int = 8) -> list[FilingExcerpt]:
    out: list[FilingExcerpt] = []
    seen: set[str] = set()
    for item in items:
        key = item.chunk_id
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(item)
        if len(out) >= limit:
            break
    return out


def _headline_context(view: AdaptiveResearchView) -> str:
    lines = []
    for item in view.news[:8]:
        title = getattr(item, "title", "") or ""
        source = getattr(item, "source", "") or ""
        if title:
            lines.append(f"- {title} ({source})")
    return "\n".join(lines) or "- no news headlines in snapshot"


def _plan_prompt(view: AdaptiveResearchView) -> list[dict[str, str]]:
    system = (
        "You are a hedge-fund research supervisor. Create a tiny research plan for one ticker. "
        "Return only JSON. Keep the queries specific and finance-focused."
    )
    user = (
        f"Ticker: {view.ticker}\n"
        f"Market cap: {view.market_cap}\n"
        f"Snapshot counts: metrics={len(view.metrics)}, news={len(view.news)}, "
        f"prefetched_web={len(view.web_results)}, prefetched_filings={len(view.filings)}\n"
        f"Recent headlines:\n{_headline_context(view)}\n\n"
        "Return JSON with fields: focus, web_queries, filing_queries. "
        "Use 2 web_queries and 2 filing_queries."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _web_lines(results: list[WebSearchResult]) -> str:
    if not results:
        return "- no live web results"
    return "\n".join(
        f"- {item.title}: {_clip(item.snippet, 260)} | source={item.url}"
        for item in results[:8]
    )


def _filing_lines(excerpts: list[FilingExcerpt]) -> str:
    if not excerpts:
        return "- no SEC filing excerpts"
    return "\n".join(
        f"- [{item.chunk_id}] {item.form} {item.fiscal_year} {item.item}: {_clip(item.text, 300)}"
        for item in excerpts[:8]
    )


def _synthesis_prompt(
    view: AdaptiveResearchView,
    plan: _ResearchPlan,
    web_results: list[WebSearchResult],
    filing_excerpts: list[FilingExcerpt],
) -> list[dict[str, str]]:
    system = (
        "You are an adaptive equity research analyst. You must base your signal only on the "
        "retrieved web results and filing excerpts supplied below. Return only JSON.\n"
        "Rules: do not invent facts, cite source URLs or filing chunk IDs inside key_findings, "
        "and use signal bullish/bearish/neutral with confidence 0-100."
    )
    user = (
        f"Ticker: {view.ticker}\n"
        f"Research focus: {plan.focus}\n"
        f"Web queries run: {plan.web_queries}\n"
        f"Filing queries run: {plan.filing_queries}\n\n"
        f"WEB RESULTS:\n{_web_lines(web_results)}\n\n"
        f"SEC FILING EXCERPTS:\n{_filing_lines(filing_excerpts)}\n\n"
        "Return JSON with fields: signal, confidence, reasoning, key_findings. "
        "Reasoning should be 2-4 concise sentences. key_findings should be 3-6 concise cited bullets."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _fallback_memo(web_results: list[WebSearchResult], filing_excerpts: list[FilingExcerpt]) -> _ResearchMemo:
    findings: list[str] = []
    for item in web_results[:2]:
        findings.append(f"{item.title}: {_clip(item.snippet, 180)} ({item.url})")
    for item in filing_excerpts[:2]:
        findings.append(f"[{item.chunk_id}] {_clip(item.text, 180)}")
    return _ResearchMemo(
        signal="neutral",
        confidence=50 if findings else 0,
        reasoning=(
            "Adaptive research retrieved evidence but the synthesis model was unavailable, "
            "so the agent stayed neutral and preserved the cited findings."
            if findings
            else "Adaptive research could not retrieve enough web or filing evidence."
        ),
        key_findings=findings,
    )


def _run_retrieval(ticker: str, plan: _ResearchPlan) -> tuple[list[WebSearchResult], list[FilingExcerpt]]:
    token = current_fetch_owner.set(AGENT_ID)
    try:
        web_results: list[WebSearchResult] = []
        for query in (plan.web_queries or [])[:3]:
            progress.update_status(AGENT_ID, ticker, f"Web search: {_clip(query, 38)}")
            try:
                web_results.extend(get_web_research(ticker=ticker, question=query, limit=4))
            except Exception as exc:
                logger.warning("adaptive web search failed for %s query %r: %s", ticker, query, exc)

        filing_excerpts: list[FilingExcerpt] = []
        if plan.filing_queries:
            progress.update_status(AGENT_ID, ticker, "Searching SEC filings")
            try:
                filing_excerpts = get_sec_filing_excerpts(
                    ticker=ticker,
                    queries=plan.filing_queries[:4],
                    per_query=2,
                    max_excerpts=8,
                )
            except Exception as exc:
                logger.warning("adaptive SEC search failed for %s: %s", ticker, exc)
    finally:
        current_fetch_owner.reset(token)
    return _dedupe_web(web_results), _dedupe_filings(filing_excerpts)


def run_adaptive_research_agent(ctx: RunContext, view: AdaptiveResearchView, recorder=None) -> Signal:
    progress.update_status(AGENT_ID, view.ticker, "Planning research")
    try:
        plan: _ResearchPlan = call_llm(
            prompt=_plan_prompt(view),
            pydantic_model=_ResearchPlan,
            agent_name=AGENT_ID,
            state=_state(ctx),  # type: ignore[arg-type]
            default_factory=lambda: _default_plan(view.ticker),
            seed=ctx.derive_seed(),
            recorder=recorder,
            ticker=view.ticker,
        )
    except Exception:
        logger.exception("adaptive research plan failed for %s", view.ticker)
        plan = _default_plan(view.ticker)

    if not plan.web_queries and not plan.filing_queries:
        plan = _default_plan(view.ticker)

    web_results, filing_excerpts = _run_retrieval(view.ticker, plan)
    web_results = _dedupe_web([*web_results, *view.web_results])
    filing_excerpts = _dedupe_filings([*filing_excerpts, *view.filings])

    if not web_results and not filing_excerpts:
        return Signal.abstained(
            agent_id=AGENT_ID,
            ticker=view.ticker,
            reason="Adaptive research could not retrieve web or SEC evidence",
        )

    progress.update_status(AGENT_ID, view.ticker, "Synthesizing research")
    fallback = _fallback_memo(web_results, filing_excerpts)
    try:
        memo: _ResearchMemo = call_llm(
            prompt=_synthesis_prompt(view, plan, web_results, filing_excerpts),
            pydantic_model=_ResearchMemo,
            agent_name=AGENT_ID,
            state=_state(ctx),  # type: ignore[arg-type]
            default_factory=lambda: fallback,
            seed=ctx.derive_seed(),
            recorder=recorder,
            ticker=view.ticker,
        )
    except Exception:
        logger.exception("adaptive research synthesis failed for %s", view.ticker)
        memo = fallback

    confidence = max(0.0, min(1.0, float(memo.confidence or 0) / 100.0))
    progress.update_status(AGENT_ID, view.ticker, "Done")
    web_sources = [
        WebSourceCitation(title=r.title or r.url, url=r.url, snippet=_clip(r.snippet, 200))
        for r in web_results
        if r.url
    ][:8]
    filing_sources = [
        FilingCitation(
            chunk_id=e.chunk_id,
            form=e.form,
            fiscal_year=e.fiscal_year,
            item=e.item,
            url=e.url,
            snippet=_clip(e.text, 200),
        )
        for e in filing_excerpts
    ][:8]
    return Signal(
        agent_id=AGENT_ID,
        ticker=view.ticker,
        direction=memo.signal,
        confidence=confidence,
        reasoning=memo.reasoning,
        web_sources=web_sources,
        filing_sources=filing_sources,
        key_factors=[
            f"focus={plan.focus or 'adaptive research'}",
            f"web_queries={len(plan.web_queries or [])}",
            f"filing_queries={len(plan.filing_queries or [])}",
            f"web_sources={len(web_results)}",
            f"filing_sources={len(filing_excerpts)}",
            *(memo.key_findings or [])[:6],
        ],
    )
