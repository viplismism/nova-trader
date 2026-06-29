"""Supervisor → specialists → bear → synthesis debate engine.

Ported from the reference `analyst_debate` research pod and adapted into Nova as
an additive "research desk" mode. Nova's deterministic analyst engine is
untouched; this runs as a separate, Anthropic-native path. Reads
``ANTHROPIC_API_KEY`` from the environment (Nova's .env is authoritative).
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Literal

from pydantic import BaseModel, Field

from anthropic import AsyncAnthropic

from src.debate.filings_rag import FilingStore


# Supervisor / bear / synthesizer run on the reasoning model; the 4 parallel
# specialists default to a faster, cheaper model. Both are env-overridable.
REASONING_MODEL = os.getenv("NOVA_DEBATE_REASONING_MODEL", "claude-opus-4-8")
SPECIALIST_MODEL = os.getenv("NOVA_DEBATE_SPECIALIST_MODEL", "claude-sonnet-4-6")

# The bear (Opus + multi-round live web search) is the slowest, least-bounded step.
# Cap it so a runaway search loop can't hang the whole debate — on timeout we degrade
# to a placeholder bear and still produce a memo. Synth is bounded too.
BEAR_TIMEOUT_S = int(os.getenv("NOVA_DEBATE_BEAR_TIMEOUT", "300"))
SYNTH_TIMEOUT_S = int(os.getenv("NOVA_DEBATE_SYNTH_TIMEOUT", "180"))

WEB_SEARCH_TOOL = {"type": "web_search_20260209", "name": "web_search"}  # dynamic filtering built in

# Client-side custom tool: retrieval over the company's real 10-K / 10-Q filings.
SEARCH_FILINGS_TOOL = {
    "name": "search_filings",
    "description": (
        "Search this company's most recent 10-K and 10-Q filings for relevant passages. "
        "Returns real excerpts, each prefixed with a citation ID in square brackets "
        "(e.g. [NVDA-10K-0042]). You MUST cite that exact ID verbatim as the `source` for any "
        "claim you draw from a passage. Run several focused queries to cover your mandate."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to look for, e.g. 'data center segment revenue'"}
        },
        "required": ["query"],
        "additionalProperties": False,
    },
}


Confidence = Literal["high", "medium", "low"]


class FocusAreas(BaseModel):
    fundamental: str
    sentiment: str
    valuation: str
    macro: str


class Plan(BaseModel):
    restated_question: str
    focus_areas: FocusAreas


class Finding(BaseModel):
    claim: str
    evidence: str
    source: str = Field(description="URL or document name backing this finding")
    confidence: Confidence


class SpecialistDraft(BaseModel):
    agent: str
    stance: Literal["bullish", "bearish", "neutral"]
    key_findings: list[Finding]
    summary: str


class Refutation(BaseModel):
    target_claim: str
    refutation: str
    evidence: str
    severity: Confidence


class BearCase(BaseModel):
    refutations: list[Refutation]
    disconfirming_evidence: list[str]
    biggest_risk: str


class Memo(BaseModel):
    ticker: str
    conviction: Confidence
    directional_lean: Literal["constructive", "neutral", "cautious"]
    bull_case: str
    bear_case: str
    base_case: str
    key_risks: list[str]
    what_would_change_my_mind: list[str]
    citations: list[str]



SYSTEM_PREAMBLE = """You are one analyst on a disciplined equity-research pod. The pod produces \
decision-ready research memos for a human portfolio manager who signs off before any trade. \
You are a copilot that compresses analyst hours — NOT an autopilot, and NOT a trade-execution system.

Operating rules that bind every member of this pod:
1. CITATION CONTRACT. Every material claim must carry concrete evidence (a number where possible) \
and a source (a URL or a named document). Never state a figure you cannot attribute.
2. RECENCY. Prefer the most recent primary or reputable sources. Flag when data is stale or estimated.
3. NUMERIC HONESTY. Do not invent precision. If a figure is approximate or model-derived, say so and \
label confidence accordingly (high / medium / low) based on source quality and recency.
4. NO HOUSE VIEW PRESSURE. Represent your own slice of the analysis honestly. Specialists argue their \
angle; they do not pre-compromise toward a consensus. The bear actively tries to break the thesis. \
The synthesizer judges the evidence rather than averaging opinions.
5. RISK FIRST. Surface disconfirming evidence, customer/sector concentration, and what is already \
priced in. High conviction requires that no high-severity risk survives unrebutted.
6. OUTPUT DISCIPLINE. Be sharp and specific. A few well-evidenced findings beat a wall of text. \
Return exactly the requested structured object.

This is research support. It does not, on its own, constitute investment advice or alpha; a human \
makes the call."""


def system_blocks(role_instructions: str) -> list[dict]:
    """Cached shared preamble (stable prefix) + per-role instructions (volatile suffix)."""
    return [
        {"type": "text", "text": SYSTEM_PREAMBLE, "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": role_instructions},
    ]


# --------------------------------------------------------------------------------------
# Usage tracking — lets us confirm prompt caching is actually happening.
# --------------------------------------------------------------------------------------
class Usage:
    def __init__(self) -> None:
        self.input = 0
        self.output = 0
        self.cache_read = 0
        self.cache_write = 0

    def add(self, u) -> None:
        if u is None:
            return
        self.input += getattr(u, "input_tokens", 0) or 0
        self.output += getattr(u, "output_tokens", 0) or 0
        self.cache_read += getattr(u, "cache_read_input_tokens", 0) or 0
        self.cache_write += getattr(u, "cache_creation_input_tokens", 0) or 0

    def reset(self) -> None:
        self.input = self.output = self.cache_read = self.cache_write = 0

    def __str__(self) -> str:
        return (
            f"input={self.input:,}  output={self.output:,}  "
            f"cache_write={self.cache_write:,}  cache_read={self.cache_read:,}"
        )


USAGE = Usage()


def _noop(**_kw) -> None:  # default progress sink (CLI doesn't need live events)
    pass


class DebateCancelled(Exception):
    """Raised when a caller requests the debate stop (e.g. the browser disconnected)."""


def _check_cancel(cancel) -> None:
    if cancel is not None and cancel.is_set():
        raise DebateCancelled()


def _effort_config(effort: str | None) -> dict:
    """output_config for the research loops. Omitted when effort is None (e.g. Haiku
    fast-mode specialists, which reject the effort parameter)."""
    return {"output_config": {"effort": effort}} if effort else {}


def _thinking_config(thinking: bool) -> dict:
    """thinking param for the research loops. Omitted in fast mode — Haiku 4.5 does
    not support adaptive thinking (it 400s)."""
    return {"thinking": {"type": "adaptive"}} if thinking else {}


def _emit_web_activity(resp, emit, phase: str) -> None:
    """Surface what the server-side web_search actually did — the queries it ran and
    the pages it pulled — into the activity log."""
    for b in resp.content:
        bt = getattr(b, "type", "")
        if bt == "server_tool_use" and getattr(b, "name", "") == "web_search":
            q = (getattr(b, "input", {}) or {}).get("query", "")
            if q:
                emit(type="tool_call", agent=phase, tool="web_search", query=q)
        elif bt == "web_search_tool_result":
            content = getattr(b, "content", None) or []
            urls = []
            for item in content:
                u = getattr(item, "url", None) or getattr(item, "title", None)
                if u:
                    urls.append(u)
            if urls:
                emit(type="tool_call", agent=phase, tool="web_search", results=urls[:6])


async def web_research(
    client: AsyncAnthropic, role_instructions: str, user: str, model: str,
    max_rounds: int = 6, emit=_noop, record=_noop, phase: str = "",
    effort: str | None = "medium", cancel=None, thinking: bool = True,
) -> str:
    """Run the server-side web_search loop and return the agent's free-form findings text."""
    messages: list[dict] = [{"role": "user", "content": user}]
    resp = None
    for _ in range(max_rounds):
        _check_cancel(cancel)
        resp = await client.messages.create(
            model=model,
            max_tokens=12000,
            system=system_blocks(role_instructions),
            messages=messages,
            tools=[WEB_SEARCH_TOOL],
            **_thinking_config(thinking),
            **_effort_config(effort),
        )
        USAGE.add(resp.usage)
        _emit_web_activity(resp, emit, phase)
        text_so_far = "".join(b.text for b in resp.content if b.type == "text")
        record(phase=phase, kind="web_research", model=model, prompt=user, response=text_so_far, usage=resp.usage)
        # Server-side tool loop hit its iteration cap — re-send to resume (no extra user msg).
        if resp.stop_reason == "pause_turn":
            messages = [
                {"role": "user", "content": user},
                {"role": "assistant", "content": resp.content},
            ]
            continue
        break
    return "".join(b.text for b in resp.content if b.type == "text")


def make_filings_executor(store: FilingStore, emit=_noop, label: str = ""):
    """Client-side executor for the search_filings tool — returns real, citable excerpts.
    Emits each query + the chunk IDs it returned so the activity log shows the retrieval."""

    def execute(name: str, tool_input: dict) -> str:
        if name != "search_filings":
            return f"unknown tool: {name}"
        query = tool_input.get("query", "")
        hits = store.search(query, k=5)
        emit(type="tool_call", agent=label, tool="search_filings", query=query,
             results=[h.chunk_id for h in hits])
        if not hits:
            return "No matching passages found in the indexed filings."
        return "\n\n".join(
            f"[{h.chunk_id}] ({h.form} {h.fiscal} - {h.item})\n{h.text}" for h in hits
        )

    return execute


async def agent_with_tool(
    client: AsyncAnthropic,
    role_instructions: str,
    user: str,
    model: str,
    tool_defs: list[dict],
    executor,
    max_rounds: int = 8,
    record=_noop,
    phase: str = "",
    effort: str | None = "medium",
    cancel=None,
    thinking: bool = True,
) -> str:
    """Manual agentic loop for a client-side tool (the RAG tool). Returns the agent's findings text."""
    messages: list[dict] = [{"role": "user", "content": user}]
    resp = None
    for _ in range(max_rounds):
        _check_cancel(cancel)
        resp = await client.messages.create(
            model=model,
            max_tokens=12000,
            system=system_blocks(role_instructions),
            messages=messages,
            tools=tool_defs,
            **_thinking_config(thinking),
            **_effort_config(effort),
        )
        USAGE.add(resp.usage)
        text_so_far = "".join(b.text for b in resp.content if b.type == "text")
        record(phase=phase, kind="filings_research", model=model, prompt=user, response=text_so_far, usage=resp.usage)
        if resp.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": resp.content})
            results = [
                {"type": "tool_result", "tool_use_id": b.id, "content": executor(b.name, b.input)}
                for b in resp.content
                if b.type == "tool_use"
            ]
            messages.append({"role": "user", "content": results})
            continue
        break
    return "".join(b.text for b in resp.content if b.type == "text")


async def structure(
    client: AsyncAnthropic, model: str, schema: type[BaseModel], role_instructions: str, user: str,
    record=_noop, phase: str = "",
):
    """Coerce free-form analysis into a validated Pydantic object (no tools, structured output)."""
    resp = await client.messages.parse(
        model=model,
        max_tokens=4000,
        system=system_blocks(role_instructions),
        messages=[{"role": "user", "content": user}],
        output_format=schema,
    )
    USAGE.add(resp.usage)
    out = resp.parsed_output
    try:
        record(phase=phase, kind="structure", model=model, prompt=user,
               response=out.model_dump_json(indent=2) if out is not None else "", usage=resp.usage)
    except Exception:  # pragma: no cover - recording must never break the run
        pass
    if out is None:
        # parsed_output is None on a refusal or a max_tokens truncation. Raise so the
        # caller degrades (fallback draft / placeholder bear / degraded memo) rather
        # than crashing on out.model_dump().
        raise ValueError(f"structured output was empty/refused (phase={phase}, model={model})")
    return out



SPECIALISTS = [
    ("fundamental", "Financials, revenue growth, margins, guidance, balance sheet, segment trends from filings & earnings calls"),
    ("sentiment",   "Recent news flow, analyst rating changes, management tone shifts, narrative momentum"),
    ("valuation",   "Current multiples vs history & peers, what is priced in, downside/upside scenarios"),
    ("macro",       "Sector tailwinds/headwinds, competitive landscape, regulation, demand drivers"),
]


def _default_plan(ticker: str, question: str) -> Plan:
    """Generic plan if the supervisor refuses/fails — keeps the debate running."""
    return Plan(
        restated_question=f"For {ticker}: {question}",
        focus_areas=FocusAreas(
            fundamental=f"Revenue growth, margins, guidance, balance sheet, and segment trends relevant to: {question}",
            sentiment=f"Recent news, analyst rating changes, and narrative momentum relevant to: {question}",
            valuation=f"Current multiples vs history/peers and what is priced in, relevant to: {question}",
            macro=f"Sector tailwinds/headwinds, competition, regulation, and demand drivers relevant to: {question}",
        ),
    )


def _fallback_draft(key: str) -> SpecialistDraft:
    """Neutral draft if a specialist refuses/errors — one bad specialist must not abort
    the run or orphan the other parallel tasks."""
    return SpecialistDraft(
        agent=f"{key} analyst",
        stance="neutral",
        key_findings=[],
        summary=f"The {key} specialist could not complete its research this run.",
    )


async def make_plan(client: AsyncAnthropic, ticker: str, question: str, horizon: str, record=_noop) -> Plan:
    role = "You are the SUPERVISOR / PM. Decompose the question into one specific, non-generic research mandate per specialist."
    user = (
        f"Ticker: {ticker}\nQuestion: \"{question}\" (horizon: {horizon}).\n"
        "Restate the question precisely, then write a focused mandate for each of: fundamental, "
        "sentiment, valuation, macro. Tell each exactly what to dig into for THIS question."
    )
    try:
        return await structure(client, REASONING_MODEL, Plan, role, user, record=record, phase="supervisor")
    except DebateCancelled:
        raise
    except Exception:
        return _default_plan(ticker, question)


async def run_specialist(
    client, key: str, mandate: str, plan: Plan, ticker, question, horizon, source: str, store,
    emit=_noop, record=_noop, spec_model: str = SPECIALIST_MODEL, effort: str | None = "medium",
    cancel=None, thinking: bool = True,
) -> SpecialistDraft:
    """Resilient wrapper: a single specialist failing (refusal, parse error, API error)
    must NOT raise — otherwise it aborts run_debate and orphans the other parallel tasks.
    On error it emits 'done' with a neutral fallback draft. Cancellation still propagates."""
    try:
        return await _run_specialist_inner(
            client, key, mandate, plan, ticker, question, horizon, source, store,
            emit=emit, record=record, spec_model=spec_model, effort=effort, cancel=cancel, thinking=thinking,
        )
    except DebateCancelled:
        raise
    except Exception:
        fb = _fallback_draft(key)
        emit(type="specialist", key=key, status="done", draft=fb.model_dump())
        return fb


async def _run_specialist_inner(
    client, key: str, mandate: str, plan: Plan, ticker, question, horizon, source: str, store,
    emit=_noop, record=_noop, spec_model: str = SPECIALIST_MODEL, effort: str | None = "medium",
    cancel=None, thinking: bool = True,
) -> SpecialistDraft:
    focus = getattr(plan.focus_areas, key)
    _check_cancel(cancel)
    emit(type="specialist", key=key, status="start")

    # 1) research — filings RAG (cited spans) by default, web search as fallback
    if source == "filings" and store is not None and len(store) > 0:
        research_role = (
            f"You are the {key.upper()} analyst. Your standing mandate: {mandate}. "
            "Use the search_filings tool to ground every figure in the company's ACTUAL 10-K / 10-Q. "
            "Do not use outside knowledge or invent URLs — if a fact is not in the filings, say so."
        )
        research_user = (
            f"Ticker: {ticker}. Question: \"{question}\" (horizon: {horizon}).\n"
            f"Supervisor's specific focus for you: {focus}\n\n"
            "Run several search_filings queries, then produce 3-5 sharp findings from YOUR angle. "
            "Each finding must quote/derive its evidence from a returned passage and use that passage's "
            "bracketed chunk ID (e.g. NVDA-10K-0042) as its source."
        )
        findings_text = await agent_with_tool(
            client, research_role, research_user, spec_model,
            [SEARCH_FILINGS_TOOL], make_filings_executor(store, emit, key),
            record=record, phase=key, effort=effort, cancel=cancel, thinking=thinking,
        )
        source_rule = (
            "Every `source` MUST be a filing chunk ID that appeared in your research "
            "(e.g. 'NVDA-10K-0042'). Never invent a source or use a URL."
        )
    else:
        research_role = (
            f"You are the {key.upper()} analyst. Your standing mandate: {mandate}. "
            "Use web_search to gather the most recent facts; attach numbers and sources to every point."
        )
        research_user = (
            f"Ticker: {ticker}. Question: \"{question}\" (horizon: {horizon}).\n"
            f"Supervisor's specific focus for you: {focus}\n\n"
            "Research this now with web_search. Produce 3-5 sharp, sourced findings from YOUR angle only, "
            "each with concrete evidence (numbers) and a source URL. State your overall stance."
        )
        findings_text = await web_research(client, research_role, research_user, spec_model,
                                           emit=emit, record=record, phase=key, effort=effort,
                                           cancel=cancel, thinking=thinking)
        source_rule = "Every `source` should be the URL you found the fact at."

    # 2) structure
    struct_role = (
        f"You are the {key.upper()} analyst. Convert your research into the required structured draft. "
        + source_rule
    )
    struct_user = (
        f"Here is your research on {ticker} re: \"{question}\":\n\n{findings_text}\n\n"
        "Return your structured draft: agent name, stance (bullish/bearish/neutral), 3-5 key_findings "
        "(each claim/evidence/source/confidence), and a summary."
    )
    draft = await structure(client, spec_model, SpecialistDraft, struct_role, struct_user,
                            record=record, phase=key)
    emit(type="specialist", key=key, status="done", draft=draft.model_dump())
    return draft


async def run_bear(client, drafts: list[SpecialistDraft], ticker, question, horizon,
                   emit=_noop, record=_noop, cancel=None, bear_model: str = REASONING_MODEL,
                   bear_web: bool = True) -> BearCase:
    dossier = json.dumps([d.model_dump() for d in drafts], indent=2)
    if bear_web:
        # 1) adversarial research with live web search (the slow but thorough path)
        research_role = (
            "You are the BEAR / CHALLENGER. Default to skepticism. Attack the strongest version of the bull "
            "thesis and hunt with web_search for disconfirming data (demand softening, competition, "
            "valuation risk, concentration, accounting/guidance flags)."
        )
        research_user = (
            f"The specialists produced these drafts on {ticker} re: \"{question}\" (horizon {horizon}):\n\n"
            f"{dossier}\n\nActively look for contradicting evidence now. Find at least 2 SPECIFIC "
            "disconfirming data points (with sources), or state none could be found and why."
        )
        bear_text = await web_research(client, research_role, research_user, bear_model,
                                       max_rounds=4, emit=emit, record=record, phase="bear", cancel=cancel)
    else:
        # Quick path: no live web search (that's the slow part) — challenge the assembled
        # thesis directly from the specialist drafts. Fast, still adversarial.
        emit(type="tool_call", agent="bear", tool="reasoning", query="challenging the thesis from the drafts (no web)")
        bear_text = (
            "Quick mode: no live web search. Build the strongest adversarial case strictly from the "
            "specialist drafts below — surface internal contradictions, the weakest-supported claims, "
            "concentration/valuation/competition risks, and what evidence is missing.\n\n" + dossier
        )

    # 2) structure
    struct_role = "You are the BEAR / CHALLENGER. Convert your attack into the required structured object."
    struct_user = (
        f"Specialist drafts:\n{dossier}\n\nYour adversarial research:\n{bear_text}\n\n"
        "Return: refutations (each target_claim/refutation/evidence/severity), disconfirming_evidence "
        "(list), and the single biggest_risk to the thesis."
    )
    return await structure(client, bear_model, BearCase, struct_role, struct_user, record=record, phase="bear")


async def run_synth(client, drafts: list[SpecialistDraft], bear: BearCase, ticker, question, horizon, record=_noop) -> Memo:
    dossier = json.dumps([d.model_dump() for d in drafts], indent=2)
    bear_json = json.dumps(bear.model_dump(), indent=2)
    role = (
        "You are the SYNTHESIZER / PM. Reconcile the specialist drafts and the bear case into a "
        "decision-ready memo. Weight by evidence quality and recency — judge, do not average. Every "
        "assertion must trace to a specialist finding or a bear point; introduce no new uncited claims. "
        "HIGH conviction requires that the bear surfaced no high-severity unrebutted risk."
    )
    user = (
        f"Ticker {ticker}. Question: \"{question}\" (horizon {horizon}).\n\n"
        f"SPECIALIST DRAFTS:\n{dossier}\n\nBEAR CASE:\n{bear_json}\n\n"
        "Return the memo: conviction, directional_lean, bull_case, bear_case, base_case, key_risks, "
        "what_would_change_my_mind, and the aggregated citations actually used."
    )
    return await structure(client, REASONING_MODEL, Memo, role, user, record=record, phase="synthesize")


def _degraded_memo(ticker: str, drafts: list[SpecialistDraft], bear: BearCase) -> Memo:
    """A best-effort memo assembled from the specialist drafts + bear case, used only
    if the synthesizer times out/fails — so the run always returns a result."""
    bull = sum(1 for d in drafts if d.stance == "bullish")
    bearish = sum(1 for d in drafts if d.stance == "bearish")
    lean = "constructive" if bull > bearish else "cautious" if bearish > bull else "neutral"
    cites: list[str] = []
    for d in drafts:
        for f in d.key_findings:
            if f.source:
                cites.append(f.source)
    bull_summaries = "; ".join(d.summary for d in drafts if d.stance == "bullish")
    return Memo(
        ticker=ticker,
        conviction="low",
        directional_lean=lean,
        bull_case=bull_summaries or "See the specialist drafts above for the constructive case.",
        bear_case=bear.biggest_risk or "; ".join(bear.disconfirming_evidence[:2]) or "See the bear case above.",
        base_case="Synthesis step did not complete — this is a degraded memo assembled directly from the "
                  "specialist drafts and the bear case. Re-run to get the full synthesized memo.",
        key_risks=[r for r in (bear.disconfirming_evidence[:5] or [bear.biggest_risk]) if r],
        what_would_change_my_mind=["Re-run the debate — the synthesis step timed out or failed this time."],
        citations=list(dict.fromkeys(cites))[:12],
    )


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------
async def run_debate(ticker: str, question: str, horizon: str, specialist_source: str = "filings",
                     emit=_noop, record=_noop, cancel=None, fast: bool = False,
                     bear_model: str | None = None, bear_web: bool = True):
    """Run the full debate. Returns (result_dict, filings_store_or_None).

    ``emit(**event)`` streams progress (phases, specialists, tool_call activity).
    ``record(**call)`` captures every LLM call (phase, model, prompt, response, usage)
    so the caller can persist the full trajectory for later model training.
    ``cancel`` (a threading.Event) stops the run at the next checkpoint.
    ``fast`` runs the 4 specialists on Haiku with no effort param — quicker/cheaper."""
    client = AsyncAnthropic()  # reads ANTHROPIC_API_KEY from env
    USAGE.reset()  # per-run token accounting (NOTE: process-global — concurrent debates share it)
    timings: dict[str, float] = {}
    store = None
    spec_model = "claude-haiku-4-5" if fast else SPECIALIST_MODEL
    spec_effort = None if fast else "medium"   # Haiku rejects the effort param
    spec_thinking = not fast                   # Haiku rejects adaptive thinking
    bear_m = bear_model or REASONING_MODEL     # quick tier runs the bear on Sonnet

    if specialist_source == "filings":
        t = time.perf_counter()
        print(f"[index]      fetching & indexing {ticker} 10-K / 10-Q from SEC EDGAR ...", flush=True)
        emit(type="phase", phase="index", status="start", detail=f"fetching {ticker} 10-K / 10-Q from SEC EDGAR")
        try:
            store = await asyncio.to_thread(FilingStore.from_ticker, ticker)
            if len(store) == 0:
                raise RuntimeError("no chunks indexed")
            print(f"             indexed {len(store)} passages from {store.summary()}", flush=True)
            emit(type="phase", phase="index", status="done",
                 detail=f"indexed {len(store)} passages from {store.summary()}")
        except Exception as e:  # network down, ticker not in EDGAR, etc.
            print(f"             WARNING: filings index unavailable ({e}); falling back to web search", flush=True)
            specialist_source, store = "web", None
            emit(type="phase", phase="index", status="warn", detail=f"filings unavailable ({e}); using web search")
        timings["index"] = time.perf_counter() - t

    t = time.perf_counter()
    print(f"\n[plan]       supervisor decomposing the question for {ticker} ...", flush=True)
    emit(type="phase", phase="plan", status="start", detail="supervisor decomposing the question")
    plan = await make_plan(client, ticker, question, horizon, record=record)
    timings["plan"] = time.perf_counter() - t
    emit(type="phase", phase="plan", status="done", plan=plan.model_dump(), source=specialist_source)

    t = time.perf_counter()
    label = "filings RAG" if specialist_source == "filings" else "web search"
    print(f"[research]   4 specialists researching in parallel ({label}) ...", flush=True)
    emit(type="phase", phase="research", status="start", detail=f"4 specialists researching in parallel ({label})")
    tasks = [
        asyncio.create_task(
            run_specialist(client, key, mandate, plan, ticker, question, horizon, specialist_source, store,
                           emit, record, spec_model=spec_model, effort=spec_effort, cancel=cancel, thinking=spec_thinking)
        )
        for key, mandate in SPECIALISTS
    ]
    drafts: list[SpecialistDraft] = []
    try:
        for fut in asyncio.as_completed(tasks):  # stream each card as it finishes
            drafts.append(await fut)
    finally:
        # Defensive: if anything still escaped (cancel/unexpected), don't leave the
        # other specialist tasks running detached (orphaned tasks keep billing).
        for task in tasks:
            if not task.done():
                task.cancel()
    timings["research"] = time.perf_counter() - t
    print("             stances -> " + ", ".join(f"{d.agent}:{d.stance}" for d in drafts), flush=True)
    emit(type="phase", phase="research", status="done")

    t = time.perf_counter()
    print("[debate]     bear challenger attacking the assembled thesis ...", flush=True)
    emit(type="phase", phase="debate", status="start", detail="bear challenger attacking the thesis")
    try:
        bear = await asyncio.wait_for(
            run_bear(client, drafts, ticker, question, horizon, emit=emit, record=record, cancel=cancel,
                     bear_model=bear_m, bear_web=bear_web),
            timeout=BEAR_TIMEOUT_S,
        )
        emit(type="phase", phase="debate", status="done", bear=bear.model_dump())
    except DebateCancelled:
        raise
    except Exception as e:  # timeout, web_search unavailable, etc. — degrade, don't hang/fail the run.
        reason = f"timed out after {BEAR_TIMEOUT_S}s" if isinstance(e, asyncio.TimeoutError) else f"{type(e).__name__}"
        bear = BearCase(
            refutations=[],
            disconfirming_evidence=[f"Bear step unavailable ({reason}); memo lacks a full adversarial web challenge."],
            biggest_risk="No adversarial bear challenge was completed for this run.",
        )
        emit(type="phase", phase="debate", status="warn",
             detail=f"bear challenge skipped ({reason})", bear=bear.model_dump())
    timings["debate"] = time.perf_counter() - t

    t = time.perf_counter()
    print("[synthesize] PM reconciling into a cited, conviction-scored memo ...", flush=True)
    emit(type="phase", phase="synthesize", status="start", detail="PM reconciling into a cited memo")
    try:
        memo = await asyncio.wait_for(
            run_synth(client, drafts, bear, ticker, question, horizon, record=record),
            timeout=SYNTH_TIMEOUT_S,
        )
    except DebateCancelled:
        raise
    except Exception as e:  # never leave the run without a memo — build a degraded one from drafts+bear.
        memo = _degraded_memo(ticker, drafts, bear)
        emit(type="phase", phase="synthesize", status="warn",
             detail=f"synthesis degraded ({'timeout' if isinstance(e, asyncio.TimeoutError) else type(e).__name__})")
    timings["synthesize"] = time.perf_counter() - t
    timings["total"] = sum(timings.values())
    emit(type="phase", phase="synthesize", status="done", memo=memo.model_dump(), timings=timings)

    # Map filing chunk-ID citations to their EDGAR document URLs so the UI can link them.
    citation_urls: dict[str, str] = {}
    if store is not None:
        seen_sources = set(memo.citations or [])
        for d in drafts:
            for f in d.key_findings:
                seen_sources.add(f.source)
        for cid in seen_sources:
            ch = store.get(str(cid))
            if ch is not None and getattr(ch, "url", ""):
                citation_urls[str(cid)] = ch.url

    result = {
        "input": {"ticker": ticker, "question": question, "horizon": horizon},
        "specialist_source": specialist_source,
        "plan": plan.model_dump(),
        "specialist_drafts": [d.model_dump() for d in drafts],
        "bear_case": bear.model_dump(),
        "memo": memo.model_dump(),
        "timings": timings,
        "citation_urls": citation_urls,
    }
    return result, store

