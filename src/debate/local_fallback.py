"""Provider-free debate fallback for demos when Anthropic credits are unavailable.

Keeps the same event/result shape as the full debate (src/debate/engine.py) but
uses only local EDGAR retrieval + BM25 search — no LLM calls. This preserves the
SEC-filings showcase instead of failing a browser run on a billing dip.

Ported from the reference research-pod server fallback.
"""

from __future__ import annotations

import asyncio
import time

from src.debate.filings_rag import FilingStore


def _excerpt_text(chunk) -> str:
    text = " ".join(getattr(chunk, "text", "").split())
    return text[:520].rstrip() + ("..." if len(text) > 520 else "")


def _stance_for(texts: list[str]) -> str:
    joined = " ".join(texts).lower()
    risk_terms = sum(joined.count(t) for t in ("risk", "competition", "uncertain", "decline", "litigation", "regulation"))
    pos_terms = sum(joined.count(t) for t in ("growth", "increase", "demand", "margin", "cash", "profit"))
    if pos_terms > risk_terms + 2:
        return "bullish"
    if risk_terms > pos_terms + 2:
        return "bearish"
    return "neutral"


async def run_local_debate_fallback(ticker: str, question: str, horizon: str, emit) -> dict:
    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    emit(type="phase", phase="index", status="start", detail=f"fetching {ticker} 10-K / 10-Q from SEC EDGAR")
    try:
        store = await asyncio.to_thread(FilingStore.from_ticker, ticker)
        emit(type="phase", phase="index", status="done",
             detail=f"indexed {len(store)} passages from {store.summary()}")
    except Exception as exc:
        emit(type="error", message=f"Local fallback could not fetch filings: {type(exc).__name__}: {exc}")
        raise
    timings["index"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    focus = {
        "fundamental": f"Use filings to identify operating results, segment demand, margins, and cash generation relevant to: {question}",
        "sentiment": f"Use management language and risk-factor tone in filings to assess narrative risk for: {question}",
        "valuation": f"Use filings evidence on growth, margins, capital intensity, and risks to frame what may already be priced in for: {question}",
        "macro": f"Use filings to identify sector, regulation, supply-chain, customer, and competitive forces relevant to: {question}",
    }
    plan = {
        "restated_question": f"For {ticker}, assess {question} over {horizon} using latest SEC filings.",
        "focus_areas": focus,
    }
    emit(type="phase", phase="plan", status="done", plan=plan, source="filings")
    timings["plan"] = time.perf_counter() - t0

    queries = {
        "fundamental": "revenue growth margins cash flow segment demand operating results",
        "sentiment": "management discussion outlook risk factors uncertainty demand customer",
        "valuation": "capital expenditures free cash flow margins liquidity debt market risk",
        "macro": "competition regulation supply chain customer concentration industry demand",
    }
    drafts = []
    t0 = time.perf_counter()
    emit(type="phase", phase="research", status="start", detail="4 specialists researching (local filings fallback)")
    for key, query in queries.items():
        emit(type="specialist", key=key, status="start")
        hits = store.search(query, k=3)
        texts = [_excerpt_text(h) for h in hits]
        stance = _stance_for(texts)
        findings = [
            {
                "claim": f"{getattr(h, 'form', 'Filing')} {getattr(h, 'item', 'section')} contains relevant {key} evidence.",
                "evidence": _excerpt_text(h),
                "source": getattr(h, "chunk_id", f"{ticker}-FILINGS"),
                "confidence": "medium",
            }
            for h in hits
        ]
        draft = {
            "agent": key,
            "stance": stance,
            "key_findings": findings,
            "summary": f"Local filings fallback found {len(findings)} cited passages; stance is {stance}.",
        }
        drafts.append(draft)
        emit(type="specialist", key=key, status="done", draft=draft)
    timings["research"] = time.perf_counter() - t0
    emit(type="phase", phase="research", status="done")

    t0 = time.perf_counter()
    bear = {
        "refutations": [{
            "target_claim": "A high-conviction directional call",
            "refutation": "This fallback is filings-only — no live web research, estimate revisions, price action, or fresh news.",
            "evidence": "Generated after Anthropic credits were unavailable; use as a source-grounded preview only.",
            "severity": "medium",
        }],
        "disconfirming_evidence": [
            "Primary filings are historical and may lag current market conditions.",
            "No external web-search challenge was run in fallback mode.",
        ],
        "biggest_risk": "The memo may miss recent events after the latest 10-K/10-Q.",
    }
    emit(type="phase", phase="debate", status="done", bear=bear)
    timings["debate"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    bullish = sum(1 for d in drafts if d["stance"] == "bullish")
    bearish = sum(1 for d in drafts if d["stance"] == "bearish")
    lean = "constructive" if bullish > bearish else "cautious" if bearish > bullish else "neutral"
    citations = list(dict.fromkeys(f["source"] for d in drafts for f in d["key_findings"]))
    memo = {
        "ticker": ticker,
        "conviction": "low",
        "directional_lean": lean,
        "bull_case": "Filing passages support the constructive side where operating demand, margins, liquidity, or cash generation are discussed.",
        "bear_case": "Constrained because Anthropic credits were unavailable; no live web research or adversarial model challenge ran.",
        "base_case": f"Treat this as a filings-grounded preview for {question}, not the full research-pod debate.",
        "key_risks": bear["disconfirming_evidence"],
        "what_would_change_my_mind": [
            "Restore Anthropic credits and run the full debate with web research + filing citations.",
            "Compare the filing evidence against fresh news, estimate revisions, and price action.",
        ],
        "citations": citations[:12],
    }
    timings["synthesize"] = time.perf_counter() - t0
    timings["total"] = sum(timings.values())
    emit(type="phase", phase="synthesize", status="done", memo=memo, timings=timings)

    return {
        "input": {"ticker": ticker, "question": question, "horizon": horizon},
        "specialist_source": "filings",
        "plan": plan,
        "specialist_drafts": drafts,
        "bear_case": bear,
        "memo": memo,
        "timings": timings,
    }
