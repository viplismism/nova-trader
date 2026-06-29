"""Thin browser wrapper around the Nova Trader engine."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import io
import json
import os
import queue
import re
import secrets
import threading
from importlib.resources import files
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, PlainTextResponse, StreamingResponse

from src.chat.context import build_context
from src.chat.models import DEFAULT_AGENTS, ChatSettings
from src.chat.rendering import recommendation_verdict_text
from src.chat.signal_card import build_qa_messages, build_signal_cards, signal_cards_context_text
from src.debate import USAGE as DEBATE_USAGE, run_debate
from src.debate.engine import DebateCancelled
from src.debate.local_fallback import run_local_debate_fallback
from src.debate.recorder import DebateRecorder
from src.engine import run_engine
from src.runs import RunRecorder, runs_root
from src.schemas.signals import Recommendation
from src.utils.llm import provider_has_credentials, required_api_key_name, stream_chat
from src.utils.progress import progress

# override=True so the project's .env is authoritative for the demo — otherwise a
# stale ANTHROPIC_API_KEY (or other key) exported in the shell silently shadows it.
load_dotenv(override=True)


_RUN_CACHE: dict[str, Recommendation] = {}
_RUN_CACHE_LOCK = threading.Lock()
_MAX_CACHED_RUNS = 24

# Active debate runs by run_id → cancel Event. A debate runs in a worker thread
# decoupled from the SSE connection, so a dropped/closed stream does NOT abort it
# (the result is saved to disk and recoverable). Only an explicit Close (the cancel
# endpoint) stops it. This is what makes "finished but can't see it" recoverable.
_ACTIVE_DEBATES: dict[str, threading.Event] = {}
_ACTIVE_DEBATES_LOCK = threading.Lock()

_QA_SYSTEM = (
    "You are Nova, the assistant inside Nova Trader. Answer from the supplied signal-card context only. "
    "Use short, clear paragraphs. If the run does not contain enough evidence, say exactly what is missing. "
    "Do not change the final action, confidence, or risk limits. This is research support, not personal financial advice."
)


def create_app() -> FastAPI:
    app = FastAPI(title="Nova Trader", version="0.1.0")

    # Optional shared-password gate (HTTP Basic). Enabled only when NOVA_ACCESS_PASSWORD
    # is set, so local dev stays open. Guards every route except the health probe, so a
    # public demo URL can't run anything (and spend API credits) without the password.
    access_password = os.environ.get("NOVA_ACCESS_PASSWORD", "")
    if access_password:
        @app.middleware("http")
        async def _require_password(request: Request, call_next):
            if request.url.path == "/api/health":
                return await call_next(request)
            header = request.headers.get("authorization", "")
            ok = False
            if header.startswith("Basic "):
                with contextlib.suppress(Exception):
                    _, _, supplied = base64.b64decode(header[6:]).decode().partition(":")
                    ok = secrets.compare_digest(supplied, access_password)
            if not ok:
                return PlainTextResponse(
                    "Authentication required", status_code=401,
                    headers={"WWW-Authenticate": 'Basic realm="Nova Trader"'},
                )
            return await call_next(request)

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return files("src.web.static").joinpath("index.html").read_text()

    @app.get("/api/health")
    async def health() -> dict[str, str]:
        return {"ok": "true"}

    @app.get("/api/config")
    async def config(provider: str | None = None, model: str | None = None) -> dict[str, Any]:
        provider = provider or _default_provider()
        model = model or _default_model()
        key_name = required_api_key_name(provider)
        return {
            "provider": provider,
            "model": model,
            "credential_key": key_name,
            "has_credentials": provider_has_credentials(provider),
        }

    @app.get("/api/run")
    async def run_analysis(
        tickers: str = Query(..., min_length=1),
        provider: str | None = Query(None),
        model: str | None = Query(None),
        portfolio_mode: str = Query("research"),
        agents: str = Query(""),
    ) -> StreamingResponse:
        ticker_list = _parse_tickers(tickers)
        provider = provider or _default_provider()
        model = model or _default_model()
        if not provider_has_credentials(provider):
            key_name = required_api_key_name(provider) or "provider credentials"
            raise HTTPException(
                status_code=400,
                detail=f"{key_name} is not configured. Add it to .env or choose a provider with credentials.",
            )
        agent_list = _parse_agents(agents)
        settings = ChatSettings(
            provider=provider,
            model=model,
            portfolio_mode=_portfolio_mode(portfolio_mode),
            agents=agent_list or DEFAULT_AGENTS.copy(),
        )
        ctx = build_context(ticker_list, settings)
        selected_agents = ctx.request.selected_agents or None
        events: queue.Queue[dict[str, Any]] = queue.Queue()
        done = threading.Event()

        def on_progress(agent_name: str, ticker: str | None, status: str, analysis: str | None, timestamp: str) -> None:
            events.put({
                "type": "activity",
                "agent": agent_name,
                "ticker": ticker,
                "status": status,
                "analysis": analysis,
                "timestamp": timestamp,
            })

        def worker() -> None:
            progress.reset_telemetry()
            progress.register_handler(on_progress)
            events.put({
                "type": "run",
                "run_id": ctx.run_id,
                "tickers": ticker_list,
                "provider": provider,
                "model": model,
                "portfolio_mode": settings.portfolio_mode,
                "agents": selected_agents or DEFAULT_AGENTS,
            })
            try:
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    recommendation = run_engine(
                        ctx,
                        selected_agents=selected_agents,
                        record=True,
                    )
                _remember_run(recommendation)
                events.put({
                    "type": "done",
                    "run_id": recommendation.run_id,
                    "summary": recommendation.summary,
                    "verdict": recommendation_verdict_text(recommendation),
                    "cards": [card.model_dump(mode="json") for card in build_signal_cards(recommendation)],
                    "llm_logs": _public_llm_logs(recommendation.run_id),
                    "saved_to": str(runs_root() / recommendation.run_id),
                })
            except Exception as exc:  # pragma: no cover - exact provider/network errors vary
                events.put({"type": "failure", "message": _friendly_error(exc)})
            finally:
                progress.unregister_handler(on_progress)
                done.set()

        thread = threading.Thread(target=worker, name=f"nova-web-run-{ctx.run_id}", daemon=True)
        thread.start()
        return StreamingResponse(_stream_events(events, done), media_type="text/event-stream")

    @app.get("/api/ask")
    async def ask_run(
        run_id: str = Query(..., min_length=1),
        question: str = Query(..., min_length=1),
        provider: str | None = Query(None),
        model: str | None = Query(None),
    ) -> StreamingResponse:
        run_id = _validate_run_id(run_id)
        provider = provider or _default_provider()
        model = model or _default_model()
        if not provider_has_credentials(provider):
            key_name = required_api_key_name(provider) or "provider credentials"
            raise HTTPException(
                status_code=400,
                detail=f"{key_name} is not configured. Add it to .env or choose a provider with credentials.",
            )
        if run_id.startswith("debate-") and DebateRecorder.exists(run_id):
            context = "Grounded Nova Trader research-desk memo:\n" + _debate_context_text(run_id)
        else:
            context = "Grounded Nova Trader run context:\n" + signal_cards_context_text(_load_recommendation(run_id))
        events: queue.Queue[dict[str, Any]] = queue.Queue()
        done = threading.Event()
        messages = build_qa_messages(_QA_SYSTEM, question, context)

        def worker() -> None:
            try:
                for channel, chunk in stream_chat(messages, provider=provider, model=model):
                    if channel == "answer" and chunk:
                        events.put({"type": "token", "text": chunk})
                events.put({"type": "done"})
            except Exception as exc:  # pragma: no cover - exact provider/network errors vary
                events.put({"type": "failure", "message": _friendly_error(exc)})
            finally:
                done.set()

        thread = threading.Thread(target=worker, name=f"nova-web-ask-{run_id}", daemon=True)
        thread.start()
        return StreamingResponse(_stream_events(events, done), media_type="text/event-stream")

    @app.get("/api/debate")
    async def run_research_desk(
        request: Request,
        ticker: str = Query(..., min_length=1),
        question: str = Query(..., min_length=1),
        horizon: str = Query("6-12 months"),
        source: str = Query("filings"),
        speed: str = Query("full"),
    ) -> StreamingResponse:
        """Supervisor → specialists → bear → synthesis debate, streamed as SSE.

        Anthropic-native (separate from the deterministic /api/run). On a credit
        dip it degrades to a local filings-only fallback instead of failing. A closed
        tab (Close) cancels the run so it stops spending tokens; a passive drop does not.

        Speed tiers:
          full  — specialists Sonnet, bear Opus + web        (most thorough, ~8 min)
          fast  — specialists Haiku, bear Opus + web          (~5 min)
          quick — specialists Haiku, bear Sonnet, no web      (quickest, ~3-4 min)
        """
        symbol = _parse_tickers(ticker)[0]
        question = question.strip()
        source = source if source in {"filings", "web"} else "filings"
        speed = speed if speed in {"full", "fast", "quick"} else "full"
        fast = speed in {"fast", "quick"}
        bear_model = "claude-sonnet-4-6" if speed == "quick" else None
        bear_web = speed != "quick"   # quick skips the slow live-web bear search
        if not provider_has_credentials("Anthropic"):
            raise HTTPException(
                status_code=400,
                detail="ANTHROPIC_API_KEY is not configured. The research-desk debate runs on Claude.",
            )
        # Bounded: a debate is decoupled from the stream, so if the browser disconnects
        # nobody drains this. Cap it and drop (never block the worker) when full — the
        # full result is still saved to disk and recoverable by run_id, and stream
        # termination is driven by done.set(), not by these events.
        events: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=4096)
        done = threading.Event()
        cancel = threading.Event()
        recorder = DebateRecorder(symbol, question, horizon, source)
        with _ACTIVE_DEBATES_LOCK:
            _ACTIVE_DEBATES[recorder.run_id] = cancel

        def _put(event: dict[str, Any]) -> None:
            try:
                events.put_nowait(event)
            except queue.Full:
                pass

        def emit(**event: Any) -> None:
            _put(event)
            recorder.event(**event)  # tap the stream for the saved trajectory (always captured)

        def worker() -> None:
            async def driver() -> None:
                try:
                    # run_id up front so the browser can recover the result if the stream drops.
                    emit(type="start", ticker=symbol, question=question, horizon=horizon,
                         source=source, speed=speed, run_id=recorder.run_id)
                    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                        result, _store = await run_debate(symbol, question, horizon, source,
                                                          emit=emit, record=recorder.record, cancel=cancel,
                                                          fast=fast, bear_model=bear_model, bear_web=bear_web)
                    run_id = recorder.save(result, str(DEBATE_USAGE))
                    _put({"type": "done", "result": result, "usage": str(DEBATE_USAGE),
                          "run_id": run_id, "llm_logs": recorder.grouped_llm_logs()})
                except DebateCancelled:
                    _put({"type": "failure", "message": "Debate cancelled."})
                except Exception as exc:  # pragma: no cover - provider/network errors vary
                    if _is_billing_or_quota_error(exc):
                        emit(type="phase", phase="plan", status="warn",
                             detail="Anthropic credits unavailable; using local SEC-filings fallback")
                        try:
                            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                                result = await run_local_debate_fallback(symbol, question, horizon, emit)
                            run_id = recorder.save(result, "local fallback; no Anthropic tokens")
                            _put({"type": "done", "result": result,
                                  "usage": "local fallback; no Anthropic tokens", "run_id": run_id,
                                  "llm_logs": recorder.grouped_llm_logs()})
                        except Exception as exc2:  # noqa: BLE001
                            _put({"type": "failure", "message": _friendly_error(exc2)})
                    else:
                        _put({"type": "failure", "message": _friendly_error(exc)})
                except BaseException as exc:  # noqa: BLE001 - never let the stream hang
                    _put({"type": "failure", "message": _friendly_error(exc)})

            try:
                asyncio.run(driver())
            except BaseException as exc:  # asyncio.run itself failing must not hang the stream
                _put({"type": "failure", "message": _friendly_error(exc)})
            finally:
                # Guarantee the stream terminates and the registry never leaks, even if
                # asyncio.run() failed before driver()'s own cleanup could run.
                done.set()
                with _ACTIVE_DEBATES_LOCK:
                    _ACTIVE_DEBATES.pop(recorder.run_id, None)

        thread = threading.Thread(target=worker, name=f"nova-web-debate-{symbol}", daemon=True)
        thread.start()
        return StreamingResponse(
            _stream_events(events, done),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/debate/cancel")
    async def cancel_debate(run_id: str = Query(..., min_length=1)) -> dict[str, str]:
        """Explicit stop (the Close button) — signals the worker to stop at its next
        checkpoint so it stops spending tokens. A passive stream drop does NOT do this."""
        run_id = _validate_run_id(run_id)
        with _ACTIVE_DEBATES_LOCK:
            ev = _ACTIVE_DEBATES.get(run_id)
        if ev is not None:
            ev.set()
            return {"status": "cancelling", "run_id": run_id}
        return {"status": "not_active", "run_id": run_id}

    @app.get("/api/debate/recent")
    async def debate_recent() -> dict[str, Any]:
        """Saved debates (newest first) so the UI can re-open a past run by id."""
        return {"runs": DebateRecorder.list_recent(20)}

    @app.get("/api/recent")
    async def signals_recent() -> dict[str, Any]:
        """Saved Signals runs (newest first) so the UI can re-open a past run by id."""
        return {"runs": RunRecorder.list_recent(20)}

    @app.get("/api/run/result")
    async def run_result(run_id: str = Query(..., min_length=1)) -> dict[str, Any]:
        """Re-open a saved Signals run: returns the same card payload the live stream
        delivers on 'done', so the browser can render a past session."""
        run_id = _validate_run_id(run_id)
        if not RunRecorder.exists(run_id):
            return {"status": "not_found", "run_id": run_id}
        recommendation = _load_recommendation(run_id)
        return {
            "status": "done",
            "run_id": run_id,
            "cards": [card.model_dump(mode="json") for card in build_signal_cards(recommendation)],
            "summary": recommendation.summary,
            "llm_logs": _public_llm_logs(run_id),
        }

    @app.get("/api/debate/result")
    async def debate_result(run_id: str = Query(..., min_length=1)) -> dict[str, Any]:
        """Fetch a debate's saved result by run_id — lets the browser recover the memo
        if the live stream dropped before delivering the 'done' event."""
        run_id = _validate_run_id(run_id)
        with _ACTIVE_DEBATES_LOCK:
            still_running = run_id in _ACTIVE_DEBATES
        if DebateRecorder.exists(run_id):
            return {"status": "done", "result": DebateRecorder.load(run_id)}
        if still_running:
            return {"status": "pending", "run_id": run_id}
        return {"status": "not_found", "run_id": run_id}

    return app


app = create_app()


def launch_web(host: str = "127.0.0.1", port: int = 8000, *, reload: bool = False) -> int:
    load_dotenv(override=True)
    import uvicorn

    uvicorn.run("src.web.server:app", host=host, port=port, reload=reload)
    return 0


def _default_provider() -> str:
    return os.getenv("NOVA_MODEL_PROVIDER", "Anthropic")


def _default_model() -> str:
    return os.getenv("NOVA_MODEL_NAME", "claude-opus-4-8")


def _portfolio_mode(value: str) -> str:
    normalized = str(value or "research").strip().lower()
    if normalized not in {"research", "long_only", "long_short"}:
        raise HTTPException(status_code=400, detail="portfolio_mode must be research, long_only, or long_short")
    return normalized


def _parse_tickers(raw: str) -> list[str]:
    tickers = [part.strip().upper() for part in re.split(r"[\s,]+", raw or "") if part.strip()]
    cleaned: list[str] = []
    for ticker in tickers:
        if not re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,9}", ticker):
            raise HTTPException(status_code=400, detail=f"Invalid ticker: {ticker}")
        if ticker not in cleaned:
            cleaned.append(ticker)
    if not cleaned:
        raise HTTPException(status_code=400, detail="At least one ticker is required")
    if len(cleaned) > 8:
        raise HTTPException(status_code=400, detail="Use 8 tickers or fewer for the live demo")
    return cleaned


def _parse_agents(raw: str) -> list[str]:
    return [part.strip() for part in re.split(r"[\s,]+", raw or "") if part.strip()]


_RUN_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,80}$")


def _validate_run_id(run_id: str) -> str:
    """Reject anything that isn't a plain run-id token before it touches the filesystem.
    Blocks path traversal (``..``, ``/``, absolute paths) in the run_id query param that
    flows into runs_root()/<run_id>/... reads."""
    rid = (run_id or "").strip()
    if not _RUN_ID_RE.match(rid) or rid in {".", ".."}:
        raise HTTPException(status_code=400, detail="Invalid run_id")
    return rid


def _remember_run(recommendation: Recommendation) -> None:
    with _RUN_CACHE_LOCK:
        _RUN_CACHE[recommendation.run_id] = recommendation
        while len(_RUN_CACHE) > _MAX_CACHED_RUNS:
            oldest = next(iter(_RUN_CACHE))
            _RUN_CACHE.pop(oldest, None)


def _load_recommendation(run_id: str) -> Recommendation:
    with _RUN_CACHE_LOCK:
        cached = _RUN_CACHE.get(run_id)
    if cached is not None:
        return cached
    if not RunRecorder.exists(run_id):
        raise HTTPException(status_code=404, detail=f"No run found for {run_id}")
    try:
        recommendation = RunRecorder.load_recommendation(run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Run {run_id} has no recommendation") from exc
    _remember_run(recommendation)
    return recommendation


def _debate_context_text(run_id: str) -> str:
    """Compact grounding text from a saved debate, so the chat dock can answer
    follow-ups about a research-desk memo (not just deterministic Signals runs)."""
    d = DebateRecorder.load(run_id)
    memo = d.get("memo", {})
    lines = [
        f"Ticker: {d.get('input', {}).get('ticker', '')}",
        f"Question: {d.get('input', {}).get('question', '')}",
        f"Conviction: {memo.get('conviction', '')} | Lean: {memo.get('directional_lean', '')}",
        f"Bull: {memo.get('bull_case', '')}",
        f"Base: {memo.get('base_case', '')}",
        f"Bear: {memo.get('bear_case', '')}",
    ]
    if memo.get("key_risks"):
        lines.append("Key risks: " + "; ".join(memo["key_risks"]))
    for draft in d.get("specialist_drafts", []):
        lines.append(f"\n[{draft.get('agent', '')}] {draft.get('stance', '')}: {draft.get('summary', '')}")
        for f in draft.get("key_findings", [])[:4]:
            lines.append(f"  - {f.get('claim', '')} (src {f.get('source', '')})")
    if memo.get("citations"):
        lines.append("\nCitations: " + ", ".join(memo["citations"][:15]))
    return "\n".join(lines)


def _public_llm_logs(run_id: str) -> dict[str, list[dict[str, Any]]]:
    grouped = RunRecorder.load_llm_calls(run_id)
    public: dict[str, list[dict[str, Any]]] = {}
    for agent_id, calls in grouped.items():
        rows: list[dict[str, Any]] = []
        for call in calls:
            rows.append({
                "agent_id": call.get("agent_id", agent_id),
                "ticker": call.get("ticker"),
                "provider": call.get("provider"),
                "model": call.get("model"),
                "attempt": call.get("attempt"),
                "latency_ms": call.get("latency_ms"),
                "prompt_tokens": call.get("prompt_tokens"),
                "completion_tokens": call.get("completion_tokens"),
                "error": call.get("error"),
                "seed": call.get("seed"),
                "system_fingerprint": call.get("system_fingerprint"),
                # Full prompt/response (newlines preserved) so the inspector can show
                # the complete call, plus short previews for the collapsed view.
                "prompt": _full_text(call.get("prompt")),
                "response": _full_text(call.get("response")),
                "prompt_preview": _preview(call.get("prompt")),
                "response_preview": _preview(call.get("response")),
            })
        public[agent_id] = rows
    return public


def _full_text(value: Any, limit: int = 60000) -> str:
    """Full text of a recorded prompt/response with newlines intact (capped only to
    avoid a pathological payload). Unlike _preview, does not collapse whitespace."""
    if value is None:
        return ""
    if not isinstance(value, str):
        value = json.dumps(value, default=str, indent=2)
    if len(value) > limit:
        return value[:limit].rstrip() + f"\n… [truncated at {limit} chars]"
    return value


def _preview(value: Any, limit: int = 360) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = json.dumps(value, default=str)
    value = " ".join(str(value).split())
    if len(value) > limit:
        return value[: limit - 1].rstrip() + "..."
    return value


async def _stream_events(events: queue.Queue[dict[str, Any]], done: threading.Event):
    idle = 0
    while True:
        try:
            event = events.get_nowait()
        except queue.Empty:
            if done.is_set():
                break
            await asyncio.sleep(0.08)
            idle += 1
            # Keepalive comment during long silent phases (bear/synth) so proxies and
            # the browser don't drop an idle connection. SSE comments start with ':'.
            if idle % 75 == 0:  # ~every 6s
                yield ": keepalive\n\n"
            continue
        idle = 0
        yield _sse(event, event.get("type", "message"))
    while not events.empty():
        event = events.get_nowait()
        yield _sse(event, event.get("type", "message"))


def _sse(payload: dict[str, Any], event: str = "message") -> str:
    return f"event: {event}\ndata: {json.dumps(payload, default=str)}\n\n"


def _friendly_error(exc: Exception) -> str:
    text = " ".join(str(exc).split())
    if not text:
        return exc.__class__.__name__
    return text[:500]


def _is_billing_or_quota_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(term in text for term in ("credit balance", "quota", "billing", "insufficient"))
