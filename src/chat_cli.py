"""Chat-style terminal interface for Nova Trader."""

from __future__ import annotations

import re
import shlex
import threading
from dataclasses import dataclass, field
from datetime import date, datetime, timezone

from dateutil.relativedelta import relativedelta
from rich import box
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src.engine import run_engine
from src.runs import RunRecorder, runs_root
from src.schemas.context import ModelConfig, RunContext, RunRequest
from src.schemas.portfolio import Portfolio, Position, RealizedGains
from src.schemas.signals import Recommendation
from src.schemas.snapshot import MarketSnapshot
from src.utils.progress import progress

TEAL = "#7fbbb3"
SAGE = "#a7c080"
ROSE = "#e67e80"
AMBER = "#dbbc7f"
ASH = "grey58"
INK = "grey85"

DEFAULT_AGENTS = [
    "technical",
    "fundamentals",
    "growth",
    "valuation",
    "news_sentiment",
    "insider_sentiment",
    "warren_buffett",
]

PROVIDERS = [
    "OpenAI",
    "MiniMax",
    "DeepSeek",
    "Groq",
    "xAI",
    "OpenRouter",
    "Azure OpenAI",
    "Ollama",
]


@dataclass
class ChatSettings:
    provider: str
    model: str
    agents: list[str] = field(default_factory=lambda: DEFAULT_AGENTS.copy())
    initial_cash: float = 100_000.0
    margin_requirement: float = 0.5


@dataclass
class ChatEvent:
    kind: str
    title: str
    body: str = ""


def _default_start_date() -> date:
    return (datetime.now() - relativedelta(months=3)).date()


def _default_end_date() -> date:
    return datetime.now().date()


def _normalize_provider(value: str) -> str:
    compact = value.strip().lower().replace("_", " ").replace("-", " ")
    aliases = {
        "openai": "OpenAI",
        "mini max": "MiniMax",
        "minimax": "MiniMax",
        "deepseek": "DeepSeek",
        "deep seek": "DeepSeek",
        "groq": "Groq",
        "xai": "xAI",
        "x ai": "xAI",
        "openrouter": "OpenRouter",
        "open router": "OpenRouter",
        "azure openai": "Azure OpenAI",
        "azure": "Azure OpenAI",
        "ollama": "Ollama",
    }
    return aliases.get(compact, value.strip())


def _extract_tickers(text: str) -> list[str]:
    ignored = {
        "ANALYZE",
        "RUN",
        "SHOW",
        "RERUN",
        "MODEL",
        "PROVIDER",
        "AGENTS",
        "DETAIL",
        "DETAILS",
        "WHY",
        "HELP",
        "EXIT",
        "QUIT",
        "SET",
        "SETUP",
        "USE",
        "WITH",
        "AND",
        "ME",
        "FOR",
        "THE",
        "LAST",
    }
    found = re.findall(r"\b[A-Z][A-Z0-9.]{0,5}\b", text.upper())
    tickers: list[str] = []
    for token in found:
        if token in ignored:
            continue
        if token not in tickers:
            tickers.append(token)
    return tickers


def _is_analysis_prompt(text: str) -> bool:
    lower = text.lower()
    if any(word in lower for word in ("analyze", "analyse", "recommend", "evaluate", "check", "run ")):
        return True
    compact = re.sub(r"[\s,]+", "", text)
    return bool(compact) and bool(re.fullmatch(r"[A-Za-z0-9.,\s]+", text)) and any(
        token.isupper() for token in re.findall(r"\b[A-Z][A-Z0-9.]{0,5}\b", text)
    )


def _provider_choices() -> str:
    return ", ".join(PROVIDERS)


def _model_choices_for(provider: str) -> list[str]:
    try:
        from src.llm.models import AVAILABLE_MODELS, OLLAMA_MODELS
    except Exception:
        return []

    return [
        model.model_name
        for model in AVAILABLE_MODELS + OLLAMA_MODELS
        if model.provider.value.lower() == provider.lower() and model.model_name
    ]


def _build_context(tickers: list[str], settings: ChatSettings) -> RunContext:
    portfolio = Portfolio(
        cash=settings.initial_cash,
        margin_requirement=settings.margin_requirement,
        margin_used=0.0,
        positions={ticker: Position() for ticker in tickers},
        realized_gains={ticker: RealizedGains() for ticker in tickers},
    )
    request = RunRequest(
        tickers=tickers,
        start_date=_default_start_date(),
        end_date=_default_end_date(),
        portfolio=portfolio,
        model=ModelConfig(provider=settings.provider, name=settings.model),
        selected_agents=settings.agents,
    )
    return RunContext(request=request, as_of=datetime.now(timezone.utc))


def _build_context_from_metadata(meta: dict) -> tuple[RunContext, MarketSnapshot]:
    request = RunRequest(
        tickers=meta["tickers"],
        start_date=date.fromisoformat(meta["start_date"]),
        end_date=date.fromisoformat(meta["end_date"]),
        portfolio=Portfolio(
            cash=100_000.0,
            margin_requirement=0.5,
            positions={ticker: Position() for ticker in meta["tickers"]},
            realized_gains={ticker: RealizedGains() for ticker in meta["tickers"]},
        ),
        model=ModelConfig(**meta["model"]),
        selected_agents=meta.get("selected_agents", []),
    )
    ctx = RunContext(request=request, as_of=datetime.now(timezone.utc), seed=meta.get("seed"))
    snapshot = MarketSnapshot.model_validate(RunRecorder.load_snapshot_dict(meta["run_id"]))
    return ctx, snapshot


def _status_color(value: str) -> str:
    normalized = value.lower()
    if normalized in {"buy", "bullish", "ok"}:
        return SAGE
    if normalized in {"sell", "short", "bearish", "failed"}:
        return ROSE
    return AMBER


def _field_value(obj: object, name: str, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _shorten(text: str, limit: int = 220) -> str:
    text = " ".join(str(text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _recommendation_summary_text(recommendation: Recommendation) -> str:
    lines = [
        f"Run {recommendation.run_id}",
        f"Summary: {recommendation.summary}",
        "",
        f"{'Ticker':<8} {'Consensus':<10} {'Action':<8} {'Conf':>5}  Decision",
        "-" * 74,
    ]

    for ticker in recommendation.tickers:
        consensus = recommendation.consensus.get(ticker)
        decision = recommendation.decisions.per_ticker.get(ticker)
        if not decision:
            continue
        consensus_text = str(_field_value(consensus, "direction", "n/a")).upper()
        action = str(_field_value(decision, "action", "hold")).upper()
        confidence = float(_field_value(decision, "confidence", 0.0))
        reasoning = str(_field_value(decision, "reasoning", "") or "")
        lines.append(f"{ticker:<8} {consensus_text:<10} {action:<8} {confidence:>4.0%}  {reasoning}")

    plan = recommendation.decisions.hedge_plan
    if plan.pairs or plan.blocked_longs:
        lines.extend(["", f"Hedge plan: {plan.status}"])
        for pair in plan.pairs:
            lines.append(
                f"  LONG {pair.long_quantity} {pair.long_ticker} vs "
                f"SHORT {pair.short_quantity} {pair.short_ticker}"
            )
        if plan.blocked_longs:
            lines.append(f"  Blocked longs: {', '.join(plan.blocked_longs)}")

    lines.append("")
    lines.append("Analyst reasoning")
    lines.append("-" * 74)
    for ticker in recommendation.tickers:
        lines.append(f"{ticker}")
        ticker_signals = [signal for signal in recommendation.signals if signal.ticker == ticker]
        if not ticker_signals:
            lines.append("  No analyst signals.")
            continue
        for signal in ticker_signals:
            status = signal.status
            direction = signal.direction.upper()
            confidence = f"{signal.confidence:.0%}" if status == "ok" else status
            reasoning = signal.error or signal.reasoning or "No reasoning supplied."
            lines.append(
                f"  - {signal.agent_id:<18} {direction:<8} {confidence:<9} {_shorten(reasoning)}"
            )

    ok = sum(1 for signal in recommendation.signals if signal.status == "ok")
    abstained = sum(1 for signal in recommendation.signals if signal.status == "abstained")
    failed = sum(1 for signal in recommendation.signals if signal.status == "failed")
    lines.extend([
        "",
        f"{len(recommendation.signals)} signals: {ok} ok, {abstained} abstained, {failed} failed",
        f"Saved: {runs_root() / recommendation.run_id}/",
    ])
    return "\n".join(lines)


def _ticker_details_text(recommendation: Recommendation, ticker: str) -> str:
    ticker = ticker.upper()
    if ticker not in recommendation.tickers:
        return f"{ticker} was not part of run {recommendation.run_id}."

    consensus = recommendation.consensus.get(ticker)
    decision = recommendation.decisions.per_ticker.get(ticker)
    lines = [f"{ticker} details from run {recommendation.run_id}", ""]
    if consensus:
        lines.append(
            "Consensus: "
            f"{_field_value(consensus, 'direction', 'n/a')} "
            f"({_field_value(consensus, 'confidence', 0.0):.0%})"
        )
    if decision:
        lines.append(
            "Decision: "
            f"{_field_value(decision, 'action', 'hold').upper()} "
            f"({_field_value(decision, 'confidence', 0.0):.0%})"
        )
        reasoning = _field_value(decision, "reasoning", "")
        if reasoning:
            lines.append(f"Reason: {reasoning}")

    lines.extend(["", "Analyst reasoning"])
    for signal in [s for s in recommendation.signals if s.ticker == ticker]:
        status = signal.status
        direction = signal.direction.upper()
        confidence = f"{signal.confidence:.0%}" if status == "ok" else status
        reasoning = signal.error or signal.reasoning or "No reasoning supplied."
        lines.append(f"- {signal.agent_id}: {direction} {confidence}")
        lines.append(f"  {_shorten(reasoning, 420)}")
    return "\n".join(lines)


def _render_recommendation(console: Console, recommendation: Recommendation) -> None:
    console.print()
    console.print(
        Panel(
            f"[bold {TEAL}]Run[/bold {TEAL}] {recommendation.run_id}\n"
            f"[{ASH}]summary[/{ASH}] {escape(recommendation.summary)}",
            border_style=TEAL,
            padding=(1, 2),
        )
    )

    table = Table(box=box.SIMPLE, show_edge=False, pad_edge=False, padding=(0, 2))
    table.add_column("Ticker", style=f"bold {TEAL}", no_wrap=True)
    table.add_column("Consensus", no_wrap=True)
    table.add_column("Action", no_wrap=True)
    table.add_column("Confidence", justify="right", no_wrap=True)
    table.add_column("Decision", style=INK, overflow="fold")

    for ticker in recommendation.tickers:
        consensus = recommendation.consensus.get(ticker)
        decision = recommendation.decisions.per_ticker.get(ticker)
        if not decision:
            continue
        consensus_text = _field_value(consensus, "direction", "n/a")
        action = _field_value(decision, "action", "hold")
        confidence = float(_field_value(decision, "confidence", 0.0))
        reasoning = _field_value(decision, "reasoning", "")
        table.add_row(
            escape(ticker),
            f"[{_status_color(consensus_text)}]{consensus_text.upper()}[/{_status_color(consensus_text)}]",
            f"[bold {_status_color(action)}]{action.upper()}[/bold {_status_color(action)}]",
            f"{confidence:.0%}",
            escape(reasoning or ""),
        )

    console.print(table)

    plan = recommendation.decisions.hedge_plan
    if plan.pairs or plan.blocked_longs:
        hedge = Table(box=box.SIMPLE, show_edge=False, pad_edge=False, padding=(0, 2))
        hedge.add_column("Hedge", style=f"bold {TEAL}", no_wrap=True)
        hedge.add_column("Details", style=INK)
        for pair in plan.pairs:
            hedge.add_row(
                "pair",
                f"LONG {pair.long_quantity} {escape(pair.long_ticker)} vs SHORT {pair.short_quantity} {escape(pair.short_ticker)}",
            )
        if plan.blocked_longs:
            hedge.add_row("blocked", escape(", ".join(plan.blocked_longs)))
        console.print(hedge)

    ok = sum(1 for signal in recommendation.signals if signal.status == "ok")
    abstained = sum(1 for signal in recommendation.signals if signal.status == "abstained")
    failed = sum(1 for signal in recommendation.signals if signal.status == "failed")
    console.print(
        f"[{ASH}]saved to {runs_root() / recommendation.run_id}/  "
        f"{len(recommendation.signals)} signals: "
        f"[{SAGE}]{ok} ok[/{SAGE}], "
        f"[{AMBER}]{abstained} abstained[/{AMBER}], "
        f"[{ROSE}]{failed} failed[/{ROSE}][/{ASH}]"
    )
    console.print()


class NovaChat:
    def __init__(self, console: Console, settings: ChatSettings):
        self.console = console
        self.settings = settings
        self.last_run_id: str | None = None
        self.last_recommendation: Recommendation | None = None
        self._app = None
        self._transcript_area = None
        self._status_area = None
        self._input_area = None
        self._header_area = None
        self._transcript: list[str] = []
        self._status_lines: dict[str, str] = {}
        self._busy = False
        self._ui_lock = threading.Lock()

    def run(self) -> int:
        try:
            from prompt_toolkit.application import Application
            from prompt_toolkit.key_binding import KeyBindings
            from prompt_toolkit.layout import HSplit, Layout, VSplit, Window
            from prompt_toolkit.layout.controls import FormattedTextControl
            from prompt_toolkit.styles import Style
            from prompt_toolkit.widgets import Frame, TextArea
        except ImportError:
            self.console.print("[red]Chat CLI requires prompt_toolkit. Run `uv pip install -e .`.[/red]")
            return 2

        self._transcript_area = TextArea(
            text="",
            read_only=True,
            scrollbar=True,
            wrap_lines=True,
            focusable=False,
        )
        self._status_area = TextArea(
            text="",
            read_only=True,
            scrollbar=True,
            wrap_lines=True,
            focusable=False,
        )
        self._input_area = TextArea(
            height=3,
            prompt="nova › ",
            multiline=False,
            wrap_lines=False,
        )

        def submit(buffer):
            text = buffer.text.strip()
            buffer.reset()
            if text:
                self._dispatch_tui(text)
            return True

        self._input_area.buffer.accept_handler = submit
        self._header_area = Window(
            FormattedTextControl(lambda: self._header_fragments()),
            height=3,
            style="class:header",
        )

        footer = Window(
            FormattedTextControl(
                " Enter submit   Ctrl-C/Ctrl-D exit   commands: analyze AAPL,NVDA | model MiniMax MiniMax-M2.7 | show last | help "
            ),
            height=1,
            style="class:footer",
        )

        root = HSplit(
            [
                self._header_area,
                VSplit(
                    [
                        Frame(self._transcript_area, title="Transcript"),
                        Frame(self._status_area, title="Run Status", width=42),
                    ],
                    padding=1,
                ),
                Frame(self._input_area, title="Message"),
                footer,
            ]
        )

        bindings = KeyBindings()

        @bindings.add("c-c")
        @bindings.add("c-d")
        def _(event):
            event.app.exit(result=0)

        style = Style.from_dict(
            {
                "header": "bg:#111111 #7fbbb3 bold",
                "footer": "bg:#111111 #888888",
                "frame.border": "#7fbbb3",
                "frame.label": "#7fbbb3 bold",
            }
        )

        self._app = Application(
            layout=Layout(root, focused_element=self._input_area),
            key_bindings=bindings,
            full_screen=True,
            mouse_support=True,
            style=style,
        )
        self._emit(ChatEvent("assistant", "Ready", "Ask for a recommendation or change the run settings. Try `analyze AAPL,NVDA` or `help`."))
        self._refresh_tui()
        return int(self._app.run() or 0)

    def _header_fragments(self):
        busy = "running" if self._busy else "ready"
        agents = "default analyst set" if self.settings.agents == DEFAULT_AGENTS else ",".join(self.settings.agents)
        return [
            ("class:header", "  ◆ Nova Trader  "),
            ("#888888", "chat workspace\n"),
            ("#aaaaaa", f"  model "),
            ("#ffffff bold", f"{self.settings.provider} / {self.settings.model}"),
            ("#aaaaaa", "   agents "),
            ("#ffffff", agents[:48]),
            ("#aaaaaa", "   status "),
            ("#a7c080 bold" if not self._busy else "#dbbc7f bold", busy),
        ]

    def _emit(self, event: ChatEvent) -> None:
        with self._ui_lock:
            if event.kind == "tool":
                self._status_lines[event.title] = event.body
            else:
                prefix = {
                    "user": "You",
                    "assistant": "Nova",
                    "result": "Recommendation",
                    "error": "Error",
                    "system": "System",
                }.get(event.kind, event.kind.title())
                block = f"{prefix}: {event.title}"
                if event.body:
                    block += f"\n{event.body}"
                self._transcript.append(block)
        self._refresh_tui()

    def _refresh_tui(self) -> None:
        if not self._app or not self._transcript_area or not self._status_area:
            return
        from prompt_toolkit.document import Document

        with self._ui_lock:
            transcript = "\n\n".join(self._transcript[-80:])
            status = self._status_text()
        self._transcript_area.buffer.set_document(Document(transcript, cursor_position=len(transcript)), bypass_readonly=True)
        self._status_area.buffer.set_document(Document(status, cursor_position=len(status)), bypass_readonly=True)
        self._app.invalidate()

    def _status_text(self) -> str:
        lines = [
            f"Model: {self.settings.provider}",
            f"Name:  {self.settings.model}",
            f"Last:  {self.last_run_id or 'none'}",
            "",
        ]
        if self._busy:
            lines.append("Run: running")
        else:
            lines.append("Run: ready")
        lines.append("")
        if not self._status_lines:
            lines.append("No active tool events.")
        else:
            for label, status in list(self._status_lines.items())[-28:]:
                marker = "✓" if status.lower() == "done" else "•"
                lines.append(f"{marker} {label}\n  {status}")
        return "\n".join(lines)

    def _dispatch_tui(self, text: str) -> None:
        if self._busy:
            self._emit(ChatEvent("system", "A run is already in progress.", "Wait for it to finish before sending another command."))
            return

        lower = text.lower().strip()
        if lower in {"exit", "quit", "/exit", "/quit"}:
            if self._app:
                self._app.exit(result=0)
            return

        self._emit(ChatEvent("user", text))
        tickers = _extract_tickers(text)

        if lower in {"help", "/help", "?"}:
            self._emit(ChatEvent("assistant", "Available commands", self._help_text()))
            return
        if lower in {"hi", "hello", "hey", "yo"}:
            self._emit(ChatEvent("assistant", "Hi.", self._intro_text()))
            return
        if lower in {"status", "/status", "settings", "/settings"}:
            self._emit(ChatEvent("assistant", "Current settings", self._settings_text()))
            return
        if tickers and lower.startswith(("details", "detail", "why", "explain")):
            self._emit_ticker_details(tickers[0])
            return
        if lower.startswith(("explain", "what is", "what are", "how does")):
            self._emit(ChatEvent("assistant", "How Nova works", self._product_explanation_text()))
            return
        if lower.startswith(("what's", "whats", "what is this", "what's this")):
            self._emit(ChatEvent("assistant", "What this is", self._product_explanation_text()))
            return
        if lower.startswith(("model ", "/model ", "use model ", "use ")):
            self._set_model_from_command(text, emit_to_tui=True)
            return
        if lower.startswith(("agents ", "/agents ")):
            self._set_agents_from_command(text, emit_to_tui=True)
            return
        if lower in {"show last", "/show last"}:
            if self.last_run_id:
                self._show_run_tui(self.last_run_id)
            else:
                self._emit(ChatEvent("assistant", "No previous run in this session."))
            return
        if lower.startswith(("show ", "/show ")):
            self._show_run_tui(text.split(" ", 1)[1].strip())
            return
        if lower in {"rerun last", "/rerun last"}:
            if self.last_run_id:
                self._run_in_thread("rerun", [self.last_run_id])
            else:
                self._emit(ChatEvent("assistant", "No previous run in this session."))
            return
        if lower.startswith(("rerun ", "/rerun ")):
            self._run_in_thread("rerun", [text.split(" ", 1)[1].strip()])
            return

        if tickers and _is_analysis_prompt(text):
            self._run_in_thread("analyze", tickers)
            return

        self._emit(
            ChatEvent(
                "assistant",
                "I can help with Nova recommendations.",
                self._intro_text(),
            )
        )

    def _run_in_thread(self, mode: str, values: list[str]) -> None:
        self._busy = True
        self._status_lines.clear()
        self._refresh_tui()
        thread = threading.Thread(target=self._run_worker, args=(mode, values), daemon=True)
        thread.start()

    def _run_worker(self, mode: str, values: list[str]) -> None:
        try:
            if mode == "analyze":
                self._analyze_tui(values)
            elif mode == "rerun":
                self._rerun_tui(values[0])
        finally:
            self._busy = False
            self._refresh_tui()

    def handle(self, text: str) -> int | None:
        lower = text.lower().strip()
        tickers = _extract_tickers(text)
        if lower in {"exit", "quit", "/exit", "/quit"}:
            return 0
        if lower in {"help", "/help", "?"}:
            self._print_help()
            return None
        if lower in {"hi", "hello", "hey", "yo"}:
            self.console.print(self._intro_text())
            return None
        if lower in {"status", "/status", "settings", "/settings"}:
            self._print_settings()
            return None
        if tickers and lower.startswith(("details", "detail", "why", "explain")):
            self._print_ticker_details(tickers[0])
            return None
        if lower.startswith(("explain", "what is", "what are", "how does")):
            self._print_product_explanation()
            return None
        if lower.startswith(("what's", "whats", "what is this", "what's this")):
            self._print_product_explanation()
            return None
        if lower.startswith(("model ", "/model ", "use model ", "use ")):
            self._handle_model(text)
            return None
        if lower.startswith(("agents ", "/agents ")):
            self._handle_agents(text)
            return None
        if lower in {"show last", "/show last"}:
            if self.last_run_id:
                self._show_run(self.last_run_id)
            else:
                self.console.print(f"[{AMBER}]No previous run in this session.[/{AMBER}]")
            return None
        if lower.startswith(("show ", "/show ")):
            self._handle_show(text)
            return None
        if lower in {"rerun last", "/rerun last"}:
            if self.last_run_id:
                self._rerun(self.last_run_id)
            else:
                self.console.print(f"[{AMBER}]No previous run in this session.[/{AMBER}]")
            return None
        if lower.startswith(("rerun ", "/rerun ")):
            self._handle_rerun(text)
            return None

        if tickers and _is_analysis_prompt(text):
            self._analyze(tickers)
            return None

        self.console.print(self._intro_text())
        return None

    def _help_text(self) -> str:
        return "\n".join(
            [
                "analyze AAPL,NVDA          run a recommendation",
                "details AAPL               show analyst reasoning from the last run",
                "explain NVDA               explain a ticker decision from the last run",
                "show <run_id>              print a saved recommendation",
                "show last                  print the last run from this session",
                "rerun <run_id>             replay a saved run snapshot",
                "model MiniMax MiniMax-M2.7 switch provider/model",
                "agents technical,valuation choose analyst agents",
                "status                     show current model and agents",
                "exit                       close Nova",
            ]
        )

    def _intro_text(self) -> str:
        return "\n".join(
            [
                "This is Nova Trader, a portfolio-aware recommendation agent.",
                "Ask `analyze AAPL,NVDA` to run the analyst pipeline.",
                "After a run, ask `details AAPL` or `explain NVDA` to inspect the reasoning.",
                "Use `model MiniMax MiniMax-M2.7`, `agents technical,valuation`, or `help` to control the workspace.",
            ]
        )

    def _product_explanation_text(self) -> str:
        return "\n".join(
            [
                "Nova is a portfolio-aware recommendation agent.",
                "",
                "A run builds one market snapshot, slices typed views for analyst agents,",
                "aggregates their signals, applies risk limits, and then asks the portfolio",
                "manager for a hedged decision.",
                "",
                "The transcript shows the conversation and final decision. The status pane",
                "shows live tool and analyst progress while a run is active.",
                "",
                "Use `analyze AAPL,NVDA` to run it, `details AAPL` to inspect reasoning,",
                "`model MiniMax MiniMax-M2.7` to switch models, and `show last` to inspect",
                "the latest recommendation.",
            ]
        )

    def _settings_text(self) -> str:
        agent_label = "default analyst set" if self.settings.agents == DEFAULT_AGENTS else ",".join(self.settings.agents)
        return "\n".join(
            [
                f"provider: {self.settings.provider}",
                f"model:    {self.settings.model}",
                f"agents:   {agent_label}",
                f"last run: {self.last_run_id or 'none'}",
            ]
        )

    def _set_model_from_command(self, text: str, *, emit_to_tui: bool = False) -> tuple[str, str]:
        cleaned = text.strip()
        for prefix in ("/model", "model", "use model", "use"):
            if cleaned.lower().startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
                break
        if not cleaned:
            message = f"Available providers: {_provider_choices()}"
            if emit_to_tui:
                self._emit(ChatEvent("assistant", message))
            else:
                self.console.print(f"[{ASH}]{message}[/{ASH}]")
            return self.settings.provider, self.settings.model

        parts = shlex.split(cleaned)
        provider = _normalize_provider(parts[0])
        if provider == "Azure" and len(parts) > 1 and parts[1].lower() == "openai":
            provider = "Azure OpenAI"
            parts = parts[1:]
        model = parts[1] if len(parts) > 1 else None
        choices = _model_choices_for(provider)
        if not model:
            model = choices[0] if choices else self.settings.model
        self.settings.provider = provider
        self.settings.model = model

        body = f"Model set to {provider} / {model}"
        if choices:
            body += f"\nKnown models: {', '.join(choices)}"
        if emit_to_tui:
            self._emit(ChatEvent("assistant", "Model updated", body))
        else:
            self.console.print(f"[{SAGE}]Model set to {escape(provider)} / {escape(model)}[/{SAGE}]", highlight=False)
            if choices:
                self.console.print(f"[{ASH}]Known models: {escape(', '.join(choices))}[/{ASH}]", highlight=False)
        self._refresh_tui()
        return provider, model

    def _set_agents_from_command(self, text: str, *, emit_to_tui: bool = False) -> list[str]:
        cleaned = text.split(" ", 1)[1] if " " in text else ""
        agents = [agent.strip() for agent in cleaned.split(",") if agent.strip()]
        if not agents:
            self.settings.agents = DEFAULT_AGENTS.copy()
            message = "Agents reset to default."
        else:
            self.settings.agents = agents
            message = f"Agents set to {','.join(agents)}"

        if emit_to_tui:
            self._emit(ChatEvent("assistant", "Agents updated", message))
        else:
            self.console.print(f"[{SAGE}]{escape(message)}[/{SAGE}]", highlight=False)
        self._refresh_tui()
        return self.settings.agents

    def _emit_ticker_details(self, ticker: str) -> None:
        if not self.last_recommendation:
            self._emit(
                ChatEvent(
                    "assistant",
                    "No recommendation loaded.",
                    "Run `analyze AAPL,NVDA` first, or use `show <run_id>` to load a saved run.",
                )
            )
            return
        self._emit(ChatEvent("assistant", f"Details for {ticker.upper()}", _ticker_details_text(self.last_recommendation, ticker)))

    def _print_ticker_details(self, ticker: str) -> None:
        if not self.last_recommendation:
            self.console.print(f"[{AMBER}]No recommendation loaded. Run analyze first or show a saved run.[/{AMBER}]")
            return
        self.console.print(_ticker_details_text(self.last_recommendation, ticker))

    def _print_header(self) -> None:
        title = Text()
        title.append("Nova Trader", style=f"bold {TEAL}")
        title.append("  chat", style=ASH)
        agent_label = "default analyst set" if self.settings.agents == DEFAULT_AGENTS else ",".join(self.settings.agents)
        body = (
            f"[{ASH}]Ask for recommendations, change models, and inspect saved runs.[/{ASH}]\n\n"
            f"[{ASH}]model[/{ASH}] {escape(self.settings.provider)} / {escape(self.settings.model)}\n"
            f"[{ASH}]agents[/{ASH}] {escape(agent_label)}\n\n"
            "[bold]Try[/bold]\n"
            "  analyze AAPL,NVDA\n"
            "  model MiniMax MiniMax-M2.7\n"
            "  show last\n"
            "  help"
        )
        self.console.print()
        self.console.print(Panel(body, title=title, border_style=TEAL, padding=(1, 2)))
        self.console.print()

    def _print_help(self) -> None:
        self.console.print(
            Panel(
                self._help_text(),
                title="Commands",
                border_style=TEAL,
                padding=(1, 2),
            )
        )

    def _print_product_explanation(self) -> None:
        self.console.print(
            Panel(
                self._product_explanation_text(),
                title="How Nova Works",
                border_style=TEAL,
                padding=(1, 2),
            )
        )

    def _print_settings(self) -> None:
        self.console.print(
            Panel(
                escape(self._settings_text()),
                title="Settings",
                border_style=TEAL,
                padding=(1, 2),
            )
        )

    def _handle_model(self, text: str) -> None:
        self._set_model_from_command(text)

    def _handle_agents(self, text: str) -> None:
        self._set_agents_from_command(text)

    def _handle_show(self, text: str) -> None:
        run_id = text.split(" ", 1)[1].strip()
        if run_id == "last" and self.last_run_id:
            run_id = self.last_run_id
        self._show_run(run_id)

    def _handle_rerun(self, text: str) -> None:
        run_id = text.split(" ", 1)[1].strip()
        if run_id == "last" and self.last_run_id:
            run_id = self.last_run_id
        self._rerun(run_id)

    def _progress_handler(self, agent_name: str, ticker: str | None, status: str, *_) -> None:
        label = agent_name.replace("_", " ").title()
        scope = f" [{ticker}]" if ticker else ""
        color = SAGE if status.lower() == "done" else ASH
        self.console.print(f"[{color}]tool[/{color}] {escape(label)}{escape(scope)}: {escape(status)}")

    def _progress_handler_tui(self, agent_name: str, ticker: str | None, status: str, *_) -> None:
        label = agent_name.replace("_", " ").title()
        scope = f" [{ticker}]" if ticker else ""
        self._emit(ChatEvent("tool", f"{label}{scope}", status))

    def _analyze(self, tickers: list[str]) -> None:
        self.console.print()
        self.console.print(f"[bold {TEAL}]user[/bold {TEAL}] analyze {escape(','.join(tickers))}")
        self.console.print(
            f"[bold {TEAL}]nova[/bold {TEAL}] Building snapshot and running "
            f"{len(self.settings.agents)} agents with {escape(self.settings.provider)} / {escape(self.settings.model)}."
        )

        ctx = _build_context(tickers, self.settings)
        progress.register_handler(self._progress_handler)
        try:
            recommendation = run_engine(
                ctx,
                selected_agents=ctx.request.selected_agents or None,
                record=True,
            )
        except Exception as exc:
            self.console.print(f"[{ROSE}]Run failed: {exc}[/{ROSE}]")
            return
        finally:
            progress.unregister_handler(self._progress_handler)

        self.last_run_id = recommendation.run_id
        self.last_recommendation = recommendation
        _render_recommendation(self.console, recommendation)

    def _analyze_tui(self, tickers: list[str]) -> None:
        self._emit(
            ChatEvent(
                "assistant",
                f"Analyzing {','.join(tickers)}",
                f"Building a market snapshot and running {len(self.settings.agents)} agents with "
                f"{self.settings.provider} / {self.settings.model}.",
            )
        )

        ctx = _build_context(tickers, self.settings)
        progress.register_handler(self._progress_handler_tui)
        try:
            recommendation = run_engine(
                ctx,
                selected_agents=ctx.request.selected_agents or None,
                record=True,
            )
        except Exception as exc:
            self._emit(ChatEvent("error", "Run failed", str(exc)))
            return
        finally:
            progress.unregister_handler(self._progress_handler_tui)

        self.last_run_id = recommendation.run_id
        self.last_recommendation = recommendation
        self._emit(ChatEvent("result", f"Run {recommendation.run_id}", _recommendation_summary_text(recommendation)))

    def _show_run(self, run_id: str) -> None:
        if not RunRecorder.exists(run_id):
            self.console.print(f"[{ROSE}]No run found at {runs_root() / run_id}[/{ROSE}]")
            return
        try:
            recommendation = Recommendation.model_validate(RunRecorder.load_recommendation_dict(run_id))
        except FileNotFoundError:
            self.console.print(f"[{ROSE}]Run {run_id} has no recommendation.json.[/{ROSE}]")
            return
        self.last_run_id = run_id
        self.last_recommendation = recommendation
        _render_recommendation(self.console, recommendation)

    def _show_run_tui(self, run_id: str) -> None:
        if not RunRecorder.exists(run_id):
            self._emit(ChatEvent("error", "Run not found", str(runs_root() / run_id)))
            return
        try:
            recommendation = Recommendation.model_validate(RunRecorder.load_recommendation_dict(run_id))
        except FileNotFoundError:
            self._emit(ChatEvent("error", f"Run {run_id} has no recommendation.json."))
            return
        self.last_run_id = run_id
        self.last_recommendation = recommendation
        self._emit(ChatEvent("result", f"Run {recommendation.run_id}", _recommendation_summary_text(recommendation)))

    def _rerun(self, run_id: str) -> None:
        if not RunRecorder.exists(run_id):
            self.console.print(f"[{ROSE}]No run found at {runs_root() / run_id}[/{ROSE}]")
            return
        self.console.print(f"[bold {TEAL}]nova[/bold {TEAL}] Replaying saved snapshot {run_id}.")
        meta = RunRecorder.load_metadata(run_id)
        ctx, snapshot = _build_context_from_metadata(meta)
        progress.register_handler(self._progress_handler)
        try:
            recommendation = run_engine(
                ctx,
                selected_agents=ctx.request.selected_agents or None,
                snapshot=snapshot,
                record=True,
            )
        except Exception as exc:
            self.console.print(f"[{ROSE}]Rerun failed: {exc}[/{ROSE}]")
            return
        finally:
            progress.unregister_handler(self._progress_handler)

        self.last_run_id = recommendation.run_id
        self.last_recommendation = recommendation
        _render_recommendation(self.console, recommendation)

    def _rerun_tui(self, run_id: str) -> None:
        if not RunRecorder.exists(run_id):
            self._emit(ChatEvent("error", "Run not found", str(runs_root() / run_id)))
            return
        self._emit(ChatEvent("assistant", f"Replaying saved snapshot {run_id}."))
        meta = RunRecorder.load_metadata(run_id)
        ctx, snapshot = _build_context_from_metadata(meta)
        progress.register_handler(self._progress_handler_tui)
        try:
            recommendation = run_engine(
                ctx,
                selected_agents=ctx.request.selected_agents or None,
                snapshot=snapshot,
                record=True,
            )
        except Exception as exc:
            self._emit(ChatEvent("error", "Rerun failed", str(exc)))
            return
        finally:
            progress.unregister_handler(self._progress_handler_tui)

        self.last_run_id = recommendation.run_id
        self.last_recommendation = recommendation
        self._emit(ChatEvent("result", f"Run {recommendation.run_id}", _recommendation_summary_text(recommendation)))


def launch_chat(console: Console, provider: str, model: str) -> int:
    settings = ChatSettings(provider=provider, model=model)
    return NovaChat(console, settings).run()
