"""Chat-style terminal interface for Nova Trader."""

from __future__ import annotations

import re
import shlex
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
        consensus = recommendation.consensus.per_ticker.get(ticker)
        decision = recommendation.decisions.per_ticker.get(ticker)
        if not decision:
            continue
        consensus_text = consensus.direction if consensus else "n/a"
        table.add_row(
            escape(ticker),
            f"[{_status_color(consensus_text)}]{consensus_text.upper()}[/{_status_color(consensus_text)}]",
            f"[bold {_status_color(decision.action)}]{decision.action.upper()}[/bold {_status_color(decision.action)}]",
            f"{decision.confidence:.0%}",
            escape(decision.reasoning or ""),
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

    def run(self) -> int:
        try:
            from prompt_toolkit import PromptSession
            from prompt_toolkit.formatted_text import HTML
        except ImportError:
            self.console.print("[red]Chat CLI requires prompt_toolkit. Run `uv pip install -e .`.[/red]")
            return 2

        self._print_header()
        session = PromptSession()

        while True:
            try:
                prompt = HTML("<ansiteal>nova</ansiteal> <ansiblue>›</ansiblue> ")
                text = session.prompt(prompt).strip()
            except (EOFError, KeyboardInterrupt):
                self.console.print()
                return 0

            if not text:
                continue
            exit_code = self.handle(text)
            if exit_code is not None:
                return exit_code

    def handle(self, text: str) -> int | None:
        lower = text.lower().strip()
        if lower in {"exit", "quit", "/exit", "/quit"}:
            return 0
        if lower in {"help", "/help", "?"}:
            self._print_help()
            return None
        if lower in {"status", "/status", "settings", "/settings"}:
            self._print_settings()
            return None
        if lower.startswith(("explain", "what is", "what are", "how does")):
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

        tickers = _extract_tickers(text)
        if tickers and _is_analysis_prompt(text):
            self._analyze(tickers)
            return None

        self.console.print(
            f"[{AMBER}]I need one or more tickers. Try `analyze AAPL,NVDA`, "
            "`model MiniMax MiniMax-M2.7`, or `help`.[/{AMBER}]"
        )
        return None

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
                "\n".join(
                    [
                        "[bold]analyze AAPL,NVDA[/bold]       run a recommendation",
                        "[bold]show <run_id>[/bold]           print a saved recommendation",
                        "[bold]show last[/bold]               print the last run from this session",
                        "[bold]rerun <run_id>[/bold]          replay a saved run snapshot",
                        "[bold]model MiniMax MiniMax-M2.7[/bold]  switch provider/model",
                        "[bold]agents technical,valuation[/bold] choose analyst agents",
                        "[bold]status[/bold]                  show current model and agents",
                        "[bold]exit[/bold]                    close Nova",
                    ]
                ),
                title="Commands",
                border_style=TEAL,
                padding=(1, 2),
            )
        )

    def _print_product_explanation(self) -> None:
        self.console.print(
            Panel(
                "\n".join(
                    [
                        "Nova is a portfolio-aware recommendation agent.",
                        "",
                        "A run builds one market snapshot, slices typed views for analyst agents,",
                        "aggregates their signals, applies risk limits, and then asks the portfolio",
                        "manager for a hedged decision.",
                        "",
                        "Use `analyze AAPL,NVDA` to run it, `model MiniMax MiniMax-M2.7` to switch",
                        "models, and `show last` to inspect the latest recommendation.",
                    ]
                ),
                title="How Nova Works",
                border_style=TEAL,
                padding=(1, 2),
            )
        )

    def _print_settings(self) -> None:
        agent_label = "default analyst set" if self.settings.agents == DEFAULT_AGENTS else ",".join(self.settings.agents)
        self.console.print(
            Panel(
                f"[{ASH}]provider[/{ASH}] {escape(self.settings.provider)}\n"
                f"[{ASH}]model[/{ASH}] {escape(self.settings.model)}\n"
                f"[{ASH}]agents[/{ASH}] {escape(agent_label)}\n"
                f"[{ASH}]last run[/{ASH}] {escape(self.last_run_id or 'none')}",
                title="Settings",
                border_style=TEAL,
                padding=(1, 2),
            )
        )

    def _handle_model(self, text: str) -> None:
        cleaned = text.strip()
        for prefix in ("/model", "model", "use model", "use"):
            if cleaned.lower().startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
                break
        if not cleaned:
            self.console.print(f"[{ASH}]Available providers: {_provider_choices()}[/{ASH}]")
            return

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
        self.console.print(f"[{SAGE}]Model set to {escape(provider)} / {escape(model)}[/{SAGE}]", highlight=False)
        if choices:
            self.console.print(f"[{ASH}]Known models: {escape(', '.join(choices))}[/{ASH}]", highlight=False)

    def _handle_agents(self, text: str) -> None:
        cleaned = text.split(" ", 1)[1] if " " in text else ""
        agents = [agent.strip() for agent in cleaned.split(",") if agent.strip()]
        if not agents:
            self.settings.agents = DEFAULT_AGENTS.copy()
            self.console.print(f"[{SAGE}]Agents reset to default.[/{SAGE}]", highlight=False)
            return
        self.settings.agents = agents
        self.console.print(f"[{SAGE}]Agents set to {escape(','.join(agents))}[/{SAGE}]", highlight=False)

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


def launch_chat(console: Console, provider: str, model: str) -> int:
    settings = ChatSettings(provider=provider, model=model)
    return NovaChat(console, settings).run()
