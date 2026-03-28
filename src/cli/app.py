"""Nova Trader — Interactive CLI (dexter-style).

A rich terminal UI with:
  - ASCII banner + model display
  - Chat-style interaction (type tickers, get analysis)
  - Real-time agent progress (⏺ agent → ⎿ result)
  - Animated spinner during processing
  - Markdown-rendered output
  - /commands for model selection, help, etc.
"""

import os
import sys
import json
import time
import threading
from datetime import datetime
from dateutil.relativedelta import relativedelta
from typing import Optional

from dotenv import load_dotenv
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.markdown import Markdown
from rich.columns import Columns
from rich.style import Style as RichStyle
from rich import box
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.formatted_text import HTML
from langchain_core.messages import HumanMessage

from src.agents.portfolio_manager import portfolio_management_agent
from src.agents.risk_manager import risk_management_agent
from src.graph.state import AgentState
from src.utils.analysts import ANALYST_ORDER, get_analyst_nodes
from src.orchestrator.pipeline import Pipeline
from src.utils.progress import progress

load_dotenv()

console = Console()

# ── Theme Colors ──────────────────────────────────────────
PRIMARY = "bold dodger_blue2"
SUCCESS = "bold green"
ERROR = "bold red"
WARNING = "bold yellow"
MUTED = "dim"
ACCENT = "bold cyan"

# ── Spinner Verbs ─────────────────────────────────────────
THINKING_VERBS = [
    "Analyzing", "Evaluating", "Researching", "Computing",
    "Assessing", "Examining", "Investigating", "Processing",
    "Crunching numbers", "Consulting models", "Running signals",
    "Scoring metrics", "Checking fundamentals", "Reading filings",
]

BANNER = r"""
[dodger_blue2]
 ███╗   ██╗ ██████╗ ██╗   ██╗ █████╗    ████████╗██████╗  █████╗ ██████╗ ███████╗██████╗
 ████╗  ██║██╔═══██╗██║   ██║██╔══██╗   ╚══██╔══╝██╔══██╗██╔══██╗██╔══██╗██╔════╝██╔══██╗
 ██╔██╗ ██║██║   ██║██║   ██║███████║      ██║   ██████╔╝███████║██║  ██║█████╗  ██████╔╝
 ██║╚██╗██║██║   ██║╚██╗ ██╔╝██╔══██║      ██║   ██╔══██╗██╔══██║██║  ██║██╔══╝  ██╔══██╗
 ██║ ╚████║╚██████╔╝ ╚████╔╝ ██║  ██║      ██║   ██║  ██║██║  ██║██████╔╝███████╗██║  ██║
 ╚═╝  ╚═══╝ ╚═════╝   ╚═══╝  ╚═╝  ╚═╝      ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝ ╚══════╝╚═╝  ╚═╝
[/dodger_blue2]"""


class NovaTraderCLI:
    """Interactive terminal UI for Nova Trader."""

    def __init__(self):
        self.model_provider = os.getenv("MODEL_PROVIDER", "ollama")
        self.model_name = os.getenv("MODEL_NAME", "llama3.1:8b")
        self.initial_cash = 100_000.0
        self.show_reasoning = False
        self.history = InMemoryHistory()
        self.session = PromptSession(history=self.history)
        self.agent_events: list[dict] = []
        self._spinner_active = False

    # ── Display ───────────────────────────────────────────

    def show_banner(self):
        """Show the intro banner with model info."""
        console.print(BANNER)
        console.print(
            f"  [dim]Model:[/dim] [bold]{self.model_provider}[/bold] / "
            f"[cyan]{self.model_name}[/cyan]  "
            f"[dim]│  Type [bold white]/help[/bold white] for commands  │  "
            f"[bold white]/model[/bold white] to change model[/dim]\n"
        )

    def show_help(self):
        """Show available commands."""
        help_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
        help_table.add_column("Command", style="bold cyan", min_width=20)
        help_table.add_column("Description", style="dim")
        help_table.add_row("/help", "Show this help")
        help_table.add_row("/model", "Change LLM provider and model")
        help_table.add_row("/reasoning", "Toggle show reasoning on/off")
        help_table.add_row("/cash <amount>", "Set initial cash (default: 100,000)")
        help_table.add_row("AAPL NVDA TSLA", "Analyze one or more tickers")
        help_table.add_row("exit / quit / q", "Exit Nova Trader")
        console.print(Panel(help_table, title="[bold]Commands[/bold]", border_style="blue"))

    def show_model_selector(self):
        """Interactive model selection."""
        providers = [
            ("ollama", "Local models (free, private)"),
            ("groq", "Groq Cloud (free tier, fast)"),
            ("openai", "OpenAI (GPT-4o, GPT-4o-mini)"),
            ("anthropic", "Anthropic (Claude Sonnet, Haiku)"),
            ("google", "Google (Gemini Flash, Pro)"),
            ("deepseek", "DeepSeek (cheap, good)"),
            ("xai", "xAI (Grok)"),
            ("openrouter", "OpenRouter (any model)"),
        ]
        console.print("\n[bold]Select a provider:[/bold]")
        for i, (prov, desc) in enumerate(providers, 1):
            marker = " [cyan]◉[/cyan]" if prov == self.model_provider else " [dim]○[/dim]"
            console.print(f"  {marker} [bold]{i}[/bold]. {prov} [dim]— {desc}[/dim]")

        try:
            choice = self.session.prompt(
                HTML("<b><skyblue>  provider> </skyblue></b>")
            ).strip()
            if choice.isdigit() and 1 <= int(choice) <= len(providers):
                self.model_provider = providers[int(choice) - 1][0]
            elif choice in [p[0] for p in providers]:
                self.model_provider = choice
            else:
                console.print("[dim]  Cancelled.[/dim]")
                return
        except (KeyboardInterrupt, EOFError):
            console.print("[dim]  Cancelled.[/dim]")
            return

        # Model name
        default_models = {
            "ollama": "llama3.1:8b",
            "groq": "llama-3.3-70b-versatile",
            "openai": "gpt-4o-mini",
            "anthropic": "claude-3-5-haiku-20241022",
            "google": "gemini-2.0-flash",
            "deepseek": "deepseek-chat",
            "xai": "grok-2",
            "openrouter": "anthropic/claude-3.5-sonnet",
        }
        default = default_models.get(self.model_provider, "")
        try:
            model = self.session.prompt(
                HTML(f"<b><skyblue>  model name</skyblue></b> <i>[{default}]</i><b><skyblue>> </skyblue></b>")
            ).strip()
            self.model_name = model if model else default
        except (KeyboardInterrupt, EOFError):
            self.model_name = default

        console.print(
            f"\n  [dim]Using:[/dim] [bold]{self.model_provider}[/bold] / "
            f"[cyan]{self.model_name}[/cyan]\n"
        )

    # ── Agent Progress Rendering ──────────────────────────

    def _render_agent_events(self) -> Text:
        """Render current agent events as a Rich Text block."""
        output = Text()
        for event in self.agent_events:
            status = event.get("status", "running")
            name = event.get("name", "")
            ticker = event.get("ticker", "")
            detail = event.get("detail", "")

            if status == "running":
                output.append("  ⏺ ", style="bold dodger_blue2")
                output.append(f"{name}", style="bold")
                if ticker:
                    output.append(f"({ticker})", style="cyan")
                output.append("\n")
            elif status == "done":
                output.append("  ⏺ ", style="bold green")
                output.append(f"{name}", style="bold")
                if ticker:
                    output.append(f"({ticker})", style="cyan")
                output.append("\n")
                if detail:
                    output.append(f"    ⎿  {detail}\n", style="dim")
            elif status == "error":
                output.append("  ⏺ ", style="bold red")
                output.append(f"{name}", style="bold")
                output.append("\n")
                output.append(f"    ⎿  Error: {detail}\n", style="red")
        return output

    def _render_spinner(self, verb: str, elapsed: float) -> Text:
        """Render an animated spinner line."""
        frames = ["⣾", "⣽", "⣻", "⢿", "⡿", "⣟", "⣯", "⣷"]
        frame = frames[int(elapsed * 8) % len(frames)]
        t = Text()
        t.append(f"\n  {frame} ", style="bold dodger_blue2")
        t.append(f"{verb}...", style="dim italic")
        t.append(" (esc to interrupt)", style="dim")
        return t

    # ── Core Analysis ─────────────────────────────────────

    def _on_agent_update(self, agent_name: str, ticker: Optional[str], status: str, analysis: Optional[str] = None, timestamp: Optional[str] = None):
        """Callback for agent progress updates."""
        display_name = agent_name.replace("_agent", "").replace("_", " ").title()
        is_done = status.lower() == "done"

        # Update existing event
        for event in self.agent_events:
            if event["name"] == display_name and event.get("ticker") == ticker:
                if is_done:
                    event["status"] = "done"
                    if analysis:
                        event["detail"] = analysis[:80]
                else:
                    event["detail"] = status
                return

        self.agent_events.append({
            "name": display_name,
            "ticker": ticker,
            "status": "done" if is_done else "running",
            "detail": (analysis[:80] if analysis else "") if is_done else status,
        })

    def run_analysis(self, tickers: list[str]):
        """Run the full multi-agent pipeline with live progress display."""
        self.agent_events = []
        start_time = time.time()

        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - relativedelta(months=3)).strftime("%Y-%m-%d")

        # Build portfolio
        portfolio = {
            "cash": self.initial_cash,
            "margin_requirement": 0.0,
            "margin_used": 0.0,
            "positions": {
                t: {"long": 0, "short": 0, "long_cost_basis": 0.0,
                    "short_cost_basis": 0.0, "short_margin_used": 0.0}
                for t in tickers
            },
            "realized_gains": {t: {"long": 0.0, "short": 0.0} for t in tickers},
        }

        state: AgentState = {
            "messages": [HumanMessage(content="Make trading decisions based on the provided data.")],
            "data": {
                "tickers": tickers,
                "portfolio": portfolio,
                "start_date": start_date,
                "end_date": end_date,
                "analyst_signals": {},
            },
            "metadata": {
                "show_reasoning": self.show_reasoning,
                "model_name": self.model_name,
                "model_provider": self.model_provider,
            },
        }

        analyst_nodes = get_analyst_nodes()
        analysts = [(key, analyst_nodes[key][1]) for key in analyst_nodes]

        pipeline = Pipeline(
            analyst_agents=analysts,
            risk_agent=risk_management_agent,
            portfolio_agent=portfolio_management_agent,
        )

        # Register progress handler
        handler = progress.register_handler(self._on_agent_update)

        result = {"decisions": None, "analyst_signals": {}}
        error_msg = None

        def _run():
            nonlocal result, error_msg
            try:
                final_state = pipeline.run(state)
                content = final_state["messages"][-1].content
                try:
                    decisions = json.loads(content)
                except (json.JSONDecodeError, TypeError):
                    decisions = None
                result = {
                    "decisions": decisions,
                    "analyst_signals": final_state["data"].get("analyst_signals", {}),
                }
            except KeyboardInterrupt:
                error_msg = "Interrupted by user"
            except Exception as e:
                error_msg = str(e)

        # Run pipeline in background thread
        thread = threading.Thread(target=_run, daemon=True)

        import random
        verb_idx = 0

        console.print()  # blank line before output

        with Live(console=console, refresh_per_second=8, transient=True) as live:
            thread.start()
            while thread.is_alive():
                elapsed = time.time() - start_time
                # Rotate verb every 3 seconds
                verb_idx = int(elapsed / 3) % len(THINKING_VERBS)
                verb = THINKING_VERBS[verb_idx]

                display = Text()
                display.append_text(self._render_agent_events())
                display.append_text(self._render_spinner(verb, elapsed))
                live.update(display)

                try:
                    time.sleep(0.12)
                except KeyboardInterrupt:
                    error_msg = "Interrupted by user"
                    break

            thread.join(timeout=2)

        progress.unregister_handler(handler)

        elapsed = time.time() - start_time

        if error_msg:
            console.print(f"\n  [bold red]✗[/bold red] {error_msg}\n")
            return

        # Show final agent events (completed)
        for event in self.agent_events:
            event["status"] = "done"
        console.print(self._render_agent_events())

        # Performance line
        console.print(
            f"  [dim]✻ {elapsed:.1f}s[/dim]\n"
        )

        # Render decisions
        self._render_decisions(result, tickers)

    def _render_decisions(self, result: dict, tickers: list[str]):
        """Render trading decisions in a rich format."""
        decisions = result.get("decisions")
        if not decisions:
            console.print("  [bold red]No trading decisions returned.[/bold red]\n")
            return

        for ticker in tickers:
            if ticker not in decisions:
                continue
            decision = decisions[ticker]

            action = decision.get("action", "hold").upper()
            quantity = decision.get("quantity", 0)
            confidence = decision.get("confidence", 0)
            reasoning = decision.get("reasoning", "")

            # Color the action
            action_style = {
                "BUY": "bold green",
                "STRONG_BUY": "bold green",
                "SELL": "bold red",
                "STRONG_SELL": "bold red",
                "SHORT": "bold red",
                "COVER": "bold green",
                "HOLD": "bold yellow",
            }.get(action, "bold white")

            # Build signal summary table
            signals_table = Table(box=box.SIMPLE_HEAVY, show_header=True, padding=(0, 1))
            signals_table.add_column("Agent", style="bold", min_width=22)
            signals_table.add_column("Signal", justify="center", min_width=10)
            signals_table.add_column("Confidence", justify="center", min_width=12)

            analyst_signals = result.get("analyst_signals", {})
            for agent_key, agent_signals in analyst_signals.items():
                if ticker not in agent_signals:
                    continue
                if agent_key == "risk_management_agent":
                    continue

                sig = agent_signals[ticker]
                agent_name = agent_key.replace("_agent", "").replace("_", " ").title()
                signal = sig.get("signal", "").upper()
                conf = sig.get("confidence", 0)

                sig_style = {"BULLISH": "green", "BEARISH": "red", "NEUTRAL": "yellow"}.get(signal, "white")
                conf_bar = "█" * int(conf * 10) + "░" * (10 - int(conf * 10))
                signals_table.add_row(
                    agent_name,
                    f"[{sig_style}]{signal}[/{sig_style}]",
                    f"[{sig_style}]{conf_bar}[/{sig_style}] {conf:.0%}",
                )

            # Decision header
            header = Text()
            header.append(f"  ⏺ ", style="bold dodger_blue2")
            header.append(f"{ticker}", style="bold cyan")
            header.append(f"  →  ", style="dim")
            header.append(f"{action}", style=action_style)
            if quantity:
                header.append(f" {quantity} shares", style="dim")
            header.append(f"  ({confidence:.0%} confidence)", style="dim")
            console.print(header)

            # Signals table
            console.print(signals_table)

            # Reasoning
            if reasoning and self.show_reasoning:
                console.print(Panel(
                    Markdown(reasoning),
                    title=f"[dim]Reasoning[/dim]",
                    border_style="dim",
                    padding=(1, 2),
                ))
            console.print()

    # ── Command Handler ───────────────────────────────────

    def handle_command(self, cmd: str) -> bool:
        """Handle slash commands. Returns True if handled."""
        parts = cmd.strip().split()
        command = parts[0].lower()

        if command in ("exit", "quit", "q"):
            console.print("\n  [dim]Goodbye.[/dim]\n")
            return True  # Signal exit

        if command == "/help":
            self.show_help()
            return False

        if command == "/model":
            self.show_model_selector()
            return False

        if command == "/reasoning":
            self.show_reasoning = not self.show_reasoning
            state = "[green]on[/green]" if self.show_reasoning else "[red]off[/red]"
            console.print(f"  [dim]Show reasoning:[/dim] {state}\n")
            return False

        if command == "/cash" and len(parts) > 1:
            try:
                self.initial_cash = float(parts[1].replace(",", "").replace("$", ""))
                console.print(f"  [dim]Cash set to:[/dim] [bold]${self.initial_cash:,.0f}[/bold]\n")
            except ValueError:
                console.print("  [red]Invalid amount.[/red]\n")
            return False

        return False

    # ── Main Loop ─────────────────────────────────────────

    def run(self):
        """Main interactive loop."""
        self.show_banner()

        while True:
            try:
                user_input = self.session.prompt(
                    HTML("<b><skyblue>❯ </skyblue></b>"),
                ).strip()

                if not user_input:
                    continue

                # Commands
                if user_input.startswith("/") or user_input.lower() in ("exit", "quit", "q"):
                    should_exit = self.handle_command(user_input)
                    if should_exit:
                        break
                    continue

                # Treat input as tickers
                tickers = [t.upper() for t in user_input.split() if t.isalpha()]
                if not tickers:
                    console.print("  [dim]Enter ticker symbols (e.g. AAPL NVDA TSLA)[/dim]\n")
                    continue

                console.print(
                    f"  [dim]Analyzing:[/dim] [bold cyan]{' '.join(tickers)}[/bold cyan]\n"
                )
                self.run_analysis(tickers)

            except KeyboardInterrupt:
                console.print("\n  [dim]Press Ctrl+C again or type 'exit' to quit.[/dim]\n")
                continue
            except EOFError:
                break

        console.print("[dim]Session ended.[/dim]")


def main():
    cli = NovaTraderCLI()
    cli.run()


if __name__ == "__main__":
    main()
