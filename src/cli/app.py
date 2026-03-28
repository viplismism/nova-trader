"""Nova Trader — Interactive CLI.

Feynman-inspired terminal UI with:
  - Block-char ASCII banner
  - Two-column splash: system info + slash commands
  - Warm earth-tone palette (ink/stone/sage/teal/rose)
  - Agent progress events (check on done, dot while running)
  - Animated spinner during processing
  - Rich decision rendering with confidence bars
"""

import os
import sys
import json
import time
import platform
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
from src.utils.analysts import ANALYST_ORDER, ANALYST_CONFIG, get_analyst_nodes
from src.core.engine import HydraEngine
from src.core.pad import AgentPad
from src.utils.progress import progress

load_dotenv()

console = Console()

# ── Version ───────────────────────────────────────────────
VERSION = "0.1.0"

# ── Theme: warm earth tones (Feynman-inspired) ────────────
INK = "grey85"              # primary text, warm beige
STONE = "grey62"            # muted text
ASH = "grey50"              # body text, dim
DARK_ASH = "grey35"         # borders
SAGE = "#a7c080"            # muted green — success, accents
TEAL = "#7fbbb3"            # teal — headers, highlights
ROSE = "#e67e80"            # soft red — errors
AMBER = "#dbbc7f"           # warm amber — warnings

# ── Symbols ───────────────────────────────────────────────
SYM_CHECK = "✓"
SYM_CROSS = "✗"
SYM_WARN = "⚠"
SYM_DIAMOND = "◆"
SYM_DOT = "⏺"
SYM_RESULT = "⎿"
SYM_PROMPT = "❯"

# ── Spinner Verbs ─────────────────────────────────────────
THINKING_VERBS = [
    "Analyzing", "Evaluating", "Researching", "Computing",
    "Assessing", "Examining", "Investigating", "Processing",
    "Crunching numbers", "Consulting models", "Running signals",
    "Scoring metrics", "Checking fundamentals", "Reading filings",
]

# ── ASCII Banner (just "NOVA" — compact, iconic) ─────────
BANNER_LINES = [
    " ███╗   ██╗  ██████╗  ██╗   ██╗  █████╗ ",
    " ████╗  ██║ ██╔═══██╗ ██║   ██║ ██╔══██╗",
    " ██╔██╗ ██║ ██║   ██║ ██║   ██║ ███████║",
    " ██║╚██╗██║ ██║   ██║ ╚██╗ ██╔╝ ██╔══██║",
    " ██║ ╚████║ ╚██████╔╝  ╚████╔╝  ██║  ██║",
    " ╚═╝  ╚═══╝  ╚═════╝    ╚═══╝   ╚═╝  ╚═╝",
]

# ── Slash Commands ────────────────────────────────────────
SLASH_COMMANDS = [
    ("/analyze", "AAPL ...", "Analyze tickers with all agents"),
    ("/pad", "", "Show last AgentPad summary"),
    ("/execute", "", "Execute last decisions (paper)"),
    ("/model", "", "Switch LLM provider and model"),
    ("/reasoning", "", "Toggle reasoning display"),
    ("/cash", "<amount>", "Set portfolio cash"),
    ("/agents", "", "List all analyst agents"),
    ("/status", "", "Show system status"),
    ("/help", "", "Show help"),
]


def _get_system_info():
    """Gather system info for splash screen."""
    cores = os.cpu_count() or "?"
    mem_gb = "?"
    try:
        import subprocess
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True, text=True, timeout=2
        )
        if result.returncode == 0:
            mem_gb = f"{int(result.stdout.strip()) // (1024**3)}GB"
    except Exception:
        pass
    arch = platform.machine()
    return cores, mem_gb, arch


# Quant agent keys (non-template Python agents)
_QUANT_KEYS = frozenset([
    "technical_analyst", "fundamentals_analyst", "growth_analyst",
    "news_sentiment_analyst", "sentiment_analyst", "valuation_analyst",
])


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
        self.last_pad: AgentPad | None = None

    # ── Splash Screen ─────────────────────────────────────

    def show_banner(self):
        """Render full splash: banner + version + two-column info box."""
        console.print()
        for line in BANNER_LINES:
            console.print(f"  {line}", style=f"bold {TEAL}")
        console.print()

        # Version line centered under banner
        banner_w = max(len(l) for l in BANNER_LINES) + 4
        ver = f"v{VERSION}"
        pad_l = (banner_w - len(ver) - 2) // 2
        pad_r = banner_w - len(ver) - 2 - pad_l
        console.print(f"  {'─' * pad_l} {ver} {'─' * pad_r}", style=DARK_ASH)

        self._render_splash_box()
        console.print()

    def _render_splash_box(self):
        """Render Feynman-style two-column bordered info box."""
        cores, mem_gb, arch = _get_system_info()

        # Count agents
        persona_count = sum(1 for k in ANALYST_CONFIG if k not in _QUANT_KEYS)
        total_agents = persona_count + len(_QUANT_KEYS) + 2

        # Persona last names
        persona_names = [
            v["display_name"].split()[-1]
            for k, v in sorted(ANALYST_CONFIG.items(), key=lambda x: x[1]["order"])
            if k not in _QUANT_KEYS
        ]

        # Dimensions — adapt to terminal width (min 110 for two-column layout)
        term_w = min(max(console.width or 110, 110), 130)
        inner = term_w - 6  # 6 = "  │ " + " │"
        LW = inner * 46 // 100
        RW = inner - LW - 3  # 3 = " │ " separator
        hbar = "─" * (inner + 2)

        # Left column: list of (label, value) or None=blank or ("_h", text)=header
        left = []
        left.append(("model", f"{self.model_provider}/{self.model_name}"))
        left.append(("cash", f"${self.initial_cash:,.0f}"))
        left.append(("reasoning", "on" if self.show_reasoning else "off"))
        left.append(None)
        left.append(("system", f"{cores} cores · {mem_gb} · {arch}"))
        left.append(None)
        left.append(("_h", f"{total_agents} agents"))
        # Wrap persona names into rows
        pn = ", ".join(persona_names)
        chunk_w = LW - 14
        first_persona = True
        while pn:
            cut = pn[:chunk_w]
            if len(pn) > chunk_w:
                lc = cut.rfind(",")
                if lc > 0:
                    cut = pn[:lc + 1]
            lbl = "  personas" if first_persona else ""
            left.append((lbl, cut.strip()))
            pn = pn[len(cut):].strip().lstrip(",").strip()
            first_persona = False
        left.append(("  quant", "Technical, Fundamentals, Growth,"))
        left.append(("", "Sentiment, News, Valuation"))
        left.append(("  orch.", "Risk Manager, Portfolio Manager"))

        # Right column
        right = []
        right.append(("_h", "Commands"))
        for cmd, args, desc in SLASH_COMMANDS:
            usage = f"{cmd} {args}".strip() if args else cmd
            right.append(("cmd", usage, desc))
        right.append(None)
        right.append(("_n", "Type tickers directly: AAPL TSLA NVDA"))

        # Pad height
        mx = max(len(left), len(right))
        left += [None] * (mx - len(left))
        right += [None] * (mx - len(right))

        console.print(Text.assemble((f"  ┌{hbar}┐", DARK_ASH)))

        for i in range(mx):
            ll, rl = left[i], right[i]

            # Left cell
            lt = Text()
            if ll is None:
                lt.append(" " * LW)
            elif ll[0] == "_h":
                lt.append(ll[1], style=f"bold {TEAL}")
                lt.append(" " * max(0, LW - len(ll[1])))
            else:
                lbl, val = ll
                label_w = 14
                lt.append(f"{lbl:<{label_w}}", style=STONE)
                lt.append(val, style=INK)
                lt.append(" " * max(0, LW - label_w - len(val)))

            # Right cell
            rt = Text()
            if rl is None:
                rt.append(" " * RW)
            elif rl[0] == "_h":
                rt.append(rl[1], style=f"bold {TEAL}")
                rt.append(" " * max(0, RW - len(rl[1])))
            elif rl[0] == "cmd":
                _, usage, desc = rl
                cmd_w = min(22, RW // 2)
                rt.append(f"{usage:<{cmd_w}}", style=f"bold {SAGE}")
                rem = RW - cmd_w
                d = desc[:rem]
                rt.append(d, style=ASH)
                rt.append(" " * max(0, rem - len(d)))
            elif rl[0] == "_n":
                note = rl[1][:RW]
                rt.append(note, style=f"italic {ASH}")
                rt.append(" " * max(0, RW - len(note)))
            else:
                rt.append(" " * RW)

            row = Text()
            row.append("  │ ", style=DARK_ASH)
            row.append_text(lt)
            row.append(" │ ", style=DARK_ASH)
            row.append_text(rt)
            row.append(" │", style=DARK_ASH)
            console.print(row)

        console.print(Text.assemble((f"  └{hbar}┘", DARK_ASH)))

    # ── Help ──────────────────────────────────────────────

    def show_help(self):
        """Show commands in Feynman style."""
        console.print()
        console.print(f"  {SYM_DIAMOND} Commands", style=f"bold {TEAL}")
        for cmd, args, desc in SLASH_COMMANDS:
            usage = f"{cmd} {args}".strip() if args else cmd
            console.print(f"    {usage:<24}", style=f"bold {SAGE}", end="")
            console.print(desc, style=ASH)
        console.print(f"    {'exit / quit / q':<24}", style=f"bold {SAGE}", end="")
        console.print("Exit Nova Trader", style=ASH)
        console.print()

    def show_agents(self):
        """List all available analyst agents."""
        console.print()
        console.print(f"  {SYM_DIAMOND} Investor Personas", style=f"bold {TEAL}")
        for key, config in sorted(ANALYST_CONFIG.items(), key=lambda x: x[1]["order"]):
            if key in _QUANT_KEYS:
                continue
            console.print(f"    {config['display_name']:<24}", style=f"bold {SAGE}", end="")
            console.print(config.get("description", ""), style=ASH)
        console.print()
        console.print(f"  {SYM_DIAMOND} Quantitative Agents", style=f"bold {TEAL}")
        for key in sorted(_QUANT_KEYS):
            config = ANALYST_CONFIG.get(key, {})
            if config:
                console.print(f"    {config['display_name']:<24}", style=f"bold {SAGE}", end="")
                console.print(config.get("description", ""), style=ASH)
        console.print()
        console.print(f"  {SYM_DIAMOND} Orchestration", style=f"bold {TEAL}")
        console.print(f"    {'Risk Manager':<24}", style=f"bold {SAGE}", end="")
        console.print("Volatility-adjusted position sizing and risk limits", style=ASH)
        console.print(f"    {'Portfolio Manager':<24}", style=f"bold {SAGE}", end="")
        console.print("Final trading decisions via LLM consensus", style=ASH)
        console.print()

    def show_status(self):
        """Show system status panel."""
        cores, mem_gb, arch = _get_system_info()
        console.print()
        console.print(f"  {SYM_DIAMOND} Configuration", style=f"bold {TEAL}")
        console.print(f"    {'Model:':<16}{self.model_provider}/{self.model_name}", style=INK)
        console.print(f"    {'Cash:':<16}${self.initial_cash:,.0f}", style=INK)
        console.print(f"    {'Reasoning:':<16}{'on' if self.show_reasoning else 'off'}", style=INK)
        console.print()
        console.print(f"  {SYM_DIAMOND} System", style=f"bold {TEAL}")
        console.print(f"    {'Platform:':<16}{platform.system()} {arch}", style=INK)
        console.print(f"    {'Cores:':<16}{cores}", style=INK)
        console.print(f"    {'Memory:':<16}{mem_gb}", style=INK)
        console.print(f"    {'Python:':<16}{platform.python_version()}", style=INK)
        console.print()
        console.print(f"  {SYM_DIAMOND} Agents", style=f"bold {TEAL}")
        total = len(ANALYST_CONFIG) + 2
        console.print(f"    {'Total:':<16}{total} ({len(ANALYST_CONFIG)} analysts + 2 orchestration)", style=INK)
        console.print()

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
        console.print()
        console.print(f"  {SYM_DIAMOND} Select Provider", style=f"bold {TEAL}")
        for i, (prov, desc) in enumerate(providers, 1):
            if prov == self.model_provider:
                console.print(f"    [{SAGE}]◉ {i}. {prov}[/{SAGE}]  ", end="")
            else:
                console.print(f"    [{STONE}]○ {i}. {prov}[/{STONE}]  ", end="")
            console.print(desc, style=ASH)

        try:
            choice = self.session.prompt(
                HTML("<style fg='#a7c080'><b>  provider❯ </b></style>")
            ).strip()
            if choice.isdigit() and 1 <= int(choice) <= len(providers):
                self.model_provider = providers[int(choice) - 1][0]
            elif choice in [p[0] for p in providers]:
                self.model_provider = choice
            else:
                console.print(f"  [{ASH}]Cancelled.[/{ASH}]")
                return
        except (KeyboardInterrupt, EOFError):
            console.print(f"  [{ASH}]Cancelled.[/{ASH}]")
            return

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
                HTML(f"<style fg='#a7c080'><b>  model</b></style> <i>[{default}]</i><style fg='#a7c080'><b>❯ </b></style>")
            ).strip()
            self.model_name = model if model else default
        except (KeyboardInterrupt, EOFError):
            self.model_name = default

        console.print(f"\n  {SYM_CHECK} Using {self.model_provider}/{self.model_name}", style=f"bold {SAGE}")
        console.print()

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
                output.append(f"  {SYM_DOT} ", style=f"bold {TEAL}")
                output.append(name, style=f"bold {INK}")
                if ticker:
                    output.append(f" ({ticker})", style=STONE)
                output.append("\n")
            elif status == "done":
                output.append(f"  {SYM_CHECK} ", style=f"bold {SAGE}")
                output.append(name, style=INK)
                if ticker:
                    output.append(f" ({ticker})", style=STONE)
                output.append("\n")
                if detail:
                    output.append(f"    {SYM_RESULT}  {detail}\n", style=ASH)
            elif status == "error":
                output.append(f"  {SYM_CROSS} ", style=f"bold {ROSE}")
                output.append(name, style=f"bold {INK}")
                output.append("\n")
                output.append(f"    {SYM_RESULT}  Error: {detail}\n", style=ROSE)
        return output

    def _render_spinner(self, verb: str, elapsed: float) -> Text:
        """Render an animated braille spinner line."""
        frames = ["⣾", "⣽", "⣻", "⢿", "⡿", "⣟", "⣯", "⣷"]
        frame = frames[int(elapsed * 8) % len(frames)]
        t = Text()
        t.append(f"\n  {frame} ", style=f"bold {TEAL}")
        t.append(f"{verb}...", style=f"italic {ASH}")
        return t

    # ── Core Analysis ─────────────────────────────────────

    def _on_agent_update(self, agent_name: str, ticker: Optional[str], status: str,
                         analysis: Optional[str] = None, timestamp: Optional[str] = None):
        """Callback for agent progress updates."""
        display_name = agent_name.replace("_agent", "").replace("_", " ").title()
        is_done = status.lower() == "done"

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

        engine = HydraEngine(
            analyst_agents=analysts,
            risk_agent=risk_management_agent,
            portfolio_agent=portfolio_management_agent,
        )

        handler = progress.register_handler(self._on_agent_update)

        result = {"decisions": None, "analyst_signals": {}, "pad": None}
        error_msg = None

        def _run():
            nonlocal result, error_msg
            try:
                final_state, pad = engine.run(state)
                content = final_state["messages"][-1].content
                try:
                    decisions = json.loads(content)
                except (json.JSONDecodeError, TypeError):
                    decisions = None
                result = {
                    "decisions": decisions,
                    "analyst_signals": final_state["data"].get("analyst_signals", {}),
                    "pad": pad,
                }
            except KeyboardInterrupt:
                error_msg = "Interrupted by user"
            except Exception as e:
                error_msg = str(e)

        thread = threading.Thread(target=_run, daemon=True)
        verb_idx = 0

        console.print()

        with Live(console=console, refresh_per_second=8, transient=True) as live:
            thread.start()
            while thread.is_alive():
                elapsed = time.time() - start_time
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
            console.print(f"\n  {SYM_CROSS} {error_msg}", style=f"bold {ROSE}")
            console.print()
            return

        # Show final completed events
        for event in self.agent_events:
            event["status"] = "done"
        console.print(self._render_agent_events())

        console.print(f"  [{ASH}]completed in {elapsed:.1f}s[/{ASH}]")
        console.print()

        # Store the pad for later /pad and /execute commands
        pad = result.get("pad")
        if pad:
            self.last_pad = pad
            self._render_consensus(pad, tickers)

        self._render_decisions(result, tickers)

    def _render_decisions(self, result: dict, tickers: list[str]):
        """Render trading decisions."""
        decisions = result.get("decisions")
        if not decisions:
            console.print(f"  {SYM_CROSS} No trading decisions returned.", style=f"bold {ROSE}")
            console.print()
            return

        for ticker in tickers:
            if ticker not in decisions:
                continue
            decision = decisions[ticker]

            action = decision.get("action", "hold").upper()
            quantity = decision.get("quantity", 0)
            confidence_raw = decision.get("confidence", 0)
            confidence = confidence_raw / 100 if confidence_raw > 1 else confidence_raw
            reasoning = decision.get("reasoning", "")

            if action in ("BUY", "STRONG_BUY", "COVER"):
                action_color = SAGE
            elif action in ("SELL", "STRONG_SELL", "SHORT"):
                action_color = ROSE
            else:
                action_color = AMBER

            console.print(f"  {SYM_DIAMOND} {ticker}", style=f"bold {TEAL}", end="")
            console.print("  →  ", style=ASH, end="")
            console.print(action, style=f"bold {action_color}", end="")
            if quantity:
                console.print(f" {quantity} shares", style=STONE, end="")
            console.print(f"  ({confidence:.0%})", style=ASH)
            console.print()

            analyst_signals = result.get("analyst_signals", {})
            for agent_key, agent_signals in analyst_signals.items():
                if ticker not in agent_signals:
                    continue
                if agent_key == "risk_management_agent":
                    continue
                sig = agent_signals[ticker]
                agent_name = agent_key.replace("_agent", "").replace("_", " ").title()
                signal = sig.get("signal", "").upper()
                conf_raw = sig.get("confidence", 0)
                conf = conf_raw / 100 if conf_raw > 1 else conf_raw
                if signal == "BULLISH":
                    sig_color = SAGE
                elif signal == "BEARISH":
                    sig_color = ROSE
                else:
                    sig_color = AMBER
                conf_bar = "█" * int(conf * 10) + "░" * (10 - int(conf * 10))
                console.print(f"    {agent_name:<26}", style=STONE, end="")
                console.print(f"{signal:<10}", style=f"bold {sig_color}", end="")
                console.print(f"{conf_bar} {conf:.0%}", style=sig_color)

            console.print()

            if reasoning and self.show_reasoning:
                console.print(Panel(
                    Markdown(reasoning),
                    title="[dim]Reasoning[/dim]",
                    border_style=DARK_ASH,
                    padding=(1, 2),
                ))
                console.print()

    def _render_consensus(self, pad: AgentPad, tickers: list[str]):
        """Render consensus view from the AgentPad."""
        if not pad.consensus:
            return

        console.print(f"  {SYM_DIAMOND} Consensus", style=f"bold {TEAL}")
        console.print()

        for ticker in tickers:
            cons = pad.get_consensus(ticker)
            if not cons:
                continue

            signal = cons.get("signal", "neutral").upper()
            conf = cons.get("confidence", 0)
            conf_norm = conf / 100 if conf > 1 else conf
            bull = cons.get("bull_count", 0)
            bear = cons.get("bear_count", 0)
            neutral = cons.get("neutral_count", 0)
            total = cons.get("total_agents", 0)

            if signal == "BULLISH":
                sig_color = SAGE
            elif signal == "BEARISH":
                sig_color = ROSE
            else:
                sig_color = AMBER

            conf_bar = "█" * int(conf_norm * 10) + "░" * (10 - int(conf_norm * 10))

            console.print(f"    {ticker:<8}", style=f"bold {TEAL}", end="")
            console.print(f"{signal:<10}", style=f"bold {sig_color}", end="")
            console.print(f"{conf_bar} {conf_norm:.0%}", style=sig_color, end="")
            console.print(f"   [{ASH}]{bull}↑ {bear}↓ {neutral}— ({total} agents)[/{ASH}]")

        console.print()

    def show_pad(self):
        """Display the last AgentPad content."""
        if not self.last_pad:
            console.print(f"  {SYM_WARN} No analysis pad yet. Run /analyze first.", style=f"bold {AMBER}")
            console.print()
            return

        pad = self.last_pad
        console.print(f"  {SYM_DIAMOND} AgentPad  [{ASH}]session {pad.session_id}[/{ASH}]", style=f"bold {TEAL}")
        console.print()

        # Consensus
        if pad.consensus:
            console.print(f"    {'Consensus':<16}", style=f"bold {TEAL}")
            for ticker, cons in pad.consensus.items():
                signal = cons.get("signal", "neutral").upper()
                conf = cons.get("confidence", 0)
                bull = cons.get("bull_count", 0)
                bear = cons.get("bear_count", 0)
                console.print(f"      {ticker:<8} {signal:<10} conf={conf}%  {bull}↑ {bear}↓", style=STONE)
            console.print()

        # Decisions
        if pad.decisions:
            console.print(f"    {'Decisions':<16}", style=f"bold {TEAL}")
            for ticker, dec in pad.decisions.items():
                action = dec.get("action", "hold").upper()
                qty = dec.get("quantity", 0)
                console.print(f"      {ticker:<8} {action:<8} {qty} shares", style=STONE)
            console.print()

        # Agent timings
        if pad.agent_timings:
            console.print(f"    {'Timings':<16}", style=f"bold {TEAL}")
            sorted_timings = sorted(pad.agent_timings.items(), key=lambda x: x[1], reverse=True)
            for agent, secs in sorted_timings[:10]:
                name = agent.replace("_agent", "").replace("_", " ").title()
                console.print(f"      {name:<26} {secs:.1f}s", style=ASH)
            console.print()

        # Orders
        if pad.orders:
            console.print(f"    {'Orders':<16}", style=f"bold {TEAL}")
            for order in pad.orders:
                console.print(f"      {order.get('ticker'):<8} {order.get('side'):<5} "
                              f"{order.get('quantity')} shares  [{order.get('status')}]", style=STONE)
            console.print()

    def execute_last(self):
        """Execute the last analysis decisions via paper trading."""
        if not self.last_pad or not self.last_pad.decisions:
            console.print(f"  {SYM_WARN} No decisions to execute. Run /analyze first.", style=f"bold {AMBER}")
            console.print()
            return

        console.print(f"  {SYM_DIAMOND} Executing via paper trading...", style=f"bold {TEAL}")
        console.print()

        try:
            from src.execution.bridge import ExecutionBridge
            bridge = ExecutionBridge.paper()
            result = bridge.execute(self.last_pad.decisions)

            for order in result.orders:
                status_color = SAGE if order.status not in ("failed", "rejected") else ROSE
                console.print(f"    {order.ticker:<8} {order.side:<5} {order.quantity} shares  "
                              f"[{order.status}]", style=status_color)
                # Record in pad
                self.last_pad.write_order({
                    "ticker": order.ticker,
                    "side": order.side,
                    "quantity": order.quantity,
                    "status": order.status,
                    "order_id": order.order_id,
                })

            if result.errors:
                for err in result.errors:
                    console.print(f"    {SYM_CROSS} {err}", style=f"bold {ROSE}")

            console.print()
            console.print(f"  [{ASH}]{result.success_count} succeeded, "
                          f"{result.failed_count} failed[/{ASH}]")

        except ImportError:
            console.print(f"  {SYM_CROSS} alpaca-py not installed. Run: pip install alpaca-py", style=f"bold {ROSE}")
        except ValueError as e:
            console.print(f"  {SYM_CROSS} {e}", style=f"bold {ROSE}")
        except Exception as e:
            console.print(f"  {SYM_CROSS} Execution error: {e}", style=f"bold {ROSE}")

        console.print()

    # ── Command Handler ───────────────────────────────────

    def handle_command(self, cmd: str) -> bool:
        """Handle slash commands. Returns True if should exit."""
        parts = cmd.strip().split()
        command = parts[0].lower()

        if command in ("exit", "quit", "q"):
            console.print()
            console.print(f"  [{ASH}]Goodbye.[/{ASH}]")
            console.print()
            return True

        if command == "/help":
            self.show_help()
            return False

        if command == "/model":
            self.show_model_selector()
            return False

        if command == "/reasoning":
            self.show_reasoning = not self.show_reasoning
            label = f"[{SAGE}]on[/{SAGE}]" if self.show_reasoning else f"[{ROSE}]off[/{ROSE}]"
            console.print(f"  {SYM_CHECK} Show reasoning: {label}", style=f"bold {SAGE}")
            console.print()
            return False

        if command == "/cash" and len(parts) > 1:
            try:
                self.initial_cash = float(parts[1].replace(",", "").replace("$", ""))
                console.print(f"  {SYM_CHECK} Cash set to ${self.initial_cash:,.0f}", style=f"bold {SAGE}")
            except ValueError:
                console.print(f"  {SYM_CROSS} Invalid amount.", style=f"bold {ROSE}")
            console.print()
            return False

        if command == "/agents":
            self.show_agents()
            return False

        if command == "/status":
            self.show_status()
            return False

        if command == "/pad":
            self.show_pad()
            return False

        if command == "/execute":
            self.execute_last()
            return False

        if command == "/analyze" and len(parts) > 1:
            tickers = [t.upper() for t in parts[1:] if t.isalpha()]
            if tickers:
                ticker_str = " ".join(tickers)
                console.print(f"  [{ASH}]Analyzing:[/{ASH}] [bold {TEAL}]{ticker_str}[/bold {TEAL}]")
                self.run_analysis(tickers)
            return False

        console.print(f"  {SYM_WARN} Unknown command: {command}", style=f"bold {AMBER}")
        console.print()
        return False

    # ── Main Loop ─────────────────────────────────────────

    def run(self):
        """Main interactive loop."""
        self.show_banner()

        while True:
            try:
                user_input = self.session.prompt(
                    HTML("<style fg='#a7c080'><b>❯ </b></style>"),
                ).strip()

                if not user_input:
                    continue

                if user_input.startswith("/") or user_input.lower() in ("exit", "quit", "q"):
                    should_exit = self.handle_command(user_input)
                    if should_exit:
                        break
                    continue

                tickers = [t.upper() for t in user_input.split() if t.isalpha()]
                if not tickers:
                    console.print(f"  [{ASH}]Enter ticker symbols (e.g. AAPL NVDA TSLA) or /help[/{ASH}]")
                    console.print()
                    continue

                ticker_str = " ".join(tickers)
                console.print(f"  [{ASH}]Analyzing:[/{ASH}] [bold {TEAL}]{ticker_str}[/bold {TEAL}]")
                self.run_analysis(tickers)

            except KeyboardInterrupt:
                console.print(f"\n  [{ASH}]Press Ctrl+C again or type 'exit' to quit.[/{ASH}]")
                console.print()
                continue
            except EOFError:
                break

        console.print(f"  [{ASH}]Session ended.[/{ASH}]")


def main():
    cli = NovaTraderCLI()
    cli.run()


if __name__ == "__main__":
    main()
