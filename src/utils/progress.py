import contextvars
import json
import re
import threading
from datetime import datetime, timezone
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.style import Style
from rich.text import Text
from rich.layout import Layout
from typing import Dict, Optional, Callable, List

console = Console()

# Attributes data fetches to the agent that triggered them. Today the snapshot phase
# does all fetching (analysts read slices), so this is set to "snapshot" there; the
# contextvar keeps attribution correct if that ever changes without threading an
# agent name through six api.py signatures.
current_fetch_owner = contextvars.ContextVar("current_fetch_owner", default="snapshot")


def _json_object_from_text(text: str) -> dict | None:
    stripped = (text or "").strip()
    if not stripped:
        return None
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(stripped[start:end + 1])
            except json.JSONDecodeError:
                return None
    return None


def _strip_think_blocks(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.IGNORECASE | re.DOTALL).strip()


class AgentProgress:
    """Manages progress tracking for multiple agents with interactive display."""

    def __init__(self):
        self.agent_status: Dict[str, Dict[str, str]] = {}
        self.started = False
        self._frame = 0
        self.update_handlers: List[Callable[[str, Optional[str], str], None]] = []
        self.live: Optional[Live] = None
        self._done_agents: set = set()
        self._active_agents: set = set()
        # Live telemetry, accumulated across the ThreadPoolExecutor workers in the
        # engine — guard every mutation with _tele_lock or concurrent += loses counts.
        self.agent_tokens: Dict[str, dict] = {}   # agent_name -> {"prompt": int, "completion": int}
        self.fetch_counts: Dict[str, int] = {}    # agent_name -> data-fetch call count
        # (agent_id, ticker) -> {"reasoning": str, "source": "reasoning_content"|"response_content"}
        # Captured at the call_llm seam so the inspector can show what an LLM agent "thought".
        self.agent_reasoning: Dict[tuple, dict] = {}
        self._tele_lock = threading.Lock()

    def register_handler(self, handler: Callable[[str, Optional[str], str], None]):
        """Register a handler to be called when agent status updates."""
        self.update_handlers.append(handler)
        return handler

    def unregister_handler(self, handler: Callable[[str, Optional[str], str], None]):
        """Unregister a previously registered handler."""
        if handler in self.update_handlers:
            self.update_handlers.remove(handler)

    def start(self):
        """Start the progress display."""
        if not self.started:
            self.live = Live(
                self._build_display(),
                console=console,
                refresh_per_second=10,
                transient=False
            )
            self.live.start()
            self.started = True

    def stop(self):
        """Stop the progress display."""
        if self.started and self.live:
            self.live.stop()
            self.started = False
            self.live = None

    def update_status(self, agent_name: str, ticker: Optional[str] = None, status: str = "", analysis: Optional[str] = None):
        """Update the status of an agent."""
        if agent_name not in self.agent_status:
            self.agent_status[agent_name] = {"status": "", "ticker": None, "analysis": None}

        prev_status = self.agent_status[agent_name]["status"]
        is_done = status.lower() == "done"

        if ticker:
            self.agent_status[agent_name]["ticker"] = ticker
        if status:
            self.agent_status[agent_name]["status"] = status
        if analysis:
            self.agent_status[agent_name]["analysis"] = analysis

        if is_done and prev_status.lower() != "done":
            self._done_agents.add(agent_name)
        elif not is_done:
            self._done_agents.discard(agent_name)
            self._active_agents.add(agent_name)
        elif status.lower() == "done":
            self._active_agents.discard(agent_name)

        timestamp = datetime.now(timezone.utc).isoformat()
        self.agent_status[agent_name]["timestamp"] = timestamp

        for handler in self.update_handlers:
            handler(agent_name, ticker, status, analysis, timestamp)

        self._refresh_display()

    def add_tokens(self, agent_name: str, prompt_tokens: int = 0, completion_tokens: int = 0) -> None:
        """Accumulate per-agent token usage. Safe to call from worker threads."""
        with self._tele_lock:
            acc = self.agent_tokens.setdefault(agent_name or "unknown", {"prompt": 0, "completion": 0})
            acc["prompt"] += int(prompt_tokens or 0)
            acc["completion"] += int(completion_tokens or 0)

    def token_total(self, agent_name: str) -> int:
        acc = self.agent_tokens.get(agent_name)
        return (acc["prompt"] + acc["completion"]) if acc else 0

    def record_fetch(self, agent_name: str, kind: str) -> None:  # noqa: ARG002 - kind kept for callers/future detail
        """Count one data-fetch call against an agent. Safe to call from worker threads."""
        with self._tele_lock:
            self.fetch_counts[agent_name or "unknown"] = self.fetch_counts.get(agent_name or "unknown", 0) + 1

    def total_fetches(self) -> int:
        with self._tele_lock:
            return sum(self.fetch_counts.values())

    def capture_reasoning(self, agent_name: str, ticker: str | None, reasoning_content, response_content) -> None:
        """Store an LLM agent's thinking, keyed by (agent, ticker). Prefers a dedicated
        reasoning stream (o1/o3/DeepSeek-R1 reasoning_content); falls back to the raw
        model output. Last write per key wins, so a retry overwrites cleanly."""
        text = (reasoning_content or "").strip()
        source = "reasoning_content"
        if not text:
            text = (response_content or "").strip()
            source = "response_content"
            parsed = _json_object_from_text(_strip_think_blocks(text))
            if isinstance(parsed, dict):
                for key in ("reasoning", "explanation", "rationale", "analysis"):
                    value = parsed.get(key)
                    if isinstance(value, str) and value.strip():
                        text = value.strip()
                        source = f"response_content.{key}"
                        break
        if not text:
            return
        with self._tele_lock:
            self.agent_reasoning[(agent_name or "unknown", ticker)] = {"reasoning": text[:4000], "source": source}

    def reasoning_snapshot(self) -> dict:
        with self._tele_lock:
            return {k: dict(v) for k, v in self.agent_reasoning.items()}

    def reset_telemetry(self) -> None:
        """Clear token + fetch + reasoning accumulators. Call at the start of each run
        so the long-lived chat session never shows stale cross-run state."""
        with self._tele_lock:
            self.agent_tokens.clear()
            self.fetch_counts.clear()
            self.agent_reasoning.clear()

    def get_all_status(self):
        """Get the current status of all agents as a dictionary."""
        return {
            agent_name: {
                "ticker": info["ticker"],
                "status": info["status"],
                "display_name": self._get_display_name(agent_name)
            }
            for agent_name, info in self.agent_status.items()
        }

    def _get_display_name(self, agent_name: str) -> str:
        """Convert agent_name to a display-friendly format."""
        return agent_name.replace("_agent", "").replace("_", " ").title()

    def _get_spinner_frame(self) -> str:
        """Get current spinner frame for animation."""
        frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        return frames[self._frame % len(frames)]

    def _get_status_info(self, status: str) -> tuple[str, str]:
        """Get symbol and color for status."""
        status_lower = status.lower()
        if status_lower == "done":
            return "✓", "green"
        elif status_lower == "error":
            return "✗", "red"
        elif any(kw in status_lower for kw in ["fetching", "fetch"]):
            return self._get_spinner_frame(), "cyan"
        elif any(kw in status_lower for kw in ["running", "calling", "scoring", "deciding", "classifying"]):
            return self._get_spinner_frame(), "yellow"
        elif any(kw in status_lower for kw in ["computing", "analyzing", "processing"]):
            return self._get_spinner_frame(), "magenta"
        return "○", "white"

    def _build_status_line(self, agent_name: str, info: dict) -> Text:
        """Build a single status line for an agent."""
        status = info["status"]
        ticker = info.get("ticker")
        symbol, color = self._get_status_info(status)
        agent_display = self._get_display_name(agent_name)

        line = Text()
        line.append(f" {symbol} ", style=f"bold {color}")

        name_style = "bold white" if status.lower() == "done" else "bold cyan"
        line.append(f"{agent_display:<20}", style=name_style)

        if ticker:
            line.append(f" [{ticker}]", style="cyan bold")

        status_style = f"bold {color}" if status.lower() == "done" else color
        line.append(f" {status}", style=status_style)

        return line

    def _build_display(self) -> Panel:
        """Build the main display panel."""
        lines = []

        title_bar = Text()
        title_bar.append("◆ ", style="bold cyan")
        title_bar.append("Nova Trader", style="bold cyan underline")
        title_bar.append("  ·  Recommendation Engine", style="dim")
        lines.append(title_bar)
        lines.append(Text())

        done_count = len(self._done_agents)
        total_count = len(self.agent_status)
        active_count = total_count - done_count

        summary = Text()
        if done_count == total_count and total_count > 0:
            summary.append(" ● ", style="bold green")
            summary.append(" All agents complete  ", style="green")
        elif active_count > 0:
            summary.append(" ◐ ", style="bold yellow")
            summary.append(f" {active_count} agent(s) running  ", style="yellow")
            if done_count > 0:
                summary.append(f"· {done_count} done", style="dim")
        else:
            summary.append(" ○ ", style="white")
            summary.append(" Initializing...", style="white")

        lines.append(summary)
        lines.append(Text())

        def sort_key(item):
            agent_name = item[0]
            if "snapshot" in agent_name:
                return (0, agent_name)
            elif "risk_management" in agent_name:
                return (4, agent_name)
            elif "portfolio_management" in agent_name:
                return (5, agent_name)
            else:
                return (1, agent_name)

        for agent_name, info in sorted(self.agent_status.items(), key=sort_key):
            lines.append(self._build_status_line(agent_name, info))

        content = Text("\n").join(lines)

        self._frame += 1

        border_style = "green" if done_count == total_count and total_count > 0 else "cyan"

        return Panel(
            content,
            border_style=border_style,
            padding=(1, 2),
            width=75,
        )

    def _refresh_display(self):
        """Refresh the display."""
        if self.started and self.live:
            self.live.update(self._build_display())


progress = AgentProgress()
