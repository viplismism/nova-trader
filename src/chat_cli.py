"""Chat-style terminal interface for Nova Trader."""

from __future__ import annotations

import contextlib
import io
import re
import shlex
import threading
import time

from rich.console import Console, Group
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.text import Text

# Strips SGR color codes to measure visible width when padding rendered lines.
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

from src.chat.context import (
    build_context as _build_context,
    build_context_from_metadata as _build_context_from_metadata,
)
from src.chat.models import ChatEvent, ChatSettings, DEFAULT_AGENTS, IntentRoute
from src.chat.rendering import (
    answer_renderable as _answer_renderable,
    clean_action as _clean_action,
    event_renderable as _event_renderable,
    field_value as _field_value,
    recommendation_renderable as _recommendation_renderable,
    recommendation_summary_text as _recommendation_summary_text,
    recommendation_verdict_text as _recommendation_verdict_text,
    render_recommendation as _render_recommendation,
    simple_response as _simple_response,
    status_color as _status_color,
    stream_markdown_display_text as _stream_markdown_display_text,
    ticker_details_renderable as _ticker_details_renderable,
    ticker_details_text as _ticker_details_text,
)
from src.chat.routing import (
    default_router_model as _default_router_model,
    extract_tickers as _extract_tickers,
    fallback_ticker_route as _fallback_ticker_route,
    is_analysis_prompt as _is_analysis_prompt,
    model_choices_for as _model_choices_for,
    normalize_provider as _normalize_provider,
    provider_choices as _provider_choices,
)
from src.chat.theme import AMBER, ASH, INK, PTK_ASH, PTK_BG, PTK_INK, ROSE, SAGE, TEAL
from src.engine import run_engine
from src.runs import RunRecorder, runs_root
from src.schemas.signals import Recommendation
from src.utils.progress import progress


class NovaChat:
    def __init__(self, console: Console, settings: ChatSettings):
        self.console = console
        self.settings = settings
        self.last_run_id: str | None = None
        self.last_recommendation: Recommendation | None = None
        self._app = None
        self._transcript_window = None
        self._status_window = None
        self._input_area = None
        self._header_area = None
        self._transcript: list = []  # list of Rich renderables
        self._status_lines: dict[str, str] = {}
        self._status_takeaway: dict[str, str] = {}  # agent label -> stance (for the chatter feed)
        self._status_tokens: dict[str, int] = {}  # agent label -> live token total
        self._status_events: list[tuple[str, str, str | None]] = []
        self._run_tickers: list[str] = []
        self._run_decisions: str = ""
        self._transcript_scroll = 0  # lines above the bottom; 0 means follow newest.
        self._busy = False
        self._ui_lock = threading.Lock()
        self._transcript_nlines = 0
        self._status_nlines = 0
        self._stream_active = False
        self._recv_answer = ""      # received from the model
        self._recv_reason = ""
        self._disp_answer = ""      # revealed on screen by the animator
        self._disp_reason = ""
        self._recv_done = False
        self._last_reasoning = ""   # reasoning kept visible in the status pane after a stream
        self._stream_width = 80
        self._static_prefix = ""
        self._animator: threading.Thread | None = None
        self._inspector_mode = False
        self._inspector_scroll = 0
        self._inspector_selection = 0
        self._inspector_agent_keys: list[str] = []  # ordered list labels, set at paint time
        self._inspector_list_window = None
        self._inspector_detail_window = None
        self._active_run_id: str | None = None       # captured BEFORE the engine runs, for live tailing
        self._llm_cache: dict[str, tuple] = {}         # run_id -> (mtime, size, grouped) so we re-parse only on growth
        self._heartbeat: threading.Thread | None = None

    def run(self) -> int:
        try:
            from prompt_toolkit.application import Application
            from prompt_toolkit.data_structures import Point
            from prompt_toolkit.formatted_text import ANSI
            from prompt_toolkit.filters import Condition
            from prompt_toolkit.key_binding import KeyBindings
            from prompt_toolkit.layout import ConditionalContainer, HSplit, Layout, VSplit, Window
            from prompt_toolkit.layout.controls import FormattedTextControl
            from prompt_toolkit.layout.margins import ScrollbarMargin
            from prompt_toolkit.styles import Style
            from prompt_toolkit.widgets import Frame, TextArea
        except ImportError:
            self.console.print("[red]Chat CLI requires prompt_toolkit. Run `uv pip install -e .`.[/red]")
            return 2

        # Read-only, color-capable panes: Rich renderables -> ANSI -> prompt_toolkit.
        transcript_control = FormattedTextControl(
            text=lambda: ANSI(self._transcript_ansi()),
            focusable=False,
            get_cursor_position=lambda: Point(0, self._transcript_nlines),
        )
        self._transcript_window = Window(
            transcript_control,
            wrap_lines=False,
            right_margins=[ScrollbarMargin(display_arrows=False)],
        )
        status_control = FormattedTextControl(
            text=lambda: ANSI(self._status_ansi()),
            focusable=False,
            get_cursor_position=lambda: Point(0, self._status_nlines),
        )
        self._status_window = Window(
            status_control,
            wrap_lines=False,
            right_margins=[ScrollbarMargin(display_arrows=False)],
        )
        # Left = clickable agent list (raw fragments carry mouse handlers; cannot be ANSI).
        # Right = the selected agent's detail, ANSI so the line-padding overwrite fix applies.
        self._inspector_list_window = Window(
            FormattedTextControl(text=lambda: self._inspector_list_fragments(), focusable=False),
            wrap_lines=False,
            width=32,
            right_margins=[ScrollbarMargin(display_arrows=False)],
        )
        self._inspector_detail_window = Window(
            FormattedTextControl(text=lambda: ANSI(self._inspector_detail_ansi()), focusable=False),
            wrap_lines=False,
            right_margins=[ScrollbarMargin(display_arrows=False)],
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
            height=2,
            style="class:header",
        )

        footer = Window(
            FormattedTextControl(lambda: self._footer_fragments()),
            height=1,
            style="class:footer",
        )

        main_view = VSplit(
            [
                Frame(self._transcript_window, title="conversation"),
                Frame(self._status_window, title="run status", width=44),
            ],
            padding=1,
        )
        inspector_view = Frame(
            VSplit(
                [
                    Frame(self._inspector_list_window, title="agents"),
                    Frame(self._inspector_detail_window, title="detail"),
                ],
                padding=1,
            ),
            title="inspector",
        )
        # ConditionalContainer (not layout reassignment): a redraw mid-stream simply
        # re-selects the active branch, so the animator's invalidate() never races a
        # half-built layout, and header/input/footer stay mounted so input keeps focus.
        root = HSplit(
            [
                self._header_area,
                ConditionalContainer(main_view, filter=Condition(lambda: not self._inspector_mode)),
                ConditionalContainer(inspector_view, filter=Condition(lambda: self._inspector_mode)),
                Frame(self._input_area, title="message"),
                footer,
            ]
        )

        bindings = KeyBindings()

        @bindings.add("c-c")
        @bindings.add("c-d")
        def _(event):
            event.app.exit(result=0)

        # Tab toggles the full-screen inspector. eager=True is required because the
        # input TextArea is focused and would otherwise consume Tab for focus traversal.
        @bindings.add("tab", eager=True)
        def _(event):
            self._toggle_inspector()

        inspector_active = Condition(lambda: self._inspector_mode)

        @bindings.add("pageup", eager=True)
        @bindings.add("escape", "v", eager=True)
        def _(event):
            if self._inspector_mode:
                self._scroll_inspector(self._pane_height(self._inspector_detail_window, 30) - 3)
                return
            self._scroll_transcript(self._transcript_page_size())

        @bindings.add("pagedown", eager=True)
        @bindings.add("c-v", eager=True)
        def _(event):
            if self._inspector_mode:
                self._scroll_inspector(-(self._pane_height(self._inspector_detail_window, 30) - 3))
                return
            self._scroll_transcript(-self._transcript_page_size())

        # Agent selection — gated by `inspector_active` so j/k/up/down are INACTIVE (and
        # fall through to the input box) when the inspector is closed. Without the filter,
        # eager j/k would swallow those letters from normal typing.
        @bindings.add("up", eager=True, filter=inspector_active)
        def _(event):
            self._select_agent(self._inspector_selection - 1)

        @bindings.add("down", eager=True, filter=inspector_active)
        def _(event):
            self._select_agent(self._inspector_selection + 1)

        # Esc/Home: in inspector mode, close it; otherwise scroll the transcript to top.
        # Guarding inside the existing escape-prefix handler avoids adding a bare
        # `escape` binding that would clash with the two-key escape sequences below.
        @bindings.add("home", eager=True)
        @bindings.add("escape", "<", eager=True)
        def _(event):
            if self._inspector_mode:
                self._toggle_inspector()
                return
            self._scroll_transcript(10_000)

        @bindings.add("end", eager=True)
        @bindings.add("escape", ">", eager=True)
        def _(event):
            if self._inspector_mode:
                self._scroll_inspector(-10_000)
                return
            self._scroll_transcript(-10_000)

        style = Style.from_dict(
            {
                "header": f"bg:{PTK_BG} {TEAL} bold",
                "footer": f"bg:{PTK_BG} {PTK_ASH}",
                "frame.border": PTK_ASH,
                "frame.label": f"{TEAL} bold",
            }
        )

        self._app = Application(
            layout=Layout(root, focused_element=self._input_area),
            key_bindings=bindings,
            full_screen=True,
            mouse_support=True,
            style=style,
        )
        self._emit(
            ChatEvent(
                "assistant",
                "Ready when you are.",
                "Type a ticker set to get a recommendation — e.g. `analyze AAPL,NVDA`.\n"
                "After a run: `details AAPL`, `show last`, switch with `model …`, or `help`.",
            )
        )
        self._refresh_tui()
        return int(self._app.run() or 0)

    def _header_fragments(self):
        agents = "default set (7)" if self.settings.agents == DEFAULT_AGENTS else ",".join(self.settings.agents)
        dot_style = (AMBER + " bold") if self._busy else (SAGE + " bold")
        status = "running" if self._busy else "ready"
        return [
            (TEAL + " bold", "  ◆ Nova Trader"),
            (PTK_ASH, "  ·  chat workspace"),
            (dot_style, "          ● " + status + "\n"),
            (PTK_ASH, "  model "),
            (PTK_INK + " bold", f"{self.settings.provider} / {self.settings.model}"),
            (PTK_ASH, "    ·    agents "),
            (PTK_INK, agents[:48]),
        ]

    def _footer_fragments(self):
        sep = (PTK_ASH, "   ")
        key = lambda label: (TEAL, label)  # noqa: E731
        return [
            (PTK_ASH, " "),
            key("enter"), (PTK_ASH, " send"), sep,
            key("tab"), (PTK_ASH, " inspect"), sep,
            key("pgup/pgdn"), (PTK_ASH, " scroll"), sep,
            key("ctrl-c"), (PTK_ASH, " quit"), sep,
            (PTK_ASH, "try: "),
            (PTK_INK, "ask a question"), sep,
            (PTK_INK, "analyze AAPL,NVDA"), sep,
            (PTK_INK, "/mode"), sep,
            (PTK_INK, "/model"), sep,
            (PTK_INK, "show last"), sep,
            (PTK_INK, "/help"),
        ]

    def _emit(self, event: ChatEvent) -> None:
        with self._ui_lock:
            if event.kind == "tool":
                self._status_lines[event.title] = event.body
                self._status_events.append((event.title, event.body, self._status_takeaway.get(event.title)))
                self._status_events = self._status_events[-60:]
            else:
                self._transcript.append(_event_renderable(event.kind, event.title, event.body))
        self._refresh_tui()

    def _emit_result(self, recommendation: Recommendation) -> None:
        with self._ui_lock:
            self._transcript.append(_recommendation_renderable(recommendation))
        self._refresh_tui()

    def _emit_answer(self, text: str) -> None:
        with self._ui_lock:
            self._transcript.append(_answer_renderable(text))
        self._refresh_tui()

    def _prime_analysis_status(self, tickers: list[str]) -> None:
        with self._ui_lock:
            self._run_tickers = tickers
            self._run_decisions = ""
            self._status_events.append(("Run", f"Starting {','.join(tickers)}", None))
            self._status_events = self._status_events[-60:]
        self._refresh_tui()

    def _finish_analysis_status(self, recommendation: Recommendation) -> None:
        verdict = []
        for ticker in recommendation.tickers:
            decision = recommendation.decisions.per_ticker.get(ticker)
            if decision:
                verdict.append(f"{ticker} {str(_field_value(decision, 'action', 'hold')).upper()}")
        with self._ui_lock:
            self._run_decisions = ", ".join(verdict)
            for label in list(self._status_lines):
                if label != "Run" and self._status_lines[label].lower() != "done":
                    self._status_lines[label] = "Done"
            for label in list(self._status_lines):
                if label != "Run" and " [" not in label and any(existing.startswith(f"{label} [") for existing in self._status_lines):
                    del self._status_lines[label]
            self._status_events.append(("Run", "Finished " + (self._run_decisions or ", ".join(recommendation.tickers)), None))
            self._status_events = self._status_events[-60:]
        self._refresh_tui()

    # --- streaming answers ------------------------------------------------
    # Receiving (network) is decoupled from displaying. The worker appends raw tokens
    # to _recv_*; a steady ~60fps animator reveals _disp_* toward _recv_* so the text
    # flows smoothly as a typewriter even when the model delivers tokens in bursts.
    _FRAME = 0.016  # ~60fps

    def _begin_stream(self) -> None:
        # Render the existing transcript once so per-frame redraws only re-render the
        # small live tail (reasoning box + answer), keeping streaming cheap.
        with self._ui_lock:
            items = list(self._transcript[-60:])
        width = self._pane_width(self._transcript_window, 80)
        prefix = self._render_ansi(Group(*self._interleave(items)), width) if items else ""
        with self._ui_lock:
            self._stream_active = True
            self._recv_answer = self._recv_reason = ""
            self._disp_answer = self._disp_reason = ""
            self._recv_done = False
            self._last_reasoning = ""
            self._stream_width = width
            self._static_prefix = prefix
        self._animator = threading.Thread(target=self._animate, daemon=True)
        self._animator.start()
        self._refresh_tui()

    def _animate(self) -> None:
        """Reveal displayed text toward received text at a steady rate (typewriter)."""
        def advance(disp: str, recv: str) -> str:
            pending = len(recv) - len(disp)
            if pending <= 0:
                return disp
            # Steady typewriter: a few chars/frame so a big burst never dumps at once,
            # capped so we still finish a long buffer in reasonable time (~60fps).
            step = min(max(3, pending // 20), 12)
            return recv[: len(disp) + step]

        while True:
            with self._ui_lock:
                if not self._stream_active:
                    return
                self._disp_reason = advance(self._disp_reason, self._recv_reason)
                self._disp_answer = advance(self._disp_answer, self._recv_answer)
                caught_up = (
                    self._disp_reason == self._recv_reason
                    and self._disp_answer == self._recv_answer
                )
                done = self._recv_done
            self._refresh_tui()
            if done and caught_up:
                return
            time.sleep(self._FRAME)

    def _finalize_stream(self) -> str:
        with self._ui_lock:
            answer = self._recv_answer.strip()
            reasoning = self._recv_reason.strip()
            # Reasoning lives in the run-status box, not the conversation. Keep the final
            # reasoning there so it stays readable; only the answer goes in the transcript.
            self._last_reasoning = reasoning
            if answer:
                self._transcript.append(_answer_renderable(answer))
            self._stream_active = False
            self._recv_answer = self._recv_reason = ""
            self._disp_answer = self._disp_reason = ""
        self._refresh_tui()
        return answer

    def _stream_answer(self, messages: list[dict[str, str]]) -> str:
        """Stream a free-form answer from the configured model into the transcript.

        Thinking tokens stream into a small reasoning box; the user-facing reply streams
        into the answer and is boxed as a paragraph when finished.
        """
        from src.utils.llm import stream_chat

        self._begin_stream()
        try:
            for channel, chunk in stream_chat(messages, provider=self.settings.provider, model=self.settings.model):
                with self._ui_lock:
                    if channel == "reasoning":
                        self._recv_reason += chunk
                    else:
                        self._recv_answer += chunk
        except Exception as exc:
            with self._ui_lock:
                self._stream_active = False
                self._recv_answer = self._recv_reason = ""
                self._disp_answer = self._disp_reason = ""
            if self._animator:
                self._animator.join(timeout=1.0)
            self._emit(ChatEvent("error", "Could not reach the model", str(exc)))
            return ""
        with self._ui_lock:
            self._recv_done = True
        if self._animator:
            self._animator.join(timeout=10.0)  # let the typewriter finish revealing
        return self._finalize_stream()

    def _refresh_tui(self) -> None:
        if not self._app:
            return
        self._app.invalidate()

    def _transcript_page_size(self) -> int:
        try:
            info = self._transcript_window.render_info
            if info is not None and info.window_height:
                return max(3, info.window_height - 3)
        except Exception:
            pass
        return 20

    def _scroll_transcript(self, delta: int) -> None:
        with self._ui_lock:
            self._transcript_scroll = max(0, self._transcript_scroll + delta)
        self._refresh_tui()

    # --- inspector (tab to open) ------------------------------------------
    # A full-screen view modeled on the workflows viewer: a live header (model ·
    # tokens · fetches), the phases, an interleaved agent-activity log showing how
    # the analysts work together, and — once a run finishes — signals grouped by agent.

    def _toggle_inspector(self) -> None:
        # No _ui_lock needed: ConditionalContainer just re-selects a branch on the
        # next pull-render; the animator only mutates _disp_*/_recv_* + invalidates.
        self._inspector_mode = not self._inspector_mode
        if not self._inspector_mode:
            self._inspector_scroll = 0
            self._inspector_selection = 0  # reopen always starts on the first agent
            if self._app:
                self._app.layout.focus(self._input_area)
        self._refresh_tui()

    def _scroll_inspector(self, delta: int) -> None:
        with self._ui_lock:
            self._inspector_scroll = max(0, self._inspector_scroll + delta)
        self._refresh_tui()

    def _select_agent(self, idx: int) -> None:
        with self._ui_lock:
            if self._inspector_agent_keys:
                self._inspector_selection = max(0, min(len(self._inspector_agent_keys) - 1, idx))
                self._inspector_scroll = 0
        self._refresh_tui()

    def _llm_calls_for_run(self) -> dict[str, list[dict]]:
        """Per-agent llm.jsonl records for the active/last run. mtime+size cached so the
        file is re-parsed only when it grows, and the disk read happens off the UI lock."""
        with self._ui_lock:
            run_id = self._active_run_id or self.last_run_id
        if not run_id:
            return {}
        path = runs_root() / run_id / "llm.jsonl"
        try:
            st = path.stat()
        except FileNotFoundError:
            return {}
        cached = self._llm_cache.get(run_id)
        if cached and cached[0] == st.st_mtime and cached[1] == st.st_size:
            return cached[2]
        grouped = RunRecorder.load_llm_calls(run_id)  # I/O off the UI lock
        self._llm_cache[run_id] = (st.st_mtime, st.st_size, grouped)
        return grouped

    def _inspector_list_fragments(self):
        """Left pane: one clickable row per agent that has reported activity. Returns raw
        (style, text, mouse_handler) fragments — mouse handlers can't ride on ANSI text."""
        from prompt_toolkit.mouse_events import MouseEventType

        with self._ui_lock:
            events = list(self._status_events)
            takeaways = dict(self._status_takeaway)
            seen: dict[str, str] = {}
            for label, status, _ in events:
                if label != "Run":
                    seen[label] = status  # last status per label, preserving first-seen order
            keys = list(seen.keys())
            self._inspector_agent_keys = keys  # set under lock so click indices map to painted rows
            if self._inspector_selection >= len(keys):
                self._inspector_selection = max(0, len(keys) - 1)
            sel = self._inspector_selection

        if not keys:
            return [(ASH, "no agents yet".ljust(31) + "\n")]

        def make_handler(i):
            def handler(mouse_event):
                if mouse_event.event_type == MouseEventType.MOUSE_UP:
                    self._select_agent(i)
                    return None
                return NotImplemented  # let the Window handle wheel-scroll etc.
            return handler

        frags = []
        for i, label in enumerate(keys):
            marker, mcolor = self._activity_marker(seen[label], takeaways.get(label))
            name = self._clip_text(self._activity_label(label), 26)
            prefix = "› " if i == sel else "  "
            line = (prefix + marker + " " + name).ljust(31)  # pad to width-1 so stale cells clear
            style = f"bg:{PTK_INK} bold" if i == sel else f"{mcolor}"
            frags.append((style, line + "\n", make_handler(i)))
        return frags

    @staticmethod
    def _slice_lines_from_top(text: str, height: int, scroll: int) -> tuple[str, int]:
        """Top-anchored window — the detail view reads top-down (header first), unlike the
        log tail. scroll = lines hidden above the top; clamped so we never scroll past end."""
        lines = text.split("\n") if text else [""]
        total = len(lines)
        visible = max(1, height)
        max_scroll = max(0, total - visible)
        scroll = min(max(0, scroll), max_scroll)
        return "\n".join(lines[scroll:scroll + visible]), scroll

    def _inspector_detail_ansi(self) -> str:
        with self._ui_lock:
            rec = self.last_recommendation
            events = list(self._status_events)
            takeaways = dict(self._status_takeaway)
            tokens = dict(self._status_tokens)
            run_tickers = list(self._run_tickers)
            status_lines = list(self._status_lines.items())
            reasoning = self._disp_reason if self._stream_active else self._last_reasoning
            scroll = self._inspector_scroll
            keys = list(self._inspector_agent_keys)
            sel = self._inspector_selection
            agent_reasoning = progress.reasoning_snapshot()
        fetches = progress.total_fetches()
        llm_by_agent = self._llm_calls_for_run()  # disk I/O, off the UI lock
        width = self._pane_width(self._inspector_detail_window, 60)
        height = self._pane_height(self._inspector_detail_window, 30)
        if keys and 0 <= sel < len(keys):
            body = self._selected_agent_renderable(keys[sel], rec, events, takeaways, tokens, agent_reasoning, llm_by_agent)
        else:
            # No agent selected yet → reuse the untouched overview (also the guarded test path).
            body = self._inspector_renderable(rec, events, takeaways, tokens, fetches, run_tickers, status_lines, reasoning)
        text = self._render_ansi(body, width)
        total = text.count("\n") + 1
        sliced, clamped = self._slice_lines_from_top(text, max(1, height), scroll)
        if clamped != scroll:
            with self._ui_lock:
                self._inspector_scroll = clamped
        return sliced

    def _agent_id_to_label(self, agent_id: str, ticker: str | None = None) -> str:
        """Normalize a raw agent_id (+ ticker) to the same humanized label the list uses,
        so signals, the reasoning store, and llm.jsonl all reconcile on one key."""
        base = self._activity_label(agent_id.replace("_", " ").title())
        return f"{base} [{ticker}]" if ticker else base

    def _selected_agent_renderable(self, label, rec, events, takeaways, tokens, agent_reasoning, llm_by_agent) -> Group:
        """Right pane: everything known about ONE agent — what it's doing, what it was
        asked, what it thought, and the WHY (signal + factor breakdown)."""
        norm = self._activity_label(label)

        def matches(agent_id: str, ticker: str | None = None) -> bool:
            return self._agent_id_to_label(agent_id, ticker) == norm or self._agent_id_to_label(agent_id) == norm

        parts: list = []
        status = next((s for lbl, s, _ in reversed(events) if lbl == label), "")
        head = Text()
        head.append(norm, style=f"bold {TEAL}")
        action, acolor = self._activity_action(status, takeaways.get(label))
        head.append("   " + action, style=acolor)
        parts += [head, Text("")]

        agent_events = [(s, takeaway) for lbl, s, takeaway in events if lbl == label]
        if agent_events:
            parts.append(Text("activity", style=f"bold {TEAL}"))
            recent_events = agent_events[-8:]
            done_events = [item for item in recent_events if item[0].lower() == "done"]
            other_events = [item for item in recent_events if item[0].lower() != "done"]
            for event_status, event_takeaway in done_events + other_events:
                action, acolor = self._activity_action(event_status, event_takeaway)
                row = Text("• ", style=ASH)
                row.append(action, style=acolor)
                cleaned = _clean_action(event_status)
                if cleaned and cleaned != action:
                    row.append(": ", style=ASH)
                    row.append(cleaned, style=INK)
                parts.append(row)
            parts.append(Text(""))

        # Reasoning, in priority order:
        # 1) council narrative for the risk/portfolio managers (rec.risk_reasoning /
        #    rec.portfolio_reasoning — how the committee weighed the whole board);
        # 2) the clean explain-only narration on an analyst's Signal;
        # 3) otherwise the captured model reasoning/explanation text.
        low_label = label.lower()
        explain_text = ""
        explain_header = "analyst reasoning"
        if rec is not None and "risk" in low_label and getattr(rec, "risk_reasoning", ""):
            explain_text, explain_header = rec.risk_reasoning, "council reasoning"
        elif rec is not None and "portfolio" in low_label and getattr(rec, "portfolio_reasoning", ""):
            explain_text, explain_header = rec.portfolio_reasoning, "council reasoning"
        elif rec is not None and getattr(rec, "signals", None):
            for s in rec.signals:
                if matches(str(_field_value(s, "agent_id", "agent")), str(_field_value(s, "ticker", "")) or None):
                    explain_text = str(_field_value(s, "explain_reasoning", "") or "")
                    if explain_text:
                        break
        if explain_text:
            parts.append(Text("reasoning", style=f"bold {TEAL}"))
            parts.append(Text(self._clip_text(explain_text, 1100), style=INK))
            parts.append(Text(""))
        else:
            for (aid, tkr), info in agent_reasoning.items():
                if matches(aid, tkr):
                    parts.append(Text("reasoning", style=f"bold {TEAL}"))
                    parts.append(Text(self._clip_text(info["reasoning"], 800), style=INK))
                    if info["source"] == "response_content":
                        parts.append(Text("  raw model output; no dedicated reasoning stream", style=ASH))
                    parts.append(Text(""))
                    break

        # The WHY — the agent's signal + its factor breakdown. Works for every agent: the
        # 6 deterministic analysts already publish their sub-scores in Signal.key_factors.
        sigs = []
        if rec is not None and getattr(rec, "signals", None):
            sigs = [
                s for s in rec.signals
                if matches(str(_field_value(s, "agent_id", "agent")), str(_field_value(s, "ticker", "")) or None)
            ]
        low = label.lower()
        if sigs:
            parts.append(Text("signal & why", style=f"bold {TEAL}"))
            for sig in sigs:
                direction = str(_field_value(sig, "direction", "neutral"))
                confidence = float(_field_value(sig, "confidence", 0.0) or 0.0)
                parts.append(Text.assemble(("• ticker: ", ASH), (str(_field_value(sig, "ticker", "")), INK)))
                verdict = Text("• verdict: ", style=ASH)
                verdict.append(f"{direction} ", style=_status_color(direction))
                verdict.append(f"{confidence:.0%}", style=INK)
                parts.append(verdict)
                reason = str(_field_value(sig, "reasoning", "") or "")
                if reason:
                    parts.append(Text.assemble(("• reason: ", ASH), (self._clip_text(reason, 220), INK)))
                factors = list(_field_value(sig, "key_factors", []) or [])
                if factors:
                    parts.append(Text("• factors:", style=ASH))
                    for factor in factors:
                        raw = str(factor)
                        if "=" in raw:
                            key, value = raw.split("=", 1)
                            parts.append(Text.assemble(("  - " + key.strip().replace("_", " ")[:16].ljust(16), ASH), (value.strip(), INK)))
                        else:
                            parts.append(Text("  - " + raw, style=ASH))
        elif "portfolio" in low and rec is not None:
            # The portfolio manager has no Signal — its "why" is the per-ticker decision.
            decisions = getattr(getattr(rec, "decisions", None), "per_ticker", {}) or {}
            if decisions:
                parts.append(Text("decision & why", style=f"bold {TEAL}"))
                for tkr, dec in decisions.items():
                    action = str(_field_value(dec, "action", "hold")).upper()
                    confidence = float(_field_value(dec, "confidence", 0.0) or 0.0)
                    parts.append(Text.assemble(("• ticker: ", ASH), (str(tkr), INK)))
                    action_row = Text("• action: ", style=ASH)
                    action_row.append(action, style=_status_color(action))
                    action_row.append(f" {confidence:.0%}", style=INK)
                    parts.append(action_row)
                    qty = _field_value(dec, "quantity", 0)
                    if qty:
                        parts.append(Text.assemble(("• quantity: ", ASH), (str(qty), INK)))
                    reason = str(_field_value(dec, "reasoning", "") or "")
                    if reason:
                        parts.append(Text.assemble(("• reason: ", ASH), (self._clip_text(reason, 220), INK)))
        elif "risk" in low and rec is not None:
            # The risk manager's "why" is the per-ticker sizing limits it set.
            limits = getattr(getattr(rec, "limits", None), "per_ticker", {}) or {}
            if limits:
                parts.append(Text("risk limits", style=f"bold {TEAL}"))
                for tkr, lim in limits.items():
                    parts.append(Text.assemble(("• ticker: ", ASH), (str(tkr), INK)))
                    parts.append(Text.assemble(("• max position: ", ASH), (f"${_field_value(lim, 'max_position_dollars', 0):,.0f}", INK)))
                    parts.append(Text.assemble(("• max shares: ", ASH), (str(int(_field_value(lim, "max_shares", 0) or 0)), INK)))
                    parts.append(Text.assemble(("• volatility: ", ASH), (f"{float(_field_value(lim, 'annualized_volatility', 0.0) or 0.0):.0%}", INK)))
        if not parts[2:]:  # nothing beyond the header
            parts.append(Text("state", style=f"bold {TEAL}"))
            row = Text("  ")
            row.append("waiting", style=AMBER if self._busy else ASH)
            row.append("  no activity, reasoning, or signal details recorded yet", style=ASH)
            parts.append(row)

        return Group(*parts)

    def _inspector_renderable(
        self, rec, events, takeaways, tokens, fetches, run_tickers, status_lines, reasoning
    ) -> Group:
        if rec is None and not status_lines and not run_tickers:
            return Group(Text("No run yet — analyze a ticker set first, then press tab to inspect.", style=ASH))

        parts: list = []
        # Header status line — Running/complete · model · live tokens · fetches.
        tok_total = sum(tokens.values())
        head = Text()
        if self._busy:
            head.append("● Running", style=f"bold {AMBER}")
        else:
            head.append("● complete", style=f"bold {SAGE}")
        head.append(f"   ·   {self.settings.provider} / {self.settings.model}", style=ASH)
        if tok_total:
            head.append(f"   ·   {self._fmt_tok(tok_total)}", style=TEAL)
        if fetches:
            head.append(f"   ·   {fetches} fetches", style=TEAL)
        if run_tickers:
            head.append(f"   ·   {', '.join(run_tickers)}", style=INK)
        parts.append(head)
        parts.append(Text(""))

        # Phases (reuse the same helpers as the compact pane).
        if status_lines or run_tickers:
            n_tickers = max(1, len(run_tickers))
            expected_analysts = n_tickers * max(1, len(self.settings.agents))
            rows = [
                ("Snapshot", self._phase_counts(status_lines, "snapshot", expected=n_tickers)),
                ("Analysts", self._phase_counts(status_lines, "analysis", expected=expected_analysts)),
                ("Risk", self._phase_counts(status_lines, "risk", expected=n_tickers)),
                ("Portfolio", self._phase_counts(status_lines, "portfolio", expected=n_tickers)),
            ]
            phase_line = Text("phases   ", style=f"bold {TEAL}")
            for label, (d, t, a) in rows:
                marker = "✓" if (t and d >= t) else ("•" if (a or d) else "○")
                color = SAGE if marker == "✓" else (AMBER if marker == "•" else ASH)
                phase_line.append(f"{marker} {label} ", style=color)
                if marker == "•" and t:
                    phase_line.append(f"{d}/{t}  ", style=ASH)
                else:
                    phase_line.append(" ", style=ASH)
            parts.append(phase_line)
            parts.append(Text(""))

        # Live agent-activity log — the interleaved feed of every agent's progress,
        # so you can watch how the analysts work together. Newest at the bottom.
        if events:
            parts.append(Text("agent activity", style=f"bold {TEAL}"))
            for label, status, takeaway in events[-60:]:
                if label == "Run":
                    row = Text("  ")
                    row.append("▸ run  ", style=f"bold {TEAL}")
                    row.append(status, style=INK)
                    parts.append(row)
                    continue
                marker, mcolor = self._activity_marker(status, takeaway)
                action, acolor = self._activity_action(status, takeaway)
                row = Text("  ")
                row.append(f"{marker} ", style=f"bold {mcolor}")
                row.append(f"{self._clip_text(self._activity_label(label), 26):<26}", style=INK)
                row.append(action, style=acolor)
                tok = tokens.get(label)
                if tok:
                    row.append(f"   {self._fmt_tok(tok)}", style=ASH)
                parts.append(row)
            parts.append(Text(""))

        # Signals grouped by agent — available once the recommendation is built.
        if rec is not None and getattr(rec, "signals", None):
            by_agent: dict[str, list] = {}
            for sig in rec.signals:
                by_agent.setdefault(_field_value(sig, "agent_id", "agent"), []).append(sig)
            parts.append(Text("signals by agent", style=f"bold {TEAL}"))
            for agent_id, sigs in by_agent.items():
                parts.append(Text(self._activity_label(agent_id.replace("_", " ").title()), style=f"bold {INK}"))
                for sig in sigs:
                    direction = str(_field_value(sig, "direction", "neutral"))
                    confidence = _field_value(sig, "confidence", 0.0) or 0.0
                    reason = str(_field_value(sig, "reasoning", "") or "")
                    row = Text("  ")
                    row.append(f"{str(_field_value(sig, 'ticker', '')):<6}", style=INK)
                    row.append(f"{direction:<8}", style=_status_color(direction))
                    row.append(f"{float(confidence):.0%}  ", style=ASH)
                    row.append(self._clip_text(reason, 64), style=INK)
                    parts.append(row)
                parts.append(Text(""))

        # Model reasoning — only present for providers that stream thinking; this is
        # the one place run-status reasoning is surfaced (kept out of the compact pane).
        if reasoning:
            parts.append(Text("model reasoning", style=f"bold {TEAL}"))
            parts.append(Text(reasoning.strip(), style=ASH))

        return Group(*parts)

    def _render_ansi(self, renderable, width: int) -> str:
        width = max(20, width)
        buffer = Console(
            file=io.StringIO(),
            width=width,
            force_terminal=True,
            color_system="truecolor",
            highlight=False,
        )
        buffer.print(renderable)
        raw = buffer.file.getvalue().rstrip("\n")
        # Pad each line to the full pane width so a shorter new render fully overwrites
        # the previous one — otherwise prompt_toolkit leaves stale cells (e.g. the
        # "modeMiniMaxax" corruption when switching providers).
        padded = []
        for line in raw.split("\n"):
            visible = len(_ANSI_RE.sub("", line))
            if visible < width:
                line += " " * (width - visible)
            padded.append(line)
        return "\n".join(padded)

    def _pane_width(self, window, default: int) -> int:
        try:
            info = window.render_info
            if info is not None and info.window_width:
                return max(20, info.window_width - 1)
        except Exception:
            pass
        return default

    def _pane_height(self, window, default: int) -> int:
        try:
            info = window.render_info
            if info is not None and info.window_height:
                return max(3, info.window_height)
        except Exception:
            pass
        return default

    @staticmethod
    def _slice_lines_from_bottom(text: str, height: int, scroll: int) -> tuple[str, int]:
        lines = text.split("\n") if text else [""]
        total = len(lines)
        visible = max(1, height)
        max_scroll = max(0, total - visible)
        scroll = min(max(0, scroll), max_scroll)
        end = total - scroll
        start = max(0, end - visible)
        return "\n".join(lines[start:end]), scroll

    def _run_thinking_renderable(self) -> Group:
        with self._ui_lock:
            events = list(self._status_events)
            tickers = ",".join(self._run_tickers)
        latest = ""
        for label, status, _ in reversed(events):
            if label == "Run":
                continue
            latest = self._activity_label(label)
            break
        title = Text()
        title.append("• ", style=f"bold {AMBER}")
        title.append("Thinking through the run", style=f"bold {AMBER}")
        if tickers:
            title.append(f" for {tickers}", style=INK)
        title.append("...", style=AMBER)
        if latest:
            detail = Text()
            detail.append("└ ", style=ASH)
            detail.append(f"Watching {latest}", style=ASH)
            return Group(title, detail)
        return Group(title)

    def _transcript_ansi(self) -> str:
        with self._ui_lock:
            streaming = self._stream_active
            scroll = self._transcript_scroll
            show_run_thinking = self._busy and bool(self._run_tickers) and not streaming
            if streaming:
                prefix = self._static_prefix
                width = self._stream_width
                live = self._live_stream_renderable(self._disp_answer)
            else:
                items = list(self._transcript[-60:])
        if streaming:
            live_text = self._render_ansi(live, width)
            text = f"{prefix}\n\n{live_text}" if prefix else live_text
        else:
            width = self._pane_width(self._transcript_window, 80)
            render_items = self._interleave(items)
            if show_run_thinking:
                if render_items:
                    render_items.append(Text(""))
                render_items.append(self._run_thinking_renderable())
            group = Group(*render_items) if render_items else Text("")
            text = self._render_ansi(group, width)
        height = self._pane_height(self._transcript_window, 30)
        sliced, clamped_scroll = self._slice_lines_from_bottom(text, height, scroll)
        if clamped_scroll != scroll:
            with self._ui_lock:
                self._transcript_scroll = clamped_scroll
        self._transcript_nlines = sliced.count("\n")
        return sliced

    @staticmethod
    def _live_stream_renderable(answer: str) -> Group:
        # The answer streams here; the model's thinking streams in the run-status box.
        if answer:
            safe_answer = _stream_markdown_display_text(answer)
            try:
                body = Markdown(safe_answer, style=INK)
            except Exception:
                body = Text(safe_answer, style=INK)
        else:
            body = Text("thinking", style=ASH)  # immediate feedback before the first token
        return Group(body, Text("▌", style=f"bold {TEAL}"))  # streaming caret

    @staticmethod
    def _interleave(items: list) -> list:
        spaced: list = []
        for index, item in enumerate(items):
            if index:
                spaced.append(Text(""))
            spaced.append(item)
        return spaced

    def _status_ansi(self) -> str:
        with self._ui_lock:
            status_lines = list(self._status_lines.items())
            takeaways = dict(self._status_takeaway)
            events = list(self._status_events)
            thinking = self._stream_active
            reasoning = self._disp_reason if self._stream_active else self._last_reasoning
            run_tickers = list(self._run_tickers)
            run_decisions = self._run_decisions
            tokens = dict(self._status_tokens)
        fetches = progress.total_fetches()
        width = self._pane_width(self._status_window, 40)
        text = self._render_ansi(
            self._status_renderable(
                status_lines, takeaways, events, reasoning, thinking,
                run_tickers, run_decisions, tokens=tokens, fetches=fetches,
            ),
            width,
        )
        self._status_nlines = text.count("\n")
        return text

    @staticmethod
    def _chatter_name(label: str) -> str:
        # lowercase the agent name but keep the [TICKER] tag uppercase
        if " [" in label:
            name, tag = label.split(" [", 1)
            return f"{name.lower()} [{tag}"
        return label.lower()

    @staticmethod
    def _fmt_tok(n: int) -> str:
        return f"{n / 1000:.1f}k tok" if n >= 1000 else f"{n} tok"

    @staticmethod
    def _clip_text(value: str, limit: int = 31) -> str:
        value = " ".join(str(value or "").split())
        if len(value) <= limit:
            return value
        return value[: max(1, limit - 1)].rstrip() + "…"

    @staticmethod
    def _is_analysis_event(label: str) -> bool:
        low = label.lower()
        return any(key in low for key in ("snapshot", "technical", "fundamentals", "growth", "valuation", "news sentiment", "insider sentiment", "warren buffett", "risk manager", "portfolio manager"))

    @classmethod
    def _todo_rows(cls, status_lines: list[tuple[str, str]]) -> list[tuple[str, str]]:
        rows = [(label, status) for label, status in status_lines if cls._is_analysis_event(label)]
        scoped_bases = {label.split(" [", 1)[0] for label, _ in rows if " [" in label}
        return [
            (label, status)
            for label, status in rows
            if " [" in label or label not in scoped_bases
        ]

    @staticmethod
    def _phase_counts(status_lines: list[tuple[str, str]], phase: str, expected: int | None = None) -> tuple[int, int, bool]:
        phase = phase.lower()
        if phase == "snapshot":
            rows = [(label, status) for label, status in status_lines if label.lower().startswith("snapshot")]
        elif phase == "risk":
            rows = [(label, status) for label, status in status_lines if label.lower().startswith("risk manager")]
        elif phase == "portfolio":
            rows = [(label, status) for label, status in status_lines if label.lower().startswith("portfolio manager")]
        else:
            rows = [
                (label, status)
                for label, status in status_lines
                if NovaChat._is_analysis_event(label)
                and not label.lower().startswith(("snapshot", "risk manager", "portfolio manager"))
            ]
        done = sum(1 for _, status in rows if status.lower() == "done")
        total = expected if expected is not None else len(rows)
        active = any(status.lower() != "done" for _, status in rows)
        return done, total, active

    @staticmethod
    def _phase_row(label: str, done: int, total: int, active: bool, detail: str = "") -> Text:
        if total and done >= total:
            marker, color, state = "✓", SAGE, "complete"
        elif active or done:
            marker, color, state = "•", AMBER, f"{done}/{total}" if total else "running"
        else:
            marker, color, state = "○", ASH, "queued"
        row = Text()
        row.append(f"{marker} ", style=f"bold {color}")
        row.append(f"{label:<10}", style=color if state != "queued" else ASH)
        row.append(" ", style=ASH)
        row.append(detail or state, style=INK if state == "complete" else color)
        return row

    @staticmethod
    def _activity_label(label: str) -> str:
        scope = ""
        name = label
        if " [" in label:
            name, rest = label.split(" [", 1)
            scope = f" [{rest}"
        aliases = {
            "Snapshot": "Snapshot",
            "Technical": "Technical Agent",
            "Fundamentals": "Fundamentals Agent",
            "Growth": "Growth Agent",
            "Valuation": "Valuation Agent",
            "Warren Buffett": "Buffett Agent",
            "News Sentiment": "News Agent",
            "Insider Sentiment": "Insider Agent",
            "Risk Manager": "Risk Manager",
            "Portfolio Manager": "Portfolio",
        }
        return aliases.get(name, name) + scope

    @staticmethod
    def _activity_action(status: str, takeaway: str | None = None) -> tuple[str, str]:
        if takeaway:
            return takeaway, _status_color(takeaway)
        low = status.lower()
        if low.startswith("starting"):
            return low, ASH
        if status.lower() == "done":
            return "done", ASH
        if "fetching prices" in low:
            return "prices", AMBER
        if "fetching financials" in low:
            return "financials", AMBER
        if "fetching line items" in low:
            return "line items", AMBER
        if "fetching market cap" in low:
            return "market cap", AMBER
        if "fetching news" in low:
            return "news", AMBER
        if "fetching insider" in low:
            return "insiders", AMBER
        if "calling llm" in low:
            return "model", AMBER
        if "classif" in low:
            return "headlines", AMBER
        if "insider" in low:
            return "trades", AMBER
        if "deciding" in low:
            return "deciding", AMBER
        if "valuation" in low:
            return "valuation", AMBER
        if "growth" in low:
            return "growth", AMBER
        if "trend" in low or "momentum" in low:
            return "technicals", AMBER
        if "scor" in low:
            return "scoring", AMBER
        return _clean_action(status), AMBER

    @staticmethod
    def _activity_marker(status: str, takeaway: str | None = None) -> tuple[str, str]:
        if status.lower() in {"error", "failed"} or takeaway in {"failed", "error"}:
            return "✗", ROSE
        if status.lower() == "done":
            return "✓", SAGE
        return "•", AMBER

    def _status_renderable(
        self,
        status_lines: list[tuple[str, str]],
        takeaways: dict[str, str] | None = None,
        events: list[tuple[str, str, str | None]] | None = None,
        reasoning: str = "",  # noqa: ARG002 - model reasoning is intentionally hidden here.
        thinking: bool = False,
        run_tickers: list[str] | None = None,
        run_decisions: str = "",
        tokens: dict[str, int] | None = None,
        fetches: int = 0,
    ) -> Group:
        takeaways = takeaways or {}
        events = events or []
        run_tickers = run_tickers or []
        tokens = tokens or {}
        analysis_lines = self._todo_rows(status_lines)
        parts: list = [
            Text.assemble((f"{'model':<6}", ASH), (self.settings.provider, INK)),
            Text.assemble((f"{'name':<6}", ASH), (self.settings.model, INK)),
            Text.assemble((f"{'mode':<6}", ASH), (self.settings.portfolio_mode, INK)),
            Text.assemble((f"{'last':<6}", ASH), (self.last_run_id or "none", INK)),
            Text(""),
        ]

        done = sum(1 for _, status in analysis_lines if status.lower() == "done")
        total = len(analysis_lines)
        if thinking and not total:
            summary = Text.assemble(("▸ ", AMBER + " bold"), ("answering", AMBER))
        elif self._busy:
            summary = Text.assemble(("▸ ", AMBER + " bold"), ("running", AMBER))
            if total:
                summary.append(f"  ·  {done}/{total}", style=ASH)
            if fetches:
                summary.append(f"  ·  {fetches} fetches", style=TEAL)
        elif total:
            summary = Text.assemble(("● ", SAGE + " bold"), (f"complete · {done}/{total}", SAGE))
            if run_decisions:
                summary.append(f" · {self._clip_text(run_decisions, 18)}", style=AMBER)
        else:
            summary = Text.assemble(("○ ", ASH), ("idle", ASH))
        parts.extend([summary, Text("")])

        # PHASES checklist — four rows (Snapshot, Analysts, Risk, Portfolio) derived
        # from the same status_lines via the _phase_counts/_phase_row helpers.
        if status_lines or run_tickers:
            n_tickers = max(1, len(run_tickers))
            expected_analysts = n_tickers * max(1, len(self.settings.agents))
            sd, st, sa = self._phase_counts(status_lines, "snapshot", expected=n_tickers)
            ad, at, aa = self._phase_counts(status_lines, "analysis", expected=expected_analysts)
            rd, rt, ra = self._phase_counts(status_lines, "risk", expected=n_tickers)
            qd, qt, qa = self._phase_counts(status_lines, "portfolio", expected=n_tickers)
            parts.append(Text("phases", style=f"bold {TEAL}"))
            parts.append(self._phase_row("Snapshot", sd, st, sa, detail=(f"{fetches} fetches" if (sa and fetches) else "")))
            parts.append(self._phase_row("Analysts", ad, at, aa))
            parts.append(self._phase_row("Risk", rd, rt, ra))
            parts.append(self._phase_row("Portfolio", qd, qt, qa))
            parts.append(Text(""))

        # ACTIVE-AGENT card — the most-recent still-running analyst, mirroring the
        # workflows viewer: name · model · live tokens · current action/stance.
        active = next(
            ((lbl, stt) for lbl, stt in reversed(status_lines)
             if self._is_analysis_event(lbl) and stt.lower() != "done"),
            None,
        )
        if active and self._busy:
            lbl, stt = active
            card = Text()
            card.append("● ", style=f"bold {AMBER}")
            card.append(self._clip_text(self._activity_label(lbl), 28), style=f"bold {AMBER}")
            card.append(f"\n  {self.settings.model}", style=ASH)
            n = tokens.get(lbl, 0)
            if n:
                card.append(f"  ·  {self._fmt_tok(n)}", style=TEAL)
            action, acolor = self._activity_action(stt, takeaways.get(lbl))
            card.append(f"\n  {action}", style=acolor)
            parts.extend([card, Text("")])

        router_lines = [(label, status) for label, status in status_lines if label.lower().startswith("intent router")]
        if analysis_lines:
            parts.append(Text("todo", style=f"bold {TEAL}"))
            for label, status in analysis_lines[-14:]:
                marker, marker_color = self._activity_marker(status, takeaways.get(label))
                row = Text()
                checkbox = "[x]" if marker == "✓" else "[ ]" if marker == "•" else "[!]"
                row.append(f"{checkbox} ", style=f"bold {marker_color}")
                row.append(self._clip_text(self._activity_label(label), 34), style=INK if marker == "✓" else marker_color)
                parts.append(row)
        elif router_lines:
            parts.append(Text("router", style=f"bold {TEAL}"))
            for label, status in router_lines[-3:]:
                takeaway = takeaways.get(label)
                is_done = status.lower() == "done"
                marker, mcolor = ("✓", SAGE) if is_done else ("•", AMBER)
                row = Text()
                row.append(f"{marker} ", style=f"bold {mcolor}")
                row.append(self._chatter_name(label), style=INK if is_done else mcolor)
                row.append("  ", style=ASH)
                row.append(takeaway or _clean_action(status), style=f"bold {_status_color(takeaway or status)}")
                parts.append(row)

        if not status_lines and not run_tickers and not thinking:
            parts.append(Text("no analyst activity yet", style=ASH))

        return Group(*parts)

    def _dispatch_tui(self, text: str) -> None:
        if self._busy:
            self._emit(ChatEvent("system", "A run is already in progress.", "Wait for it to finish before sending another command."))
            return

        raw = text.strip()
        lower = raw.lower()
        if lower in {"exit", "quit"}:
            if self._app:
                self._app.exit(result=0)
            return

        self._emit(ChatEvent("user", raw))

        simple = _simple_response(raw)
        if simple is not None:
            self._emit_answer(simple)
            return

        if lower in {"help", "?", "status", "settings", "show last", "rerun last"} or lower.startswith(
            ("model ", "provider ", "mode ", "portfolio ", "agents ", "show ", "rerun ")
        ):
            self._handle_slash(raw)
            return

        # Slash input is always a command — never sent to the model.
        if raw.startswith("/"):
            self._handle_slash(raw[1:].strip())
            return

        # A few bare natural-language verbs still drive the engine directly.
        tickers = _extract_tickers(raw)
        if tickers and lower.startswith(("details", "detail", "why", "explain")):
            self._emit_ticker_details(tickers[0])
            return
        if tickers and _is_analysis_prompt(raw):
            self._run_in_thread("analyze", tickers)
            return
        if tickers:
            self._run_in_thread("route", [raw])
            return

        # Everything else is conversation — one streaming LLM call answers it.
        self._run_in_thread("ask", [raw])

    def _handle_slash(self, body: str) -> None:
        """Route a /command. Unknown commands get a hint, never the model."""
        if self._app and body.lower() in {"exit", "quit"}:
            self._app.exit(result=0)
            return

        parts = body.split(maxsplit=1)
        cmd = parts[0].lower() if parts else ""
        rest = parts[1].strip() if len(parts) > 1 else ""

        if cmd in {"", "help", "?", "commands"}:
            self._emit(ChatEvent("assistant", "Commands", self._help_text()))
        elif cmd in {"status", "settings"}:
            self._emit(ChatEvent("assistant", "Current settings", self._settings_text()))
        elif cmd == "model":
            if rest:
                self._set_model_from_command(f"model {rest}", emit_to_tui=True)
            else:
                choices = _model_choices_for(self.settings.provider)
                lines = [
                    f"Current: {self.settings.provider} / {self.settings.model}",
                    "Usage: /model <provider> <model>",
                ]
                if choices:
                    lines.append(f"Models for {self.settings.provider}: {', '.join(choices)}")
                self._emit(ChatEvent("assistant", "Model", "\n".join(lines)))
        elif cmd == "provider":
            if rest:
                self._set_provider_from_command(f"provider {rest}", emit_to_tui=True)
            else:
                self._emit(ChatEvent("assistant", "Provider", f"Current: {self.settings.provider}\nUsage: /provider <name>\nAvailable: {_provider_choices()}"))
        elif cmd in {"mode", "portfolio"}:
            if rest:
                self._set_portfolio_mode_from_command(f"mode {rest}", emit_to_tui=True)
            else:
                self._emit(
                    ChatEvent(
                        "assistant",
                        "Portfolio mode",
                        (
                            f"Current: {self.settings.portfolio_mode}\n"
                            "Usage: /mode research | /mode long_only | /mode long_short\n"
                            "research shows the recommendation directly; long_short requires a short hedge for new buys."
                        ),
                    )
                )
        elif cmd == "agents":
            if rest:
                self._set_agents_from_command(f"agents {rest}", emit_to_tui=True)
            else:
                label = "default analyst set" if self.settings.agents == DEFAULT_AGENTS else ",".join(self.settings.agents)
                self._emit(ChatEvent("assistant", "Agents", f"Current: {label}\nUsage: /agents technical,valuation (or blank to reset)"))
        elif cmd == "reasoning":
            if rest:
                self._set_reasoning_from_command(f"reasoning {rest}", emit_to_tui=True)
            else:
                state = "on" if self.settings.show_reasoning else "off"
                self._emit(ChatEvent("assistant", "Reasoning", f"Current: {state}\nUsage: /reasoning on | /reasoning off"))
        elif cmd == "show":
            run_id = self.last_run_id if rest in {"", "last"} else rest
            if run_id:
                self._show_run_tui(run_id)
            else:
                self._emit(ChatEvent("assistant", "No previous run in this session."))
        elif cmd == "rerun":
            run_id = self.last_run_id if rest in {"", "last"} else rest
            if run_id:
                self._run_in_thread("rerun", [run_id])
            else:
                self._emit(ChatEvent("assistant", "No previous run in this session."))
        elif cmd in {"analyze", "analyse", "run"}:
            tickers = _extract_tickers(rest)
            if tickers:
                self._run_in_thread("analyze", tickers)
            else:
                self._emit(ChatEvent("assistant", "Usage", "/analyze AAPL,NVDA"))
        elif cmd in {"details", "detail", "explain", "why"}:
            tickers = _extract_tickers(rest)
            if tickers:
                self._emit_ticker_details(tickers[0])
            else:
                self._emit(ChatEvent("assistant", "Usage", "/details AAPL"))
        else:
            self._emit(ChatEvent("system", f"Unknown command /{cmd}", "Type /help to see what Nova can do."))

    def _run_in_thread(self, mode: str, values: list[str]) -> None:
        self._busy = True
        self._status_lines.clear()
        self._status_takeaway.clear()
        self._status_tokens.clear()
        self._status_events.clear()
        progress.reset_telemetry()
        self._last_reasoning = ""
        # Drop the previous run's inspector data so a new run never briefly resolves stale
        # llm.jsonl / selection between dispatch and the worker building its context.
        self._active_run_id = None
        self._llm_cache.clear()
        self._inspector_selection = 0
        self._refresh_tui()
        self._start_heartbeat()
        thread = threading.Thread(target=self._run_worker, args=(mode, values), daemon=True)
        thread.start()

    def _start_heartbeat(self) -> None:
        """Repaint a few times a second while busy. Analyst LLM calls are non-streaming,
        so without this the inspector's live thinking / llm.jsonl tail would only update
        when the next progress event fires — stalling visibly during a long single call."""
        if self._heartbeat and self._heartbeat.is_alive():
            return

        def beat() -> None:
            while self._busy:
                time.sleep(0.4)
                self._refresh_tui()

        self._heartbeat = threading.Thread(target=beat, daemon=True)
        self._heartbeat.start()

    def _run_worker(self, mode: str, values: list[str]) -> None:
        try:
            if mode == "analyze":
                self._analyze_tui(values)
            elif mode == "rerun":
                self._rerun_tui(values[0])
            elif mode == "ask":
                self._ask_tui(values[0])
            elif mode == "route":
                self._route_tui(values[0])
        finally:
            self._busy = False
            self._refresh_tui()

    # --- free-form financial Q&A -----------------------------------------
    _CHAT_SYSTEM = (
        "You are Nova, the assistant inside Nova Trader — a portfolio-aware equity "
        "research and recommendation tool with optional long/short portfolio mode. "
        "Talk like a helpful, sharp colleague: greet back naturally, answer finance/markets/investing/risk/portfolio questions "
        "directly and conversationally, and keep it concise. If the user clearly wants a "
        "recommendation on specific tickers, tell them to run `analyze AAPL,NVDA` (the "
        "engine does the real multi-analyst run). For non-finance topics, briefly decline "
        "and steer back. You are not a licensed advisor — no personalized guarantees. "
        "You may use light markdown (bold, short bullet lists) but keep answers tight."
    )

    def _chat_messages(self, question: str) -> list[dict[str, str]]:
        messages = [{"role": "system", "content": self._CHAT_SYSTEM}]
        if self.last_recommendation is not None:
            messages.append({
                "role": "system",
                "content": "Context from the latest run:\n" + _recommendation_summary_text(self.last_recommendation),
            })
        messages.append({"role": "user", "content": question})
        return messages

    def _ask_tui(self, question: str) -> None:
        # The status box shows the live thinking; the answer streams in the conversation.
        self._stream_answer(self._chat_messages(question))

    def _route_tui(self, text: str) -> None:
        tickers = _extract_tickers(text)
        with self._ui_lock:
            self._status_lines["Intent Router"] = "classifying request"
        self._refresh_tui()
        route = self._route_intent(text, tickers)
        with self._ui_lock:
            self._status_lines["Intent Router"] = "done"
            self._status_takeaway["Intent Router"] = route.route
        self._refresh_tui()

        if route.route == "details":
            self._emit_ticker_details(tickers[0])
        elif route.route == "analyze" and tickers:
            self._analyze_tui(tickers)
        else:
            self._ask_tui(text)

    def _stream_verdict(self, recommendation: Recommendation) -> None:
        self._emit_answer(_recommendation_verdict_text(recommendation))

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
        if lower.startswith(("provider ", "/provider ")):
            self._set_provider_from_command(text)
            return None
        if lower.startswith(("mode ", "/mode ", "portfolio ", "/portfolio ")):
            self._set_portfolio_mode_from_command(text)
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
        if tickers and lower.startswith(("details", "detail", "why", "explain")):
            self._print_ticker_details(tickers[0])
            return None
        if tickers and _is_analysis_prompt(text):
            self._analyze(tickers)
            return None
        if tickers:
            route = self._route_intent(text, tickers)
            if route.route == "details":
                self._print_ticker_details(tickers[0])
            elif route.route == "analyze":
                self._analyze(tickers)
            else:
                self.console.print(self._intro_text())
            return None
        if lower.startswith(("explain", "what is", "what are", "how does")):
            self._print_product_explanation()
            return None
        if lower.startswith(("what's", "whats", "what is this", "what's this")):
            self._print_product_explanation()
            return None

        self.console.print(self._intro_text())
        return None

    def _help_text(self) -> str:
        return "\n".join(
            [
                "ask anything (finance)     just type a question — Nova answers, streamed",
                "analyze AAPL,NVDA          run a recommendation",
                "details AAPL               show analyst reasoning from the last run",
                "explain NVDA               explain a ticker decision from the last run",
                "show <run_id> | show last  print a saved / the latest recommendation",
                "rerun <run_id>             replay a saved run snapshot",
                "/model OpenAI gpt-4.1-mini switch provider + model",
                "/provider OpenAI           switch provider (keeps a default model)",
                "/mode research             choose research, long_only, or long_short",
                "/agents technical,valuation choose analyst agents",
                "/reasoning on|off          per-analyst LLM 'why' narration in the inspector",
                "/status                    show current model and agents",
                "/help · /exit              this help · close Nova",
            ]
        )

    def _intro_text(self) -> str:
        return "\n".join(
            [
                "This is Nova Trader, a portfolio-aware recommendation agent.",
                "Ask any finance question and Nova answers, or `analyze AAPL,NVDA` to run the analyst pipeline.",
                "After a run, ask `details AAPL` or `explain NVDA` to inspect the reasoning.",
                "Use `/model`, `/provider`, `/agents`, or `/help` to control the workspace.",
            ]
        )

    def _product_explanation_text(self) -> str:
        return "\n".join(
            [
                "Nova is a portfolio-aware recommendation agent.",
                "",
                "A run builds one market snapshot, slices typed views for analyst agents,",
                "aggregates their signals, applies risk limits, and then asks the portfolio",
                "manager for a decision. The default research mode shows the recommendation",
                "directly; long_short mode requires a short hedge before opening a new long.",
                "",
                "The transcript shows the conversation and final decision. The status pane",
                "shows live tool and analyst progress while a run is active.",
                "",
                "Use `analyze AAPL,NVDA` to run it, `details AAPL` to inspect reasoning,",
                "`model OpenAI gpt-4.1-mini` to switch models, and `show last` to inspect",
                "the latest recommendation.",
            ]
        )

    def _settings_text(self) -> str:
        agent_label = "default analyst set" if self.settings.agents == DEFAULT_AGENTS else ",".join(self.settings.agents)
        router_provider, router_model = self._router_model()
        return "\n".join(
            [
                f"provider: {self.settings.provider}",
                f"model:    {self.settings.model}",
                f"mode:     {self.settings.portfolio_mode}",
                f"router:   {router_provider} / {router_model}",
                f"agents:   {agent_label}",
                f"reasoning: {'on' if self.settings.show_reasoning else 'off'}",
                f"last run: {self.last_run_id or 'none'}",
            ]
        )

    def _router_model(self) -> tuple[str, str]:
        if self.settings.router_provider and self.settings.router_model:
            return self.settings.router_provider, self.settings.router_model
        return _default_router_model(self.settings.provider, self.settings.model)

    def _route_intent(self, text: str, tickers: list[str]) -> IntentRoute:
        provider, model = self._router_model()
        prompt = [
            {
                "role": "system",
                "content": (
                    "You are Nova Trader's fast intent router. Return only JSON with keys "
                    "route, confidence, and reason. route must be one of: analyze, chat, details. "
                    "Use analyze when the user wants market analysis, a recommendation, trade setup, "
                    "short/put/call/buy/sell/risk view, or asks what is going on with listed tickers. "
                    "Use details when they ask to inspect or explain a loaded run/ticker. "
                    "Use chat for product questions, greetings, general education, or anything that does not need the analyst pipeline."
                ),
            },
            {
                "role": "user",
                "content": f"User text: {text}\nExtracted tickers: {', '.join(tickers) or 'none'}",
            },
        ]
        try:
            from src.utils.llm import _call_json_model

            route, _ = _call_json_model(
                prompt=prompt,
                pydantic_model=IntentRoute,
                model_name=model,
                model_provider=provider,
                api_keys=None,
                seed=0,
            )
            return route
        except Exception:
            return _fallback_ticker_route(text, tickers)

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

    def _set_provider_from_command(self, text: str, *, emit_to_tui: bool = False) -> str:
        cleaned = text.strip()
        for prefix in ("/provider", "provider"):
            if cleaned.lower().startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
                break
        if not cleaned:
            message = f"Available providers: {_provider_choices()}"
            if emit_to_tui:
                self._emit(ChatEvent("assistant", "Pick a provider", message))
            else:
                self.console.print(f"[{ASH}]{message}[/{ASH}]")
            return self.settings.provider

        provider = _normalize_provider(cleaned)
        choices = _model_choices_for(provider)
        self.settings.provider = provider
        if choices:
            self.settings.model = choices[0]

        body = f"Provider set to {provider}"
        if choices:
            body += f" · model {self.settings.model}\nKnown models: {', '.join(choices)}"
        if emit_to_tui:
            self._emit(ChatEvent("assistant", "Provider updated", body))
        else:
            self.console.print(f"[{SAGE}]Provider set to {escape(provider)}[/{SAGE}]", highlight=False)
            if choices:
                self.console.print(f"[{ASH}]model {escape(self.settings.model)} · known: {escape(', '.join(choices))}[/{ASH}]", highlight=False)
        self._refresh_tui()
        return provider

    def _set_portfolio_mode_from_command(self, text: str, *, emit_to_tui: bool = False) -> str:
        cleaned = text.strip()
        for prefix in ("/portfolio", "portfolio", "/mode", "mode"):
            if cleaned.lower().startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
                break

        aliases = {
            "research": "research",
            "default": "research",
            "long": "long_only",
            "longonly": "long_only",
            "long_only": "long_only",
            "long-only": "long_only",
            "long short": "long_short",
            "longshort": "long_short",
            "long_short": "long_short",
            "long-short": "long_short",
            "hedged": "long_short",
        }
        mode = aliases.get(cleaned.lower().replace("/", " ").strip()) if cleaned else None
        if mode is None:
            message = (
                f"Current: {self.settings.portfolio_mode}\n"
                "Usage: /mode research | /mode long_only | /mode long_short"
            )
            if emit_to_tui:
                self._emit(ChatEvent("assistant", "Portfolio mode", message))
            else:
                self.console.print(f"[{ASH}]{escape(message)}[/{ASH}]", highlight=False)
            return self.settings.portfolio_mode

        self.settings.portfolio_mode = mode  # type: ignore[assignment]
        explanation = {
            "research": "research mode: show the recommendation directly; no forced hedge block.",
            "long_only": "long-only mode: size buys/sells without requiring a paired short.",
            "long_short": "long/short mode: opening buys require a bearish short hedge candidate.",
        }[mode]
        if emit_to_tui:
            self._emit(ChatEvent("assistant", "Portfolio mode updated", explanation))
        else:
            self.console.print(f"[{SAGE}]{escape(explanation)}[/{SAGE}]", highlight=False)
        self._refresh_tui()
        return self.settings.portfolio_mode

    def _set_reasoning_from_command(self, text: str, *, emit_to_tui: bool = False) -> bool:
        cleaned = text.strip()
        for prefix in ("/reasoning", "reasoning"):
            if cleaned.lower().startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
                break
        aliases = {
            "on": True, "true": True, "yes": True, "1": True,
            "off": False, "false": False, "no": False, "0": False,
        }
        enabled = aliases.get(cleaned.lower()) if cleaned else None
        if enabled is None:
            state = "on" if self.settings.show_reasoning else "off"
            msg = f"Current: {state}\nUsage: /reasoning on | /reasoning off"
            if emit_to_tui:
                self._emit(ChatEvent("assistant", "Reasoning", msg))
            else:
                self.console.print(f"[{ASH}]{escape(msg)}[/{ASH}]", highlight=False)
            return self.settings.show_reasoning
        self.settings.show_reasoning = enabled
        explanation = (
            "reasoning on: each analyst narrates its numbers in the inspector (adds an LLM call per analyst)."
            if enabled
            else "reasoning off: analysts return numbers only — fast pure-quant runs, no LLM narration."
        )
        if emit_to_tui:
            self._emit(ChatEvent("assistant", "Reasoning updated", explanation))
        else:
            self.console.print(f"[{SAGE}]{escape(explanation)}[/{SAGE}]", highlight=False)
        self._refresh_tui()
        return self.settings.show_reasoning

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
        with self._ui_lock:
            self._transcript.append(_event_renderable("assistant", f"Details for {ticker.upper()}"))
            self._transcript.append(_ticker_details_renderable(self.last_recommendation, ticker))
        self._refresh_tui()

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
            f"[{ASH}]mode[/{ASH}] {escape(self.settings.portfolio_mode)}\n"
            f"[{ASH}]agents[/{ASH}] {escape(agent_label)}\n\n"
            "[bold]Try[/bold]\n"
            "  analyze AAPL,NVDA\n"
            "  mode long_short\n"
            "  model OpenAI gpt-4.1-mini\n"
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

    def _progress_handler_tui(self, agent_name: str, ticker: str | None, status: str, *rest) -> None:
        analysis = rest[0] if rest else None
        original_ticker = ticker
        if ticker is None:
            with self._ui_lock:
                if len(self._run_tickers) == 1 and agent_name != "snapshot":
                    ticker = self._run_tickers[0]
        label = agent_name.replace("_", " ").title()
        scope = f" [{ticker}]" if ticker else ""
        full = f"{label}{scope}"
        suppress_emit = False
        with self._ui_lock:
            if status.lower() == "done" and original_ticker is None and ticker is None:
                prefix = f"{label} ["
                for existing in list(self._status_lines):
                    if existing.startswith(prefix):
                        self._status_lines[existing] = "Done"
                        suppress_emit = True
            if analysis:  # the engine reports each agent's stance on completion
                self._status_takeaway[full] = str(analysis)
            tok = progress.token_total(agent_name)
            if tok:
                self._status_tokens[full] = tok
        if suppress_emit:
            self._refresh_tui()
            return
        self._emit(ChatEvent("tool", full, status))

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
        self._prime_analysis_status(tickers)

        ctx = _build_context(tickers, self.settings)
        with self._ui_lock:
            self._active_run_id = ctx.run_id  # set BEFORE the engine writes llm.jsonl, so live tailing has a path
        progress.register_handler(self._progress_handler_tui)
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
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
        self._finish_analysis_status(recommendation)
        self._stream_verdict(recommendation)
        self._emit_result(recommendation)

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
        self._emit_result(recommendation)

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
        meta = RunRecorder.load_metadata(run_id)
        ctx, snapshot = _build_context_from_metadata(meta)
        self._prime_analysis_status(ctx.request.tickers)
        with self._ui_lock:
            self._active_run_id = ctx.run_id
        progress.register_handler(self._progress_handler_tui)
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
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
        self._finish_analysis_status(recommendation)
        self._stream_verdict(recommendation)
        self._emit_result(recommendation)


def launch_chat(console: Console, provider: str, model: str, portfolio_mode: str = "research") -> int:
    allowed_modes = {"research", "long_only", "long_short"}
    mode = portfolio_mode if portfolio_mode in allowed_modes else "research"
    settings = ChatSettings(provider=provider, model=model, portfolio_mode=mode)  # type: ignore[arg-type]
    return NovaChat(console, settings).run()
