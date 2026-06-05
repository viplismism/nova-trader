"""Regression guards for the chat streaming + rendering layer.

These cover behaviors that had no tests: the reasoning/answer stream split, the
typewriter animator, ANSI line-padding (status-pane corruption fix), number-free
agent verbs, and the recommendation log view at a narrow pane width (the gap fix).

Kept in a separate file so it doesn't collide with the actively-edited
test_chat_cli.py.
"""

import re
from unittest.mock import patch

from rich.console import Console

import src.chat_cli as cc
from src.chat_cli import ChatSettings, NovaChat, _clean_action
from src.schemas.signals import Consensus, Decisions, Recommendation, Signal, TickerDecision
from src.utils.llm import _ThinkTagSplitter

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text: str) -> str:
    return _ANSI.sub("", text)


def _merge(chunks: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Collapse consecutive same-channel pieces for readable assertions."""
    merged: list[list] = []
    for channel, piece in chunks:
        if merged and merged[-1][0] == channel:
            merged[-1][1] += piece
        else:
            merged.append([channel, piece])
    return [(c, t) for c, t in merged]


# --- think-tag splitter (reasoning vs answer) -------------------------------

def test_think_splitter_separates_reasoning_from_answer():
    s = _ThinkTagSplitter()
    out = s.feed("<think>let me reason</think>The answer is 42.") + s.flush()
    assert _merge(out) == [("reasoning", "let me reason"), ("answer", "The answer is 42.")]


def test_think_splitter_handles_tag_split_across_chunks():
    s = _ThinkTagSplitter()
    out = s.feed("Hello <th") + s.feed("ink>secret</thi") + s.feed("nk> world") + s.flush()
    assert _merge(out) == [("answer", "Hello "), ("reasoning", "secret"), ("answer", " world")]


def test_think_splitter_plain_text_is_all_answer():
    s = _ThinkTagSplitter()
    out = s.feed("just plain text") + s.flush()
    assert _merge(out) == [("answer", "just plain text")]


# --- number-free agent verbs for the chatter feed --------------------------

def test_clean_action_strips_numbers_into_plain_verbs():
    assert _clean_action("Adj. limit 19.4%, available $19393") == "sizing positions"
    assert _clean_action("Calling LLM") == "reasoning with the model"
    assert _clean_action("Fetching prices") == "gathering data"
    assert _clean_action("Classifying 10 headlines") == "reading the news"
    # whatever verb is chosen, it must never leak digits
    for raw in ("Adj. limit 19.4%, available $19393", "Classifying 10 headlines", "Analyzing 38 insider trades"):
        assert not any(ch.isdigit() for ch in _clean_action(raw))


# --- ANSI line padding (status-pane corruption fix) ------------------------

def test_render_ansi_pads_every_line_to_pane_width():
    chat = NovaChat(Console(), ChatSettings(provider="MiniMax", model="MiniMax-M2.7"))
    from rich.text import Text

    width = 24
    out = chat._render_ansi(Text("short"), width)
    for line in out.split("\n"):
        assert len(_plain(line)) == width, f"line not padded to {width}: {len(_plain(line))!r}"


# --- typewriter animator (smooth reveal, capped step) ----------------------

def test_animator_reveals_burst_progressively_and_capped():
    chat = NovaChat(Console(), ChatSettings(provider="x", model="y"))
    seen: list[int] = []
    chat._refresh_tui = lambda: seen.append(len(chat._disp_answer))

    def one_big_burst(messages, *, provider, model, temperature=0.3):
        yield ("answer", "X" * 300)  # everything at once — worst case

    with patch("src.utils.llm.stream_chat", one_big_burst, create=True):
        out = chat._stream_answer([{"role": "user", "content": "hi"}])

    assert out == "X" * 300
    growth = [b - a for a, b in zip(seen, seen[1:]) if b > a]
    assert len(growth) >= 5, f"answer did not reveal gradually: {seen}"
    assert max(growth) <= 12, f"reveal step not capped (got {max(growth)})"


def test_stream_split_routes_reasoning_and_answer_to_boxes():
    chat = NovaChat(Console(), ChatSettings(provider="x", model="y"))

    def stream(messages, *, provider, model, temperature=0.3):
        yield ("reasoning", "weighing options")
        yield ("answer", "Go long.")

    with patch("src.utils.llm.stream_chat", stream, create=True):
        out = chat._stream_answer([{"role": "user", "content": "hi"}])

    assert out == "Go long."
    # the answer lands in the transcript; the reasoning is kept for the status box
    assert chat._last_reasoning.strip() == "weighing options"


# --- recommendation log view at a narrow pane (the gap regression) ----------

def _recommendation() -> Recommendation:
    return Recommendation(
        run_id="r1",
        as_of="2026-06-01T00:00:00Z",
        tickers=["AAPL"],
        signals=[
            Signal(agent_id="insider_sentiment", ticker="AAPL", direction="bullish", confidence=0.74,
                   reasoning="62 insider buys vs 38 sells (net signal +24%)"),
            Signal(agent_id="warren_buffett", ticker="AAPL", direction="neutral", confidence=0.62,
                   reasoning="Exceptional business quality but high D/E 2.48 and tight liquidity raise leverage concerns."),
        ],
        consensus={"AAPL": Consensus(ticker="AAPL", direction="neutral", confidence=0.58, weighted_score=0.0)},
        decisions=Decisions(per_ticker={"AAPL": TickerDecision(ticker="AAPL", action="hold", confidence=0.58, reasoning="Neutral consensus")}),
        summary="AAPL hold",
    )


def test_recommendation_renders_at_narrow_width_without_gap():
    # Design-agnostic invariants: at a cramped pane width the run view still shows the
    # key facts, surfaces the analysts, and has no tall blank gap (the narrow-width bug).
    chat = NovaChat(Console(), ChatSettings(provider="MiniMax", model="MiniMax-M2.7"))
    out = _plain(chat._render_ansi(cc._recommendation_renderable(_recommendation()), 34))
    body = " ".join(out.replace("│", " ").split())

    assert "AAPL" in body and "HOLD" in body            # the decision is visible
    assert "insider sentiment" in body                  # analysts are surfaced
    blanks = max((len(g) for g in _runs_of_blanks(out)), default=0)
    assert blanks <= 3, f"unexpected vertical gap of {blanks} blank rows"


def _runs_of_blanks(rendered: str):
    run = []
    for line in rendered.split("\n"):
        if line.strip("│ ") == "":
            run.append(line)
        else:
            if run:
                yield run
            run = []
    if run:
        yield run
