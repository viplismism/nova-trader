from rich.console import Console

from src.chat_cli import (
    ChatSettings,
    IntentRoute,
    NovaChat,
    _answer_renderable,
    _build_context,
    _default_router_model,
    _extract_tickers,
    _fallback_ticker_route,
    _is_analysis_prompt,
    _recommendation_summary_text,
    _recommendation_verdict_text,
    _render_recommendation,
    _simple_response,
    _stream_markdown_display_text,
    _ticker_details_text,
)
from src.schemas.signals import Consensus, Decisions, HedgePlan, Recommendation, Signal, TickerDecision


def test_extract_tickers_from_natural_prompt():
    assert _extract_tickers("analyze AAPL and NVDA for me") == ["AAPL", "NVDA"]


def test_extract_tickers_deduplicates_symbols():
    assert _extract_tickers("run TSLA, tsla, MSFT") == ["TSLA", "MSFT"]


def test_extract_tickers_maps_company_names_and_ignores_pronouns():
    text = "what should you do if we analyze NVIDIA and apple?"

    assert _extract_tickers(text) == ["NVDA", "AAPL"]


def test_explain_setup_is_not_treated_as_ticker_request():
    assert _extract_tickers("explain the setup") == []
    assert not _is_analysis_prompt("explain the setup")


def test_bare_ticker_list_is_analysis_prompt():
    assert _is_analysis_prompt("AAPL,NVDA")


def test_natural_ticker_trade_question_routes_to_analysis():
    text = "i want to know what's going on with the aapl and how should i back this out? short or put?"

    assert _extract_tickers(text) == ["AAPL"]
    assert _fallback_ticker_route(text, ["AAPL"]).route == "analyze"


def test_router_model_prefers_small_fast_model_for_known_provider():
    assert _default_router_model("OpenAI", "gpt-5.2") == ("OpenAI", "gpt-4o-mini")
    assert _default_router_model("MiniMax", "MiniMax-M2.7") == ("MiniMax", "MiniMax-M2.7-highspeed")


def test_route_intent_uses_classifier_model(monkeypatch):
    def fake_call(**kwargs):
        assert kwargs["model_name"] == "gpt-4o-mini"
        assert kwargs["model_provider"] == "OpenAI"
        return IntentRoute(route="analyze", confidence=0.99, reason="ticker question"), {}

    monkeypatch.setattr("src.utils.llm._call_json_model", fake_call)
    chat = NovaChat(Console(record=True), ChatSettings(provider="OpenAI", model="gpt-5.2"))

    route = chat._route_intent("what is going on with AAPL?", ["AAPL"])

    assert route.route == "analyze"


def test_chat_model_command_updates_provider_and_model():
    console = Console(record=True)
    chat = NovaChat(console, ChatSettings(provider="OpenAI", model="gpt-4o-mini"))

    chat.handle("model MiniMax MiniMax-M2.7")

    assert chat.settings.provider == "MiniMax"
    assert chat.settings.model == "MiniMax-M2.7"


def test_chat_mode_command_updates_portfolio_mode():
    console = Console(record=True)
    chat = NovaChat(console, ChatSettings(provider="OpenAI", model="gpt-4o-mini"))

    chat.handle("mode long_short")

    assert chat.settings.portfolio_mode == "long_short"
    assert "opening buys require a bearish short hedge" in console.export_text()


def test_build_context_passes_portfolio_mode_to_engine():
    settings = ChatSettings(provider="OpenAI", model="gpt-4o-mini", portfolio_mode="long_only")

    ctx = _build_context(["NVDA"], settings)

    assert ctx.request.portfolio_mode == "long_only"


def test_chat_agents_command_updates_agent_list():
    console = Console(record=True)
    chat = NovaChat(console, ChatSettings(provider="OpenAI", model="gpt-4o-mini"))

    chat.handle("agents technical,valuation")

    assert chat.settings.agents == ["technical", "valuation"]


def test_show_last_without_run_is_friendly_message():
    console = Console(record=True)
    chat = NovaChat(console, ChatSettings(provider="OpenAI", model="gpt-4o-mini"))

    chat.handle("show last")

    assert "No previous run" in console.export_text()


def test_render_recommendation_uses_ticker_keyed_consensus():
    console = Console(record=True)
    recommendation = Recommendation(
        run_id="test-run",
        as_of="2026-05-31T00:00:00Z",
        tickers=["AAPL"],
        signals=[
            Signal(
                agent_id="technical",
                ticker="AAPL",
                direction="neutral",
                confidence=0.5,
            )
        ],
        consensus={
            "AAPL": Consensus(
                ticker="AAPL",
                direction="neutral",
                confidence=0.5,
                weighted_score=0.0,
            )
        },
        decisions=Decisions(
            per_ticker={
                "AAPL": TickerDecision(
                    ticker="AAPL",
                    action="hold",
                    confidence=0.5,
                    reasoning="Neutral consensus",
                )
            }
        ),
        summary="AAPL: consensus=neutral, action=hold",
    )

    _render_recommendation(console, recommendation)

    rendered = console.export_text()
    assert "AAPL" in rendered
    assert "HOLD" in rendered
    assert "SIGNAL SNAPSHOT" in rendered
    assert "saved to" not in rendered


def test_render_recommendation_explains_blocked_hedge():
    console = Console(record=True)
    recommendation = Recommendation(
        run_id="test-run",
        as_of="2026-05-31T00:00:00Z",
        tickers=["NVDA"],
        signals=[
            Signal(
                agent_id="technical",
                ticker="NVDA",
                direction="bullish",
                confidence=0.75,
            )
        ],
        consensus={
            "NVDA": Consensus(
                ticker="NVDA",
                direction="bullish",
                confidence=0.75,
                weighted_score=0.5,
            )
        },
        decisions=Decisions(
            per_ticker={
                "NVDA": TickerDecision(
                    ticker="NVDA",
                    action="hold",
                    confidence=0.75,
                    reasoning="Blocked: no short hedge candidate available",
                )
            },
            hedge_plan=HedgePlan(status="blocked", blocked_longs=["NVDA"]),
        ),
        summary="NVDA: consensus=bullish, action=hold",
    )

    _render_recommendation(console, recommendation)

    rendered = console.export_text()
    assert "blocked (no short hedge): NVDA" in rendered


def test_recommendation_summary_text_is_plain_transcript_output():
    recommendation = Recommendation(
        run_id="test-run",
        as_of="2026-05-31T00:00:00Z",
        tickers=["AAPL"],
        signals=[
            Signal(
                agent_id="technical",
                ticker="AAPL",
                direction="neutral",
                confidence=0.5,
            )
        ],
        consensus={
            "AAPL": Consensus(
                ticker="AAPL",
                direction="neutral",
                confidence=0.5,
                weighted_score=0.0,
            )
        },
        decisions=Decisions(
            per_ticker={
                "AAPL": TickerDecision(
                    ticker="AAPL",
                    action="hold",
                    confidence=0.5,
                    reasoning="Neutral consensus",
                )
            }
        ),
        summary="AAPL: consensus=neutral, action=hold",
    )

    text = _recommendation_summary_text(recommendation)

    assert "Run test-run" in text
    assert "AAPL" in text
    assert "HOLD" in text
    assert "Analyst reasoning" in text
    assert "technical" in text


def test_ticker_details_text_shows_agent_reasoning():
    recommendation = Recommendation(
        run_id="test-run",
        as_of="2026-05-31T00:00:00Z",
        tickers=["AAPL"],
        signals=[
            Signal(
                agent_id="technical",
                ticker="AAPL",
                direction="bullish",
                confidence=0.75,
                reasoning="Momentum and trend are constructive.",
            )
        ],
        consensus={
            "AAPL": Consensus(
                ticker="AAPL",
                direction="bullish",
                confidence=0.75,
                weighted_score=0.5,
            )
        },
        decisions=Decisions(
            per_ticker={
                "AAPL": TickerDecision(
                    ticker="AAPL",
                    action="hold",
                    confidence=0.75,
                    reasoning="No hedge available.",
                )
            }
        ),
        summary="AAPL: consensus=bullish, action=hold",
    )

    text = _ticker_details_text(recommendation, "AAPL")

    assert "AAPL details" in text
    assert "technical" in text
    assert "Momentum and trend" in text


def test_recommendation_verdict_uses_structured_run_not_model_prompt():
    recommendation = Recommendation(
        run_id="test-run",
        as_of="2026-05-31T00:00:00Z",
        tickers=["AAPL"],
        signals=[
            Signal(
                agent_id="technical",
                ticker="AAPL",
                direction="bullish",
                confidence=0.75,
                reasoning="Momentum and trend are constructive.",
            ),
            Signal(
                agent_id="growth",
                ticker="AAPL",
                direction="bearish",
                confidence=0.86,
                reasoning="Growth score weakened.",
            ),
        ],
        consensus={
            "AAPL": Consensus(
                ticker="AAPL",
                direction="neutral",
                confidence=0.6,
                weighted_score=0.0,
            )
        },
        decisions=Decisions(
            per_ticker={
                "AAPL": TickerDecision(
                    ticker="AAPL",
                    action="hold",
                    confidence=0.6,
                    reasoning="Neutral consensus with conflicting signals.",
                )
            }
        ),
        summary="AAPL: consensus=neutral, action=hold",
    )

    text = _recommendation_verdict_text(recommendation)

    assert "**AAPL: HOLD.**" in text
    assert "\n\nDecision note:" in text
    assert "\n\nSignal read:" in text
    assert "\n\nWhat the agents thought:" in text
    assert "Momentum and trend are constructive" in text
    assert "Technical" in text
    assert "Growth" in text
    assert "short-vs-put" in text


def test_unknown_chat_prompt_gets_intro_instead_of_ticker_error():
    console = Console(record=True)
    chat = NovaChat(console, ChatSettings(provider="OpenAI", model="gpt-4o-mini"))

    chat.handle("hi")
    chat.handle("what's this is?")

    rendered = console.export_text()
    assert "portfolio-aware recommendation agent" in rendered


def test_stream_markdown_hides_incomplete_markers():
    assert _stream_markdown_display_text("**Nova Trader") == "Nova Trader"
    assert _stream_markdown_display_text("Use `analyze AAPL") == "Use analyze AAPL"
    assert _stream_markdown_display_text("**Nova** uses `signals`") == "**Nova** uses `signals`"


def test_answer_renderable_is_not_boxed_panel():
    renderable = _answer_renderable("**Nova** uses `signals`.")

    assert renderable.__class__.__name__ != "Panel"


def test_simple_questions_do_not_need_model_streaming():
    assert _simple_response("hi") is not None
    assert _simple_response("what's this is?") is not None
    assert _simple_response("Should I buy AAPL?") is None


def test_status_pane_collapses_completed_manager_child_rows():
    chat = NovaChat(Console(record=True, width=44), ChatSettings(provider="MiniMax", model="MiniMax-M2.7"))
    chat._prime_analysis_status(["NVDA"])

    chat._progress_handler_tui("risk_manager", "NVDA", "Adj. limit 20%, available $20000")
    chat._progress_handler_tui("risk_manager", None, "Done")

    status_lines = list(chat._status_lines.items())
    renderable = chat._status_renderable(status_lines, {}, chat._status_events, run_tickers=["NVDA"])
    console = Console(record=True, width=44)
    console.print(renderable)
    text = console.export_text()

    assert "todo" in text
    assert "[x]" in text
    assert "Risk Manager [NVDA]" in text
    assert "done" not in text.lower()
    assert "workflow" not in text
    assert text.count("Risk Manager") == 1


def test_single_ticker_progress_scopes_unscoped_updates():
    chat = NovaChat(Console(record=True, width=44), ChatSettings(provider="MiniMax", model="MiniMax-M2.7"))
    chat._prime_analysis_status(["NVDA"])

    chat._progress_handler_tui("growth", None, "Error - retry 1/3")

    assert "Growth [NVDA]" in chat._status_lines
    assert "Growth" not in chat._status_lines


def test_status_activity_uses_short_agent_labels():
    chat = NovaChat(Console(record=True, width=44), ChatSettings(provider="MiniMax", model="MiniMax-M2.7"))
    chat._prime_analysis_status(["NVDA"])

    chat._progress_handler_tui("warren_buffett", "NVDA", "Calling LLM")

    renderable = chat._status_renderable(list(chat._status_lines.items()), {}, chat._status_events, run_tickers=["NVDA"])
    console = Console(record=True, width=44)
    console.print(renderable)
    text = console.export_text()

    assert "Buffett Agent [NVDA]" in text
    assert "Calling LLM" not in text
    assert "Warren Buffett" not in text


def test_status_panel_hides_model_reasoning():
    chat = NovaChat(Console(record=True, width=44), ChatSettings(provider="MiniMax", model="MiniMax-M2.7"))

    renderable = chat._status_renderable([], {}, [], "private chain of thought", thinking=True)
    console = Console(record=True, width=44)
    console.print(renderable)
    text = console.export_text()

    assert "private chain" not in text
    assert "reasoning" not in text.lower()


def test_transcript_slice_scrolls_from_bottom():
    text = "\n".join(f"line {idx}" for idx in range(1, 11))

    bottom, scroll = NovaChat._slice_lines_from_bottom(text, height=4, scroll=0)
    older, older_scroll = NovaChat._slice_lines_from_bottom(text, height=4, scroll=3)

    assert bottom.splitlines() == ["line 7", "line 8", "line 9", "line 10"]
    assert scroll == 0
    assert older.splitlines() == ["line 4", "line 5", "line 6", "line 7"]
    assert older_scroll == 3


def test_scroll_transcript_updates_offset_without_app():
    chat = NovaChat(Console(record=True), ChatSettings(provider="MiniMax", model="MiniMax-M2.7"))

    chat._scroll_transcript(20)
    chat._scroll_transcript(-5)
    chat._scroll_transcript(-100)

    assert chat._transcript_scroll == 0


def test_run_thinking_renderable_is_ephemeral_status_not_chat_message():
    chat = NovaChat(Console(record=True), ChatSettings(provider="MiniMax", model="MiniMax-M2.7"))
    chat._prime_analysis_status(["NVDA"])
    chat._progress_handler_tui("technical", "NVDA", "Computing trend/momentum/vol")

    assert chat._transcript == []
    console = Console(record=True)
    console.print(chat._run_thinking_renderable())
    text = console.export_text()

    assert "Thinking through the run" in text
    assert "Watching Technical Agent [NVDA]" in text


# --- telemetry accumulator (progress singleton) -----------------------------

def test_progress_accumulates_tokens_and_fetches():
    from src.utils.progress import progress

    progress.reset_telemetry()
    progress.add_tokens("technical", 1200, 300)
    progress.add_tokens("technical", 100, 0)
    assert progress.token_total("technical") == 1600
    progress.record_fetch("snapshot", "prices")
    progress.record_fetch("snapshot", "news")
    assert progress.total_fetches() == 2
    progress.reset_telemetry()
    assert progress.token_total("technical") == 0
    assert progress.total_fetches() == 0


def test_progress_token_accumulation_is_thread_safe():
    import threading

    from src.utils.progress import progress

    progress.reset_telemetry()
    threads = [
        threading.Thread(target=lambda: [progress.add_tokens("x", 10, 0) for _ in range(100)])
        for _ in range(20)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert progress.token_total("x") == 20_000
    progress.reset_telemetry()


# --- compact pane: phases + active-agent card -------------------------------

def test_status_pane_shows_phases_checklist():
    chat = NovaChat(Console(record=True, width=44), ChatSettings(provider="MiniMax", model="MiniMax-M2.7"))
    chat._busy = True
    chat._prime_analysis_status(["NVDA", "AAPL"])
    chat._progress_handler_tui("snapshot", "NVDA", "Done")
    chat._progress_handler_tui("snapshot", "AAPL", "Done")
    chat._progress_handler_tui("technical", "NVDA", "Computing trend")

    renderable = chat._status_renderable(
        list(chat._status_lines.items()), {}, chat._status_events,
        run_tickers=["NVDA", "AAPL"], fetches=12,
    )
    console = Console(record=True, width=44)
    console.print(renderable)
    text = console.export_text()

    assert "phases" in text
    assert "Snapshot" in text
    assert "Analysts" in text
    assert "Portfolio" in text
    assert "12 fetches" in text


def test_finish_analysis_marks_lingering_rows_done():
    from src.schemas.signals import TickerDecision

    chat = NovaChat(Console(record=True, width=44), ChatSettings(provider="MiniMax", model="MiniMax-M2.7"))
    chat._prime_analysis_status(["NVDA"])
    chat._progress_handler_tui("portfolio_manager", "NVDA", "Deciding")
    chat._progress_handler_tui("risk_manager", "NVDA", "Sizing positions")
    chat._progress_handler_tui("risk_manager", None, "Done")
    rec = Recommendation(
        run_id="r1",
        as_of="2026-05-31T00:00:00Z",
        tickers=["NVDA"],
        signals=[],
        consensus={},
        decisions=Decisions(per_ticker={"NVDA": TickerDecision(ticker="NVDA", action="hold", confidence=0.5)}),
        summary="",
    )

    chat._finish_analysis_status(rec)

    assert chat._status_lines["Portfolio Manager [NVDA]"] == "Done"
    assert chat._status_lines["Risk Manager [NVDA]"] == "Done"
    assert "Risk Manager" not in chat._status_lines


def test_active_agent_card_shows_model_and_tokens():
    chat = NovaChat(Console(record=True, width=44), ChatSettings(provider="MiniMax", model="MiniMax-M2.7"))
    chat._busy = True
    chat._prime_analysis_status(["NVDA"])
    chat._progress_handler_tui("valuation", "NVDA", "Calling LLM")

    renderable = chat._status_renderable(
        list(chat._status_lines.items()), {}, chat._status_events,
        run_tickers=["NVDA"], tokens={"Valuation [NVDA]": 12100},
    )
    console = Console(record=True, width=44)
    console.print(renderable)
    text = console.export_text()

    assert "Valuation Agent [NVDA]" in text
    assert "MiniMax-M2.7" in text
    assert "12.1k tok" in text


# --- inspector --------------------------------------------------------------

def test_inspector_handles_no_run():
    chat = NovaChat(Console(record=True, width=80), ChatSettings(provider="MiniMax", model="MiniMax-M2.7"))
    renderable = chat._inspector_renderable(None, [], {}, {}, 0, [], [], "")
    console = Console(record=True, width=80)
    console.print(renderable)
    assert "No run yet" in console.export_text()


def test_inspector_shows_agent_activity_log():
    chat = NovaChat(Console(record=True, width=92), ChatSettings(provider="MiniMax", model="MiniMax-M2.7"))
    chat._busy = True
    chat._prime_analysis_status(["NVDA"])
    chat._progress_handler_tui("technical", "NVDA", "Computing trend", "bullish")
    chat._progress_handler_tui("technical", "NVDA", "Done", "bullish")

    renderable = chat._inspector_renderable(
        None, list(chat._status_events), dict(chat._status_takeaway), {}, 12,
        ["NVDA"], list(chat._status_lines.items()), "",
    )
    console = Console(record=True, width=92)
    console.print(renderable)
    text = console.export_text()

    assert "Running" in text
    assert "12 fetches" in text
    assert "agent activity" in text
    assert "Technical Agent [NVDA]" in text


def test_inspector_groups_signals_by_agent():
    chat = NovaChat(Console(record=True, width=92), ChatSettings(provider="MiniMax", model="MiniMax-M2.7"))
    rec = Recommendation(
        run_id="r1",
        as_of="2026-05-31T00:00:00Z",
        tickers=["AAPL", "NVDA"],
        signals=[
            Signal(agent_id="technical", ticker="AAPL", direction="bullish", confidence=0.7, reasoning="Trend up"),
            Signal(agent_id="technical", ticker="NVDA", direction="bearish", confidence=0.6, reasoning="Overbought"),
            Signal(agent_id="valuation", ticker="AAPL", direction="neutral", confidence=0.5, reasoning="Fair value"),
        ],
        consensus={},
        decisions=Decisions(per_ticker={}),
        summary="",
    )
    renderable = chat._inspector_renderable(rec, [], {}, {}, 4, ["AAPL", "NVDA"], [], "")
    console = Console(record=True, width=92)
    console.print(renderable)
    text = console.export_text()

    assert "signals by agent" in text
    assert "Technical Agent" in text
    assert "Valuation Agent" in text
    assert "bullish" in text
    assert "bearish" in text


# --- per-agent inspector: reasoning store, recorder reader, detail view ------

def test_capture_reasoning_prefers_reasoning_content_and_resets():
    from src.utils.progress import progress

    progress.reset_telemetry()
    # response_content fallback when no dedicated reasoning stream
    progress.capture_reasoning("warren_buffett", "AAPL", None, '{"signal":"bullish"}')
    snap = progress.reasoning_snapshot()
    assert snap[("warren_buffett", "AAPL")]["source"] == "response_content"
    # reasoning_content wins over response_content
    progress.capture_reasoning("warren_buffett", "AAPL", "moat + margin of safety", '{"signal":"bullish"}')
    snap = progress.reasoning_snapshot()
    assert snap[("warren_buffett", "AAPL")]["reasoning"] == "moat + margin of safety"
    assert snap[("warren_buffett", "AAPL")]["source"] == "reasoning_content"
    progress.reset_telemetry()
    assert progress.reasoning_snapshot() == {}


def test_capture_reasoning_extracts_json_reasoning_field():
    from src.utils.progress import progress

    progress.reset_telemetry()
    progress.capture_reasoning("warren_buffett", "AAPL", None, '{"signal":"bullish","reasoning":"Moat is durable."}')
    snap = progress.reasoning_snapshot()
    progress.reset_telemetry()

    assert snap[("warren_buffett", "AAPL")]["reasoning"] == "Moat is durable."
    assert snap[("warren_buffett", "AAPL")]["source"] == "response_content.reasoning"


def test_capture_reasoning_extracts_explanation_after_think_block():
    from src.utils.progress import progress

    progress.reset_telemetry()
    progress.capture_reasoning(
        "news_sentiment",
        "AAPL",
        None,
        '<think>private scratchpad</think> {"explanation": "Headlines are mostly neutral, so the signal stays neutral."}',
    )
    snap = progress.reasoning_snapshot()
    progress.reset_telemetry()

    assert snap[("news_sentiment", "AAPL")]["reasoning"] == "Headlines are mostly neutral, so the signal stays neutral."
    assert snap[("news_sentiment", "AAPL")]["source"] == "response_content.explanation"


def test_load_llm_calls_groups_and_tolerates_torn_line(tmp_path):
    from src.runs import RunRecorder

    run_dir = tmp_path / "run-1"
    run_dir.mkdir()
    (run_dir / "llm.jsonl").write_text(
        '{"agent_id": "warren_buffett", "ticker": "AAPL", "response": "{}"}\n'
        '{"agent_id": "warren_buffett", "ticker": "NVDA", "response": "{}"}\n'
        '{"agent_id": "technical", "ticker": "AAPL"}\n'
        '{"agent_id": "warren_buffett", "ticker": "MSF'  # torn final line, no newline
    )
    grouped = RunRecorder.load_llm_calls("run-1", base_dir=tmp_path)
    assert len(grouped["warren_buffett"]) == 2  # torn line skipped, not raised
    assert len(grouped["technical"]) == 1
    assert RunRecorder.load_llm_calls("missing", base_dir=tmp_path) == {}


def test_inspector_list_fragments_are_clickable_and_mark_selection():
    chat = NovaChat(Console(record=True, width=44), ChatSettings(provider="MiniMax", model="MiniMax-M2.7"))
    assert chat._inspector_list_fragments()[0][1].strip() == "no agents yet"

    chat._prime_analysis_status(["NVDA"])
    chat._progress_handler_tui("technical", "NVDA", "Done", "bearish")
    chat._progress_handler_tui("valuation", "NVDA", "Done", "bullish")
    frags = chat._inspector_list_fragments()

    assert all(len(f) == 3 and callable(f[2]) for f in frags)  # (style, text, mouse_handler)
    assert chat._inspector_agent_keys == ["Technical [NVDA]", "Valuation [NVDA]"]  # paint order
    assert "›" in frags[0][1] and "bg:" in frags[0][0]  # selected row marked
    assert "›" not in frags[1][1]


def test_selected_agent_renderable_shows_factor_breakdown_for_quant_agent():
    chat = NovaChat(Console(record=True, width=64), ChatSettings(provider="MiniMax", model="MiniMax-M2.7"))
    chat._prime_analysis_status(["NVDA"])
    chat._progress_handler_tui("technical", "NVDA", "Done", "bearish")
    rec = Recommendation(
        run_id="r1", as_of="2026-05-31T00:00:00Z", tickers=["NVDA"],
        signals=[Signal(
            agent_id="technical", ticker="NVDA", direction="bearish", confidence=0.61,
            reasoning="Ensemble of 5 strategies: bearish @ 61%",
            key_factors=["trend=bearish (55%)", "momentum=bearish (60%)", "volatility=0.27"],
        )],
        consensus={}, decisions=Decisions(per_ticker={}), summary="",
    )
    console = Console(record=True, width=64)
    console.print(chat._selected_agent_renderable("Technical [NVDA]", rec, list(chat._status_events), {}, {}, {}, {}))
    text = console.export_text()

    assert "signal & why" in text
    assert "bearish" in text
    assert "reason" in text
    assert "Ensemble of 5 strategies" in text
    assert "trend" in text and "bearish (55%)" in text   # the WHY: deterministic sub-scores surfaced
    assert "momentum" in text and "bearish (60%)" in text


def test_selected_agent_renderable_shows_llm_thinking():
    chat = NovaChat(Console(record=True, width=64), ChatSettings(provider="MiniMax", model="MiniMax-M2.7"))
    chat._prime_analysis_status(["NVDA"])
    chat._progress_handler_tui("warren_buffett", "NVDA", "Done", "neutral")
    agent_reasoning = {("warren_buffett", "NVDA"): {"reasoning": "wide moat but rich multiple", "source": "reasoning_content"}}
    console = Console(record=True, width=64)
    console.print(chat._selected_agent_renderable("Warren Buffett [NVDA]", None, list(chat._status_events), {}, {}, agent_reasoning, {}))
    text = console.export_text()

    assert "reasoning" in text
    assert "wide moat but rich multiple" in text


def test_selected_agent_renderable_hides_llm_audit_metadata():
    chat = NovaChat(Console(record=True, width=76), ChatSettings(provider="MiniMax", model="MiniMax-M2.7"))
    chat._prime_analysis_status(["NVDA"])
    chat._progress_handler_tui("growth", "NVDA", "Error - retry 1/3")
    llm_by_agent = {
        "growth": [{
            "ticker": "NVDA",
            "model": "MiniMax-M2.7",
            "prompt": [{"role": "user", "content": "FINAL signal (do not change): bearish"}],
            "response": "",
            "latency_ms": 11719,
            "attempt": 1,
            "error": "Connection error.",
        }]
    }
    console = Console(record=True, width=76)
    console.print(chat._selected_agent_renderable("Growth [NVDA]", None, list(chat._status_events), {}, {}, {}, llm_by_agent))
    text = console.export_text()

    assert "llm log" not in text
    assert "MiniMax-M2.7" not in text
    assert "11.7s" not in text
    assert "Connection error." not in text
    assert "FINAL signal" not in text
    assert "\nprompt\n" not in text
    assert "\nresponse\n" not in text


def test_selected_agent_renderable_shows_portfolio_decision():
    from src.schemas.signals import TickerDecision

    chat = NovaChat(Console(record=True, width=64), ChatSettings(provider="MiniMax", model="MiniMax-M2.7"))
    chat._prime_analysis_status(["NVDA"])
    chat._progress_handler_tui("portfolio_manager", "NVDA", "Done")
    rec = Recommendation(
        run_id="r1", as_of="2026-05-31T00:00:00Z", tickers=["NVDA"], signals=[],
        decisions=Decisions(per_ticker={"NVDA": TickerDecision(
            ticker="NVDA", action="buy", quantity=12, confidence=0.66, reasoning="Within risk limit")}),
        consensus={}, summary="",
    )
    console = Console(record=True, width=64)
    console.print(chat._selected_agent_renderable("Portfolio Manager [NVDA]", rec, list(chat._status_events), {}, {}, {}, {}))
    text = console.export_text()

    assert "decision & why" in text
    assert "• ticker:" in text
    assert "• action:" in text
    assert "• quantity:" in text
    assert "• reason:" in text
    assert "BUY" in text
    assert "Within risk limit" in text


def test_selected_agent_renderable_shows_risk_limits():
    from src.schemas.signals import Limits, TickerLimit

    chat = NovaChat(Console(record=True, width=64), ChatSettings(provider="MiniMax", model="MiniMax-M2.7"))
    chat._prime_analysis_status(["NVDA"])
    chat._progress_handler_tui("risk_manager", "NVDA", "Done")
    rec = Recommendation(
        run_id="r1", as_of="2026-05-31T00:00:00Z", tickers=["NVDA"], signals=[],
        limits=Limits(per_ticker={"NVDA": TickerLimit(
            ticker="NVDA", current_price=120.0, max_position_dollars=20000, max_shares=166, annualized_volatility=0.42)}),
        consensus={}, decisions=Decisions(per_ticker={}), summary="",
    )
    console = Console(record=True, width=64)
    console.print(chat._selected_agent_renderable("Risk Manager [NVDA]", rec, list(chat._status_events), {}, {}, {}, {}))
    text = console.export_text()

    assert "risk limits" in text
    assert "• max position:" in text
    assert "$20,000" in text
    assert "• max shares:" in text
    assert "166" in text


def test_status_pane_still_hides_reasoning_after_inspector_split():
    # Regression: the Stage-1..4 work must not leak model thinking into the COMPACT pane.
    chat = NovaChat(Console(record=True, width=44), ChatSettings(provider="MiniMax", model="MiniMax-M2.7"))
    renderable = chat._status_renderable([], {}, [], "private chain of thought", thinking=True)
    console = Console(record=True, width=44)
    console.print(renderable)
    text = console.export_text()
    assert "private chain" not in text
    assert "reasoning" not in text.lower()


# --- explain-only LLM reasoning layer over deterministic agents --------------

from types import SimpleNamespace


def _explain_ctx(show_reasoning=True):
    return SimpleNamespace(
        request=SimpleNamespace(show_reasoning=show_reasoning,
                                model=SimpleNamespace(name="gpt-4.1", provider="OpenAI")),
        derive_seed=lambda: 0,
    )


def _spec(agent_id="technical", display_name="Technical Analyst"):
    return SimpleNamespace(agent_id=agent_id, display_name=display_name)


def _ok_signal():
    return Signal(agent_id="technical", ticker="AAPL", direction="bullish", confidence=0.62,
                  reasoning="Ensemble bullish @ 62%", key_factors=["trend=bullish (62%)", "momentum=bullish (58%)"])


def test_explain_preserves_direction_and_confidence(monkeypatch):
    from src.agents import explain
    monkeypatch.setattr(explain, "call_llm", lambda **kw: explain._Explanation(explanation="narrated why"))
    sig = _ok_signal()
    out = explain.add_explain_reasoning(sig, _spec(), None, _explain_ctx(True))
    assert out.direction == "bullish"          # verdict immutable
    assert out.confidence == 0.62
    assert out.status == "ok"
    assert out.key_factors == sig.key_factors
    assert out.reasoning == sig.reasoning
    assert out.explain_reasoning == "narrated why"


def test_explain_toggle_off_makes_zero_calls(monkeypatch):
    from src.agents import explain
    calls = {"n": 0}

    def boom(**kw):
        calls["n"] += 1
        raise AssertionError("call_llm must not run when reasoning is off")

    monkeypatch.setattr(explain, "call_llm", boom)
    sig = _ok_signal()
    out = explain.add_explain_reasoning(sig, _spec(), None, _explain_ctx(False))
    assert calls["n"] == 0
    assert out is sig


def test_explain_skips_non_ok_and_buffett(monkeypatch):
    from src.agents import explain
    calls = {"n": 0}
    monkeypatch.setattr(explain, "call_llm", lambda **kw: calls.__setitem__("n", calls["n"] + 1) or explain._Explanation(explanation="x"))
    failed = Signal.failed(agent_id="technical", ticker="AAPL", error="boom")
    assert explain.add_explain_reasoning(failed, _spec(), None, _explain_ctx(True)) is failed
    buffett = _ok_signal().model_copy(update={"agent_id": "warren_buffett"})
    assert explain.add_explain_reasoning(buffett, _spec("warren_buffett", "Buffett"), None, _explain_ctx(True)) is buffett
    assert calls["n"] == 0


def test_explain_graceful_degradation(monkeypatch):
    from src.agents import explain

    def raise_it(**kw):
        raise RuntimeError("provider down")

    monkeypatch.setattr(explain, "call_llm", raise_it)
    sig = _ok_signal()
    out = explain.add_explain_reasoning(sig, _spec(), None, _explain_ctx(True))
    assert out is sig                      # run never breaks
    assert out.explain_reasoning == ""     # falls back to deterministic content downstream


def test_view_slice_is_empty_safe():
    from src.agents.explain import _view_slice
    from src.schemas.views import PriceView, InsiderView
    assert isinstance(_view_slice(PriceView(ticker="AAPL", prices=[])), str)
    assert isinstance(_view_slice(InsiderView(ticker="AAPL", trades=[])), str)
    assert _view_slice(object()) == "no extra view summary"


def test_reasoning_toggle_setter():
    chat = NovaChat(Console(record=True, width=44), ChatSettings(provider="MiniMax", model="MiniMax-M2.7"))
    assert chat.settings.show_reasoning is True
    assert chat._set_reasoning_from_command("reasoning off") is False
    assert chat.settings.show_reasoning is False
    assert chat._set_reasoning_from_command("reasoning on") is True
    assert chat.settings.show_reasoning is True


def test_settings_text_shows_reasoning_state():
    chat = NovaChat(Console(record=True, width=44), ChatSettings(provider="MiniMax", model="MiniMax-M2.7"))
    assert "reasoning: on" in chat._settings_text()
    chat.settings.show_reasoning = False
    assert "reasoning: off" in chat._settings_text()


def test_inspector_renders_clean_explain_reasoning_not_raw_json():
    # Review-required: the deterministic agent's narration must render as PROSE, and the
    # raw-JSON capture must be suppressed when explain_reasoning is present.
    from src.utils.progress import progress
    progress.reset_telemetry()
    # simulate the noisy raw-JSON capture that call_llm would store for a non-reasoning model
    progress.capture_reasoning("technical", "NVDA", None, '{"explanation": "Momentum is fading"}')
    chat = NovaChat(Console(record=True, width=72), ChatSettings(provider="MiniMax", model="MiniMax-M2.7"))
    chat._prime_analysis_status(["NVDA"])
    chat._progress_handler_tui("technical", "NVDA", "Done", "bearish")
    rec = Recommendation(
        run_id="r1", as_of="2026-05-31T00:00:00Z", tickers=["NVDA"],
        signals=[Signal(agent_id="technical", ticker="NVDA", direction="bearish", confidence=0.61,
                        reasoning="Ensemble bearish", key_factors=["trend=bearish (55%)"],
                        explain_reasoning="Momentum has rolled over while the longer trend stays intact, so the net read is a cautious bearish.")],
        consensus={}, decisions=Decisions(per_ticker={}), summary="",
    )
    console = Console(record=True, width=72)
    console.print(chat._selected_agent_renderable("Technical [NVDA]", rec, list(chat._status_events), {}, {}, progress.reasoning_snapshot(), {}))
    text = console.export_text()
    progress.reset_telemetry()
    assert "reasoning" in text
    assert "Momentum has rolled over" in text   # clean prose shown
    assert '{"explanation"' not in text          # raw JSON suppressed


# --- council reasoning (risk + portfolio managers) --------------------------

def _council_ctx(show_reasoning=True, tickers=("NVDA",)):
    return SimpleNamespace(
        request=SimpleNamespace(show_reasoning=show_reasoning, portfolio_mode="research",
                                model=SimpleNamespace(name="gpt-4.1", provider="OpenAI")),
        tickers=list(tickers),
        derive_seed=lambda: 0,
    )


def _council_inputs():
    from src.schemas.signals import Limits, TickerLimit, Decisions, TickerDecision, Consensus
    signals = [
        Signal(agent_id="technical", ticker="NVDA", direction="bearish", confidence=0.61),
        Signal(agent_id="valuation", ticker="NVDA", direction="bullish", confidence=0.55),
    ]
    consensus = {"NVDA": Consensus(ticker="NVDA", direction="neutral", confidence=0.5,
                                   weighted_score=0.0, bull_count=1, bear_count=1, neutral_count=0)}
    limits = Limits(per_ticker={"NVDA": TickerLimit(ticker="NVDA", current_price=120.0,
                    max_position_dollars=18000, max_shares=150, annualized_volatility=0.42, correlation_multiplier=0.9)})
    decisions = Decisions(per_ticker={"NVDA": TickerDecision(ticker="NVDA", action="hold",
                          confidence=0.5, reasoning="Neutral consensus")})
    return signals, consensus, limits, decisions


def test_council_reasoning_toggle_off_zero_calls(monkeypatch):
    from src.agents import explain
    calls = {"n": 0}
    monkeypatch.setattr(explain, "call_llm", lambda **kw: calls.__setitem__("n", calls["n"] + 1))
    rr, pr = explain.add_council_reasoning(_council_ctx(False), *_council_inputs(), None)
    assert (rr, pr) == ("", "")
    assert calls["n"] == 0


def test_council_reasoning_generates_two_narratives(monkeypatch):
    from src.agents import explain
    seen = []

    def fake(**kw):
        seen.append(kw["agent_name"])
        return explain._Explanation(explanation=f"committee note for {kw['agent_name']}")

    monkeypatch.setattr(explain, "call_llm", fake)
    signals, consensus, limits, decisions = _council_inputs()
    rr, pr = explain.add_council_reasoning(_council_ctx(True), signals, consensus, limits, decisions, None)
    assert "risk_manager" in rr and "portfolio_manager" in pr
    assert seen == ["risk_manager", "portfolio_manager"]
    # explain-only: the inputs are untouched (council returns strings, never mutates)
    assert decisions.per_ticker["NVDA"].action == "hold"
    assert limits.per_ticker["NVDA"].max_shares == 150


def test_council_reasoning_graceful_on_error(monkeypatch):
    from src.agents import explain

    def raise_it(**kw):
        raise RuntimeError("provider down")

    monkeypatch.setattr(explain, "call_llm", raise_it)
    rr, pr = explain.add_council_reasoning(_council_ctx(True), *_council_inputs(), None)
    assert (rr, pr) == ("", "")  # never breaks the run


def test_inspector_shows_council_reasoning_for_portfolio():
    chat = NovaChat(Console(record=True, width=78), ChatSettings(provider="MiniMax", model="MiniMax-M2.7"))
    chat._prime_analysis_status(["NVDA"])
    chat._progress_handler_tui("portfolio_manager", "NVDA", "Done")
    signals, consensus, limits, decisions = _council_inputs()
    rec = Recommendation(run_id="r1", as_of="2026-05-31T00:00:00Z", tickers=["NVDA"],
                         signals=signals, consensus=consensus, limits=limits, decisions=decisions,
                         summary="", portfolio_reasoning="The committee held given a split, low-conviction board.")
    console = Console(record=True, width=78)
    console.print(chat._selected_agent_renderable("Portfolio Manager [NVDA]", rec, list(chat._status_events), {}, {}, {}, {}))
    text = console.export_text()
    assert "reasoning" in text
    assert "council reasoning" not in text
    assert "The committee held given a split" in text
    assert "decision & why" in text  # deterministic block still present below
