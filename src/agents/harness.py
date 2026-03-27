"""Generic template-driven agent harness.

Replaces the 14 individual LLM-based agent Python files with a single runner
that loads YAML config + Jinja2 prompt templates to create any investor-persona agent.

The flow for every template agent:
    1. Fetch data (metrics, line items, market cap, optional insider/news)
    2. Run scoring functions (composable, declared in YAML)
    3. Aggregate scores with per-agent weights
    4. Render Jinja2 prompt template with agent persona + analysis data
    5. Call LLM for signal (bullish/bearish/neutral + confidence + reasoning)
    6. Write to shared state
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
from typing_extensions import Literal

from src.graph.state import AgentState, show_agent_reasoning
from src.tools.api import (
    get_financial_metrics,
    get_market_cap,
    search_line_items,
    get_insider_trades,
    get_company_news,
)
from src.utils.llm import call_llm
from src.utils.progress import progress
from src.utils.api_key import get_api_key_from_state
from src.agents.scoring import SCORING_FUNCTIONS


# ── Shared output model ──────────────────────

class AgentSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: int = Field(description="Confidence 0-100")
    reasoning: str = Field(description="Short justification for the decision")


# ── Jinja2 environment (loaded once) ─────────

TEMPLATES_DIR = Path(__file__).parent / "templates" / "prompts"
_jinja_env = Environment(
    loader=FileSystemLoader(str(TEMPLATES_DIR)),
    keep_trailing_newline=True,
    trim_blocks=True,
    lstrip_blocks=True,
)


def render_prompt(template_name: str, context: dict) -> str:
    """Render a Jinja2 prompt template with the given context."""
    tmpl = _jinja_env.get_template(template_name)
    return tmpl.render(**context)


# ── Data fetching ────────────────────────────

def fetch_agent_data(
    ticker: str,
    end_date: str,
    start_date: str | None,
    config: dict,
    api_key: str | None,
    agent_id: str,
) -> dict[str, Any]:
    """Fetch all data an agent needs based on its YAML config."""
    data_config = config.get("data", {})
    period = data_config.get("period", "annual")
    limit = data_config.get("limit", 10)
    line_items = data_config.get("line_items", [])

    progress.update_status(agent_id, ticker, "Fetching financial metrics")
    metrics = get_financial_metrics(ticker, end_date, period=period, limit=limit, api_key=api_key)

    progress.update_status(agent_id, ticker, "Gathering financial line items")
    financial_line_items = search_line_items(
        ticker, line_items, end_date, period=period, limit=limit, api_key=api_key,
    ) if line_items else []

    progress.update_status(agent_id, ticker, "Getting market cap")
    market_cap = get_market_cap(ticker, end_date, api_key=api_key)

    result = {
        "metrics": metrics,
        "financial_line_items": financial_line_items,
        "market_cap": market_cap,
    }

    # Optional extra data sources
    extras = data_config.get("extra", [])

    if "insider_trades" in extras:
        progress.update_status(agent_id, ticker, "Fetching insider trades")
        result["insider_trades"] = get_insider_trades(
            ticker, end_date, limit=data_config.get("insider_limit", 100), api_key=api_key,
        )

    if "company_news" in extras:
        progress.update_status(agent_id, ticker, "Fetching company news")
        result["company_news"] = get_company_news(
            ticker, end_date, limit=data_config.get("news_limit", 10), api_key=api_key,
        )

    return result


# ── Scoring phase ────────────────────────────

def run_scoring(data: dict, config: dict, agent_id: str, ticker: str) -> dict[str, dict]:
    """Run all scoring functions declared in the agent's YAML config.

    Returns: {"roic": {"score": X, "max_score": Y, "details": "..."}, ...}
    """
    scoring_config = config.get("scoring", [])
    results = {}

    for entry in scoring_config:
        func_name = entry["function"]
        params = entry.get("params", {})

        fn = SCORING_FUNCTIONS.get(func_name)
        if not fn:
            results[func_name] = {"score": 0, "max_score": 0, "details": f"Unknown scoring function: {func_name}"}
            continue

        progress.update_status(agent_id, ticker, f"Scoring: {func_name}")

        # Build kwargs from data based on what the function needs
        kwargs = _build_scoring_kwargs(fn, data, params)

        try:
            results[func_name] = fn(**kwargs)
        except Exception as e:
            results[func_name] = {"score": 0, "max_score": 0, "details": f"Error in {func_name}: {e}"}

    return results


def _build_scoring_kwargs(fn, data: dict, params: dict) -> dict:
    """Map data dict fields to scoring function parameter names."""
    import inspect
    sig = inspect.signature(fn)
    kwargs = {}

    # Standard data mappings
    data_map = {
        "metrics": data.get("metrics", []),
        "financial_line_items": data.get("financial_line_items", []),
        "insider_trades": data.get("insider_trades", []),
        "company_news": data.get("company_news", []),
        "market_cap": data.get("market_cap"),
    }

    for param_name in sig.parameters:
        if param_name in data_map:
            kwargs[param_name] = data_map[param_name]
        elif param_name in params:
            kwargs[param_name] = params[param_name]

    return kwargs


# ── Score aggregation ────────────────────────

def aggregate_scores(scoring_results: dict[str, dict], config: dict) -> tuple[float, float, str]:
    """Aggregate scoring results using weights from config.

    Returns: (total_score, max_score, signal)
    """
    weights = config.get("weights", {})
    thresholds = config.get("thresholds", {"bullish": 0.70, "bearish": 0.30})

    total_weighted = 0.0
    total_max_weighted = 0.0

    for func_name, result in scoring_results.items():
        weight = weights.get(func_name, 1.0)
        total_weighted += result.get("score", 0) * weight
        total_max_weighted += result.get("max_score", 0) * weight

    # Normalize to 0-1 range
    if total_max_weighted > 0:
        normalized = total_weighted / total_max_weighted
    else:
        normalized = 0.5

    # Map to signal
    bullish_threshold = thresholds.get("bullish", 0.70)
    bearish_threshold = thresholds.get("bearish", 0.30)

    if normalized >= bullish_threshold:
        signal = "bullish"
    elif normalized <= bearish_threshold:
        signal = "bearish"
    else:
        signal = "neutral"

    return total_weighted, total_max_weighted, signal


# ── LLM prompt & call ───────────────────────

def call_agent_llm(
    ticker: str,
    signal: str,
    score: float,
    max_score: float,
    scoring_results: dict,
    market_cap: float | None,
    config: dict,
    state: AgentState,
    agent_id: str,
) -> AgentSignal:
    """Render Jinja prompt, call LLM, return signal."""
    persona = config.get("persona", {})
    template_file = config.get("prompt_template", "default.j2")

    # Build context for Jinja template
    template_context = {
        "name": persona.get("name", "Analyst"),
        "philosophy": persona.get("philosophy", ""),
        "focus_metrics": persona.get("focus_metrics", []),
        "signal_rules": persona.get("signal_rules", ""),
        "ticker": ticker,
        "signal_hint": signal,
        "score": f"{score:.1f}",
        "max_score": f"{max_score:.1f}",
        "score_pct": f"{(score / max_score * 100):.0f}" if max_score > 0 else "0",
        "scoring_results": scoring_results,
        "market_cap": market_cap,
    }

    # Render the system prompt via Jinja2
    system_prompt = render_prompt(template_file, template_context)

    template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human",
         "Ticker: {ticker}\n"
         "Analysis Score: {score}/{max_score} ({score_pct}%)\n"
         "Scoring Details:\n{facts}\n\n"
         "Return exactly:\n"
         "{{\n"
         '  "signal": "bullish" | "bearish" | "neutral",\n'
         '  "confidence": <int 0-100>,\n'
         '  "reasoning": "<brief justification under 150 chars>"\n'
         "}}")
    ])

    # Build facts string from scoring results
    facts_lines = []
    for name, result in scoring_results.items():
        facts_lines.append(f"- {name}: {result.get('score', 0)}/{result.get('max_score', 0)} — {result.get('details', '')}")

    prompt = template.invoke({
        "ticker": ticker,
        "score": f"{score:.1f}",
        "max_score": f"{max_score:.1f}",
        "score_pct": f"{(score / max_score * 100):.0f}" if max_score > 0 else "0",
        "facts": "\n".join(facts_lines),
    })

    progress.update_status(agent_id, ticker, f"Generating {persona.get('name', 'agent')} analysis")

    def _default():
        return AgentSignal(signal="neutral", confidence=50, reasoning="Insufficient data")

    return call_llm(
        prompt=prompt,
        pydantic_model=AgentSignal,
        agent_name=agent_id,
        state=state,
        default_factory=_default,
    )


# ── Main entry: create agent function from config ──

def make_agent_from_config(config: dict):
    """Create an agent function from a YAML config dict.

    Returns a function with signature (state: AgentState) -> dict
    that can be plugged directly into the Pipeline.
    """
    agent_key = config["id"]
    agent_id = f"{agent_key}_agent"

    def agent_fn(state: AgentState, agent_id: str = agent_id) -> dict:
        data = state["data"]
        end_date = data["end_date"]
        start_date = data.get("start_date")
        tickers = data["tickers"]
        api_key = get_api_key_from_state(state, "FINANCIAL_DATASETS_API_KEY")

        analysis = {}

        for ticker in tickers:
            # 1. Fetch data
            fetched = fetch_agent_data(ticker, end_date, start_date, config, api_key, agent_id)

            # 2. Score
            scoring_results = run_scoring(fetched, config, agent_id, ticker)

            # 3. Aggregate
            total_score, max_score, signal = aggregate_scores(scoring_results, config)

            # 4. LLM call
            result = call_agent_llm(
                ticker=ticker,
                signal=signal,
                score=total_score,
                max_score=max_score,
                scoring_results=scoring_results,
                market_cap=fetched.get("market_cap"),
                config=config,
                state=state,
                agent_id=agent_id,
            )

            analysis[ticker] = {
                "signal": result.signal,
                "confidence": result.confidence,
                "reasoning": result.reasoning,
            }

            progress.update_status(agent_id, ticker, "Done", analysis=result.reasoning)

        # Write to state
        message = HumanMessage(content=json.dumps(analysis), name=agent_id)

        if state["metadata"]["show_reasoning"]:
            show_agent_reasoning(analysis, config.get("persona", {}).get("name", agent_key))

        state["data"]["analyst_signals"][agent_id] = analysis
        progress.update_status(agent_id, None, "Done")

        return {"messages": [message], "data": state["data"]}

    # Preserve name for debugging
    agent_fn.__name__ = agent_id
    agent_fn.__qualname__ = agent_id

    return agent_fn
