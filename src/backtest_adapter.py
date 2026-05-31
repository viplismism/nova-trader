"""Adapter so the backtester can drive the new engine.

The backtester's `AgentController.run_agent` expects a callable with
the legacy `run_hedge_fund` signature. We translate that into a
`RunContext` -> `run_engine` call and return a dict matching the legacy
output shape (`decisions` + `analyst_signals`).
"""

from __future__ import annotations

from datetime import date, datetime, timezone

from src.schemas.context import ModelConfig, RunContext, RunRequest
from src.schemas.portfolio import Portfolio
from src.engine import run_engine


def run_hedge_fund(
    *,
    tickers: list[str],
    start_date: str,
    end_date: str,
    portfolio: dict,
    model_name: str = "gpt-4.1",
    model_provider: str = "OpenAI",
    selected_analysts: list[str] | None = None,
    show_reasoning: bool = False,
) -> dict:
    """Drop-in replacement for the legacy run_hedge_fund using the new engine."""
    request = RunRequest(
        tickers=tickers,
        start_date=date.fromisoformat(start_date),
        end_date=date.fromisoformat(end_date),
        portfolio=Portfolio.from_legacy_dict(portfolio),
        model=ModelConfig(provider=model_provider, name=model_name),
        show_reasoning=show_reasoning,
        selected_agents=list(selected_analysts or []),
    )
    ctx = RunContext(request=request, as_of=datetime.now(timezone.utc))

    # Backtests spin up a fresh RunContext per simulated day — recording each one
    # would flood ~/.nova-trader/runs/. Disable per-day recording here.
    rec = run_engine(ctx, selected_agents=request.selected_agents or None, record=False)

    # Translate Recommendation back to legacy dict shape.
    decisions: dict[str, dict] = {}
    for ticker, d in rec.decisions.per_ticker.items():
        decisions[ticker] = {
            "action": d.action,
            "quantity": d.quantity,
            "confidence": round(d.confidence * 100),
            "reasoning": d.reasoning,
        }

    analyst_signals: dict[str, dict] = {}
    for sig in rec.signals:
        analyst_signals.setdefault(sig.agent_id, {})[sig.ticker] = {
            "signal": sig.direction,
            "confidence": round(sig.confidence * 100),
            "reasoning": sig.reasoning,
        }
    # Risk manager data — under its legacy key so backtester display can pick it up.
    risk_dict: dict[str, dict] = {}
    for ticker, lim in rec.limits.per_ticker.items():
        risk_dict[ticker] = {
            "current_price": lim.current_price,
            "remaining_position_limit": lim.remaining_position_limit,
            "annualized_volatility": lim.annualized_volatility,
        }
    if risk_dict:
        analyst_signals["risk_management_agent"] = risk_dict

    return {
        "decisions": decisions,
        "analyst_signals": analyst_signals,
        "hedge_plan": rec.decisions.hedge_plan.model_dump(),
    }
