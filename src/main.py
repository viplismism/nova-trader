"""Nova Trader — Multi-agent trading system entry point.

Runs the analyst pipeline:
  analysts (parallel) → risk manager → portfolio manager

No LangGraph — just ThreadPoolExecutor for parallel agents
and sequential calls for risk/portfolio management.
"""

import sys
import json
import argparse
from datetime import datetime
from dateutil.relativedelta import relativedelta

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from colorama import Fore, Style, init

from src.agents.portfolio_manager import portfolio_management_agent
from src.agents.risk_manager import risk_management_agent
from src.graph.state import AgentState
from src.utils.display import print_trading_output
from src.utils.analysts import ANALYST_ORDER, get_analyst_nodes
from src.utils.progress import progress
from src.orchestrator.pipeline import Pipeline
from src.cli.input import parse_cli_inputs

# Load environment variables from .env file
load_dotenv()
init(autoreset=True)


def parse_hedge_fund_response(response):
    """Parses a JSON string and returns a dictionary."""
    try:
        return json.loads(response)
    except json.JSONDecodeError as e:
        print(f"JSON decoding error: {e}\nResponse: {repr(response)}")
        return None
    except TypeError as e:
        print(f"Invalid response type (expected string, got {type(response).__name__}): {e}")
        return None
    except Exception as e:
        print(f"Unexpected error while parsing response: {e}\nResponse: {repr(response)}")
        return None


def run_hedge_fund(
    tickers: list[str],
    start_date: str,
    end_date: str,
    portfolio: dict,
    show_reasoning: bool = False,
    selected_analysts: list[str] = [],
    model_name: str = "gpt-4.1",
    model_provider: str = "OpenAI",
):
    """Run the full multi-agent trading pipeline.

    This is the main entry point called by both the CLI and the backtester.
    Replaces the LangGraph workflow with a simple Pipeline orchestrator.
    """
    progress.start()

    try:
        # Build the initial state (same structure agents expect)
        state: AgentState = {
            "messages": [
                HumanMessage(
                    content="Make trading decisions based on the provided data.",
                )
            ],
            "data": {
                "tickers": tickers,
                "portfolio": portfolio,
                "start_date": start_date,
                "end_date": end_date,
                "analyst_signals": {},
            },
            "metadata": {
                "show_reasoning": show_reasoning,
                "model_name": model_name,
                "model_provider": model_provider,
            },
        }

        # Get analyst nodes from the registry
        analyst_nodes = get_analyst_nodes()

        # Default to all analysts when none provided
        if not selected_analysts:
            selected_analysts = list(analyst_nodes.keys())

        # Build list of (name, function) tuples for selected analysts
        analysts = [
            (key, analyst_nodes[key][1])
            for key in selected_analysts
            if key in analyst_nodes
        ]

        # Create and run the pipeline
        pipeline = Pipeline(
            analyst_agents=analysts,
            risk_agent=risk_management_agent,
            portfolio_agent=portfolio_management_agent,
        )

        final_state = pipeline.run(state)

        return {
            "decisions": parse_hedge_fund_response(final_state["messages"][-1].content),
            "analyst_signals": final_state["data"]["analyst_signals"],
        }
    finally:
        progress.stop()


def main():
    """CLI entry point."""
    inputs = parse_cli_inputs(
        description="Run the Nova Trader multi-agent trading system",
        require_tickers=True,
        default_months_back=None,
        include_graph_flag=False,  # No graph visualization without LangGraph
        include_reasoning_flag=True,
    )

    tickers = inputs.tickers

    # Construct initial portfolio
    portfolio = {
        "cash": inputs.initial_cash,
        "margin_requirement": inputs.margin_requirement,
        "margin_used": 0.0,
        "positions": {
            ticker: {
                "long": 0,
                "short": 0,
                "long_cost_basis": 0.0,
                "short_cost_basis": 0.0,
                "short_margin_used": 0.0,
            }
            for ticker in tickers
        },
        "realized_gains": {
            ticker: {
                "long": 0.0,
                "short": 0.0,
            }
            for ticker in tickers
        },
    }

    result = run_hedge_fund(
        tickers=tickers,
        start_date=inputs.start_date,
        end_date=inputs.end_date,
        portfolio=portfolio,
        show_reasoning=inputs.show_reasoning,
        selected_analysts=inputs.selected_analysts,
        model_name=inputs.model_name,
        model_provider=inputs.model_provider,
    )
    print_trading_output(result)


if __name__ == "__main__":
    main()
