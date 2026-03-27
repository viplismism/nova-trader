"""
Lightweight multi-agent orchestration pipeline.

Replaces LangGraph's StateGraph with direct parallel execution
using Python's ThreadPoolExecutor. The pattern is simple:

    analysts (parallel) → risk manager → portfolio manager

Each agent:
  1. Reads shared state (tickers, dates, portfolio, metadata)
  2. Calls APIs, runs calculations, invokes LLMs
  3. Writes its signal to state["data"]["analyst_signals"][agent_id]
  4. Returns a dict with updated messages

Agents communicate through the shared state dict, not through messages.
Thread safety: each analyst writes to its own key in analyst_signals,
so parallel execution is safe under CPython's GIL.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

from src.graph.state import AgentState


class Pipeline:
    """Multi-agent orchestration pipeline.

    Usage:
        pipeline = Pipeline(
            analyst_agents=[("warren_buffett", wb_agent), ("technicals", tech_agent)],
            risk_agent=risk_management_agent,
            portfolio_agent=portfolio_management_agent,
        )
        final_state = pipeline.run(state)
    """

    def __init__(
        self,
        analyst_agents: list[tuple[str, Callable]],
        risk_agent: Callable,
        portfolio_agent: Callable,
        max_workers: int = 8,
    ):
        self.analyst_agents = analyst_agents
        self.risk_agent = risk_agent
        self.portfolio_agent = portfolio_agent
        self.max_workers = max_workers

    def run(self, state: AgentState) -> AgentState:
        """Execute the full pipeline: analysts → risk → portfolio."""

        # Phase 1: Run all analyst agents in parallel
        self._run_analysts_parallel(state)

        # Phase 2: Risk assessment (needs all analyst signals)
        self._run_sequential("risk_manager", self.risk_agent, state)

        # Phase 3: Portfolio decisions (needs risk output)
        self._run_sequential("portfolio_manager", self.portfolio_agent, state)

        return state

    def _run_analysts_parallel(self, state: AgentState) -> None:
        """Run analyst agents in parallel using ThreadPoolExecutor.

        Each agent writes to state["data"]["analyst_signals"][agent_id],
        so there are no write conflicts between threads.
        """
        workers = min(len(self.analyst_agents), self.max_workers)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(fn, state): name
                for name, fn in self.analyst_agents
            }

            for future in as_completed(futures):
                name = futures[future]
                try:
                    result = future.result()
                    if result:
                        self._collect_message(state, result)
                except Exception as e:
                    print(f"[Pipeline] Agent '{name}' failed: {e}")

    def _run_sequential(self, name: str, fn: Callable, state: AgentState) -> None:
        """Run a single agent sequentially and collect its message."""
        try:
            result = fn(state)
            if result:
                self._collect_message(state, result)
        except Exception as e:
            print(f"[Pipeline] Agent '{name}' failed: {e}")

    @staticmethod
    def _collect_message(state: AgentState, result: dict) -> None:
        """Extract the agent's new message from its return value.

        Agents return {"messages": state["messages"] + [new_msg], ...}.
        We only need the last element (the new message).
        """
        if "messages" in result:
            msgs = result["messages"]
            if msgs:
                state["messages"].append(msgs[-1])
