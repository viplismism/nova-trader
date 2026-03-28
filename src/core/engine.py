"""HydraEngine — multi-headed agent orchestrator.

Named after the Hydra: many autonomous heads, one body.
Each "head" is an analyst agent that executes independently and in parallel.
The body (engine) coordinates spawning, aggregation, risk, decisions,
and optional trade execution.

Architecture:
    ┌─────────────┐
    │  HydraEngine │
    │              │
    │  spawn()     │──→ ThreadPoolExecutor
    │              │       ├─ Agent 1 (head)  ──→ AgentPad
    │              │       ├─ Agent 2 (head)  ──→ AgentPad
    │              │       ├─ ...              ──→ AgentPad
    │              │       └─ Agent N (head)  ──→ AgentPad
    │              │
    │  aggregate() │──→ AgentPad.compute_consensus()
    │              │
    │  assess()    │──→ Risk Manager ──→ AgentPad.risk
    │              │
    │  decide()    │──→ Portfolio Manager ──→ AgentPad.decisions
    │              │
    │  execute()   │──→ Broker (optional) ──→ AgentPad.orders
    └─────────────┘
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

from langchain_core.messages import HumanMessage

from src.core.pad import AgentPad
from src.graph.state import AgentState
from src.utils.progress import progress


class HydraEngine:
    """Orchestrates multi-agent analysis with AgentPad aggregation.

    Usage:
        engine = HydraEngine(
            analyst_agents=[("warren_buffett", wb_fn), ("technicals", tech_fn)],
            risk_agent=risk_management_agent,
            portfolio_agent=portfolio_management_agent,
            broker=AlpacaBroker(),  # optional
        )
        pad = engine.run(state)
    """

    def __init__(
        self,
        analyst_agents: list[tuple[str, Callable]],
        risk_agent: Callable,
        portfolio_agent: Callable,
        broker=None,
        max_workers: int = 8,
    ):
        self.analyst_agents = analyst_agents
        self.risk_agent = risk_agent
        self.portfolio_agent = portfolio_agent
        self.broker = broker
        self.max_workers = max_workers

    def run(self, state: AgentState, execute: bool = False) -> tuple[AgentState, AgentPad]:
        """Execute the full Hydra pipeline.

        Args:
            state: The AgentState with tickers, portfolio, metadata.
            execute: If True and broker is set, place real orders.

        Returns:
            Tuple of (final_state, pad) — state for backward compat,
            pad for the enriched analysis workspace.
        """
        tickers = state["data"]["tickers"]
        pad = AgentPad(tickers=tickers)

        # Phase 1: Spawn analyst heads in parallel
        self._spawn_analysts(state, pad)

        # Phase 2: Compute consensus from collected signals
        pad.compute_consensus()

        # Sync pad signals back to state for risk/portfolio agents
        state["data"]["analyst_signals"] = pad.to_analyst_signals()

        # Phase 3: Risk assessment
        self._run_agent("risk_manager", self.risk_agent, state, pad)

        # Sync risk data into pad
        risk_signals = state["data"]["analyst_signals"].get("risk_management_agent", {})
        for ticker, risk_data in risk_signals.items():
            pad.write_risk(ticker, risk_data)

        # Phase 4: Portfolio decisions
        self._run_agent("portfolio_manager", self.portfolio_agent, state, pad)

        # Extract decisions from the portfolio manager's message
        self._extract_decisions(state, pad)

        # Phase 5: Execute trades (optional)
        if execute and self.broker and pad.decisions:
            self._execute_orders(pad)

        return state, pad

    def _spawn_analysts(self, state: AgentState, pad: AgentPad) -> None:
        """Spawn all analyst agents in parallel.

        Each agent writes to state["data"]["analyst_signals"][agent_id]
        and we mirror the signals into the pad for aggregation.
        """
        workers = min(len(self.analyst_agents), self.max_workers)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {}
            for name, fn in self.analyst_agents:
                t0 = time.monotonic()
                future = pool.submit(self._run_analyst, name, fn, state)
                futures[future] = (name, t0)

            for future in as_completed(futures):
                name, t0 = futures[future]
                elapsed = time.monotonic() - t0
                pad.record_timing(name, round(elapsed, 2))

                try:
                    result = future.result()
                    if result:
                        self._collect_message(state, result)
                except Exception as e:
                    progress.update_status(name, None, f"Error: {e}")

        # Mirror all analyst signals from state into pad
        for agent_id, ticker_signals in state["data"]["analyst_signals"].items():
            if agent_id == "risk_management_agent":
                continue
            for ticker, signal in ticker_signals.items():
                pad.write_signal(ticker, agent_id, signal)

    def _run_analyst(self, name: str, fn: Callable, state: AgentState) -> dict | None:
        """Run a single analyst agent with error handling."""
        try:
            return fn(state)
        except Exception as e:
            progress.update_status(name, None, f"Failed: {e}")
            return None

    def _run_agent(self, name: str, fn: Callable, state: AgentState, pad: AgentPad) -> None:
        """Run a sequential agent (risk/portfolio) with timing."""
        t0 = time.monotonic()
        try:
            result = fn(state)
            if result:
                self._collect_message(state, result)
        except Exception as e:
            progress.update_status(name, None, f"Failed: {e}")
        pad.record_timing(name, round(time.monotonic() - t0, 2))

    def _extract_decisions(self, state: AgentState, pad: AgentPad) -> None:
        """Extract portfolio decisions from the last message in state."""
        import json
        if not state["messages"]:
            return

        last_msg = state["messages"][-1]
        try:
            decisions = json.loads(last_msg.content)
            if isinstance(decisions, dict):
                for ticker, decision in decisions.items():
                    if isinstance(decision, dict):
                        pad.write_decision(ticker, decision)
        except (json.JSONDecodeError, TypeError, AttributeError):
            pass

    def _execute_orders(self, pad: AgentPad) -> None:
        """Execute trading decisions through the broker."""
        from src.execution.base import signal_to_orders

        orders = signal_to_orders(pad.decisions, self.broker)
        for order in orders:
            pad.write_order({
                "ticker": order.ticker,
                "side": order.side,
                "quantity": order.quantity,
                "order_type": order.order_type,
                "status": order.status,
                "order_id": order.order_id,
                "filled_price": order.filled_price,
            })

    @staticmethod
    def _collect_message(state: AgentState, result: dict) -> None:
        """Extract agent's new message and append to state."""
        if "messages" in result:
            msgs = result["messages"]
            if msgs:
                state["messages"].append(msgs[-1])
