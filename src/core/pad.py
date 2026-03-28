"""AgentPad — centralised knowledge workspace for multi-agent analysis.

Every agent (analyst, risk manager, portfolio manager) writes its output here.
The pad accumulates signals, computes consensus, and stores final decisions
and execution records in a single auditable structure.

Thread-safe: each agent writes only to its own key within a ticker bucket.
Read access is lock-free because dict reads in CPython are atomic at the
bytecode level and writers never mutate existing entries.

The pad is JSON-serializable for logging, debugging, and audit trails.
"""

import json
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Literal


class AgentPad:
    """Central knowledge store for a single analysis session.

    Lifecycle:
        pad = AgentPad(tickers=["AAPL", "NVDA"])
        # Phase 1 — analysts write signals
        pad.write_signal("AAPL", "warren_buffett_agent", {...})
        # Phase 2 — consensus computed automatically
        pad.compute_consensus()
        # Phase 3 — risk manager writes limits
        pad.write_risk("AAPL", {...})
        # Phase 4 — portfolio manager writes decisions
        pad.write_decision("AAPL", {...})
        # Phase 5 — broker executes (optional)
        pad.write_order({...})
    """

    def __init__(self, tickers: list[str], session_id: str | None = None):
        self.session_id = session_id or uuid.uuid4().hex[:12]
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.tickers = list(tickers)

        # Core data stores — all keyed by ticker
        self.signals: dict[str, dict[str, dict]] = {t: {} for t in tickers}
        self.consensus: dict[str, dict] = {}
        self.risk: dict[str, dict] = {}
        self.decisions: dict[str, dict] = {}
        self.orders: list[dict] = []

        # Metadata
        self.agent_timings: dict[str, float] = {}
        self._lock = threading.Lock()

    # ── Write Methods (thread-safe) ───────────────────────

    def write_signal(self, ticker: str, agent_id: str, signal: dict) -> None:
        """Record an analyst signal for a ticker.

        Args:
            ticker: Stock symbol.
            agent_id: Unique agent identifier (e.g. "warren_buffett_agent").
            signal: Dict with at minimum {"signal": str, "confidence": int}.
        """
        if ticker not in self.signals:
            self.signals[ticker] = {}
        self.signals[ticker][agent_id] = signal

    def write_risk(self, ticker: str, risk_data: dict) -> None:
        """Record risk manager output for a ticker."""
        self.risk[ticker] = risk_data

    def write_decision(self, ticker: str, decision: dict) -> None:
        """Record portfolio manager decision for a ticker."""
        self.decisions[ticker] = decision

    def write_order(self, order: dict) -> None:
        """Append an executed order record."""
        with self._lock:
            self.orders.append(order)

    def record_timing(self, agent_id: str, elapsed_seconds: float) -> None:
        """Record how long an agent took to execute."""
        self.agent_timings[agent_id] = elapsed_seconds

    # ── Read Methods ──────────────────────────────────────

    def get_signals(self, ticker: str) -> dict[str, dict]:
        """Get all analyst signals for a ticker."""
        return self.signals.get(ticker, {})

    def get_consensus(self, ticker: str) -> dict:
        """Get computed consensus for a ticker."""
        return self.consensus.get(ticker, {})

    def get_risk(self, ticker: str) -> dict:
        """Get risk data for a ticker."""
        return self.risk.get(ticker, {})

    def get_decision(self, ticker: str) -> dict:
        """Get portfolio decision for a ticker."""
        return self.decisions.get(ticker, {})

    # ── Consensus Computation ─────────────────────────────

    def compute_consensus(self) -> None:
        """Aggregate analyst signals into a consensus view per ticker.

        For each ticker, counts bullish/bearish/neutral signals
        and computes a confidence-weighted consensus.
        """
        for ticker in self.tickers:
            signals = self.signals.get(ticker, {})
            if not signals:
                continue

            bull_count = 0
            bear_count = 0
            neutral_count = 0
            total_confidence = 0.0
            weighted_score = 0.0  # +1 bull, -1 bear, 0 neutral

            for agent_id, sig in signals.items():
                signal_val = sig.get("signal", "neutral").lower()
                confidence = sig.get("confidence", 50)
                # Normalize to 0-1 range
                if confidence > 1:
                    confidence = confidence / 100.0

                if signal_val == "bullish":
                    bull_count += 1
                    weighted_score += confidence
                elif signal_val == "bearish":
                    bear_count += 1
                    weighted_score -= confidence
                else:
                    neutral_count += 1

                total_confidence += confidence

            total_agents = bull_count + bear_count + neutral_count
            if total_agents == 0:
                continue

            avg_confidence = total_confidence / total_agents

            # Consensus signal based on weighted score
            if weighted_score > 0.1:
                consensus_signal = "bullish"
            elif weighted_score < -0.1:
                consensus_signal = "bearish"
            else:
                consensus_signal = "neutral"

            self.consensus[ticker] = {
                "signal": consensus_signal,
                "confidence": round(avg_confidence * 100),
                "weighted_score": round(weighted_score, 3),
                "bull_count": bull_count,
                "bear_count": bear_count,
                "neutral_count": neutral_count,
                "total_agents": total_agents,
            }

    # ── State Bridge ──────────────────────────────────────

    def to_analyst_signals(self) -> dict[str, dict]:
        """Export signals in the format expected by AgentState.

        Returns dict keyed by agent_id → {ticker → signal_dict},
        matching state["data"]["analyst_signals"] shape.
        """
        result: dict[str, dict] = {}
        for ticker, agents in self.signals.items():
            for agent_id, signal in agents.items():
                if agent_id not in result:
                    result[agent_id] = {}
                result[agent_id][ticker] = signal

        # Include risk data if present
        if self.risk:
            result["risk_management_agent"] = self.risk

        return result

    @classmethod
    def from_analyst_signals(cls, tickers: list[str], analyst_signals: dict) -> "AgentPad":
        """Create an AgentPad from existing analyst_signals dict.

        This bridges the old Pipeline output into the new AgentPad format.
        """
        pad = cls(tickers=tickers)
        for agent_id, ticker_signals in analyst_signals.items():
            if agent_id == "risk_management_agent":
                for ticker, risk_data in ticker_signals.items():
                    pad.write_risk(ticker, risk_data)
            else:
                for ticker, signal in ticker_signals.items():
                    pad.write_signal(ticker, agent_id, signal)
        pad.compute_consensus()
        return pad

    # ── Serialization ─────────────────────────────────────

    def to_dict(self) -> dict:
        """Full JSON-serializable snapshot of the pad."""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "tickers": self.tickers,
            "signals": self.signals,
            "consensus": self.consensus,
            "risk": self.risk,
            "decisions": self.decisions,
            "orders": self.orders,
            "agent_timings": self.agent_timings,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize pad to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def summary(self) -> dict[str, Any]:
        """Compact summary for CLI display."""
        return {
            "session_id": self.session_id,
            "tickers": self.tickers,
            "agents_reported": sum(len(s) for s in self.signals.values()),
            "consensus": self.consensus,
            "decisions": self.decisions,
            "orders_placed": len(self.orders),
        }
