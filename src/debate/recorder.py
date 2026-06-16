"""Persist a debate run as a training-ready trajectory.

Writes, under ``runs_root()/<run_id>/`` (run_id prefixed ``debate-``):
  - debate.json      — input, plan, specialist drafts, bear case, memo, timings, usage
  - trajectory.jsonl — one line per LLM call (phase, kind, model, prompt, response, usage)
  - events.jsonl     — the full emit stream (phases, specialists, tool calls) for replay

The trajectory + events are the raw signal for later model training. Recording must
never break a run, so every method swallows its own errors.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from src.runs import runs_root


def _usage_dict(usage: Any) -> dict[str, int]:
    if usage is None:
        return {}
    return {
        "input_tokens": getattr(usage, "input_tokens", 0) or 0,
        "output_tokens": getattr(usage, "output_tokens", 0) or 0,
        "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", 0) or 0,
        "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", 0) or 0,
    }


def _short(value: Any, limit: int = 360) -> str:
    text = " ".join(str(value or "").split())
    return text[: limit - 1].rstrip() + "…" if len(text) > limit else text


class DebateRecorder:
    """Collects LLM calls + events during a debate, then persists the trajectory."""

    def __init__(self, ticker: str, question: str, horizon: str, source: str,
                 base_dir: Path | None = None) -> None:
        self.run_id = f"debate-{ticker.lower()}-{uuid.uuid4().hex[:8]}"
        self.dir = (base_dir or runs_root()) / self.run_id
        self.input = {"ticker": ticker, "question": question, "horizon": horizon, "source": source}
        self.calls: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []

    # passed as record= to run_debate
    def record(self, **call: Any) -> None:
        try:
            self.calls.append({
                "phase": call.get("phase", ""),
                "kind": call.get("kind", ""),
                "model": call.get("model", ""),
                "prompt": str(call.get("prompt", "")),
                "response": str(call.get("response", "")),
                "usage": _usage_dict(call.get("usage")),
            })
        except Exception:  # pragma: no cover
            pass

    # tap the emit stream
    def event(self, **ev: Any) -> None:
        try:
            self.events.append(ev)
        except Exception:  # pragma: no cover
            pass

    def save(self, result: dict, usage_str: str = "") -> str:
        """Persist the run; returns the run_id. Never raises."""
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            payload = {"run_id": self.run_id, "input": self.input, "usage": usage_str, **result}
            (self.dir / "debate.json").write_text(json.dumps(payload, default=str, indent=2))
            with (self.dir / "trajectory.jsonl").open("w") as fh:
                for c in self.calls:
                    fh.write(json.dumps(c, default=str) + "\n")
            with (self.dir / "events.jsonl").open("w") as fh:
                for e in self.events:
                    fh.write(json.dumps(e, default=str) + "\n")
        except Exception:  # pragma: no cover
            pass
        return self.run_id

    def grouped_llm_logs(self) -> dict[str, list[dict[str, Any]]]:
        """Per-phase LLM calls for the inspector, mirroring the deterministic llm_logs shape.
        Keyed by phase (fundamental/sentiment/valuation/macro/supervisor/bear/synthesize)."""
        grouped: dict[str, list[dict[str, Any]]] = {}
        for i, c in enumerate(self.calls):
            u = c.get("usage", {})
            grouped.setdefault(c["phase"], []).append({
                "provider": "Anthropic",
                "model": c.get("model", ""),
                "attempt": grouped.get(c["phase"]) and len(grouped[c["phase"]]) + 1 or 1,
                "kind": c.get("kind", ""),
                "prompt_tokens": u.get("input_tokens", 0),
                "completion_tokens": u.get("output_tokens", 0),
                "latency_ms": None,
                "error": None,
                "prompt": c.get("prompt", ""),
                "response": c.get("response", ""),
                "response_preview": _short(c.get("response", "")),
            })
        return grouped

    @classmethod
    def load(cls, run_id: str, base_dir: Path | None = None) -> dict:
        path = (base_dir or runs_root()) / run_id / "debate.json"
        return json.loads(path.read_text())

    @classmethod
    def exists(cls, run_id: str, base_dir: Path | None = None) -> bool:
        return ((base_dir or runs_root()) / run_id / "debate.json").is_file()
