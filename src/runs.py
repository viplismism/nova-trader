"""Persistent run records.

Every run writes to ~/.nova-trader/runs/<run_id>/. This is the audit
artifact: snapshot, per-agent views, every signal, every LLM call,
and the final recommendation are all on disk.

File layout
-----------
    ~/.nova-trader/runs/<run_id>/
        metadata.json         — run_id, as_of, model, tickers, seed, code_sha, command
        snapshot.json         — the MarketSnapshot (raw API data, fetched once)
        views.jsonl           — one line per (agent, ticker): the typed view passed in
        signals.jsonl         — one Signal per line, in completion order
        llm.jsonl             — one record per LLM call (prompt, response, fingerprint…)
        recommendation.json   — the final Recommendation
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel


def runs_root() -> Path:
    """Resolve the runs directory. Honors NOVA_RUNS_DIR env var, else ~/.nova-trader/runs/."""
    override = os.environ.get("NOVA_RUNS_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return Path.home() / ".nova-trader" / "runs"


def _dump(obj: Any) -> Any:
    """Best-effort JSON-serializable conversion."""
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json")
    if isinstance(obj, dict):
        return {k: _dump(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_dump(v) for v in obj]
    if isinstance(obj, (datetime,)):
        return obj.isoformat()
    return obj


class RunRecorder:
    """Writes the audit trail for a single run.

    All writes are append-only and thread-safe. JSONL files are appended
    line by line so a crash mid-run leaves partial-but-readable output.
    """

    def __init__(self, run_id: str, base_dir: Path | None = None) -> None:
        self.run_id = run_id
        self.dir = (base_dir or runs_root()) / run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    # ── Single-shot writes ─────────────────────────────────

    def write_metadata(self, payload: dict | BaseModel) -> None:
        self._write_json("metadata.json", payload)

    def write_snapshot(self, snapshot: BaseModel) -> None:
        self._write_json("snapshot.json", snapshot)

    def write_recommendation(self, recommendation: BaseModel) -> None:
        self._write_json("recommendation.json", recommendation)

    # ── Append-only writes ──────────────────────────────────

    def append_view(self, agent_id: str, ticker: str, view: BaseModel) -> None:
        self._append_jsonl("views.jsonl", {
            "agent_id": agent_id,
            "ticker": ticker,
            "view": _dump(view),
        })

    def append_signal(self, signal: BaseModel) -> None:
        self._append_jsonl("signals.jsonl", _dump(signal))

    def append_llm_call(
        self,
        *,
        agent_id: str,
        ticker: str | None,
        model: str,
        provider: str,
        prompt: Any,
        response: str,
        seed: int | None,
        system_fingerprint: str | None,
        latency_ms: float,
        prompt_tokens: int | None,
        completion_tokens: int | None,
        attempt: int = 1,
        error: str | None = None,
    ) -> None:
        self._append_jsonl("llm.jsonl", {
            "ts": datetime.now(timezone.utc).isoformat(),
            "agent_id": agent_id,
            "ticker": ticker,
            "provider": provider,
            "model": model,
            "prompt": _dump(prompt),
            "response": response,
            "seed": seed,
            "system_fingerprint": system_fingerprint,
            "latency_ms": round(latency_ms, 2),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "attempt": attempt,
            "error": error,
        })

    # ── Read helpers (used by show / rerun) ────────────────

    @classmethod
    def load_metadata(cls, run_id: str, base_dir: Path | None = None) -> dict:
        path = (base_dir or runs_root()) / run_id / "metadata.json"
        return json.loads(path.read_text())

    @classmethod
    def load_snapshot_dict(cls, run_id: str, base_dir: Path | None = None) -> dict:
        path = (base_dir or runs_root()) / run_id / "snapshot.json"
        return json.loads(path.read_text())

    @classmethod
    def load_recommendation_dict(cls, run_id: str, base_dir: Path | None = None) -> dict:
        path = (base_dir or runs_root()) / run_id / "recommendation.json"
        return json.loads(path.read_text())

    @classmethod
    def exists(cls, run_id: str, base_dir: Path | None = None) -> bool:
        return ((base_dir or runs_root()) / run_id).is_dir()

    # ── Internal ───────────────────────────────────────────

    def _write_json(self, name: str, payload: Any) -> None:
        path = self.dir / name
        with self._lock:
            path.write_text(json.dumps(_dump(payload), indent=2, default=str))

    def _append_jsonl(self, name: str, payload: Any) -> None:
        path = self.dir / name
        line = json.dumps(_dump(payload), default=str)
        with self._lock:
            with path.open("a") as f:
                f.write(line + "\n")
