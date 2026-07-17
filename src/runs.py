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
        self._write_recent_summary(_dump(recommendation))

    def update_recent_user(self, user: str) -> None:
        """Add attribution to the lightweight recent-run row after web auth stamps it."""
        path = self.dir / "recent.json"
        try:
            row = json.loads(path.read_text())
        except (OSError, ValueError):
            try:
                row = self._recent_row(json.loads((self.dir / "recommendation.json").read_text()))
            except (OSError, ValueError):
                return
        row["user"] = user
        self._write_json("recent.json", row)

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
    def load_recommendation(cls, run_id: str, base_dir: Path | None = None):
        """Load and validate a saved Recommendation. Single source of truth for the
        load+validate the web server and CLI both need (raises FileNotFoundError if
        the run has no recommendation.json)."""
        from src.schemas.signals import Recommendation

        return Recommendation.model_validate(cls.load_recommendation_dict(run_id, base_dir))

    @classmethod
    def load_llm_calls(cls, run_id: str, base_dir: Path | None = None) -> dict[str, list[dict]]:
        """Read llm.jsonl and group records by agent_id. Safe to call while a run is
        still writing: append_llm_call writes one whole line at a time, so at worst the
        final line is torn — we skip a JSONDecodeError rather than raise into the UI."""
        path = (base_dir or runs_root()) / run_id / "llm.jsonl"
        grouped: dict[str, list[dict]] = {}
        try:
            raw = path.read_text()
        except FileNotFoundError:
            return grouped
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            grouped.setdefault(rec.get("agent_id", "unknown"), []).append(rec)
        return grouped

    @classmethod
    def exists(cls, run_id: str, base_dir: Path | None = None) -> bool:
        return ((base_dir or runs_root()) / run_id).is_dir()

    @classmethod
    def list_recent(cls, limit: int = 20, base_dir: Path | None = None) -> list[dict[str, Any]]:
        """Saved Signals runs (newest first) as compact rows for a 'recent' picker.
        Excludes debate-* dirs (those have their own recorder) and any without a
        recommendation.json."""
        root = base_dir or runs_root()
        try:
            dirs = [d for d in root.iterdir()
                    if d.is_dir() and not d.name.startswith("debate-")
                    and (d / "recommendation.json").is_file()]
        except FileNotFoundError:
            return []
        # Sort by the immutable result timestamp. Creating a lightweight cache for an
        # older run must not make that run look new in the picker.
        dirs.sort(key=lambda d: (d / "recommendation.json").stat().st_mtime, reverse=True)
        rows: list[dict[str, Any]] = []
        for d in dirs[:limit]:
            try:
                row = json.loads((d / "recent.json").read_text())
            except (OSError, ValueError):
                try:
                    rec = json.loads((d / "recommendation.json").read_text())
                except (OSError, ValueError):
                    continue
                row = cls._recent_row(rec, d)
                # Backfill historical runs once; future picker loads read this tiny file.
                try:
                    (d / "recent.json").write_text(json.dumps(row, indent=2))
                except OSError:
                    pass
            rows.append(row)
        return rows

    @staticmethod
    def _recent_row(rec: dict[str, Any], directory: Path | None = None) -> dict[str, Any]:
        tickers = rec.get("tickers", []) or []
        consensus = rec.get("consensus", {}) or {}
        first = consensus.get(tickers[0], {}) if tickers else {}
        user = ""
        if directory is not None:
            try:
                user = json.loads((directory / "user.json").read_text()).get("user", "")
            except (OSError, ValueError):
                pass
        return {
            "run_id": rec.get("run_id", directory.name if directory else ""),
            "tickers": tickers,
            "as_of": rec.get("as_of", ""),
            "stars": first.get("stars", ""),
            "stars_label": first.get("stars_label", ""),
            "user": user,
        }

    def _write_recent_summary(self, rec: dict[str, Any]) -> None:
        self._write_json("recent.json", self._recent_row(rec))

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
