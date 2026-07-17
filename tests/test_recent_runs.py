import json

from src.debate.recorder import DebateRecorder
from src.runs import RunRecorder


def test_signal_recent_rows_are_cached_and_keep_user(tmp_path):
    recorder = RunRecorder("signal-1", base_dir=tmp_path)
    recorder.write_recommendation({
        "run_id": "signal-1",
        "as_of": "2026-07-18T00:00:00Z",
        "tickers": ["META"],
        "consensus": {"META": {"stars": 4, "stars_label": "buy"}},
    })
    recorder.update_recent_user("vipul")

    row = RunRecorder.list_recent(base_dir=tmp_path)[0]
    assert row == {
        "run_id": "signal-1",
        "tickers": ["META"],
        "as_of": "2026-07-18T00:00:00Z",
        "stars": 4,
        "stars_label": "buy",
        "user": "vipul",
    }

    # The picker should now depend only on recent.json, not the full result payload.
    (recorder.dir / "recommendation.json").write_text("not valid json")
    assert RunRecorder.list_recent(base_dir=tmp_path)[0] == row


def test_old_signal_run_is_backfilled_to_compact_recent_row(tmp_path):
    run_dir = tmp_path / "old-signal"
    run_dir.mkdir()
    (run_dir / "recommendation.json").write_text(json.dumps({
        "run_id": "old-signal",
        "as_of": "2026-07-17",
        "tickers": ["AAPL"],
        "consensus": {"AAPL": {}},
    }))
    (run_dir / "user.json").write_text(json.dumps({"user": "deep"}))

    assert RunRecorder.list_recent(base_dir=tmp_path)[0]["user"] == "deep"
    assert (run_dir / "recent.json").is_file()


def test_debate_recent_rows_are_cached_and_old_runs_are_backfilled(tmp_path):
    recorder = DebateRecorder("META", "is it expensive?", "12 months", "filings", base_dir=tmp_path)
    recorder.input["user"] = "vipul"
    recorder.save({"memo": {"conviction": "high", "directional_lean": "bearish"}})

    row = DebateRecorder.list_recent(base_dir=tmp_path)[0]
    assert row["ticker"] == "META"
    assert row["conviction"] == "high"
    assert row["user"] == "vipul"
    (recorder.dir / "debate.json").write_text("not valid json")
    assert DebateRecorder.list_recent(base_dir=tmp_path)[0] == row

    old_dir = tmp_path / "debate-old"
    old_dir.mkdir()
    (old_dir / "debate.json").write_text(json.dumps({
        "run_id": "debate-old",
        "input": {"ticker": "NVDA", "question": "why?"},
        "memo": {"conviction": "medium", "directional_lean": "bullish"},
    }))
    DebateRecorder.list_recent(base_dir=tmp_path)
    assert (old_dir / "recent.json").is_file()
