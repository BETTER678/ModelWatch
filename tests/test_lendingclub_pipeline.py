import json

import pytest

from pipeline import lendingclub_pipeline


def test_dry_run_skips_retraining_and_logs_result(tmp_path, monkeypatch):
    run_log = tmp_path / "pipeline_runs.jsonl"
    monkeypatch.setattr(lendingclub_pipeline, "RUN_LOG", run_log)
    calls = []

    def run_script(script):
        calls.append(script)
        return {
            "script": script,
            "status": "SUCCEEDED",
            "exit_code": 0,
            "stdout": "",
            "stderr": "",
        }

    monkeypatch.setattr(lendingclub_pipeline, "run_script", run_script)

    lendingclub_pipeline.run_pipeline(dry_run=True)

    record = json.loads(run_log.read_text(encoding="utf-8"))
    assert calls == ["scripts/analyze_lendingclub_drift.py"]
    assert record["status"] == "DRY_RUN"
    assert record["stages"][1]["status"] == "SKIPPED"


def test_failed_stage_is_logged_and_raised(tmp_path, monkeypatch):
    run_log = tmp_path / "pipeline_runs.jsonl"
    monkeypatch.setattr(lendingclub_pipeline, "RUN_LOG", run_log)
    monkeypatch.setattr(
        lendingclub_pipeline,
        "run_script",
        lambda script: {
            "script": script,
            "status": "FAILED",
            "exit_code": 2,
            "stdout": "",
            "stderr": "dataset missing",
        },
    )

    with pytest.raises(RuntimeError, match="exit code 2"):
        lendingclub_pipeline.run_pipeline()

    record = json.loads(run_log.read_text(encoding="utf-8"))
    assert record["status"] == "FAILED"
    assert record["stages"][0]["status"] == "FAILED"
