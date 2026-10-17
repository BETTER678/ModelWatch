
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
RUN_LOG = ROOT / "tracking" / "lendingclub" / "pipeline_runs.jsonl"

SCRIPTS = [
    "scripts/analyze_lendingclub_drift.py",
    "scripts/retrain_lendingclub.py",
]


def run_script(script_path):
    full_path = ROOT / script_path

    if not full_path.exists():
        return {
            "script": script_path,
            "status": "FAILED",
            "error": f"Script not found: {full_path}",
        }

    print(f"\n{'=' * 65}")
    print(f"Running: {script_path}")
    print("=" * 65)

    try:
        result = subprocess.run(
            [PYTHON, str(full_path)],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as error:
        return {
            "script": script_path,
            "status": "FAILED",
            "error": str(error),
        }

    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
    if result.stderr:
        print(
            result.stderr,
            file=sys.stderr,
            end="" if result.stderr.endswith("\n") else "\n",
        )

    return {
        "script": script_path,
        "status": "SUCCEEDED" if result.returncode == 0 else "FAILED",
        "exit_code": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def save_run_record(record):
    RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
    with RUN_LOG.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record) + "\n")


def run_pipeline(dry_run=False):
    started_at = datetime.now(timezone.utc)
    record = {
        "started_at": started_at.isoformat(),
        "dry_run": dry_run,
        "status": "RUNNING",
        "stages": [],
    }

    print("\nMODELWATCH - LENDINGCLUB PIPELINE")
    print("=" * 65)
    if dry_run:
        print("DRY RUN: candidate retraining is disabled.")

    try:
        for script in SCRIPTS:
            if dry_run and script == "scripts/retrain_lendingclub.py":
                record["stages"].append({
                    "script": script,
                    "status": "SKIPPED",
                    "reason": "Dry run does not create or update model artifacts.",
                })
                print(f"\nSkipping {script} (dry run).")
                continue

            stage = run_script(script)
            record["stages"].append(stage)
            if stage["status"] != "SUCCEEDED":
                detail = stage.get("error") or (
                    f"exit code {stage.get('exit_code')}"
                )
                raise RuntimeError(f"{script} failed: {detail}")

        record["status"] = "DRY_RUN" if dry_run else "SUCCEEDED"
        if dry_run:
            print("\nDry run completed. No model artifacts were changed.")
        else:
            print("\nPipeline stages completed.")
            print("Drift analysis and candidate comparison have run.")
            print(
                "Automatic promotion is disabled until a suitable "
                "untouched holdout is evaluated."
            )

    except (OSError, RuntimeError) as error:
        record["status"] = "FAILED"
        record["error"] = str(error)
        print(f"\nPIPELINE FAILED: {error}")
        raise
    finally:
        record["finished_at"] = datetime.now(timezone.utc).isoformat()
        save_run_record(record)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the LendingClub pipeline.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run read-only drift analysis and skip candidate retraining.",
    )
    run_pipeline(dry_run=parser.parse_args().dry_run)