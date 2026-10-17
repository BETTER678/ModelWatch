import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
RUN_LOG = ROOT / "tracking" / "all_layers_runs.jsonl"


def run_stage(name, script, extra_args=()):
    started = datetime.now(timezone.utc)
    command = [sys.executable, "-u", str(ROOT / script), *extra_args]
    print(f"\n{'=' * 70}\nRUNNING {name}\n{'=' * 70}", flush=True)
    try:
        completed = subprocess.run(command, cwd=ROOT, check=False)
        return {
            "name": name,
            "script": script,
            "status": "SUCCEEDED" if completed.returncode == 0 else "FAILED",
            "exit_code": completed.returncode,
            "started_at": started.isoformat(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
        }
    except OSError as error:
        return {
            "name": name,
            "script": script,
            "status": "FAILED",
            "error": str(error),
            "started_at": started.isoformat(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
        }


def main():
    parser = argparse.ArgumentParser(
        description="Run ModelWatch tabular, text-embedding, and LLM layers."
    )
    parser.add_argument("--news-limit", type=int, default=1000,
                        help="News headlines to embed; 0 means all rows.")
    parser.add_argument("--dolly-limit", type=int, default=25,
                        help="Dolly prompts to evaluate; 0 means all rows.")
    parser.add_argument("--embedding-model", default="nomic-embed-text")
    parser.add_argument("--llm-model", default="llama3.2:1b")
    args = parser.parse_args()
    if args.news_limit < 0 or args.dolly_limit < 0:
        parser.error("dataset limits must be >= 0")

    record = {
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "RUNNING",
        "dataset_limits": {
            "news": args.news_limit,
            "dolly": args.dolly_limit,
        },
        "stages": [],
    }
    stages = [
        (
            "LendingClub tabular drift and candidate evaluation",
            "pipeline/lendingclub_pipeline.py",
            (),
        ),
        (
            "News text embedding drift",
            "scripts/monitor_news_embeddings.py",
            ("--limit", str(args.news_limit), "--model", args.embedding_model),
        ),
        (
            "Dolly LLM output monitoring",
            "scripts/monitor_dolly_outputs.py",
            ("--limit", str(args.dolly_limit), "--model", args.llm_model),
        ),
    ]

    for name, script, extra_args in stages:
        stage = run_stage(name, script, extra_args)
        record["stages"].append(stage)
        if stage["status"] == "FAILED":
            print(f"\nStage failed: {name}", file=sys.stderr)

    record["finished_at"] = datetime.now(timezone.utc).isoformat()
    record["status"] = (
        "SUCCEEDED"
        if all(stage["status"] == "SUCCEEDED" for stage in record["stages"])
        else "FAILED"
    )
    RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
    with RUN_LOG.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record) + "\n")

    print("\nMODELWATCH ALL-LAYERS RESULT")
    print(json.dumps(record, indent=2))
    print(f"Run record saved to: {RUN_LOG}")
    return 0 if record["status"] == "SUCCEEDED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
