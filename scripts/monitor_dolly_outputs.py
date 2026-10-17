import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from monitoring.ollama_client import OllamaError, generate

DATA_PATH = ROOT / "data" / "llm" / "databricks-dolly-15k.jsonl"
REPORT_DIR = ROOT / "tracking" / "llm"
OUTPUT_DIR = ROOT / "data" / "llm" / "prepared"


def load_examples(limit, seed=42):
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dolly dataset not found: {DATA_PATH}")

    examples = []
    with DATA_PATH.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                example = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid Dolly JSON on line {line_number}: {error}"
                ) from error
            if example.get("instruction") and example.get("response"):
                examples.append(example)

    total_examples = len(examples)
    if limit > 0 and len(examples) > limit:
        examples = random.Random(seed).sample(examples, limit)
    return examples, total_examples


def word_f1(expected, actual):
    expected_words = re.findall(r"\w+", expected.casefold())
    actual_words = re.findall(r"\w+", actual.casefold())
    if not expected_words or not actual_words:
        return 0.0

    from collections import Counter

    overlap = sum((Counter(expected_words) & Counter(actual_words)).values())
    precision = overlap / len(actual_words)
    recall = overlap / len(expected_words)
    return 2 * precision * recall / (precision + recall) if overlap else 0.0


def build_prompt(example, variant):
    instruction = example["instruction"].strip()
    context = example.get("context", "").strip()
    if variant == "baseline":
        return f"{instruction}\n\n{context}" if context else instruction

    context_section = (
        f"Use this context when it is relevant:\n{context}\n\n"
        if context
        else ""
    )
    return (
        "Answer the instruction accurately and concisely. "
        "Do not invent facts. If the context does not provide enough "
        "information, say so.\n\n"
        f"{context_section}Instruction: {instruction}\nAnswer:"
    )


def evaluate_variant(example, model, variant):
    started = time.perf_counter()
    try:
        response, metadata = generate(
            model,
            build_prompt(example, variant),
        )
        latency = time.perf_counter() - started
        expected = example["response"]
        words = len(response.split())
        return {
            "variant": variant,
            "response": response,
            "latency_seconds": round(latency, 4),
            "prompt_tokens": metadata.get("prompt_eval_count"),
            "completion_tokens": metadata.get("eval_count"),
            "reference_word_f1": round(word_f1(expected, response), 4),
            "nonempty": bool(response.strip()),
            "within_200_words": words <= 200,
            "error": None,
        }
    except OllamaError as error:
        return {
            "variant": variant,
            "response": None,
            "latency_seconds": round(time.perf_counter() - started, 4),
            "prompt_tokens": None,
            "completion_tokens": None,
            "reference_word_f1": None,
            "nonempty": False,
            "within_200_words": False,
            "error": str(error),
        }


def summarize(results):
    variants = {}
    for name in ("baseline", "guarded"):
        rows = [row[name] for row in results]
        successful = [row for row in rows if row["error"] is None]
        variants[name] = {
            "attempts": len(rows),
            "successful": len(successful),
            "errors": len(rows) - len(successful),
            "mean_latency_seconds": (
                sum(row["latency_seconds"] for row in successful)
                / len(successful)
                if successful
                else None
            ),
            "mean_reference_word_f1": (
                sum(row["reference_word_f1"] for row in successful)
                / len(successful)
                if successful
                else None
            ),
            "nonempty_rate": (
                sum(row["nonempty"] for row in successful) / len(successful)
                if successful
                else None
            ),
            "within_200_words_rate": (
                sum(row["within_200_words"] for row in successful)
                / len(successful)
                if successful
                else None
            ),
            "prompt_tokens": sum(
                row["prompt_tokens"] or 0 for row in successful
            ),
            "completion_tokens": sum(
                row["completion_tokens"] or 0 for row in successful
            ),
        }
    return variants


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate local Ollama outputs on a fixed Dolly prompt sample."
    )
    parser.add_argument("--limit", type=int, default=25,
                        help="Number of prompts; 0 processes the whole dataset.")
    parser.add_argument("--model", default="llama3.2:1b")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be >= 0")

    examples, total_examples = load_examples(args.limit)
    if not examples:
        raise ValueError("No usable Dolly examples were found.")
    print(
        f"Dolly examples available: {total_examples:,}; "
        f"selected: {len(examples):,}"
    )
    print(f"Model: {args.model}; evaluating two prompt variants per example.")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"ollama_outputs_{run_id}.jsonl"
    results = []

    with output_path.open("w", encoding="utf-8") as output_file:
        for index, example in enumerate(examples, start=1):
            result = {
                "run_id": run_id,
                "example_index": index,
                "category": example.get("category"),
                "baseline": evaluate_variant(example, args.model, "baseline"),
                "guarded": evaluate_variant(example, args.model, "guarded"),
            }
            results.append(result)
            output_file.write(json.dumps(result, ensure_ascii=False) + "\n")
            output_file.flush()
            print(f"Evaluated {index:,}/{len(examples):,}", flush=True)

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": DATA_PATH.name,
        "dataset_rows_available": total_examples,
        "rows_evaluated": len(examples),
        "model": args.model,
        "evaluation": "paired baseline-vs-guarded prompt; temperature=0",
        "variants": summarize(results),
        "raw_outputs": str(output_path.relative_to(ROOT)),
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / "dolly_monitor_latest.json"
    report_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print("\nDOLLY LLM OUTPUT MONITOR")
    print(json.dumps(record, indent=2))
    print(f"\nSummary saved to: {report_path}")
    print(f"Responses saved to: {output_path}")

    if all(summary["successful"] == 0 for summary in record["variants"].values()):
        raise OllamaError("No Dolly prompts were successfully evaluated.")


if __name__ == "__main__":
    main()
