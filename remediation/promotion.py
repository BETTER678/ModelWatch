
from pathlib import Path
from datetime import datetime, timezone
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = ROOT / "models" / "lendingclub"
TRACKING_DIR = ROOT / "tracking" / "lendingclub"

BASELINE_MODEL = MODEL_DIR / "baseline_model.joblib"
CANDIDATE_MODEL = MODEL_DIR / "candidate_model.joblib"
BACKUP_MODEL = MODEL_DIR / "previous_model.joblib"
PROMOTION_RECORD = TRACKING_DIR / "promotion_history.json"

MIN_AP_IMPROVEMENT = 0.01
MAX_ROC_AUC_DROP = 0.01


def load_history():
    if not PROMOTION_RECORD.exists():
        return []

    with open(PROMOTION_RECORD, "r", encoding="utf-8") as file:
        history = json.load(file)

    if not isinstance(history, list):
        raise ValueError("Promotion history must contain a JSON list.")

    return history


def save_history(history):
    TRACKING_DIR.mkdir(parents=True, exist_ok=True)

    temp_path = PROMOTION_RECORD.with_suffix(".tmp")

    with open(temp_path, "w", encoding="utf-8") as file:
        json.dump(history, file, indent=4)

    temp_path.replace(PROMOTION_RECORD)


def validate_metrics(name, metrics):
    required = ("roc_auc", "average_precision")

    for key in required:
        if key not in metrics:
            raise ValueError(f"{name} metrics missing: {key}")

        value = float(metrics[key])

        if not 0 <= value <= 1:
            raise ValueError(
                f"{name} metric '{key}' must be between 0 and 1."
            )


def promote_candidate(
    baseline_metrics,
    candidate_metrics,
    evaluation_metadata,
):
    if not BASELINE_MODEL.exists():
        raise FileNotFoundError(f"Baseline not found: {BASELINE_MODEL}")

    if not CANDIDATE_MODEL.exists():
        raise FileNotFoundError(f"Candidate not found: {CANDIDATE_MODEL}")

    validate_metrics("Baseline", baseline_metrics)
    validate_metrics("Candidate", candidate_metrics)

    required_metadata = [
        "dataset_name",
        "approved_for_promotion",
        "same_dataset_for_both_models",
        "untouched_holdout",
    ]

    for key in required_metadata:
        if key not in evaluation_metadata:
            raise ValueError(f"Evaluation metadata missing: {key}")

    if not isinstance(evaluation_metadata["dataset_name"], str) or not (
        evaluation_metadata["dataset_name"].strip()
    ):
        raise ValueError("Evaluation metadata dataset_name must be non-empty.")

    for key in (
        "approved_for_promotion",
        "same_dataset_for_both_models",
        "untouched_holdout",
    ):
        if not isinstance(evaluation_metadata[key], bool):
            raise ValueError(f"Evaluation metadata '{key}' must be a boolean.")

    ap_change = (
        float(candidate_metrics["average_precision"])
        - float(baseline_metrics["average_precision"])
    )

    auc_change = (
        float(candidate_metrics["roc_auc"])
        - float(baseline_metrics["roc_auc"])
    )

    reasons = []

    if not evaluation_metadata["approved_for_promotion"]:
        reasons.append("Evaluation dataset is not approved for promotion.")

    if not evaluation_metadata["same_dataset_for_both_models"]:
        reasons.append("Models were not evaluated on the same dataset.")

    if not evaluation_metadata["untouched_holdout"]:
        reasons.append("Evaluation dataset is not an untouched holdout.")

    if ap_change < MIN_AP_IMPROVEMENT:
        reasons.append("Average Precision improvement is below threshold.")

    if auc_change < -MAX_ROC_AUC_DROP:
        reasons.append("ROC-AUC dropped beyond the allowed threshold.")

    approved = len(reasons) == 0
    promoted = False

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "evaluation_metadata": evaluation_metadata,
        "baseline_metrics": baseline_metrics,
        "candidate_metrics": candidate_metrics,
        "average_precision_change": round(ap_change, 6),
        "roc_auc_change": round(auc_change, 6),
        "thresholds": {
            "min_average_precision_improvement": MIN_AP_IMPROVEMENT,
            "max_roc_auc_drop": MAX_ROC_AUC_DROP,
        },
        "decision": "APPROVED" if approved else "REJECTED",
        "reason": (
            "All promotion checks passed."
            if approved
            else " ".join(reasons)
        ),
        "model_promoted": False,
    }

    if approved:
        # Keep a backup of the current baseline.
        shutil.copy2(BASELINE_MODEL, BACKUP_MODEL)

        try:
            shutil.copy2(CANDIDATE_MODEL, BASELINE_MODEL)
            promoted = True
        except OSError:
            # Restore the baseline if copying the candidate fails.
            shutil.copy2(BACKUP_MODEL, BASELINE_MODEL)
            raise

        record["model_promoted"] = promoted

    history = load_history()
    history.append(record)
    save_history(history)

    print("\nMODELWATCH PROMOTION RESULT")
    print("=" * 35)
    print(f"Decision: {record['decision']}")
    print(f"Reason: {record['reason']}")
    print(f"Average Precision change: {ap_change:+.4f}")
    print(f"ROC-AUC change: {auc_change:+.4f}")
    print(f"Model promoted: {promoted}")
    print(f"History saved to: {PROMOTION_RECORD}")

    return record


if __name__ == "__main__":
    print("ModelWatch promotion safety module")
    print("Import promote_candidate() to evaluate a candidate.")