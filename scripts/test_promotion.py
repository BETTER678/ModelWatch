
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from remediation.promotion import promote_candidate


def main():
    # Existing baseline evaluation metrics
    baseline_metrics = {
        "roc_auc": 0.6958,
        "average_precision": 0.3991,
    }

    # Candidate evaluation metrics
    candidate_metrics = {
        "roc_auc": 0.7078,
        "average_precision": 0.4136,
    }

    # The 2016 validation dataset was already used for model selection.
    # Therefore, it must not be treated as an untouched holdout.
    evaluation_metadata = {
        "dataset_name": "LendingClub 2016 validation",
        "approved_for_promotion": False,
        "same_dataset_for_both_models": True,
        "untouched_holdout": False,
    }

    print("MODELWATCH PROMOTION SAFETY TEST")
    print("=" * 40)

    print("\nBaseline metrics:")
    print(f"ROC-AUC: {baseline_metrics['roc_auc']:.4f}")
    print(
        "Average Precision: "
        f"{baseline_metrics['average_precision']:.4f}"
    )

    print("\nCandidate metrics:")
    print(f"ROC-AUC: {candidate_metrics['roc_auc']:.4f}")
    print(
        "Average Precision: "
        f"{candidate_metrics['average_precision']:.4f}"
    )

    print("\nEvaluation metadata:")
    for key, value in evaluation_metadata.items():
        print(f"{key}: {value}")

    print("\nRunning promotion safety checks...")

    result = promote_candidate(
        baseline_metrics=baseline_metrics,
        candidate_metrics=candidate_metrics,
        evaluation_metadata=evaluation_metadata,
    )

    assert result["decision"] == "REJECTED", (
        "Test failed: unsuitable evaluation was not rejected."
    )

    assert result["model_promoted"] is False, (
        "Test failed: model was promoted using an unsuitable holdout."
    )

    print("\nTEST RESULT: PASSED")
    print("Unsuitable evaluation was rejected.")
    print("The baseline was not replaced by this test.")


if __name__ == "__main__":
    main()