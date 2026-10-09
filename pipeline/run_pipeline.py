
import sys
import os

# Add project root to Python path
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(0, PROJECT_ROOT)

import pandas as pd

from detectors.tabular_drift import (
    detect_tabular_drift,
    calculate_domain_classifier,
    calculate_overall_verdict
)

from remediation.retrain_model import (
    retrain_and_validate
)

from tracking.mlflow_tracker import (
    log_pipeline_run
)

from tracking.run_logger import (
    log_pipeline_result
)


def run_pipeline():

    # -----------------------------------------
    # Load datasets
    # -----------------------------------------

    reference = pd.read_csv(
        os.path.join(PROJECT_ROOT, "data", "reference.csv")
    )

    current = pd.read_csv(
        os.path.join(PROJECT_ROOT, "data", "current.csv")
    )

    # -----------------------------------------
    # Detect drift
    # -----------------------------------------

    results = detect_tabular_drift(
        reference,
        current
    )

    auc, domain_status = calculate_domain_classifier(
        reference,
        current
    )

    overall = calculate_overall_verdict(
        results,
        auc
    )

    # -----------------------------------------
    # Display results
    # -----------------------------------------

    print("\nModelWatch - Automated Pipeline")
    print("=" * 65)

    print(
        f"Drifted features : "
        f"{overall['drifted_features']} / "
        f"{overall['total_features']}"
    )

    print(
        f"Drift ratio      : "
        f"{overall['drift_ratio']:.2f}"
    )

    print(
        f"Domain AUC       : "
        f"{overall['auc']:.4f}"
    )

    print(f"Verdict          : {overall['verdict']}")
    print(f"Severity         : {overall['severity']}")
    print(f"Action           : {overall['action']}")

    # -----------------------------------------
    # Retraining and validation
    # -----------------------------------------

    result = {
        "old_accuracy": None,
        "new_accuracy": None,
        "decision": "NOT_RUN"
    }

    if overall["action"] == "RETRAIN MODEL":

        print("\nDrift detected.")
        print("Starting model retraining...")

        result = retrain_and_validate()

    else:

        print("\nNo retraining required.")

    # -----------------------------------------
    # MLflow tracking
    # -----------------------------------------

    log_pipeline_run(
        drift_ratio=overall["drift_ratio"],
        domain_auc=overall["auc"],
        severity=overall["severity"],
        old_accuracy=result["old_accuracy"],
        new_accuracy=result["new_accuracy"],
        decision=result["decision"]
    )

    # -----------------------------------------
    # CSV logging
    # -----------------------------------------

    log_pipeline_result(
        drift_ratio=overall["drift_ratio"],
        domain_auc=overall["auc"],
        severity=overall["severity"],
        action=overall["action"],
        old_accuracy=result["old_accuracy"],
        new_accuracy=result["new_accuracy"],
        decision=result["decision"]
    )

    print("\nPipeline completed.")


if __name__ == "__main__":
    run_pipeline()