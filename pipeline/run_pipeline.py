import sys
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

import pandas as pd

from detectors.tabular_drift import (
    detect_tabular_drift,
    calculate_domain_classifier,
    calculate_overall_verdict
)

from remediation.retrain_model import (
    retrain_and_validate
)


# -----------------------------------------
# Load datasets
# -----------------------------------------

reference = pd.read_csv(
    "data/reference.csv"
)

current = pd.read_csv(
    "data/current.csv"
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

print(
    f"Verdict          : "
    f"{overall['verdict']}"
)

print(
    f"Severity         : "
    f"{overall['severity']}"
)

print(
    f"Action           : "
    f"{overall['action']}"
)


# -----------------------------------------
# Trigger remediation
# -----------------------------------------

if overall["action"] == "RETRAIN MODEL":

    print("\nDrift detected.")
    print("Starting model retraining...")

    result = retrain_and_validate()

    print("\nPipeline completed.")

else:

    print("\nNo retraining required.")