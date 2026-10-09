
import csv
import os
from datetime import datetime


LOG_FILE = "tracking/pipeline_history.csv"


def log_pipeline_result(
    drift_ratio,
    domain_auc,
    severity,
    action,
    old_accuracy=None,
    new_accuracy=None,
    decision="NOT_RUN"
):
    os.makedirs("tracking", exist_ok=True)

    file_exists = os.path.isfile(LOG_FILE)

    with open(LOG_FILE, "a", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "timestamp",
                "drift_ratio",
                "domain_auc",
                "severity",
                "action",
                "old_accuracy",
                "new_accuracy",
                "decision"
            ]
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow({
            "timestamp": datetime.now().isoformat(
                timespec="seconds"
            ),
            "drift_ratio": drift_ratio,
            "domain_auc": domain_auc,
            "severity": severity,
            "action": action,
            "old_accuracy": old_accuracy,
            "new_accuracy": new_accuracy,
            "decision": decision
        })

    print("Pipeline result saved to:", LOG_FILE)