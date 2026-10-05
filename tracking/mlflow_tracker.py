import mlflow


def log_pipeline_run(
    drift_ratio,
    domain_auc,
    severity,
    old_accuracy=None,
    new_accuracy=None,
    decision=None
):

    mlflow.set_experiment(
        "ModelWatch"
    )

    with mlflow.start_run():

        # Drift metrics
        mlflow.log_metric(
            "drift_ratio",
            drift_ratio
        )

        mlflow.log_metric(
            "domain_auc",
            domain_auc
        )

        # Drift information
        mlflow.set_tag(
            "severity",
            severity
        )

        # Model validation
        if old_accuracy is not None:

            mlflow.log_metric(
                "old_accuracy",
                old_accuracy
            )

        if new_accuracy is not None:

            mlflow.log_metric(
                "new_accuracy",
                new_accuracy
            )

        if decision is not None:

            mlflow.set_tag(
                "model_decision",
                decision
            )

        print("\nMLflow tracking completed.")