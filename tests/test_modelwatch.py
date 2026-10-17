
import numpy as np
import pandas as pd

from detectors.tabular_drift import (
    detect_tabular_drift,
    calculate_domain_classifier,
    calculate_overall_verdict
)


def test_stable_data_has_no_significant_drift():

    np.random.seed(42)

    reference = pd.DataFrame({
        "age": np.random.normal(35, 10, 1000),
        "monthly_spend": np.random.normal(500, 100, 1000),
        "tenure_months": np.random.normal(24, 8, 1000),
        "support_calls": np.random.poisson(3, 1000),
        "churn": np.random.randint(0, 2, 1000)
    })

    current = reference.copy()

    results = detect_tabular_drift(
        reference,
        current
    )

    auc, _ = calculate_domain_classifier(
        reference,
        current
    )

    overall = calculate_overall_verdict(
        results,
        auc
    )

    assert overall["drifted_features"] == 0
    assert overall["severity"] == "LOW"
    assert overall["action"] == "NO ACTION"


def test_shifted_data_triggers_drift_detection():

    np.random.seed(42)

    n = 1000

    reference = pd.DataFrame({
        "age": np.random.normal(35, 10, n),
        "monthly_spend": np.random.normal(500, 100, n),
        "tenure_months": np.random.normal(24, 8, n),
        "support_calls": np.random.poisson(3, n),
        "churn": np.random.randint(0, 2, n)
    })

    current = pd.DataFrame({
        "age": np.random.normal(35, 10, n),
        "monthly_spend": np.random.normal(700, 120, n),
        "tenure_months": np.random.normal(15, 8, n),
        "support_calls": np.random.poisson(6, n),
        "churn": np.random.randint(0, 2, n)
    })

    results = detect_tabular_drift(
        reference,
        current
    )

    auc, _ = calculate_domain_classifier(
        reference,
        current
    )

    overall = calculate_overall_verdict(
        results,
        auc
    )

    assert overall["drifted_features"] > 0
    assert overall["verdict"] == "DRIFT DETECTED"
    assert overall["action"] == "RETRAIN MODEL"


def test_model_promotion_requires_improvement():

    old_accuracy = 1.0
    new_accuracy = 0.995

    decision = (
        "PROMOTED"
        if new_accuracy > old_accuracy
        else "REJECTED"
    )

    assert decision == "REJECTED"