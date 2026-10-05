import numpy as np
import pandas as pd

from scipy.stats import ks_2samp

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


def calculate_psi(reference, current, bins=10):
    reference = np.asarray(reference)
    current = np.asarray(current)

    breakpoints = np.percentile(
        reference,
        np.linspace(0, 100, bins + 1)
    )

    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    reference_counts = np.histogram(
        reference,
        bins=breakpoints
    )[0]

    current_counts = np.histogram(
        current,
        bins=breakpoints
    )[0]

    reference_percent = reference_counts / len(reference)
    current_percent = current_counts / len(current)

    reference_percent = np.where(
        reference_percent == 0,
        0.0001,
        reference_percent
    )

    current_percent = np.where(
        current_percent == 0,
        0.0001,
        current_percent
    )

    psi = np.sum(
        (current_percent - reference_percent)
        * np.log(current_percent / reference_percent)
    )

    return psi


def calculate_ks(reference, current):
    statistic, p_value = ks_2samp(
        reference,
        current
    )

    return statistic, p_value


def calculate_domain_classifier(reference_df, current_df):

    reference = reference_df.copy()
    current = current_df.copy()

    reference["dataset"] = 0
    current["dataset"] = 1

    combined = pd.concat(
        [reference, current],
        ignore_index=True
    )

    X = combined.drop(
        columns=["dataset", "churn"]
    )

    y = combined["dataset"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    auc = roc_auc_score(
        y_test,
        probabilities
    )

    if auc < 0.60:
        status = "NO DRIFT"
    elif auc < 0.75:
        status = "MODERATE DRIFT"
    else:
        status = "SIGNIFICANT DRIFT"

    return auc, status


def detect_tabular_drift(reference_df, current_df):

    results = {}

    features = reference_df.select_dtypes(
        include=np.number
    ).columns

    features = [
        feature
        for feature in features
        if feature != "churn"
    ]

    for feature in features:

        psi = calculate_psi(
            reference_df[feature],
            current_df[feature]
        )

        ks_statistic, p_value = calculate_ks(
            reference_df[feature],
            current_df[feature]
        )

        if psi < 0.10:
            psi_status = "NO DRIFT"
        elif psi < 0.25:
            psi_status = "MODERATE DRIFT"
        else:
            psi_status = "SIGNIFICANT DRIFT"

        if p_value < 0.05:
            ks_status = "DRIFT"
        else:
            ks_status = "NO DRIFT"

        results[feature] = {
            "psi": round(psi, 4),
            "psi_status": psi_status,
            "ks_statistic": round(ks_statistic, 4),
            "ks_p_value": round(p_value, 4),
            "ks_status": ks_status
        }

    return results


def calculate_overall_verdict(results, auc):

    total_features = len(results)

    drifted_features = sum(
        1
        for result in results.values()
        if (
            result["psi_status"] != "NO DRIFT"
            or result["ks_status"] == "DRIFT"
        )
    )

    drift_ratio = drifted_features / total_features

    if drift_ratio >= 0.50 or auc >= 0.75:
        verdict = "DRIFT DETECTED"
        severity = "HIGH"
        action = "RETRAIN MODEL"

    elif drift_ratio > 0 or auc >= 0.60:
        verdict = "DRIFT DETECTED"
        severity = "MODERATE"
        action = "MONITOR CLOSELY"

    else:
        verdict = "NO SIGNIFICANT DRIFT"
        severity = "LOW"
        action = "NO ACTION"

    return {
        "total_features": total_features,
        "drifted_features": drifted_features,
        "drift_ratio": round(drift_ratio, 4),
        "auc": round(auc, 4),
        "verdict": verdict,
        "severity": severity,
        "action": action
    }


if __name__ == "__main__":

    reference = pd.read_csv(
        "data/reference.csv"
    )

    current = pd.read_csv(
        "data/current.csv"
    )

    # -----------------------------------------
    # PSI + KS TEST
    # -----------------------------------------

    results = detect_tabular_drift(
        reference,
        current
    )

    print("\nModelWatch - Tabular Drift Detection")
    print("=" * 65)

    for feature, result in results.items():

        print(f"\nFeature: {feature}")

        print(
            f"PSI          : {result['psi']:.4f}"
            f" -> {result['psi_status']}"
        )

        print(
            f"KS Statistic : {result['ks_statistic']:.4f}"
        )

        print(
            f"KS P-value   : {result['ks_p_value']:.4f}"
            f" -> {result['ks_status']}"
        )

    # -----------------------------------------
    # DOMAIN CLASSIFIER
    # -----------------------------------------

    auc, domain_status = calculate_domain_classifier(
        reference,
        current
    )

    print("\n" + "=" * 65)
    print("Domain Classifier")
    print("=" * 65)

    print(f"AUC    : {auc:.4f}")
    print(f"Status : {domain_status}")

    # -----------------------------------------
    # OVERALL VERDICT
    # -----------------------------------------

    overall = calculate_overall_verdict(
        results,
        auc
    )

    print("\n" + "=" * 65)
    print("ModelWatch - Overall Drift Assessment")
    print("=" * 65)

    print(
        f"Features with drift : "
        f"{overall['drifted_features']} / "
        f"{overall['total_features']}"
    )

    print(
        f"Drift ratio         : "
        f"{overall['drift_ratio']:.2f}"
    )

    print(
        f"Domain classifier   : "
        f"{overall['auc']:.4f}"
    )

    print(
        f"Overall verdict     : "
        f"{overall['verdict']}"
    )

    print(
        f"Severity            : "
        f"{overall['severity']}"
    )

    print(
        f"Recommended action  : "
        f"{overall['action']}"
    )