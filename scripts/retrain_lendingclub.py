
from pathlib import Path
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "lendingclub" / "prepared"
MODEL_DIR = ROOT / "models" / "lendingclub"

NUMERIC_FEATURES = [
    "loan_amnt",
    "int_rate",
    "annual_inc",
    "dti",
    "installment",
    "open_acc",
    "total_acc",
    "revol_bal",
    "revol_util",
]

CATEGORICAL_FEATURES = [
    "emp_length",
    "home_ownership",
    "purpose",
    "addr_state",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "target_default"


def load_years(years):
    frames = []

    for year in years:
        path = DATA_DIR / f"loans_{year}.csv"

        if not path.exists():
            raise FileNotFoundError(f"Missing dataset: {path}")

        frames.append(pd.read_csv(path))

    return pd.concat(frames, ignore_index=True)


def build_model():
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessing = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ])

    classifier = RandomForestClassifier(
        n_estimators=150,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", classifier),
    ])


def evaluate(model, data, name):
    X = data[FEATURES]
    y = data[TARGET]

    probabilities = model.predict_proba(X)[:, 1]

    roc_auc = roc_auc_score(y, probabilities)
    avg_precision = average_precision_score(y, probabilities)

    print(f"\n{name}")
    print("-" * 40)
    print(f"Rows: {len(data):,}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"Average Precision: {avg_precision:.4f}")

    return {
        "roc_auc": roc_auc,
        "average_precision": avg_precision,
    }


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Use the same 2016 evaluation data for both models.
    training_data = load_years([2014, 2015])
    validation_data = load_years([2016])

    X_train = training_data[FEATURES]
    y_train = training_data[TARGET]

    print("MODELWATCH - CANDIDATE RETRAINING")
    print("=" * 45)
    print(f"Candidate training rows: {len(training_data):,}")
    print(f"Training default rate: {y_train.mean():.2%}")
    print(f"Validation rows: {len(validation_data):,}")

    candidate = build_model()
    candidate.fit(X_train, y_train)

    candidate_path = MODEL_DIR / "candidate_model.joblib"
    joblib.dump(candidate, candidate_path)

    print(f"\nCandidate saved to: {candidate_path}")

    # Evaluate the existing baseline on the same validation data.
    baseline_path = MODEL_DIR / "baseline_model.joblib"

    if not baseline_path.exists():
        raise FileNotFoundError(
            f"Baseline model not found: {baseline_path}"
        )

    baseline = joblib.load(baseline_path)

    baseline_metrics = evaluate(
        baseline, validation_data, "BASELINE: trained on 2012-2013"
    )

    candidate_metrics = evaluate(
        candidate, validation_data, "CANDIDATE: trained on 2014-2015"
    )

    print("\nCOMPARISON")
    print("-" * 40)

    ap_change = (
        candidate_metrics["average_precision"]
        - baseline_metrics["average_precision"]
    )
    auc_change = (
        candidate_metrics["roc_auc"]
        - baseline_metrics["roc_auc"]
    )

    print(f"Average Precision change: {ap_change:+.4f}")
    print(f"ROC-AUC change: {auc_change:+.4f}")

    # Conservative example rule. This is a configurable project
    # policy, not a universal model-promotion standard.
    promote_candidate = (
        ap_change >= 0.01
        and auc_change >= -0.01
    )

    if promote_candidate:
        print("\nDECISION: CANDIDATE PASSES THE EXAMPLE RULE")
        print("Review validation quality before promoting it.")
    else:
        print("\nDECISION: DO NOT PROMOTE AUTOMATICALLY")
        print("Keep the current baseline pending further review.")

    print("\nThe baseline model was not overwritten.")
    print("The 2016 data was used for model selection,")
    print("so use a separate future holdout for final evaluation.")


if __name__ == "__main__":
    main()