
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "lendingclub" / "prepared"
MODEL_DIR = PROJECT_ROOT / "models" / "lendingclub"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "target_default"

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


def load_years(years):
    frames = []

    for year in years:
        path = DATA_DIR / f"loans_{year}.csv"

        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path.name}. Run prepare_lendingclub.py first."
            )

        frames.append(pd.read_csv(path))

    data = pd.concat(frames, ignore_index=True)

    missing = set(FEATURES + [TARGET]) - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    return data


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


def evaluate(model, data, period):
    X = data[FEATURES]
    y = data[TARGET].astype(int)

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    print(f"\n{'=' * 55}")
    print(f"EVALUATION: {period}")
    print(f"Rows: {len(data):,}")
    print(f"Default rate: {y.mean():.2%}")
    print(f"ROC-AUC: {roc_auc_score(y, probabilities):.4f}")
    print(f"PR-AUC (Average Precision): {average_precision_score(y, probabilities):.4f}")
    print("\nClassification report:")
    print(classification_report(y, predictions, zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(y, predictions))


def main():
    print("MODELWATCH - LENDINGCLUB MODEL TRAINING")

    # Fit only on the earlier time period.
    train = load_years([2012, 2013])
    print(f"\nTraining rows: {len(train):,}")
    print(f"Training default rate: {train[TARGET].mean():.2%}")

    model = build_model()
    model.fit(train[FEATURES], train[TARGET].astype(int))

    model_path = MODEL_DIR / "baseline_model.joblib"
    joblib.dump(model, model_path)
    print(f"\nSaved model: {model_path}")

    # Evaluate on later, chronologically separated periods.
    for year in [2016, 2018]:
        test = load_years([year])
        evaluate(model, test, str(year))

    print("\nTraining and evaluation complete.")


if __name__ == "__main__":
    main()