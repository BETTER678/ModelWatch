
import os
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


def retrain_and_validate():

    current_path = os.path.join(
        PROJECT_ROOT, "data", "current.csv"
    )

    holdout_path = os.path.join(
        PROJECT_ROOT, "data", "holdout.csv"
    )

    model_path = os.path.join(
        PROJECT_ROOT, "models", "current_model.pkl"
    )

    current = pd.read_csv(current_path)
    holdout = pd.read_csv(holdout_path)

    X_current = current.drop(columns=["churn"])
    y_current = current["churn"]

    X_holdout = holdout.drop(columns=["churn"])
    y_holdout = holdout["churn"]

    new_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    new_model.fit(X_current, y_current)

    new_predictions = new_model.predict(X_holdout)

    new_accuracy = accuracy_score(
        y_holdout,
        new_predictions
    )

    old_model = joblib.load(model_path)

    old_predictions = old_model.predict(X_holdout)

    old_accuracy = accuracy_score(
        y_holdout,
        old_predictions
    )

    print("\nModelWatch - Model Validation")
    print("=" * 60)
    print(f"Old model accuracy : {old_accuracy:.4f}")
    print(f"New model accuracy : {new_accuracy:.4f}")

    if new_accuracy > old_accuracy:

        joblib.dump(new_model, model_path)

        decision = "PROMOTED"

        print("\nNew model promoted.")
        print("Production model updated.")

    else:

        decision = "REJECTED"

        print("\nNew model rejected.")
        print("Existing model retained.")

    return {
        "old_accuracy": float(old_accuracy),
        "new_accuracy": float(new_accuracy),
        "decision": decision
    }


if __name__ == "__main__":
    retrain_and_validate()