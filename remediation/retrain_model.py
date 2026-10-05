import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def retrain_and_validate():

    # -----------------------------------------
    # Load datasets
    # -----------------------------------------

    current = pd.read_csv(
        "data/current.csv"
    )

    holdout = pd.read_csv(
        "data/holdout.csv"
    )

    # -----------------------------------------
    # Prepare data
    # -----------------------------------------

    X_current = current.drop(
        columns=["churn"]
    )

    y_current = current["churn"]

    X_holdout = holdout.drop(
        columns=["churn"]
    )

    y_holdout = holdout["churn"]

    # -----------------------------------------
    # Train new model
    # -----------------------------------------

    new_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    new_model.fit(
        X_current,
        y_current
    )

    # -----------------------------------------
    # Evaluate new model
    # -----------------------------------------

    new_predictions = new_model.predict(
        X_holdout
    )

    new_accuracy = accuracy_score(
        y_holdout,
        new_predictions
    )

    # -----------------------------------------
    # Load existing model
    # -----------------------------------------

    old_model = joblib.load(
        "models/current_model.pkl"
    )

    old_predictions = old_model.predict(
        X_holdout
    )

    old_accuracy = accuracy_score(
        y_holdout,
        old_predictions
    )

    # -----------------------------------------
    # Validation
    # -----------------------------------------

    print("\nModelWatch - Model Validation")
    print("=" * 60)

    print(
        f"Old model accuracy : {old_accuracy:.4f}"
    )

    print(
        f"New model accuracy : {new_accuracy:.4f}"
    )

    # -----------------------------------------
    # Model promotion
    # -----------------------------------------

    if new_accuracy > old_accuracy:

        joblib.dump(
            new_model,
            "models/current_model.pkl"
        )

        decision = "PROMOTED"

        print("\nNew model promoted.")
        print("Production model updated.")

    else:

        decision = "REJECTED"

        print("\nNew model rejected.")
        print("Existing model retained.")

    return {
        "old_accuracy": old_accuracy,
        "new_accuracy": new_accuracy,
        "decision": decision
    }


if __name__ == "__main__":

    retrain_and_validate()