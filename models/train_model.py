import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# Load reference dataset
data = pd.read_csv("data/reference.csv")


# Features and target
X = data.drop(columns=["churn"])
y = data["churn"]


# Split into training and holdout data
X_train, X_holdout, y_train, y_holdout = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# Create model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# Train model
model.fit(
    X_train,
    y_train
)


# Evaluate model
predictions = model.predict(X_holdout)

accuracy = accuracy_score(
    y_holdout,
    predictions
)


# Save model
joblib.dump(
    model,
    "models/current_model.pkl"
)


# Save holdout data
holdout = X_holdout.copy()
holdout["churn"] = y_holdout

holdout.to_csv(
    "data/holdout.csv",
    index=False
)


print("Model trained successfully.")
print(f"Holdout accuracy: {accuracy:.4f}")
print("Model saved to: models/current_model.pkl")
print("Holdout data saved to: data/holdout.csv")