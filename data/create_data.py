import numpy as np
import pandas as pd

np.random.seed(42)

n = 1000

# Reference / training data
reference = pd.DataFrame({
    "age": np.random.normal(35, 10, n).astype(int),
    "monthly_spend": np.random.normal(500, 100, n),
    "tenure_months": np.random.normal(24, 8, n).astype(int),
    "support_calls": np.random.poisson(3, n),
})

reference["churn"] = (
    (reference["support_calls"] > 4) |
    (reference["monthly_spend"] < 400)
).astype(int)

# Current / production data
current = pd.DataFrame({
    "age": np.random.normal(35, 10, n).astype(int),

    # Intentional drift
    "monthly_spend": np.random.normal(650, 120, n),

    # Intentional drift
    "tenure_months": np.random.normal(18, 8, n).astype(int),

    # Intentional drift
    "support_calls": np.random.poisson(5, n),
})

current["churn"] = (
    (current["support_calls"] > 4) |
    (current["monthly_spend"] < 400)
).astype(int)

reference.to_csv("data/reference.csv", index=False)
current.to_csv("data/current.csv", index=False)

print("Datasets created successfully.")
print(f"Reference dataset: {reference.shape}")
print(f"Current dataset: {current.shape}")