
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT_ROOT / "data" / "lendingclub" / "accepted_loans.csv"
OUTPUT = PROJECT_ROOT / "data" / "lendingclub" / "prepared"

OUTPUT.mkdir(parents=True, exist_ok=True)

if not SOURCE.exists():
    raise FileNotFoundError(f"Dataset not found: {SOURCE}")

# Keep only clear, resolved loan outcomes for this first experiment.
VALID_STATUSES = {
    "Fully Paid": 0,
    "Charged Off": 1,
}

FEATURES = [
    "loan_amnt",
    "int_rate",
    "annual_inc",
    "dti",
    "installment",
    "open_acc",
    "total_acc",
    "revol_bal",
    "revol_util",
    "emp_length",
    "home_ownership",
    "purpose",
    "addr_state",
]

REQUIRED = ["loan_status", "issue_d"] + FEATURES

# Save each year separately to support chronological experiments.
written_years = set()
total_saved = 0

for chunk in pd.read_csv(
    SOURCE,
    usecols=REQUIRED,
    chunksize=100_000,
    low_memory=True,
):
    chunk["issue_date"] = pd.to_datetime(
        chunk["issue_d"],
        format="%b-%Y",
        errors="coerce",
    )

    chunk = chunk[
        chunk["issue_date"].dt.year.between(2012, 2018)
        & chunk["loan_status"].isin(VALID_STATUSES)
    ].copy()

    if chunk.empty:
        continue

    chunk["target_default"] = chunk["loan_status"].map(VALID_STATUSES)
    chunk["year"] = chunk["issue_date"].dt.year

    chunk = chunk.drop(columns=["loan_status", "issue_d"])

    for year, yearly_data in chunk.groupby("year"):
        year = int(year)
        path = OUTPUT / f"loans_{year}.csv"

        yearly_data.to_csv(
            path,
            mode="a" if year in written_years else "w",
            header=year not in written_years,
            index=False,
        )

        written_years.add(year)
        total_saved += len(yearly_data)

print("\nLendingClub preparation complete.")
print(f"Rows saved: {total_saved:,}")
print(f"Years: {sorted(written_years)}")
print(f"Output folder: {OUTPUT}")
