
from pathlib import Path
from collections import Counter
import csv

import pandas as pd

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "lendingclub" / "accepted_loans.csv"

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}\n"
        "Check that the CSV is inside data/lendingclub "
        "and is named accepted_loans.csv."
    )

print("=" * 60)
print("MODELWATCH - LENDINGCLUB DATA INSPECTION")
print("=" * 60)

print(f"\nFile size: {DATA_PATH.stat().st_size / (1024**3):.2f} GB")

# Read only the CSV header first
with open(DATA_PATH, "r", encoding="utf-8-sig", newline="") as file:
    reader = csv.reader(file)
    columns = next(reader)

print(f"\nTotal columns: {len(columns)}")

required = ["loan_status", "issue_d"]
missing = [col for col in required if col not in columns]

if missing:
    print("\nRequired columns missing:", missing)
    print("\nFirst 40 available columns:")
    print(columns[:40])
    raise ValueError(
        "Expected columns were not found. "
        "Check the actual CSV header before proceeding."
    )

print("\nRequired columns found:", required)

# Read only the two required columns in chunks
status_counts = Counter()
year_counts = Counter()
rows_read = 0

for chunk in pd.read_csv(
    DATA_PATH,
    usecols=required,
    chunksize=100_000,
    low_memory=True
):
    rows_read += len(chunk)

    statuses = chunk["loan_status"].fillna("MISSING").astype(str)
    status_counts.update(statuses.value_counts().to_dict())

    # LendingClub issue_d commonly uses formats such as Dec-2018
    dates = pd.to_datetime(
        chunk["issue_d"],
        format="%b-%Y",
        errors="coerce"
    )

    year_counts.update(
        dates.dropna().dt.year.astype(str).value_counts().to_dict()
    )

    if rows_read % 500_000 == 0:
        print(f"Rows inspected: {rows_read:,}")

print(f"\nTotal rows inspected: {rows_read:,}")

print("\n--- LOAN STATUS COUNTS ---")
for status, count in status_counts.most_common():
    print(f"{status}: {count:,}")

print("\n--- LOAN COUNTS BY YEAR ---")
for year, count in sorted(year_counts.items()):
    print(f"{year}: {count:,}")

print("\nInspection complete.")
print("No model training or dataset modification was performed.")