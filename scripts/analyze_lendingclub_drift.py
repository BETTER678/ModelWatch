
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "lendingclub" / "prepared"

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

def load_years(years):
    frames = []

    for year in years:
        path = DATA_DIR / f"loans_{year}.csv"

        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path.name}. Run prepare_lendingclub.py first."
            )

        frames.append(pd.read_csv(path))

    return pd.concat(frames, ignore_index=True)

def calculate_psi(reference, current, bins=10):
    reference = pd.to_numeric(reference, errors="coerce").dropna()
    current = pd.to_numeric(current, errors="coerce").dropna()

    if reference.empty or current.empty:
        return np.nan

    edges = np.unique(
        np.quantile(reference, np.linspace(0, 1, bins + 1))
    )

    if len(edges) < 3:
        return 0.0

    # Extend edge bins to include current-period values.
    edges[0] = -np.inf
    edges[-1] = np.inf

    ref_counts = np.histogram(reference, bins=edges)[0]
    cur_counts = np.histogram(current, bins=edges)[0]

    ref_pct = np.clip(ref_counts / ref_counts.sum(), 1e-6, None)
    cur_pct = np.clip(cur_counts / cur_counts.sum(), 1e-6, None)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))

def main():
    print("=" * 65)
    print("MODELWATCH - LENDINGCLUB DRIFT ANALYSIS")
    print("=" * 65)

    reference = load_years([2012, 2013])
    windows = {
        "2016": load_years([2016]),
        "2018": load_years([2018]),
    }

    print(f"\nReference rows (2012-2013): {len(reference):,}")
    print(f"Reference default rate: {reference['target_default'].mean():.2%}")

    for window_name, current in windows.items():
        print(f"\n{'-' * 65}")
        print(f"CURRENT WINDOW: {window_name}")
        print(f"Rows: {len(current):,}")
        print(f"Default rate: {current['target_default'].mean():.2%}")

        results = []

        for feature in NUMERIC_FEATURES:
            ref_values = reference[feature]
            cur_values = current[feature]

            psi = calculate_psi(ref_values, cur_values)

            ref_clean = pd.to_numeric(ref_values, errors="coerce").dropna()
            cur_clean = pd.to_numeric(cur_values, errors="coerce").dropna()

            if ref_clean.empty or cur_clean.empty:
                statistic, p_value = np.nan, np.nan
            else:
                statistic, p_value = ks_2samp(ref_clean, cur_clean)

            if pd.isna(psi):
                status = "INSUFFICIENT DATA"
            elif psi < 0.1:
                status = "NO DRIFT"
            elif psi < 0.25:
                status = "MODERATE DRIFT"
            else:
                status = "SIGNIFICANT DRIFT"

            results.append({
                "feature": feature,
                "psi": psi,
                "ks_statistic": statistic,
                "ks_p_value": p_value,
                "psi_status": status,
            })

        report = pd.DataFrame(results)
        print("\nFEATURE DRIFT RESULTS")
        print(report.round(4).to_string(index=False))

        drifted = (report["psi_status"] == "SIGNIFICANT DRIFT").sum()
        print(f"\nSignificantly drifted features: {drifted}/{len(report)}")

    print("\nAnalysis complete. No model was trained or replaced.")

if __name__ == "__main__":
    main()