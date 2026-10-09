"""Application entry point for ModelWatch."""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import streamlit as st

from detectors.tabular_drift import (
    detect_tabular_drift,
    calculate_domain_classifier,
    calculate_overall_verdict
)


# -----------------------------------------
# Page configuration
# -----------------------------------------

st.set_page_config(
    page_title="ModelWatch",
    page_icon="📊",
    layout="wide"
)

st.title("ModelWatch")
st.caption("ML Monitoring and Data Drift Dashboard")


# -----------------------------------------
# Load data
# -----------------------------------------

reference_path = os.path.join(
    PROJECT_ROOT, "data", "reference.csv"
)

current_path = os.path.join(
    PROJECT_ROOT, "data", "current.csv"
)

history_path = os.path.join(
    PROJECT_ROOT, "tracking", "pipeline_history.csv"
)

if not os.path.exists(reference_path) or not os.path.exists(current_path):
    st.error("Dataset files are missing. Check data/reference.csv and data/current.csv.")
    st.stop()

reference = pd.read_csv(reference_path)
current = pd.read_csv(current_path)


# -----------------------------------------
# Run drift detection
# -----------------------------------------

results = detect_tabular_drift(
    reference,
    current
)

auc, domain_status = calculate_domain_classifier(
    reference,
    current
)

overall = calculate_overall_verdict(
    results,
    auc
)


# -----------------------------------------
# Summary metrics
# -----------------------------------------

st.subheader("Drift Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Drifted Features",
    f"{overall['drifted_features']} / {overall['total_features']}"
)

col2.metric(
    "Drift Ratio",
    f"{overall['drift_ratio']:.2f}"
)

col3.metric(
    "Domain Classifier AUC",
    f"{auc:.4f}"
)

col4.metric(
    "Severity",
    overall["severity"]
)


# -----------------------------------------
# Overall verdict
# -----------------------------------------

st.subheader("Overall Assessment")

if overall["severity"] == "HIGH":
    st.error(
        f"{overall['verdict']} — {overall['action']}"
    )
elif overall["severity"] == "MODERATE":
    st.warning(
        f"{overall['verdict']} — {overall['action']}"
    )
else:
    st.success(
        f"{overall['verdict']} — {overall['action']}"
    )


# -----------------------------------------
# Feature-level drift results
# -----------------------------------------

st.subheader("Feature-Level Drift")

drift_df = pd.DataFrame.from_dict(
    results,
    orient="index"
)

drift_df.index.name = "Feature"

st.dataframe(
    drift_df,
    use_container_width=True
)


# -----------------------------------------
# Feature distribution comparison
# -----------------------------------------

st.subheader("Feature Distribution Comparison")

feature = st.selectbox(
    "Select a feature",
    list(results.keys())
)

if feature:
    chart_data = pd.DataFrame({
        "Reference": reference[feature],
        "Current": current[feature]
    })

    st.bar_chart(
        chart_data,
        use_container_width=True
    )


# -----------------------------------------
# Pipeline history
# -----------------------------------------

st.subheader("Pipeline History")

if os.path.exists(history_path):

    history = pd.read_csv(history_path)

    if not history.empty:
        st.dataframe(
            history.sort_values(
                "timestamp",
                ascending=False
            ),
            use_container_width=True
        )
    else:
        st.info("No pipeline runs have been logged yet.")

else:
    st.info(
        "Pipeline history not found. Run python pipeline/run_pipeline.py first."
    )


# -----------------------------------------
# Dataset preview
# -----------------------------------------

with st.expander("View Dataset Samples"):

    tab1, tab2 = st.tabs([
        "Reference Data",
        "Current Data"
    ])

    with tab1:
        st.dataframe(
            reference.head(10),
            use_container_width=True
        )

    with tab2:
        st.dataframe(
            current.head(10),
            use_container_width=True
        )


st.caption("ModelWatch | Automated ML Monitoring")