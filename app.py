"""Streamlit dashboard for ModelWatch's tabular, embedding, and LLM monitors."""

import json
from pathlib import Path
import re
import subprocess
import sys

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
LENDINGCLUB_RUNS = PROJECT_ROOT / "tracking" / "lendingclub" / "pipeline_runs.jsonl"
ALL_LAYER_RUNS = PROJECT_ROOT / "tracking" / "all_layers_runs.jsonl"
NEWS_REPORT = PROJECT_ROOT / "tracking" / "news" / "embedding_drift_latest.json"
DOLLY_REPORT = PROJECT_ROOT / "tracking" / "llm" / "dolly_monitor_latest.json"


@st.cache_data(ttl=30, max_entries=8)
def load_json_report(path_string):
    path = Path(path_string)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data(ttl=30, max_entries=8)
def load_jsonl(path_string):
    path = Path(path_string)
    if not path.exists():
        return []

    records = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {path.name} on line {line_number}: {error}"
                ) from error
    return records


def latest_lendingclub_metrics(records):
    for run in reversed(records):
        drift_output = ""
        candidate_output = ""
        for stage in run.get("stages", []):
            script = stage.get("script", "")
            if script.endswith("analyze_lendingclub_drift.py"):
                drift_output = stage.get("stdout", "")
            elif script.endswith("retrain_lendingclub.py"):
                candidate_output = stage.get("stdout", "")

        if drift_output or candidate_output:
            return run, parse_lendingclub_drift(drift_output), parse_candidate_metrics(
                candidate_output
            )
    return None, [], None


def parse_lendingclub_drift(output):
    windows = []
    current_year = None

    for line in output.splitlines():
        window_match = re.search(r"CURRENT WINDOW:\s*(\d{4})", line)
        if window_match:
            current_year = window_match.group(1)
            continue

        if current_year is None:
            continue

        parts = line.split()
        if len(parts) >= 5 and parts[1].lower() != "psi":
            try:
                psi, ks_statistic, p_value = map(float, parts[1:4])
            except ValueError:
                continue
            windows.append({
                "Window": current_year,
                "Feature": parts[0],
                "PSI": psi,
                "KS statistic": ks_statistic,
                "KS p-value": p_value,
                "PSI status": " ".join(parts[4:]),
            })
    return windows


def parse_candidate_metrics(output):
    match = re.search(
        r"BASELINE:.*?ROC-AUC:\s*([\d.]+).*?Average Precision:\s*([\d.]+)"
        r".*?CANDIDATE:.*?ROC-AUC:\s*([\d.]+).*?Average Precision:\s*([\d.]+)",
        output,
        flags=re.DOTALL,
    )
    if not match:
        return None
    baseline_auc, baseline_ap, candidate_auc, candidate_ap = map(
        float, match.groups()
    )
    return pd.DataFrame([
        {"Model": "Champion/baseline", "ROC-AUC": baseline_auc,
         "Average precision": baseline_ap},
        {"Model": "Candidate", "ROC-AUC": candidate_auc,
         "Average precision": candidate_ap},
    ])


def show_timestamp(value):
    st.caption(f"Report timestamp: {value}" if value else "Timestamp unavailable")


st.set_page_config(
    page_title="ModelWatch",
    page_icon="📊",
    layout="wide",
)

st.title("ModelWatch")
st.caption(
    "Monitor LendingClub tabular drift, News embedding drift, and Dolly LLM outputs."
)

try:
    lendingclub_runs = load_jsonl(str(LENDINGCLUB_RUNS))
    all_layer_runs = load_jsonl(str(ALL_LAYER_RUNS))
    news_report = load_json_report(str(NEWS_REPORT))
    dolly_report = load_json_report(str(DOLLY_REPORT))
except (OSError, ValueError, json.JSONDecodeError) as error:
    st.error(f"Could not load a monitoring report: {error}")
    st.stop()

latest_lc_run, lendingclub_drift, candidate_metrics = latest_lendingclub_metrics(
    lendingclub_runs
)

overview_tab, lendingclub_tab, news_tab, dolly_tab = st.tabs([
    "Overview",
    "LendingClub",
    "News embeddings",
    "Dolly LLM",
])

with overview_tab:
    st.subheader("Latest monitoring status")
    status_columns = st.columns(3)
    layer_status = [
        ("LendingClub", latest_lc_run.get("status") if latest_lc_run else None),
        ("News embeddings", "REPORT READY" if news_report else None),
        ("Dolly LLM", "REPORT READY" if dolly_report else None),
    ]
    for column, (layer, status) in zip(status_columns, layer_status):
        with column:
            with st.container(border=True):
                st.markdown(f"**{layer}**")
                if status:
                    st.metric("Status", status)
                else:
                    st.metric("Status", "NOT RUN")

    st.subheader("All-layers run history")
    if all_layer_runs:
        history = pd.DataFrame([
            {
                "Run ID": record.get("run_id"),
                "Started": record.get("started_at"),
                "Finished": record.get("finished_at"),
                "Status": record.get("status"),
                "News limit": record.get("dataset_limits", {}).get("news"),
                "Dolly limit": record.get("dataset_limits", {}).get("dolly"),
            }
            for record in reversed(all_layer_runs[-20:])
        ])
        st.dataframe(history, hide_index=True, alt="Recent runs across all monitoring layers")

        most_recent = all_layer_runs[-1]
        if most_recent.get("stages"):
            st.markdown("**Latest run stages**")
            st.dataframe(
                pd.DataFrame(most_recent["stages"]),
                hide_index=True,
                alt="Status and outcome for each latest pipeline stage",
            )
    else:
        st.info("No all-layers runs have been recorded yet.")

    st.subheader("Run all three monitoring layers")
    st.write(
        "The run creates a LendingClub candidate for evaluation but does not "
        "automatically replace or promote the champion."
    )
    with st.form("all_layers_controls"):
        news_limit = st.number_input(
            "News headlines to embed (0 means all rows)",
            min_value=0,
            value=1000,
            step=100,
        )
        dolly_limit = st.number_input(
            "Dolly prompts to evaluate (0 means all rows)",
            min_value=0,
            value=25,
            step=5,
        )
        embedding_model = st.text_input(
            "Ollama embedding model",
            value="nomic-embed-text",
        )
        llm_model = st.text_input(
            "Ollama generation model",
            value="llama3.2:1b",
        )
        run_pipeline = st.form_submit_button(
            "Run all layers",
            type="primary",
        )

    if run_pipeline:
        pipeline_path = PROJECT_ROOT / "pipeline" / "run_all_layers.py"
        command = [
            sys.executable,
            str(pipeline_path),
            "--news-limit",
            str(int(news_limit)),
            "--dolly-limit",
            str(int(dolly_limit)),
            "--embedding-model",
            embedding_model.strip(),
            "--llm-model",
            llm_model.strip(),
        ]
        with st.spinner(
            "Running LendingClub, News embeddings, and Dolly monitoring. "
            "Text generation may take several minutes..."
        ):
            try:
                process = subprocess.run(
                    command,
                    cwd=PROJECT_ROOT,
                    capture_output=True,
                    text=True,
                    check=False,
                )
            except OSError as error:
                st.error(f"Could not start the all-layers pipeline: {error}")
            else:
                load_json_report.clear()
                load_jsonl.clear()
                if process.returncode == 0:
                    st.success("All three monitoring layers completed.")
                else:
                    st.error(
                        f"The all-layers pipeline failed "
                        f"(exit code {process.returncode})."
                    )
                if process.stdout:
                    st.code(process.stdout, language="text")
                if process.stderr:
                    st.code(process.stderr, language="text")
                st.info("Refresh the page to load the reports and run history.")

with lendingclub_tab:
    st.subheader("LendingClub: tabular drift and candidate evaluation")
    st.caption(
        "Drift compares loans from 2012–2013 with the 2016 and 2018 windows. "
        "The supervised candidate comparison uses the 2016 validation window "
        "and is not an independent promotion holdout."
    )

    if latest_lc_run:
        show_timestamp(latest_lc_run.get("finished_at") or latest_lc_run.get("started_at"))
        if lendingclub_drift:
            drift_frame = pd.DataFrame(lendingclub_drift)
            st.dataframe(
                drift_frame,
                hide_index=True,
                alt="LendingClub feature drift by evaluation window",
            )
        else:
            st.info(
                "No structured LendingClub drift output was found in the latest "
                "recorded run."
            )

        if candidate_metrics is not None:
            st.markdown("**Baseline and candidate metrics**")
            st.dataframe(
                candidate_metrics,
                hide_index=True,
                alt="LendingClub baseline and candidate model metrics",
            )

        with st.expander("View latest LendingClub pipeline details"):
            st.json(latest_lc_run)
    else:
        st.info("No LendingClub pipeline run has been recorded yet.")

with news_tab:
    st.subheader("News: text embedding drift")
    if not news_report:
        st.info("No News embedding report found. Run all layers to generate one.")
    else:
        show_timestamp(news_report.get("timestamp"))
        period_text = news_report.get("reference_period", {})
        period_current = news_report.get("current_period", {})

        with st.container(horizontal=True):
            st.metric("Headlines analyzed", f"{news_report.get('rows_analyzed', 0):,}")
            st.metric("Embedding dimensions", news_report.get("embedding_dimensions", "—"))
            st.metric(
                "Embedding norm PSI",
                f"{news_report.get('embedding_norm', {}).get('psi', float('nan')):.4f}",
            )
            st.metric(
                "Category total variation",
                f"{news_report.get('category_total_variation', float('nan')):.3f}",
            )

        st.write(
            f"**Reference:** {period_text.get('start', '—')} to "
            f"{period_text.get('end', '—')}  \n"
            f"**Current:** {period_current.get('start', '—')} to "
            f"{period_current.get('end', '—')}"
        )
        component_rows = news_report.get("pca_component_drift", [])
        if component_rows:
            st.markdown("**Principal-component drift**")
            st.dataframe(
                pd.DataFrame(component_rows),
                hide_index=True,
                alt="KS and PSI drift metrics for News embedding components",
            )
        reference_chars = news_report.get("headline_characters", {}).get(
            "reference_mean"
        )
        current_chars = news_report.get("headline_characters", {}).get(
            "current_mean"
        )
        if reference_chars is not None and current_chars is not None:
            st.caption(
                f"Mean headline length: {reference_chars:.1f} characters "
                f"(reference) → {current_chars:.1f} (current)."
            )
        st.caption(
            "Embedding and category shifts are monitoring signals; they do not "
            "by themselves establish downstream model degradation."
        )

with dolly_tab:
    st.subheader("Dolly: local LLM output monitoring")
    if not dolly_report:
        st.info("No Dolly monitoring report found. Run all layers to generate one.")
    else:
        show_timestamp(dolly_report.get("timestamp"))
        st.write(
            f"Dataset: **{dolly_report.get('dataset', '—')}** · "
            f"Rows evaluated: **{dolly_report.get('rows_evaluated', 0):,}** · "
            f"Model: **{dolly_report.get('model', '—')}**"
        )
        variants = dolly_report.get("variants", {})
        variant_rows = []
        for variant_name in ("baseline", "guarded"):
            metrics = variants.get(variant_name)
            if metrics:
                variant_rows.append({
                    "Prompt variant": variant_name.title(),
                    "Successful": metrics.get("successful"),
                    "Errors": metrics.get("errors"),
                    "Mean latency (s)": metrics.get("mean_latency_seconds"),
                    "Reference word F1": metrics.get("mean_reference_word_f1"),
                    "Nonempty rate": metrics.get("nonempty_rate"),
                    "Under 200 words": metrics.get("within_200_words_rate"),
                    "Prompt tokens": metrics.get("prompt_tokens"),
                    "Completion tokens": metrics.get("completion_tokens"),
                })
        if variant_rows:
            st.dataframe(
                pd.DataFrame(variant_rows),
                hide_index=True,
                alt="Latency, lexical overlap, response format, and token metrics by prompt variant",
            )
        st.caption(
            "Reference word F1 is a lexical proxy, not a human quality judgment. "
            "Review generated outputs before drawing quality conclusions."
        )
        if dolly_report.get("raw_outputs"):
            st.caption(f"Raw paired outputs: {dolly_report['raw_outputs']}")

st.divider()
st.caption("ModelWatch | Drift is a warning signal, not proof of model degradation.")
