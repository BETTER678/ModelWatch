# ModelWatch

### Multi-Layer AI Model Monitoring, Data Drift Detection, and LLM Evaluation

ModelWatch is a local AI monitoring and evaluation platform that investigates
changes in tabular data, text embeddings, and Large Language Model (LLM) outputs.
It combines data drift detection, embedding-based monitoring, and LLM response
evaluation in a unified pipeline, with a Streamlit dashboard for viewing
reports and tracking experiment history.

The project explores an important question in machine learning: how can we
identify changes in data or model behavior before trusting a model in a
real-world application?

**Project status:** Experimental / research prototype

Drift is an investigation signal, not proof of model degradation. Model
promotion is a separate, gated process and is not automatically approved by
monitoring results.

## Key Features

- **Tabular data monitoring:** Investigates changes in feature distributions
  and model evaluation metrics using LendingClub data.
- **Text embedding drift detection:** Uses local embedding models to investigate
  changes in news headline data.
- **LLM output evaluation:** Compares prompt strategies using latency, token
  usage, response checks, and reference word F1.
- **Interactive dashboard:** Displays monitoring reports and historical
  pipeline runs through Streamlit.
- **Unified monitoring pipeline:** Runs the three monitoring layers through a
  single command.
- **Local AI execution:** Uses Ollama for local embedding generation and LLM
  inference.
- **Experiment tracking:** Stores reports and run history for the monitoring
  layers.
- **Gated model promotion:** Separates monitoring evidence from model
  promotion decisions.

## Tech Stack

| Technology | Purpose |
| --- | --- |
| Python | Monitoring pipelines and evaluation |
| Pandas and NumPy | Data processing and numerical analysis |
| SciPy | Statistical tests |
| Scikit-learn | Machine learning, PCA, and evaluation utilities |
| Ollama | Local embedding and LLM inference |
| `nomic-embed-text` | Text embedding generation |
| `llama3.2:1b` | Local language model for prompt experiments |
| Streamlit | Interactive monitoring dashboard |
| Git and GitHub | Version control and collaboration |

## System Overview

```text
                       +-----------------------+
                       |     ModelWatch        |
                       |  Monitoring system    |
                       +-----------+-----------+
                                   |
               +------------------+------------------+
               |                  |                  |
               v                  v                  v
       +---------------+  +---------------+  +---------------+
       | Tabular data  |  | Text          |  | LLM output    |
       | monitoring    |  | embeddings    |  | evaluation    |
       +-------+-------+  +-------+-------+  +-------+-------+
               |                  |                  |
               v                  v                  v
       +---------------+  +---------------+  +---------------+
       | Feature drift |  | Embedding and |  | Latency,      |
       | and model     |  | category      |  | format, token |
       | metrics       |  | statistics    |  | use, and F1   |
       +-------+-------+  +-------+-------+  +-------+-------+
               +------------------+------------------+
                                  |
                                  v
                       +-----------------------+
                       | Reports and run       |
                       | history               |
                       +-----------+-----------+
                                   |
                                   v
                       +-----------------------+
                       | Streamlit dashboard   |
                       +-----------------------+
```

## The Three Monitoring Layers

### 1. Tabular Data Monitoring: LendingClub

This layer investigates changes in structured financial data using the
LendingClub accepted-loans dataset.

It:

- Compares feature distributions across different time periods.
- Uses Population Stability Index (PSI) and Kolmogorov-Smirnov (KS) statistics
  to investigate feature drift.
- Evaluates candidate models with classification metrics.
- Records run information for later inspection.

For example, if a financial dataset changes over time, ModelWatch can help
identify which features shifted and provide evaluation metrics for further
investigation. A change in feature distribution does not by itself mean model
performance has declined.

### 2. Text Embedding Monitoring: News Headlines

This layer investigates changes in text data by converting dated news headlines
into vector embeddings using a local Ollama embedding model.

It:

- Selects dated headlines across the dataset's time range.
- Generates embeddings through Ollama.
- Compares embedding norms and principal-component distributions between
  earlier and later periods.
- Examines category composition and headline length.
- Writes a report to support investigation of changes in the text data.

The default configuration analyzes up to 1,000 evenly sampled headlines. For
example, a change from predominantly business headlines to more technology
headlines may be reflected in embedding and category statistics.

Embedding drift indicates a change in representation; it does not prove that a
downstream model has become less accurate.

### 3. LLM Output Monitoring: Dolly Prompts

This layer compares LLM outputs produced by different prompt strategies on the
Dolly dataset.

It:

- Evaluates a deterministic sample of Dolly prompts.
- Compares direct prompts with guarded prompts.
- Records inference latency and Ollama-reported token counts.
- Checks whether responses are non-empty and within the configured word limit.
- Calculates reference word F1 as a lexical similarity measure.

The default configuration evaluates 25 prompts with two prompt variants. A
guarded prompt may improve response formatting while increasing inference
latency; ModelWatch records these differences for review.

Word F1 is a lexical comparison metric, not a human quality score. It does not
independently establish factual correctness, safety, or overall response
quality.

## Project Structure

```text
ModelWatch/
├── app.py
├── requirements.txt
├── api/
├── detectors/
├── monitoring/
│   └── ollama_client.py
├── pipeline/
│   ├── lendingclub_pipeline.py
│   └── run_all_layers.py
├── remediation/
│   ├── promotion.py
│   └── retrain_model.py
├── scripts/
│   ├── analyze_lendingclub_drift.py
│   ├── monitor_dolly_outputs.py
│   ├── monitor_news_embeddings.py
│   ├── prepare_lendingclub.py
│   ├── retrain_lendingclub.py
│   └── train_lendingclub.py
├── tests/
└── tracking/
```

Key files and directories:

- `app.py` — Streamlit dashboard entry point.
- `pipeline/run_all_layers.py` — Runs the tabular, embedding, and LLM
  monitoring layers.
- `pipeline/lendingclub_pipeline.py` — Runs LendingClub drift analysis and
  candidate evaluation.
- `scripts/prepare_lendingclub.py` — Prepares the LendingClub source dataset.
- `monitoring/ollama_client.py` — Client for the local Ollama API.
- `remediation/promotion.py` — Gated candidate-promotion logic.
- `tests/` — Automated test suite.
- `tracking/` — Monitoring reports and run history.

Downloaded datasets, generated reports, prepared data, and model artifacts may
be excluded from Git. Their presence in a fresh clone is not guaranteed.

## Requirements

Before getting started, install:

- Windows 10 or Windows 11.
- Python compatible with the dependencies in `requirements.txt`.
- Git.
- Ollama.
- The local embedding and language models used by the monitoring scripts.
- Sufficient disk space, memory, and local compute for the selected data and
  models.

## Installation and Setup

### 1. Clone the repository

```powershell
git clone https://github.com/BETTER678/ModelWatch.git
cd ModelWatch
```

### 2. Create and activate a virtual environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks environment activation, use the virtual environment's
Python executable directly in the commands below.

### 3. Install Python dependencies

```powershell
.\venv\Scripts\python.exe -m pip install --upgrade pip
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 4. Install Ollama and download the models

Install [Ollama](https://ollama.com/download), then download the default local
models:

```powershell
ollama pull nomic-embed-text
ollama pull llama3.2:1b
```

Confirm that the models are available:

```powershell
ollama list
```

The text monitoring layers use Ollama's local API. Ensure the Ollama service is
running before executing the pipeline.

### 5. Prepare the LendingClub dataset

Obtain the LendingClub accepted-loans CSV separately and place it at:

```text
data/lendingclub/accepted_loans.csv
```

Run the preparation script from the repository root:

```powershell
.\venv\Scripts\python.exe .\scripts\prepare_lendingclub.py
```

The preparation script retains loans marked as fully paid or charged off and
writes year-specific files under `data/lendingclub/prepared/`. Review the
dataset's terms of use and ensure sufficient storage is available.

The News and Dolly workflows also expect source data at:

- `data/news/News_Category_Dataset_v3.json`
- `data/llm/databricks-dolly-15k.jsonl`

Obtain these datasets separately and review their licenses and terms of use.

## Run ModelWatch

### Run all three monitoring layers

From the repository root:

```powershell
.\venv\Scripts\python.exe .\pipeline\run_all_layers.py
```

By default, the pipeline:

- Runs LendingClub monitoring and candidate evaluation.
- Embeds an evenly sampled set of up to 1,000 dated news headlines.
- Evaluates 25 Dolly prompts using two prompt variants.

To inspect available command-line options:

```powershell
.\venv\Scripts\python.exe .\pipeline\run_all_layers.py --help
```

To process all eligible News and Dolly rows instead of the defaults:

```powershell
.\venv\Scripts\python.exe .\pipeline\run_all_layers.py --news-limit 0 --dolly-limit 0
```

Limits can also be set independently, and the Ollama model names can be
overridden:

```powershell
.\venv\Scripts\python.exe .\pipeline\run_all_layers.py `
  --news-limit 500 `
  --dolly-limit 25 `
  --embedding-model nomic-embed-text `
  --llm-model llama3.2:1b
```

Full-data runs can take significantly longer and require additional memory,
disk space, and local compute.

### Launch the Streamlit dashboard

```powershell
.\venv\Scripts\python.exe -m streamlit run .\app.py
```

Open the local URL printed by Streamlit, usually
[http://localhost:8501](http://localhost:8501).

The dashboard displays the latest LendingClub, News embedding, and Dolly
reports, along with all-layers run history. Its run controls use the same
sample limits and local Ollama model names as the command-line pipeline.

### Run the tests

```powershell
.\venv\Scripts\python.exe -m pytest -q
```

## Reports and Monitoring Results

ModelWatch writes reports and run records to:

| Monitoring layer | Output location |
| --- | --- |
| LendingClub | `tracking/lendingclub/` |
| News embeddings | `tracking/news/` |
| Dolly LLM evaluation | `tracking/llm/` |
| All-layers run history | `tracking/all_layers_runs.jsonl` |

Sampled Dolly generations are saved under `data/llm/prepared/`. Exact metrics
and outcomes depend on the input data, model versions, sampling settings, and
execution environment.

## Model Promotion and Evaluation Safety

ModelWatch deliberately separates monitoring from model promotion. A drift
signal can indicate that further investigation is needed, but it does not
establish that a model has degraded or that a replacement model is better.

Promotion requires explicit evaluation metadata, including:

- A named and approved evaluation dataset.
- Identical evaluation data for the existing and candidate models.
- An untouched holdout dataset for independent evaluation.
- Explicit approval criteria and a separate promotion decision.

Missing evaluation metadata must not default to approval.

### LendingClub evaluation limitations

The preparation pipeline retains only loans marked fully paid or charged off.
Results therefore describe resolved loans and can be affected by
outcome-maturity and selection bias.

The 2016 LendingClub data has already been used during model selection. It is
not an independent promotion holdout. Consequently, the current pipeline does
not promote models automatically.

### Text and LLM evaluation limitations

Embedding statistics can identify distribution changes but cannot independently
establish downstream task performance.

LLM response metrics such as latency, token usage, response checks, and word F1
provide monitoring evidence. They are not substitutes for reviewed evaluation
datasets, human assessment, factuality checks, or safety testing.

## Troubleshooting

### Ollama is not responding

- Confirm that Ollama is installed and running.
- Check available models with `ollama list`.
- Verify that configured model names match installed models.
- Confirm that the local Ollama API is accessible.

### LendingClub dataset not found

Verify that the CSV exists at `data/lendingclub/accepted_loans.csv` and run the
preparation script before executing the monitoring pipeline.

### Streamlit command not found

Run Streamlit through the project's Python executable:

```powershell
.\venv\Scripts\python.exe -m streamlit run .\app.py
```

### Pipeline takes too long

- Reduce `--news-limit` or `--dolly-limit`.
- Confirm Ollama is using the expected models.
- Check memory, CPU usage, and available disk space.

### Reports are missing

Check terminal output for pipeline errors and verify that the required source
data and local models are available. A report may not be generated if its
monitoring layer fails.

## Data and Privacy

ModelWatch is designed for local experimentation using Ollama and local
pipeline execution.

- Keep credentials and private configuration out of source control.
- Do not commit large downloaded datasets or generated model binaries.
- Avoid publishing sensitive source data or model outputs.
- Review the licenses and usage terms of datasets and model assets.

Local execution reduces the need for a hosted inference service, but privacy
also depends on the data sources, local configuration, and any additional
services used.

## Future Improvements

Potential directions for further development include:

- Configurable drift thresholds and alerting.
- Improved dashboard visualizations and cross-run comparisons.
- Reproducible experiment metadata and model version tracking.
- Evaluation of drift against independently measured downstream performance.
- Reviewed LLM evaluation datasets and human assessment.
- Stronger model-promotion validation and audit logs.
- Support for additional datasets and embedding models.

These are proposed improvements rather than claims about existing
functionality.

## Contributing

Contributions, bug reports, and suggestions are welcome.

1. Fork the repository.
2. Create a feature branch.
3. Implement and test your changes.
4. Submit a pull request with a clear description of the changes.

For bug reports, include relevant logs, reproducible steps, and configuration
details. Avoid sharing private datasets, credentials, or sensitive model
outputs.

## License

No `LICENSE` file is currently included. Add a license before distributing the
project. Until then, do not assume that others have permission to reuse or
redistribute the project's code or bundled assets.

---
