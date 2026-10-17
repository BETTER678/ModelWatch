# ModelWatch

### Multi-Layer AI Model Monitoring, Data Drift Detection & LLM Evaluation

ModelWatch is a local AI monitoring platform that investigates changes in
tabular data, text embeddings, and Large Language Model (LLM) outputs. It
combines three monitoring layers into a unified pipeline with a Streamlit
dashboard for viewing reports and tracking experiment history.

**Project Status:** Experimental / Research Prototype

Drift is treated as an investigation signal, not proof of model degradation.
Model promotion is a separate, gated process.

## Key Features

- **Tabular Data Monitoring:** Investigates feature distribution changes and
  model evaluation metrics using LendingClub data.
- **Text Embedding Drift Detection:** Analyzes changes in news headline
  embeddings using local embedding models.
- **LLM Output Evaluation:** Compares prompt strategies using latency, token
  usage, response-format checks, and word F1.
- **Interactive Dashboard:** Displays monitoring reports and pipeline run
  history through Streamlit.
- **Local AI Execution:** Uses Ollama for local embedding generation and LLM
  inference.
- **Experiment Tracking:** Organizes reports for all three monitoring layers.
- **Gated Model Promotion:** Separates monitoring evidence from model
  promotion decisions.

## Tech Stack

| Technology | Purpose |
| --- | --- |
| Python | Monitoring pipelines and evaluation |
| Pandas & NumPy | Data processing and numerical analysis |
| Scikit-learn | Machine learning and evaluation utilities |
| Ollama | Local model inference |
| Nomic Embed Text | Text embedding generation |
| Llama 3.2 1B | Local LLM experiments |
| Streamlit | Interactive dashboard |
| PCA | Embedding distribution analysis |
| Git & GitHub | Version control |

## Monitoring Layers

1. **LendingClub:** Investigates tabular feature drift and evaluates candidate
   models.
2. **News Embeddings:** Analyzes changes in headline embeddings, category
   distribution, and text length.
3. **LLM Evaluation:** Compares direct and guarded prompts using latency, token
   counts, formatting checks, and reference word F1.

## Installation

Clone the repository:

```powershell
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd ModelWatch
```

Create and activate a virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

Install Ollama from [ollama.com/download](https://ollama.com/download) and pull
the required models:

```powershell
ollama pull nomic-embed-text
ollama pull llama3.2:1b
```

## Run the Pipeline

Run all three monitoring layers:

```powershell
.\venv\Scripts\python.exe .\pipeline\run_all_layers.py
```

By default, the pipeline analyzes LendingClub, embeds up to 1,000 News headlines,
and evaluates 25 Dolly prompts. Set either limit to `0` to process all eligible
rows:

```powershell
.\venv\Scripts\python.exe .\pipeline\run_all_layers.py --news-limit 0 --dolly-limit 0
```

The full-data run can take a long time and requires sufficient disk space,
memory, and local compute. You can set the limits independently or override the
Ollama model names:

```powershell
.\venv\Scripts\python.exe .\pipeline\run_all_layers.py `
  --news-limit 1000 `
  --dolly-limit 25 `
  --embedding-model nomic-embed-text `
  --llm-model llama3.2:1b
```

## Dashboard

Launch the Streamlit dashboard:

```powershell
.\venv\Scripts\python.exe -m streamlit run .\app.py
```

Open the local URL printed by Streamlit, usually
[`http://localhost:8501`](http://localhost:8501).

The dashboard shows the latest LendingClub, News embedding, and Dolly reports,
along with all-layers run history. Its run controls use the same sample limits
and local Ollama model names as the command-line pipeline.

## Reports and Generated Data

Monitoring reports are saved in:

- `tracking/lendingclub/`
- `tracking/news/`
- `tracking/llm/`

All-layers run history is recorded at `tracking/all_layers_runs.jsonl`.
Prepared Dolly prompt samples and generated responses are stored under
`data/llm/prepared/`.

Generated reports, prepared data, downloaded source data, and model binaries are
excluded from Git.

## Dataset Preparation

Place the LendingClub accepted-loans CSV at
`data/lendingclub/accepted_loans.csv`, then prepare it:

```powershell
.\venv\Scripts\python.exe .\scripts\prepare_lendingclub.py
```

Preparation retains fully paid and charged-off loans and writes year-specific
files under `data/lendingclub/prepared/`.

The News and Dolly workflows expect their source datasets at:

- `data/news/News_Category_Dataset_v3.json`
- `data/llm/databricks-dolly-15k.jsonl`

## Evaluation Notes and Limitations

- Drift does not necessarily indicate model degradation; it is a signal to
  investigate.
- LendingClub preparation retains resolved loan outcomes. Results may be
  affected by outcome-maturity and selection bias.
- The LendingClub candidate comparison uses 2016 data that has already been
  used for model selection. It is not an independent promotion holdout.
- The pipeline does not automatically promote models. Promotion requires
  explicit evaluation metadata, an approved named dataset, identical evaluation
  data for both models, and an untouched holdout.
- Missing promotion metadata cannot default to approval.
- The News report compares earlier and later sampled periods using embedding
  norm and PCA drift, category mix, and headline length.
- The Dolly report compares direct and guarded prompts using latency, Ollama
  token counts, response-format checks, and reference word F1. Word F1 is a
  lexical proxy, not a human quality judgment.
- Text monitoring provides evidence for investigation, not production
  readiness or a substitute for reviewed evaluation data.

## Future Improvements

- Configurable drift thresholds and alerts.
- Improved dashboard visualizations.
- Model version tracking and reproducible experiments.
- Independent holdout evaluation and stronger promotion safeguards.
- Human-reviewed LLM evaluation.

## License

No project license is currently included. Add an appropriate license before
permitting reuse or redistribution.

---

**ModelWatch — Monitor the signals. Investigate the changes. Evaluate before
promoting.**
