import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from monitoring.ollama_client import embed_texts

DATA_PATH = ROOT / "data" / "news" / "News_Category_Dataset_v3.json"
REPORT_DIR = ROOT / "tracking" / "news"


def calculate_psi(reference, current, bins=10):
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]

    if not len(reference) or not len(current):
        return None

    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0

    edges[0] = -np.inf
    edges[-1] = np.inf
    ref_counts = np.histogram(reference, bins=edges)[0]
    cur_counts = np.histogram(current, bins=edges)[0]
    ref_pct = np.clip(ref_counts / ref_counts.sum(), 1e-6, None)
    cur_pct = np.clip(cur_counts / cur_counts.sum(), 1e-6, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def load_news(limit):
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"News dataset not found: {DATA_PATH}")

    data = pd.read_json(DATA_PATH, lines=True)
    required = {"headline", "category", "date"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"News dataset is missing columns: {sorted(missing)}")

    data = data.dropna(subset=["headline", "date"]).copy()
    data["headline"] = data["headline"].astype(str).str.strip()
    data = data[data["headline"] != ""]
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    total_rows = len(data)
    if limit > 0 and total_rows > limit:
        sample_indices = np.linspace(
            0, total_rows - 1, num=limit, dtype=int
        )
        data = data.iloc[sample_indices].reset_index(drop=True)

    if len(data) < 2:
        raise ValueError("At least two dated news headlines are required.")

    return data, total_rows


def embed_dataset(texts, model, batch_size):
    vectors = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        vectors.extend(embed_texts(model, batch))
        print(f"Embedded {len(vectors):,}/{len(texts):,} headlines", flush=True)

    embeddings = np.asarray(vectors, dtype=np.float64)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(texts):
        raise ValueError("Embedding responses have inconsistent dimensions.")
    if not np.isfinite(embeddings).all():
        raise ValueError("Embedding response contains non-finite values.")
    return embeddings


def analyze(data, embeddings):
    split = len(data) // 2
    reference = embeddings[:split]
    current = embeddings[split:]
    reference_norms = np.linalg.norm(reference, axis=1)
    current_norms = np.linalg.norm(current, axis=1)

    component_count = min(5, embeddings.shape[1], len(embeddings) - 1)
    projected = PCA(n_components=component_count).fit_transform(embeddings)
    component_results = []
    for index in range(component_count):
        statistic, p_value = ks_2samp(
            projected[:split, index],
            projected[split:, index],
        )
        component_results.append({
            "component": index + 1,
            "ks_statistic": float(statistic),
            "p_value": float(p_value),
            "psi": calculate_psi(
                projected[:split, index],
                projected[split:, index],
            ),
        })

    category_rates = data["category"].astype(str).value_counts(normalize=True)
    reference_rates = (
        data.iloc[:split]["category"].astype(str).value_counts(normalize=True)
    )
    current_rates = (
        data.iloc[split:]["category"].astype(str).value_counts(normalize=True)
    )
    categories = set(reference_rates.index) | set(current_rates.index)
    category_tvd = 0.5 * sum(
        abs(reference_rates.get(category, 0.0) - current_rates.get(category, 0.0))
        for category in categories
    )

    centroid_norms = np.linalg.norm(reference.mean(axis=0)) * np.linalg.norm(
        current.mean(axis=0)
    )
    centroid_cosine = (
        float(np.dot(reference.mean(axis=0), current.mean(axis=0)) / centroid_norms)
        if centroid_norms
        else None
    )

    return {
        "rows_analyzed": int(len(data)),
        "reference_rows": int(split),
        "current_rows": int(len(data) - split),
        "reference_period": {
            "start": data.iloc[0]["date"].date().isoformat(),
            "end": data.iloc[split - 1]["date"].date().isoformat(),
        },
        "current_period": {
            "start": data.iloc[split]["date"].date().isoformat(),
            "end": data.iloc[-1]["date"].date().isoformat(),
        },
        "embedding_dimensions": int(embeddings.shape[1]),
        "embedding_norm": {
            "reference_mean": float(reference_norms.mean()),
            "current_mean": float(current_norms.mean()),
            "psi": calculate_psi(reference_norms, current_norms),
        },
        "mean_centroid_cosine_similarity": centroid_cosine,
        "pca_component_drift": component_results,
        "category_total_variation": float(category_tvd),
        "category_count": int(len(category_rates)),
        "headline_characters": {
            "reference_mean": float(
                data.iloc[:split]["headline"].str.len().mean()
            ),
            "current_mean": float(
                data.iloc[split:]["headline"].str.len().mean()
            ),
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Measure temporal drift in News Category headline embeddings."
    )
    parser.add_argument("--limit", type=int, default=1000,
                        help="Number of evenly sampled rows; 0 processes all rows.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--model", default="nomic-embed-text")
    args = parser.parse_args()
    if args.limit < 0 or args.batch_size < 1:
        parser.error("--limit must be >= 0 and --batch-size must be >= 1")

    data, total_rows = load_news(args.limit)
    print(f"News rows available: {total_rows:,}; selected: {len(data):,}")
    embeddings = embed_dataset(
        data["headline"].tolist(),
        args.model,
        args.batch_size,
    )
    result = analyze(data, embeddings)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": DATA_PATH.name,
        "model": args.model,
        "sample_limit": args.limit,
        "dataset_rows_available": total_rows,
        **result,
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / "embedding_drift_latest.json"
    report_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print("\nNEWS EMBEDDING DRIFT")
    print(json.dumps(result, indent=2))
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
