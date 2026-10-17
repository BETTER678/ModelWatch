import json

import numpy as np
import pandas as pd

from monitoring import ollama_client
from scripts import monitor_dolly_outputs, monitor_news_embeddings


def test_news_analyze_reports_temporal_embedding_and_category_shift():
    data = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=6),
        "headline": ["a", "bb", "ccc", "dddd", "eeeee", "ffffff"],
        "category": ["A", "A", "A", "B", "B", "B"],
    })
    embeddings = np.array([
        [1.0, 0.0, 0.0],
        [1.1, 0.0, 0.0],
        [0.9, 0.1, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.1, 0.0],
        [0.1, 0.9, 0.0],
    ])

    result = monitor_news_embeddings.analyze(data, embeddings)

    assert result["rows_analyzed"] == 6
    assert result["category_total_variation"] == 1.0
    assert result["embedding_norm"]["psi"] is not None
    assert result["pca_component_drift"]


def test_news_load_evenly_samples_date_ordered_data(tmp_path, monkeypatch):
    path = tmp_path / "news.json"
    rows = [
        {"headline": f"headline {index}", "category": "NEWS",
         "date": f"2020-01-{index + 1:02d}"}
        for index in range(6)
    ]
    path.write_text(
        "\n".join(json.dumps(row) for row in rows),
        encoding="utf-8",
    )
    monkeypatch.setattr(monitor_news_embeddings, "DATA_PATH", path)

    data, total_rows = monitor_news_embeddings.load_news(limit=3)

    assert total_rows == 6
    assert data["headline"].tolist() == [
        "headline 0", "headline 2", "headline 5"
    ]


def test_dolly_word_f1_and_example_sample_are_deterministic(tmp_path, monkeypatch):
    path = tmp_path / "dolly.jsonl"
    examples = [
        {"instruction": f"question {index}", "response": f"answer {index}"}
        for index in range(10)
    ]
    path.write_text(
        "\n".join(json.dumps(example) for example in examples),
        encoding="utf-8",
    )
    monkeypatch.setattr(monitor_dolly_outputs, "DATA_PATH", path)

    first, total = monitor_dolly_outputs.load_examples(limit=4)
    second, _ = monitor_dolly_outputs.load_examples(limit=4)

    assert total == 10
    assert first == second
    assert monitor_dolly_outputs.word_f1("a useful answer", "a useful answer") == 1
    assert monitor_dolly_outputs.word_f1("answer", "unrelated") == 0


def test_ollama_embedding_uses_batch_endpoint(monkeypatch):
    observed = {}

    def fake_post(endpoint, payload, timeout):
        observed["endpoint"] = endpoint
        observed["payload"] = payload
        return {"embeddings": [[0.1, 0.2], [0.3, 0.4]]}

    monkeypatch.setattr(ollama_client, "_post", fake_post)

    result = ollama_client.embed_texts("embed-model", ["one", "two"])

    assert result == [[0.1, 0.2], [0.3, 0.4]]
    assert observed["endpoint"] == "/api/embed"
    assert observed["payload"]["input"] == ["one", "two"]
