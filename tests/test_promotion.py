
import json

import joblib
import pytest

from remediation import promotion
from remediation.retrain_model import PROJECT_ROOT


def test_existing_model_file_exists():

    model_path = (
        PROJECT_ROOT
        + "/models/current_model.pkl"
    )

    model = joblib.load(model_path)

    assert hasattr(model, "predict")


def test_lower_accuracy_model_is_rejected():

    old_accuracy = 1.0
    new_accuracy = 0.995

    decision = (
        "PROMOTED"
        if new_accuracy > old_accuracy
        else "REJECTED"
    )

    assert decision == "REJECTED"


def test_equal_accuracy_model_is_rejected():

    old_accuracy = 1.0
    new_accuracy = 1.0

    decision = (
        "PROMOTED"
        if new_accuracy > old_accuracy
        else "REJECTED"
    )

    assert decision == "REJECTED"


def test_promotion_rejects_unapproved_evaluation_without_replacing_model(
    tmp_path, monkeypatch
):
    baseline_path = tmp_path / "baseline.joblib"
    candidate_path = tmp_path / "candidate.joblib"
    baseline_path.write_bytes(b"champion")
    candidate_path.write_bytes(b"candidate")

    monkeypatch.setattr(promotion, "BASELINE_MODEL", baseline_path)
    monkeypatch.setattr(promotion, "CANDIDATE_MODEL", candidate_path)
    monkeypatch.setattr(promotion, "BACKUP_MODEL", tmp_path / "previous.joblib")
    monkeypatch.setattr(promotion, "TRACKING_DIR", tmp_path / "tracking")
    monkeypatch.setattr(
        promotion,
        "PROMOTION_RECORD",
        tmp_path / "tracking" / "promotion_history.json",
    )

    result = promotion.promote_candidate(
        baseline_metrics={"roc_auc": 0.69, "average_precision": 0.39},
        candidate_metrics={"roc_auc": 0.71, "average_precision": 0.42},
        evaluation_metadata={
            "dataset_name": "reused validation",
            "approved_for_promotion": False,
            "same_dataset_for_both_models": True,
            "untouched_holdout": False,
        },
    )

    assert result["decision"] == "REJECTED"
    assert result["model_promoted"] is False
    assert baseline_path.read_bytes() == b"champion"
    assert not (tmp_path / "previous.joblib").exists()
    history = json.loads(
        (tmp_path / "tracking" / "promotion_history.json").read_text(
            encoding="utf-8"
        )
    )
    assert history == [result]


def test_promotion_requires_evaluation_metadata():
    with pytest.raises(TypeError, match="evaluation_metadata"):
        promotion.promote_candidate(
            baseline_metrics={"roc_auc": 0.69, "average_precision": 0.39},
            candidate_metrics={"roc_auc": 0.71, "average_precision": 0.42},
        )