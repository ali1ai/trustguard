from __future__ import annotations

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from trustguard.ml.classifier import ARTIFACT_FORMAT_VERSION, load_classifier


def test_load_project_artifact_and_score(tmp_path):
    texts = [
        "ordinary hello friend meeting tomorrow",
        "ordinary family dinner tonight",
        "spam win cash prize click link",
        "spam claim prize now click here",
        "ordinary please call me later",
        "spam free offer win cash now",
    ] * 8
    labels = [0, 0, 1, 1, 0, 1] * 8
    estimator = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
        ("classifier", LogisticRegression(random_state=42, max_iter=300)),
    ]).fit(texts, labels)
    path = tmp_path / "spam_classifier.joblib"
    joblib.dump({
        "format_version": ARTIFACT_FORMAT_VERSION,
        "task": "spam",
        "threshold": 0.6,
        "estimator": estimator,
        "metadata": {"model_name": "test-only"},
    }, path)

    loaded = load_classifier(path, "spam")
    assert loaded is not None
    assert loaded.task == "spam"
    assert loaded.threshold == 0.6
    assert 0 <= loaded.score("win a cash prize click the link") <= 1


def test_artifact_task_mismatch_is_rejected(tmp_path):
    path = tmp_path / "spam_classifier.joblib"
    joblib.dump({
        "format_version": ARTIFACT_FORMAT_VERSION,
        "task": "toxicity",
        "threshold": 0.5,
        "estimator": object(),
        "metadata": {},
    }, path)
    try:
        load_classifier(path, "spam")
    except ValueError as exc:
        assert "task mismatch" in str(exc)
    else:
        raise AssertionError("task mismatch should be rejected")
