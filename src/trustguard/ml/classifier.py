"""Safe-ish loader for project-generated sklearn model artifacts.

Only load artifacts that you generated or otherwise trust. joblib/pickle artifacts
can execute code during deserialization and must never be accepted from end users.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

ARTIFACT_FORMAT_VERSION = 1


@dataclass
class LoadedTextClassifier:
    task: str
    estimator: Any
    threshold: float
    metadata: dict[str, Any]
    path: Path

    def score(self, text: str) -> float:
        """Return the model's positive-class score, clipped to [0, 1]."""
        if not hasattr(self.estimator, "predict_proba"):
            raise ValueError(f"Classifier artifact for {self.task!r} lacks predict_proba")
        probabilities = self.estimator.predict_proba([text])[0]
        classes = list(getattr(self.estimator, "classes_", []))
        try:
            positive_index = classes.index(1)
        except ValueError as exc:
            raise ValueError(f"Classifier artifact for {self.task!r} has no positive class 1") from exc
        score = float(probabilities[positive_index])
        return max(0.0, min(1.0, score))


def load_classifier(path: str | Path, expected_task: str) -> LoadedTextClassifier | None:
    """Load a project-generated classifier artifact, returning None if absent."""
    artifact_path = Path(path)
    if not artifact_path.is_file():
        return None

    # Import lazily so API-only minimal installs can still fall back if sklearn is missing.
    try:
        import joblib
    except ImportError as exc:  # pragma: no cover - sklearn installs joblib
        raise RuntimeError("scikit-learn/joblib is required to load trained model artifacts") from exc

    payload = joblib.load(artifact_path)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid TrustGuard model artifact at {artifact_path}")
    if payload.get("format_version") != ARTIFACT_FORMAT_VERSION:
        raise ValueError(f"Unsupported model artifact version at {artifact_path}")
    if payload.get("task") != expected_task:
        raise ValueError(
            f"Model task mismatch at {artifact_path}: expected {expected_task!r}, got {payload.get('task')!r}"
        )
    estimator = payload.get("estimator")
    if estimator is None or not hasattr(estimator, "predict_proba"):
        raise ValueError(f"Invalid estimator in model artifact at {artifact_path}")
    threshold = float(payload.get("threshold", 0.5))
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"Invalid decision threshold in model artifact at {artifact_path}")
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    return LoadedTextClassifier(
        task=expected_task,
        estimator=estimator,
        threshold=threshold,
        metadata=metadata,
        path=artifact_path,
    )
