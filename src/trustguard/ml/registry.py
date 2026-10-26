"""Load available project-generated classifiers without making them mandatory."""
from __future__ import annotations

from dataclasses import dataclass, field

from trustguard.ml.classifier import LoadedTextClassifier, load_classifier
from trustguard.settings import settings


@dataclass
class ModelRegistry:
    models: dict[str, LoadedTextClassifier] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @classmethod
    def from_settings(cls) -> "ModelRegistry":
        registry = cls()
        candidates = {
            "spam": settings.spam_model_path,
            "toxicity": settings.toxicity_model_path,
        }
        for task, path in candidates.items():
            try:
                model = load_classifier(path, task)
                if model is not None:
                    registry.models[task] = model
            except Exception as exc:
                # The public demo degrades to heuristics if an artifact is missing or invalid.
                registry.warnings.append(f"trained_{task}_model_load_error:{type(exc).__name__}")
        return registry

    @property
    def available_tasks(self) -> list[str]:
        return sorted(self.models)

    def score(self, task: str, text: str) -> float | None:
        model = self.models.get(task)
        if model is None:
            return None
        return model.score(text)

    def threshold(self, task: str) -> float | None:
        model = self.models.get(task)
        return model.threshold if model else None
