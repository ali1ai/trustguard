from __future__ import annotations

from trustguard.schemas import ModerationRequest
from trustguard.services import pipeline as pipeline_module


class StubRegistry:
    warnings: list[str] = []
    available_tasks = ["spam"]

    def score(self, task: str, text: str):
        return 0.91 if task == "spam" else None

    def threshold(self, task: str):
        return 0.40 if task == "spam" else None


def test_pipeline_exposes_trained_model_score_and_routes_to_review(monkeypatch):
    monkeypatch.setattr(pipeline_module.ModelRegistry, "from_settings", classmethod(lambda cls: StubRegistry()))
    pipeline = pipeline_module.ModerationPipeline()
    result = pipeline.run(ModerationRequest(content="ordinary text with no obvious spam indicators"))

    assert result.model_backed is True
    assert result.execution_mode == "trained_ml_only"
    assert result.model_scores["spam"] == 0.91
    assert result.model_thresholds["spam"] == 0.4
    assert "trained_spam_classifier" in result.detector_sources
    assert "spam" in result.labels
    assert result.action.value == "queue_review"
    assert result.requires_human_review is True
