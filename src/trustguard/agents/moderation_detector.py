"""Provider adapter for a dedicated moderation classifier; failures degrade safely."""
from __future__ import annotations

from trustguard.settings import settings

PROVIDER_TO_LABEL = {
    "harassment": "harassment",
    "harassment/threatening": "threat",
    "hate": "hate",
    "hate/threatening": "hate",
    "violence": "violence",
    "violence/graphic": "violence",
    "sexual": "sexual",
    "self-harm": "self_harm",
    "self-harm/intent": "self_harm",
    "self-harm/instructions": "self_harm",
}


class ModerationDetector:
    def __init__(self) -> None:
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install the OpenAI SDK to enable provider moderation") from exc
        self.client = OpenAI(api_key=settings.openai_api_key)

    def detect(self, content: str) -> tuple[dict[str, float], list[str]]:
        response = self.client.moderations.create(
            model=settings.moderation_model,
            input=content,
        )
        if not response.results:
            raise RuntimeError("Moderation provider returned no results")
        result = response.results[0]
        raw_scores = result.category_scores.model_dump()
        scores = {label: 0.0 for label in (
            "spam", "harassment", "hate", "threat", "sexual", "violence", "self_harm", "toxicity", "other"
        )}
        for provider_category, label in PROVIDER_TO_LABEL.items():
            scores[label] = max(scores[label], float(raw_scores.get(provider_category, 0.0) or 0.0))
        return scores, ["provider_moderation"]
