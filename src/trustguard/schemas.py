from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class ModerationLabel(StrEnum):
    SPAM = "spam"
    HARASSMENT = "harassment"
    HATE = "hate"
    THREAT = "threat"
    SEXUAL = "sexual"
    VIOLENCE = "violence"
    SELF_HARM = "self_harm"
    TOXICITY = "toxicity"
    OTHER = "other"


class RecommendedAction(StrEnum):
    ALLOW = "allow"
    LABEL = "label"
    SOFT_HOLD = "soft_hold"
    QUEUE_REVIEW = "queue_review"


class ModerationRequest(BaseModel):
    content: str = Field(min_length=1, max_length=20_000)
    source: str = Field(default="demo", max_length=100)
    context: str | None = Field(default=None, max_length=2_000)
    content_id: str | None = Field(default=None, max_length=200)

    @field_validator("content")
    @classmethod
    def content_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("content must not be blank")
        return value


class AgentAssessment(BaseModel):
    available: bool = False
    verdict: str = "not_run"
    labels: list[str] = Field(default_factory=list)
    context_type: str = "unknown"
    confidence: float = 0.0
    rationale: str = ""
    evidence: list[str] = Field(default_factory=list)
    error: str | None = None
    audit_status: str = "not_run"
    audit_rationale: str = ""


class ModerationResult(BaseModel):
    moderation_id: str
    created_at: datetime
    policy_version: str
    labels: list[str]
    scores: dict[str, float]
    model_scores: dict[str, float] = Field(default_factory=dict)
    model_thresholds: dict[str, float] = Field(default_factory=dict)
    action: RecommendedAction
    requires_human_review: bool
    rationale: str
    evidence: list[str] = Field(default_factory=list)
    detector_sources: list[str] = Field(default_factory=list)
    contextual_assessment: AgentAssessment
    warnings: list[str] = Field(default_factory=list)
    execution_mode: str = "local_heuristics"
    model_backed: bool = False
    review_case_id: int | None = None


class ReviewDecision(BaseModel):
    decision: str = Field(pattern="^(confirmed_violation|allowed|needs_escalation)$")
    reviewer_note: str = Field(default="", max_length=2_000)


class ReviewCase(BaseModel):
    case_id: int
    created_at: str
    content: str | None
    source: str
    result: dict[str, Any]
    review_status: str
    reviewer_decision: str | None = None
    reviewer_note: str | None = None
