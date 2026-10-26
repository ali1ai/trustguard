"""Deterministic policy-to-action mapping. LLM output cannot execute enforcement actions."""
from __future__ import annotations

from trustguard.schemas import AgentAssessment, RecommendedAction

VIOLATION_THRESHOLD = 0.50
REVIEW_THRESHOLD = 0.35


def decide(
    scores: dict[str, float],
    assessment: AgentAssessment,
    signals: list[str],
    label_thresholds: dict[str, float] | None = None,
) -> tuple[list[str], RecommendedAction, bool, str, list[str]]:
    thresholds = label_thresholds or {}
    labels = sorted([
        label for label, score in scores.items()
        if score >= thresholds.get(label, VIOLATION_THRESHOLD)
    ])
    warnings: list[str] = []
    max_score = max(scores.values(), default=0.0)
    disagreement = False
    if assessment.available:
        detector_flagged = any(score >= REVIEW_THRESHOLD for label, score in scores.items() if label != "spam")
        agent_says_allowed = assessment.verdict == "likely_allowed"
        if detector_flagged and agent_says_allowed:
            disagreement = True
            warnings.append("classifier_context_disagreement")
        if assessment.verdict == "policy_violating":
            labels = sorted(set(labels).union(assessment.labels))
        # Do not let the LLM erase a strong classifier signal; route disagreements to a person.
        elif assessment.verdict == "unclear":
            warnings.append("context_ambiguous")

    if not settings_safe_to_auto_hold_spam(scores, signals):
        auto_hold_spam = False
    else:
        auto_hold_spam = True

    if disagreement or (
        assessment.available
        and (assessment.verdict == "unclear" or assessment.audit_status in {"disagreement", "uncertain", "error"})
    ):
        action = RecommendedAction.QUEUE_REVIEW
        requires_review = True
        rationale = "Signals disagree or context is ambiguous; route to a human reviewer."
    elif auto_hold_spam:
        action = RecommendedAction.SOFT_HOLD
        requires_review = True
        rationale = "Multiple strong spam indicators justify a reversible hold pending review."
    elif (
        any(scores.get(key, 0.0) >= 0.85 for key in ("threat", "hate", "harassment", "violence", "sexual", "self_harm", "toxicity"))
        or (assessment.available and assessment.verdict == "policy_violating")
    ):
        action = RecommendedAction.QUEUE_REVIEW
        requires_review = True
        rationale = "High-risk detector or policy-analysis signal detected; require human review before enforcement."
    elif labels or max_score >= REVIEW_THRESHOLD:
        action = RecommendedAction.QUEUE_REVIEW
        requires_review = True
        rationale = "Potential policy violation detected; route to the review queue."
    elif assessment.available and assessment.verdict == "likely_allowed" and assessment.confidence >= 0.75:
        action = RecommendedAction.ALLOW
        requires_review = False
        rationale = "Contextual analysis found no clear violation and the detector scores were low."
    elif max_score < REVIEW_THRESHOLD and (
        not assessment.available or (assessment.verdict == "likely_allowed" and assessment.confidence >= 0.75)
    ):
        action = RecommendedAction.ALLOW
        requires_review = False
        rationale = "No configured detector crossed the review threshold and available context analysis supports allowing the content."
    else:
        action = RecommendedAction.QUEUE_REVIEW
        requires_review = True
        rationale = "Insufficient confidence for an automated decision."

    evidence = list(dict.fromkeys(signals + assessment.evidence))[:8]
    return labels, action, requires_review, rationale, warnings + ([] if assessment.available else ["context_agent_not_run"])


def settings_safe_to_auto_hold_spam(scores: dict[str, float], signals: list[str]) -> bool:
    # A high spam score alone is not enough; require corroborating cheap signals.
    corroborators = {"multiple_urls", "promotional_phrase", "repeated_token_pattern", "multiple_email_addresses"}
    return scores.get("spam", 0.0) >= 0.90 and len(corroborators.intersection(signals)) >= 2
