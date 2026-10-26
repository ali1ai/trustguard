from trustguard.schemas import AgentAssessment, RecommendedAction
from trustguard.services.policy_engine import decide


def test_high_confidence_spam_gets_reversible_hold():
    scores = {"spam": 0.95, "harassment": 0.0, "hate": 0.0, "threat": 0.0, "sexual": 0.0, "violence": 0.0, "self_harm": 0.0, "other": 0.0}
    assessment = AgentAssessment()
    _, action, needs_review, _, _ = decide(scores, assessment, ["multiple_urls", "promotional_phrase"])
    assert action == RecommendedAction.SOFT_HOLD
    assert needs_review is True


def test_disagreement_goes_to_human_review():
    scores = {"spam": 0.0, "harassment": 0.7, "hate": 0.0, "threat": 0.0, "sexual": 0.0, "violence": 0.0, "self_harm": 0.0, "other": 0.0}
    assessment = AgentAssessment(available=True, verdict="likely_allowed", confidence=0.8)
    _, action, needs_review, _, warnings = decide(scores, assessment, [])
    assert action == RecommendedAction.QUEUE_REVIEW
    assert needs_review is True
    assert "classifier_context_disagreement" in warnings


def test_ambiguous_context_goes_to_review_even_when_classifier_scores_are_low():
    scores = {"spam": 0.0, "harassment": 0.0, "hate": 0.0, "threat": 0.0, "sexual": 0.0, "violence": 0.0, "self_harm": 0.0, "other": 0.0}
    assessment = AgentAssessment(available=True, verdict="unclear", confidence=0.3)
    _, action, needs_review, _, _ = decide(scores, assessment, [])
    assert action == RecommendedAction.QUEUE_REVIEW
    assert needs_review is True


def test_policy_agent_violation_is_not_silently_allowed_when_scores_are_low():
    scores = {"spam": 0.0, "harassment": 0.0, "hate": 0.0, "threat": 0.0, "sexual": 0.0, "violence": 0.0, "self_harm": 0.0, "other": 0.0}
    assessment = AgentAssessment(available=True, verdict="policy_violating", labels=["harassment"], confidence=0.8)
    _, action, needs_review, _, _ = decide(scores, assessment, [])
    assert action == RecommendedAction.QUEUE_REVIEW
    assert needs_review is True


def test_trained_model_threshold_can_be_applied_by_policy_engine():
    scores = {"spam": 0.41, "toxicity": 0.20}
    labels, action, needs_review, _, _ = decide(
        scores, AgentAssessment(), [], label_thresholds={"spam": 0.40}
    )
    assert "spam" in labels
    assert action.value == "queue_review"
    assert needs_review is True
