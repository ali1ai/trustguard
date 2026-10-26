from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from uuid import uuid4

from trustguard.agents.contextual_policy_agent import ContextualPolicyAgent
from trustguard.agents.moderation_detector import ModerationDetector
from trustguard.agents.policy_auditor_agent import PolicyAuditorAgent
from trustguard.ml.registry import ModelRegistry
from trustguard.schemas import AgentAssessment, ModerationRequest, ModerationResult
from trustguard.services.policy_engine import decide
from trustguard.services.signals import local_fallback_signals, spam_signals
from trustguard.services.storage import save_event
from trustguard.settings import settings


class ModerationPipeline:
    """Orchestrate local triage, optional trained models, hosted signals, agents and policy."""

    def __init__(self) -> None:
        self.model_registry = ModelRegistry.from_settings()
        self.detector = None
        self.context_agent = None
        self.auditor_agent = None
        if settings.openai_api_key and settings.enable_llm_providers:
            try:
                self.detector = ModerationDetector()
            except Exception:
                self.detector = None
            if settings.enable_context_agent:
                try:
                    self.context_agent = ContextualPolicyAgent()
                except Exception:
                    self.context_agent = None
                if settings.enable_audit_agent:
                    try:
                        self.auditor_agent = PolicyAuditorAgent()
                    except Exception:
                        self.auditor_agent = None

    def run(self, request: ModerationRequest) -> ModerationResult:
        sources: list[str] = []
        heuristic_scores, local_signals = local_fallback_signals(request.content)
        scores = dict(heuristic_scores)
        model_scores: dict[str, float] = {}
        model_thresholds: dict[str, float] = {}
        signals = list(local_signals)
        _, spam_signal_names = spam_signals(request.content)
        signals = list(dict.fromkeys(signals + spam_signal_names))
        warnings: list[str] = list(self.model_registry.warnings)
        sources.append("local_heuristics")

        # Model outputs remain separately visible. The combined score map is a conservative
        # maximum over evidence sources, not a calibrated probability or a trained ensemble.
        for task in ("spam", "toxicity"):
            try:
                model_score = self.model_registry.score(task, request.content)
                if model_score is not None:
                    model_scores[task] = model_score
                    scores[task] = max(scores.get(task, 0.0), model_score)
                    threshold = self.model_registry.threshold(task)
                    if threshold is not None:
                        model_thresholds[task] = threshold
                    sources.append(f"trained_{task}_classifier")
                    signals.append(f"trained_model_signal:{task}")
            except Exception as exc:
                warnings.append(f"trained_{task}_model_inference_error:{type(exc).__name__}")

        if self.detector is not None:
            try:
                provider_scores, provider_sources = self.detector.detect(request.content)
                for label, score in provider_scores.items():
                    scores[label] = max(scores.get(label, 0.0), score)
                sources.extend(provider_sources)
            except Exception as exc:  # fail toward review when a configured provider fails
                warnings.append(f"moderation_provider_error:{type(exc).__name__}")
                sources.append("provider_error_fallback")

        assessment = AgentAssessment()
        if self.context_agent is not None:
            try:
                assessment = self.context_agent.assess(request.content, request.context, scores)
                sources.append("contextual_policy_agent")
                needs_audit = (
                    assessment.verdict == "unclear"
                    or (max(scores.values(), default=0.0) >= 0.75 and assessment.verdict == "likely_allowed")
                    or assessment.confidence < 0.55
                )
                if self.auditor_agent is not None and needs_audit:
                    try:
                        audit = self.auditor_agent.audit(
                            request.content, request.context, assessment.model_dump(mode="json"), scores
                        )
                        assessment.audit_status = audit["status"]
                        assessment.audit_rationale = audit["rationale"]
                        assessment.evidence = list(dict.fromkeys(assessment.evidence + audit["evidence"]))[:8]
                        sources.append("conditional_policy_auditor")
                        if audit["status"] in {"disagreement", "uncertain"}:
                            assessment.verdict = "unclear"
                            assessment.labels = sorted(set(assessment.labels).union(audit["corrected_labels"]))
                            warnings.append("policy_auditor_disagreement_or_uncertainty")
                        elif audit["status"] == "agreement" and audit["corrected_verdict"] == "policy_violating":
                            assessment.labels = sorted(set(assessment.labels).union(audit["corrected_labels"]))
                    except Exception as exc:
                        assessment.audit_status = "error"
                        assessment.audit_rationale = "Independent audit was unavailable; human review may be needed."
                        warnings.append(f"policy_auditor_error:{type(exc).__name__}")
            except Exception as exc:
                assessment = AgentAssessment(error=type(exc).__name__)
                warnings.append(f"context_agent_error:{type(exc).__name__}")

        labels, action, needs_review, rationale, policy_warnings = decide(
            scores, assessment, signals, label_thresholds=model_thresholds,
        )
        warnings.extend(policy_warnings)
        if warnings and any("error" in item for item in warnings) and action.value == "allow":
            action = action.__class__.QUEUE_REVIEW
            needs_review = True
            rationale = "A detector failed; route to a reviewer rather than silently allowing content."

        now = datetime.now(timezone.utc)
        moderation_id = str(uuid4())
        content_hash = hashlib.sha256(request.content.encode("utf-8")).hexdigest()
        provider_moderation_used = "provider_moderation" in sources
        context_agent_used = assessment.available
        trained_models_used = bool(model_scores)
        model_backed = provider_moderation_used or context_agent_used or trained_models_used
        if not model_backed:
            warnings.append("demo_mode_local_heuristics_only_not_a_trained_classifier")

        if trained_models_used and (provider_moderation_used or context_agent_used):
            execution_mode = "hybrid_ml_and_llm"
        elif trained_models_used:
            execution_mode = "trained_ml_only"
        elif provider_moderation_used and context_agent_used:
            execution_mode = "hybrid_llm_and_moderation"
        elif provider_moderation_used:
            execution_mode = "provider_moderation_only"
        elif context_agent_used:
            execution_mode = "context_agent_only"
        else:
            execution_mode = "local_heuristics"

        result = ModerationResult(
            moderation_id=moderation_id,
            created_at=now,
            policy_version=settings.policy_version,
            labels=labels,
            scores={key: round(float(value), 4) for key, value in scores.items()},
            model_scores={key: round(float(value), 4) for key, value in model_scores.items()},
            model_thresholds={key: round(float(value), 4) for key, value in model_thresholds.items()},
            action=action,
            requires_human_review=needs_review,
            rationale=rationale,
            evidence=list(dict.fromkeys(signals + assessment.evidence))[:8],
            detector_sources=list(dict.fromkeys(sources)),
            contextual_assessment=assessment,
            warnings=list(dict.fromkeys(warnings)),
            execution_mode=execution_mode,
            model_backed=model_backed,
        )
        if settings.enable_review_queue or settings.save_audit_events:
            case_id = save_event(
                request.content, request.source, content_hash,
                {**result.model_dump(mode="json"), "content_id": request.content_id, "case_id": None},
                needs_review,
            )
            if settings.enable_review_queue and needs_review:
                result.review_case_id = case_id
        return result


def get_pipeline() -> ModerationPipeline:
    # Simple process-level singleton to avoid recreating provider clients per request.
    global _pipeline
    try:
        return _pipeline
    except NameError:
        _pipeline = ModerationPipeline()
        return _pipeline
