"""Cheap, explainable signals. These are triage features, not a production classifier."""
from __future__ import annotations

import re

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
REPEATED_TOKEN_RE = re.compile(r"\b(\w{2,})(?:\s+\1){3,}\b", re.IGNORECASE)

PROMO_PHRASES = (
    "click here", "buy now", "limited time", "act now", "free crypto",
    "guaranteed profit", "claim your prize", "dm me to invest", "make money fast",
    "subscribe now", "work from home and earn", "double your money",
)


def spam_signals(text: str) -> tuple[float, list[str]]:
    """Return a conservative heuristic score and short, auditable signal names."""
    signals: list[str] = []
    lowered = text.lower()
    url_count = len(URL_RE.findall(text))
    email_count = len(EMAIL_RE.findall(text))
    if url_count >= 3:
        signals.append("multiple_urls")
    if email_count >= 2:
        signals.append("multiple_email_addresses")
    matched_phrases = [phrase for phrase in PROMO_PHRASES if phrase in lowered]
    if matched_phrases:
        signals.append("promotional_phrase")
    if REPEATED_TOKEN_RE.search(text):
        signals.append("repeated_token_pattern")
    alpha = [ch for ch in text if ch.isalpha()]
    if len(alpha) >= 20 and sum(ch.isupper() for ch in alpha) / len(alpha) > 0.75:
        signals.append("excessive_uppercase")
    # Repetition is only a weak prior. A single URL or promotional phrase is not enough to auto-hold.
    score = min(0.98, 0.12 * len(signals) + (0.30 if url_count >= 5 else 0.0))
    if url_count >= 3 and matched_phrases:
        score = max(score, 0.82)
    if url_count >= 5 and len(matched_phrases) >= 2:
        score = max(score, 0.94)
    return round(score, 4), signals


def local_fallback_signals(text: str) -> tuple[dict[str, float], list[str]]:
    """Small fallback for offline demos. Never represented as equivalent to a trained model."""
    spam_score, signals = spam_signals(text)
    lowered = text.lower()
    scores = {key: 0.0 for key in (
        "spam", "harassment", "hate", "threat", "sexual", "violence", "self_harm", "toxicity", "other"
    )}
    scores["spam"] = spam_score
    # High-precision-only examples for a local demo. Context is unresolved without an LLM.
    threat_patterns = (
        r"\bi will kill you\b", r"\bi'm going to kill you\b", r"\bi am going to kill you\b",
        r"\bi will hurt you\b", r"\bi'm going to hurt you\b",
    )
    if any(re.search(pattern, lowered) for pattern in threat_patterns):
        scores["threat"] = 0.92
        signals.append("explicit_threat_phrase")
    harassment_patterns = (r"\byou are an idiot\b", r"\byou're an idiot\b", r"\byou are worthless\b")
    if any(re.search(pattern, lowered) for pattern in harassment_patterns):
        scores["harassment"] = 0.78
        signals.append("direct_insult_pattern")
    return scores, signals
