from trustguard.services.signals import local_fallback_signals, spam_signals


def test_spam_signals_are_explainable():
    score, signals = spam_signals("CLICK HERE limited time https://a.example https://b.example https://c.example")
    assert score > 0
    assert "multiple_urls" in signals
    assert "promotional_phrase" in signals


def test_normal_message_has_no_strong_spam_score():
    score, signals = spam_signals("Could you send me the agenda before tomorrow's meeting?")
    assert score < 0.5
    assert signals == []


def test_local_fallback_detects_explicit_threat_pattern():
    scores, signals = local_fallback_signals("I will kill you")
    assert scores["threat"] >= 0.9
    assert "explicit_threat_phrase" in signals
