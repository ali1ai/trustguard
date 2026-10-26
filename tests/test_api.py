from fastapi.testclient import TestClient

from trustguard.api.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_home_serves_interactive_web_interface():
    response = client.get("/")
    assert response.status_code == 200
    assert "TrustGuard" in response.text
    assert "Analyze content" in response.text


def test_public_config_does_not_expose_secrets():
    response = client.get("/api/v1/config")
    assert response.status_code == 200
    body = response.json()
    assert body["execution_mode"] in {"local_heuristics", "provider_moderation_only", "context_agent_only", "hybrid_llm_and_moderation", "trained_ml_only", "hybrid_ml_and_llm"}
    assert "api_key" not in body
    assert "review_api_token" not in body
    assert "trained_models_available" in body


def test_moderate_rejects_blank_content():
    response = client.post("/api/v1/moderate", json={"content": "   "})
    assert response.status_code == 422


def test_moderation_endpoint_returns_auditable_result():
    response = client.post(
        "/api/v1/moderate",
        json={"content": "CLICK HERE limited time https://a.example https://b.example https://c.example", "source": "test"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["action"] in {"allow", "label", "soft_hold", "queue_review"}
    assert "scores" in body
    assert "policy_version" in body
    assert body["requires_human_review"] is True
    assert "execution_mode" in body
    assert "model_backed" in body
    assert "model_scores" in body


def test_review_queue_is_disabled_by_default():
    response = client.get("/api/v1/review-queue")
    assert response.status_code in {401, 404, 503}
