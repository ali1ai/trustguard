"""Environment-backed settings for TrustGuard."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    return default if value is None else value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY") or None
    moderation_model: str = os.getenv("OPENAI_MODERATION_MODEL", "omni-moderation-latest")
    agent_model: str = os.getenv("OPENAI_AGENT_MODEL", "gpt-4.1-mini")
    enable_llm_providers: bool = _env_bool("ENABLE_LLM_PROVIDERS", False)
    enable_context_agent: bool = _env_bool("ENABLE_CONTEXT_AGENT", True)
    enable_audit_agent: bool = _env_bool("ENABLE_AUDIT_AGENT", True)
    db_path: Path = Path(os.getenv("TRUSTGUARD_DB_PATH", "./data/trustguard.db"))
    policy_version: str = os.getenv("POLICY_VERSION", "2026-10-25-demo-v2")
    review_api_token: str | None = os.getenv("REVIEW_API_TOKEN") or None
    # Public demos must not persist submitted content or enable the reviewer API by default.
    store_content: bool = _env_bool("STORE_CONTENT", False)
    enable_review_queue: bool = _env_bool("ENABLE_REVIEW_QUEUE", False)
    save_audit_events: bool = _env_bool("SAVE_AUDIT_EVENTS", False)
    spam_model_path: Path = Path(os.getenv("TRUSTGUARD_SPAM_MODEL_PATH", "models/spam_classifier.joblib"))
    toxicity_model_path: Path = Path(os.getenv("TRUSTGUARD_TOXICITY_MODEL_PATH", "models/toxicity_classifier.joblib"))


settings = Settings()
