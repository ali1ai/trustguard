from __future__ import annotations

import secrets
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from trustguard.schemas import ModerationRequest, ModerationResult, ReviewCase, ReviewDecision
from trustguard.services.pipeline import get_pipeline
from trustguard.services.storage import get_pending_cases, init_db, submit_review
from trustguard.settings import settings

_SOURCE_WEB_DIR = Path(__file__).resolve().parents[3] / "web"
_RUNTIME_WEB_DIR = Path.cwd() / "web"
WEB_DIR = _SOURCE_WEB_DIR if (_SOURCE_WEB_DIR / "index.html").exists() else _RUNTIME_WEB_DIR


@asynccontextmanager
async def lifespan(_: FastAPI):
    if settings.enable_review_queue or settings.save_audit_events:
        init_db()
    yield


app = FastAPI(
    title="TrustGuard",
    description="Privacy-conscious content risk analysis with an explicit human-review boundary.",
    version="0.1.0",
    docs_url="/docs",
    redoc_url=None,
    lifespan=lifespan,
)
if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.middleware("http")
async def security_headers(request, call_next):
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    response.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
    response.headers.setdefault("Cache-Control", "no-store" if request.url.path.startswith("/api/") else "no-cache")
    return response


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    index = WEB_DIR / "index.html"
    if not index.exists():
        raise HTTPException(status_code=503, detail="TrustGuard web interface is missing")
    return FileResponse(index, media_type="text/html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "trustguard", "version": app.version}


@app.get("/api/v1/config")
def public_config() -> dict[str, object]:
    pipeline = get_pipeline()
    trained_models = pipeline.model_registry.available_tasks
    provider_active = pipeline.detector is not None
    context_active = pipeline.context_agent is not None
    if trained_models and (provider_active or context_active):
        mode = "hybrid_ml_and_llm"
    elif trained_models:
        mode = "trained_ml_only"
    elif provider_active and context_active:
        mode = "hybrid_llm_and_moderation"
    elif provider_active:
        mode = "provider_moderation_only"
    elif context_active:
        mode = "context_agent_only"
    else:
        mode = "local_heuristics"
    return {
        "service": "TrustGuard",
        "version": app.version,
        "policy_version": settings.policy_version,
        "execution_mode": mode,
        "model_backed": mode != "local_heuristics",
        "trained_models_available": trained_models,
        "review_queue_enabled": settings.enable_review_queue,
        "content_persistence_enabled": settings.store_content,
        "notice": (
            "Local heuristic demo only. Results are illustrative and are not a trained moderation model."
            if mode == "local_heuristics"
            else "Model-assisted analysis is enabled. Scores are signals, not calibrated probabilities."
        ),
    }


@app.post("/api/v1/moderate", response_model=ModerationResult)
def moderate(payload: ModerationRequest) -> ModerationResult:
    return get_pipeline().run(payload)


def require_reviewer_token(x_reviewer_token: str | None = Header(default=None)) -> None:
    if not settings.enable_review_queue:
        raise HTTPException(status_code=404, detail="Review queue is disabled")
    expected = settings.review_api_token
    if not expected:
        raise HTTPException(status_code=503, detail="Review queue requires REVIEW_API_TOKEN to be configured")
    if not x_reviewer_token or not secrets.compare_digest(x_reviewer_token, expected):
        raise HTTPException(status_code=401, detail="Valid reviewer token required")


@app.get(
    "/api/v1/review-queue",
    response_model=list[ReviewCase],
    dependencies=[Depends(require_reviewer_token)],
)
def review_queue(limit: int = Query(default=50, ge=1, le=200)) -> list[ReviewCase]:
    return [ReviewCase(**case) for case in get_pending_cases(limit)]


@app.patch(
    "/api/v1/review-queue/{case_id}",
    dependencies=[Depends(require_reviewer_token)],
)
def review_case(case_id: int, payload: ReviewDecision) -> dict[str, str | int]:
    updated = submit_review(case_id, payload.decision, payload.reviewer_note)
    if not updated:
        raise HTTPException(status_code=404, detail="Pending review case not found")
    return {"case_id": case_id, "status": "completed", "decision": payload.decision}
