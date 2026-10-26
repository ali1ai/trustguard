# Public Demo Deployment: Render

## Scope

This runbook deploys the single-service TrustGuard UI and API from `render.yaml`. The multi-stage Docker build downloads the official UCI SMS Spam Collection, removes normalized duplicates before splitting, trains the spam baseline, and copies the model and summary report into the final runtime image. Raw data and the per-example test-error CSV are removed before the runtime image is built. The build needs network access to the official UCI dataset host. The application does not persist visitor messages by default.

## Before deploying

- Push the project to a GitHub repository.
- Confirm the repository's GitHub Actions workflows are enabled. `Train and benchmark spam model` runs when relevant training files change and can also be triggered manually.
- Keep `.env`, API keys, databases, `.venv`, `__pycache__`, `.pytest_cache`, raw datasets, and per-example error exports out of the repository.
- Keep `ENABLE_LLM_PROVIDERS=false`, `STORE_CONTENT=false`, `ENABLE_REVIEW_QUEUE=false`, and `SAVE_AUDIT_EVENTS=false` for the public demo.

## Deploy

1. Sign in to Render and connect the Git provider.
2. Create a Blueprint using this repository.
3. Confirm Render detects `render.yaml` and the Docker web service named `trustguard-demo`.
4. Start the deployment. The Docker build downloads the UCI dataset and trains a model. If that build step fails, the service should not silently deploy while claiming a trained spam model is available.
5. Visit the service URL and test the built-in sample content.
6. Check `/health`, `/api/v1/config`, and `/docs`.

## Post-deploy checks

- `GET /api/v1/config` should list `spam` in `trained_models_available` after a successful default Docker build.
- The response should identify `trained_ml_only` when the classifier is loaded and no hosted providers are enabled.
- Treat model scores as signals, not calibrated probabilities.
- No visitor message or moderation result should be persisted in default mode.
- `GET /api/v1/review-queue` stays disabled.
- The service has no external model key configured, so anonymous visitors cannot spend a hosted-model budget.

## Known platform trade-offs

Free services may spin down when idle and local runtime files are ephemeral. Cold starts are expected. The deployed container includes the trained model file, so it does not depend on a writable runtime filesystem for model availability.

## Do not enable hosted model calls yet

Before enabling `ENABLE_LLM_PROVIDERS`, add and test a public API abuse policy: global/per-client rate limits, request concurrency limits, provider timeouts, a hard usage budget or gateway quota, response-size limits, and monitoring. The initial public demo intentionally keeps external provider calls disabled.
