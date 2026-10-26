# Build a real spam-model artifact in the build stage, then keep raw training data
# out of the final runtime image. This makes online builds self-contained when the
# build environment can reach the official UCI dataset host.
FROM python:3.12-slim AS trainer
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
ARG TRUSTGUARD_TRAIN_SPAM_MODEL=1
COPY pyproject.toml README.md ./
COPY src ./src
COPY scripts ./scripts
COPY data ./data
COPY models ./models
COPY reports ./reports
RUN pip install . \
    && if [ "$TRUSTGUARD_TRAIN_SPAM_MODEL" = "1" ]; then \
         python scripts/download_sms_dataset.py \
         && python scripts/train_spam_classifier.py; \
       fi \
    && rm -f data/raw/sms_spam.csv reports/spam_test_errors.csv

FROM python:3.12-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY web ./web
COPY policies ./policies
COPY --from=trainer /app/models ./models
COPY --from=trainer /app/reports ./reports
RUN pip install . \
    && useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/data \
    && chown -R appuser:appuser /app
USER appuser
EXPOSE 8000
CMD ["sh", "-c", "uvicorn trustguard.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
