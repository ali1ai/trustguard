# TrustGuard

**TrustGuard is a privacy-conscious content-risk analysis demo for spam and policy-violating content. It combines explainable heuristics, optional locally trained classifiers, optional hosted moderation/context agents, a deterministic policy engine, and a human-review boundary. This repository is the initial public project baseline; no earlier numbered releases are required to use it.

The application does not require a hosted AI API. Running the Python app directly with no trained artifacts uses the explicitly labelled heuristic baseline. The Docker build is configured to download the official UCI SMS dataset, deduplicate it before splitting, train the spam baseline, and include the generated artifact in the final runtime image. That Docker build therefore requires external access to the UCI dataset host. We do not include fabricated model weights or benchmark results.

## What is included

- Repeatable training for spam and toxicity classification.
- Word n-gram and word-plus-character TF-IDF baselines.
- Naive Bayes and Logistic Regression candidates, selected using validation data.
- Normalized-text deduplication before splitting, conflicting-label exclusion, stratified train/validation/test split, threshold selection on validation only, and held-out test evaluation.
- Accuracy, positive precision/recall/F1, macro-F1, false-positive rate, confusion matrix, per-class report, and misclassified examples.
- Model artifacts carrying task, threshold, dataset, label rule, split, seed, and training metadata.
- Optional model loading at inference; an absent or invalid artifact degrades to the local heuristic path.
- A GitHub Actions workflow to train on the full UCI dataset and publish model/evaluation artifacts from a real-data run.
- A multi-stage Docker build that trains the spam model in the build stage by default and copies only the model and summary reports into the runtime image; raw training data is not shipped.
- Separate `model_scores` in the API response so trained-model output is not hidden inside the combined signal map.

- A guided Jupyter experiment notebook for dataset inspection, class-balance analysis, duplicate/conflicting-label checks, reproducible model training, validation-selected model comparison, held-out metrics, confusion-matrix visualization, error analysis, and sample predictions.
- Optional notebook dependencies, kept out of the default runtime install.

## Architecture

```text
Browser / API client
        |
        v
FastAPI validation
        |
        +---- explainable local heuristic signals
        +---- optional trained spam classifier
        +---- optional trained toxicity classifier
        +---- optional hosted moderation provider
        +---- optional contextual policy agent
                         |
                         +---- conditional independent auditor
        |
        v
Deterministic policy-to-action engine
        |
        +---- allow / label
        +---- reversible soft hold
        +---- human review recommendation
        |
        v
Optional explicitly enabled audit/review storage
```

The LLM is not an enforcement authority. The deterministic engine maps the signals to a recommendation, and uncertainty or disagreement goes to review. The public demo defaults to no content persistence and no reviewer endpoint.

## Technology

- Python 3.11+
- FastAPI and Pydantic
- scikit-learn, TF-IDF, Multinomial Naive Bayes, Logistic Regression
- joblib for project-generated model artifacts
- Optional Hugging Face `datasets` for streaming Civil Comments samples
- Optional OpenAI provider integration
- Docker, GitHub Actions, Render Blueprint, pytest

## Quick start (Windows PowerShell)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
uvicorn trustguard.api.main:app --reload
```

Open `http://127.0.0.1:8000`. API docs: `http://127.0.0.1:8000/docs`.

The default fallback does not require an API key or trained model. It uses a limited set of explicit patterns and is only a software smoke-test mode, not a reliable production moderation system.


## Guided ML experiment notebook

The repository includes `notebooks/01_spam_model_experiment.ipynb`. It is the **main guided ML experiment**, not the application's runtime entry point. The production API, model-training implementation, policy engine, and agents remain in `src/` and `scripts/` so the same code is testable and deployable outside Jupyter.

Install the notebook extras after creating and activating the project virtual environment:

```powershell
python -m pip install -e ".[dev,notebook]"
python scripts/download_sms_dataset.py
jupyter lab notebooks/01_spam_model_experiment.ipynb
```

Run the notebook from top to bottom. It will inspect the local dataset and call the same training function used by the command-line training script, then display the generated evaluation results. The official dataset must be downloaded separately and is excluded from Git. Do not commit local error exports or model artifacts unless you intentionally review and approve them.

## Train the spam model

The starter uses the [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection), a 5,574-message research dataset with `ham` and `spam` labels. The UCI record identifies its license as CC BY 4.0. Keep the attribution and do not claim that SMS performance automatically generalizes to reviews, forums, or social media.

```powershell
python scripts/download_sms_dataset.py
python scripts/train_spam_classifier.py
```

The download script writes `data/raw/sms_spam.csv`. Raw data is excluded from Git. The trainer normalizes message text for deduplication before splitting; identical messages are not allowed to appear across train/validation/test. Rows whose normalized text has contradictory labels are excluded and counted in the report. If automatic download is blocked, obtain the dataset from the official UCI page, extract `SMSSpamCollection`, and run:

```powershell
python scripts/convert_sms_dataset.py "C:\path\to\SMSSpamCollection"
python scripts/train_spam_classifier.py
```

Generated outputs:

- `models/spam_classifier.joblib`
- `reports/spam_evaluation.json`
- `reports/spam_test_errors.csv`
- `reports/spam_model_card.md`

The trainer compares word-only TF-IDF + Multinomial Naive Bayes with word-and-character TF-IDF + Logistic Regression. It chooses an operating threshold on the validation split, selects the candidate using validation macro-F1, and reports held-out test metrics. The test set is not used to select the model or threshold.

## Train the toxicity model (optional)

The optional baseline uses the [Civil Comments dataset card](https://huggingface.co/datasets/google/civil_comments). The dataset card identifies its text and labels as CC0 1.0. Its `toxicity` score is converted to a binary target using a configurable threshold; it does **not** mean the model can reliably separate harassment, hate, threats, and every other policy category.

```powershell
python -m pip install -e ".[training]"
python scripts/train_toxicity_classifier.py --max-rows 50000 --toxicity-threshold 0.5
```

This streams a shuffled, bounded sample instead of committing or downloading the entire corpus into the Git repository. Increase the sample only if your machine has enough RAM and time. Generated outputs use the `toxicity_` prefix under `models/` and `reports/`.

## Integrate trained models into the app

The runtime looks for these project-generated files by default:

- `models/spam_classifier.joblib`
- `models/toxicity_classifier.joblib`

Once an artifact exists, restart the app. `GET /api/v1/config` reports which model tasks loaded. The moderation response returns `model_scores`, `model_thresholds`, combined `scores`, detector sources, action, policy version, and warnings. The web interface separates model scores from combined heuristic/provider signals.

Before using a model output to inform real enforcement:

1. Inspect the held-out report and the locally generated misclassified examples.
2. Review dataset license and source attribution.
3. Review the model card and only load artifacts generated by your own trusted training process. joblib/pickle files can execute code when loaded.
4. For the public Render instance, the default multi-stage Docker build trains the spam model and copies that artifact into the runtime image. The raw dataset and per-example error CSV are removed before the runtime image is assembled. The separate GitHub Actions workflow uploads the model and aggregate report for inspection.

Do not commit raw datasets. Generated error CSVs can contain full message text and are excluded from Git and the public workflow artifact by default. Keep them local unless every row has been reviewed and disclosure is appropriate.

## Optional hosted LLM integration

To enable hosted provider moderation and contextual agents in a private environment, install the optional SDK and configure `.env`:

```powershell
python -m pip install -e ".[dev,llm]"
```

Set `OPENAI_API_KEY` and `ENABLE_LLM_PROVIDERS=true`, then restart. The public demo should keep external provider calls disabled until authentication, rate limiting, cost limits, and abuse controls are established.

## Docker

```powershell
Copy-Item .env.example .env
docker compose up --build
```

Docker runs the same single-service FastAPI app. By default, the build downloads the official UCI dataset and trains the spam classifier in an intermediate build stage. To build an offline heuristic-only image for interface development, pass `--build-arg TRUSTGUARD_TRAIN_SPAM_MODEL=0` to `docker build` (or set that build arg in your own CI). The public Render blueprint uses the default `1` setting.

## Deploy a public demo

The `render.yaml` is a Render Blueprint starting point. The multi-stage Docker build downloads the UCI SMS dataset and trains the spam model by default before building the slim runtime image. This means the first deployed image can be ML-backed without a third-party AI API key, provided the build can reach the UCI dataset host. If dataset download or training fails, the image build fails rather than silently publishing a purported ML-backed demo. Connect the GitHub repository to Render, deploy the service, then test the assigned URL, `/health`, `/api/v1/config`, and `/docs`. Render free services can spin down after inactivity; cold starts should be expected. The app defaults to no content storage and no persistent audit/review queue.

This repository has not been deployed to your hosting account. Account connection and creation of the public URL must happen in your GitHub/Render accounts.

## API examples

```http
GET /health
GET /api/v1/config
POST /api/v1/moderate
Content-Type: application/json

{"content":"Limited time offer! Claim your prize now https://a.example https://b.example https://c.example","source":"demo"}
```

### Important response semantics

- `scores` is the conservative maximum across available evidence sources; it is not a calibrated probability or a learned ensemble.
- `model_scores` contains the separately displayed output of the trained local text classifiers.
- model thresholds are selected on validation data, but scores are not independently probability-calibrated.
- `action` is a recommendation (`allow`, `label`, `soft_hold`, or `queue_review`), not a real-world enforcement action.

## Evaluation and tests

```powershell
pytest -q
ruff check src tests scripts
```

Training a classifier writes machine-readable metrics, a local-only test-error CSV, and a model card. The GitHub Actions workflow uploads only the model, aggregate JSON report, and model card. Unit tests use a small synthetic fixture only to test the training software; those tests are **not** the published model benchmark. Do not quote test-fixture metrics as evidence of real-world classifier quality.

## Current state and next milestones

The repository currently includes a single-service API and browser UI, optional locally trained spam/toxicity classifiers, a guided Jupyter experiment notebook, duplicate-safe benchmark tooling, automated training workflow, Docker configuration, and a Render deployment blueprint. The real-data benchmark and public deployment still need to be run and verified in the project owner's GitHub/hosting environment.

Next milestones:

- Run the real-data spam training workflow and review held-out metrics and errors.
- Add a context challenge set for quoted abuse, counterspeech, and prompt injection.
- Strengthen agent contract tests and secure the human-review workflow.
- Add rate limiting, cost controls, latency/quality observability, drift tracking, and model/policy version monitoring.
- Deploy and verify the public demo with a reviewed model artifact, benchmark report, model card, threat model, architecture diagram, and recorded demonstration.

## Known limitations

Running the Python API directly without a model artifact is heuristic-only; the default Docker build is configured to generate and load the spam model. SMS spam and Civil Comments are narrow historical English-language datasets; their labels and domains do not represent every community. This is a research and portfolio project, not a production enforcement service, legal-compliance product, or substitute for trained human moderators. Do not use it to make consequential decisions about real users.
