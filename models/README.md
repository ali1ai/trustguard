# Trained model artifacts

The multi-stage Docker build trains the spam classifier using the official UCI SMS Spam Collection and includes the generated `spam_classifier.joblib` in the final image. Raw dataset files and the per-example error CSV are removed before the runtime image is produced. The model and evaluation report are generated from real data only when the build can download the source dataset.

For an inspectable evaluation run, use GitHub Actions → **Train and benchmark spam model** (or run `python scripts/download_sms_dataset.py` and `python scripts/train_spam_classifier.py` locally). The workflow uploads the model, JSON metrics, and model card as a 30-day workflow artifact. The per-example error CSV is not uploaded by default because it contains raw message text; generate it locally for detailed error analysis.

Do not commit raw datasets. Do not accept joblib files from untrusted users: loading pickle-based artifacts can execute code.
