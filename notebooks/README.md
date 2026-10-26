# TrustGuard notebooks

`01_spam_model_experiment.ipynb` is the guided ML experiment for the SMS spam baseline. It reads the locally downloaded dataset, summarizes data quality and labels, calls the same training routine as `scripts/train_spam_classifier.py`, and visualizes held-out results and example errors.

The notebook is not the application runtime. Keep reusable and deployable implementation in `src/trustguard/` and command-line workflows in `scripts/`.

Install extras: `python -m pip install -e ".[dev,notebook]"`. Download the dataset using `python scripts/download_sms_dataset.py` before running the notebook. Data and per-example error exports are not intended for Git commits.
