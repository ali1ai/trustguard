# TrustGuard Phase 2 files

This overlay adds the next research artifacts to the existing TrustGuard repository without replacing existing files:

- `notebooks/02_toxicity_modeling.ipynb`: standalone Google Colab notebook for toxicity + subtype modeling.
- `docs/dataset-strategy.md`: selected data sources, label semantics, and experiment rules.
- `docs/architecture-roadmap.md`: target system architecture, component responsibilities, and evaluation sequence.

## How to add these files

1. Download and extract this ZIP.
2. Copy the ZIP's `notebooks/` and `docs/` folders into the **root of your existing local `trustguard` repository**.
3. Merge folders if Windows prompts. The new files use new names, so existing application files should not need replacing.
4. In GitHub Desktop, review the changes, commit with `Add toxicity modeling notebook and architecture plan`, and push.
5. Open `notebooks/02_toxicity_modeling.ipynb` in Colab via **File → Upload notebook**.

## Notebook notes

The notebook downloads the Civil Comments dataset from the Hugging Face Hub's Parquet conversion branch. It needs an internet-enabled Colab runtime. The full dataset download is about 422 MB compressed; defaults train on 60k comments and evaluate on 15k validation + 15k test comments. Reduce the constants in the configuration cell if memory or time is limited.

The notebook is intentionally separate from the deployed app. It generates an app-compatible binary `toxicity_classifier.joblib` artifact plus aggregate reports; subtype classifiers are evaluated and reported but are not yet integrated into the runtime API. Row-level test errors contain comment text and are excluded from the downloadable bundle.
