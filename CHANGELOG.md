# Changelog

## 0.2.2
- Add a guided Jupyter notebook for the spam model experiment and error analysis.
- Add optional Jupyter, pandas, and matplotlib dependencies.
- Ignore Jupyter checkpoint files and document the notebook workflow.

## v0.2.1

- Deduplicate normalized message text before train/validation/test splitting to reduce evaluation leakage.
- Remove contradictory-label message groups and record deduplication counts.
- Expose model decision thresholds separately from raw model scores in the API and browser UI.
- Add a GitHub Actions workflow that trains on the official UCI SMS Spam Collection, verifies the generated artifact, summarizes held-out metrics, and uploads model/report artifacts.
- Use a multi-stage Docker build that trains the spam baseline during the image build and omits raw data and per-example test errors from the final runtime image.
- Record dataset license and attribution in model metadata and model cards.

The local execution environment could not reach the external UCI dataset host, so this source release contains no real-data model artifact or fabricated benchmark numbers. The network-enabled GitHub Actions/Docker build performs the real-data training step.
