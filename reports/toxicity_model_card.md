# TrustGuard Toxicity Classifier Model Card

- **Task:** binary overall toxicity classification
- **Selected model:** `word_char_tfidf_logistic_regression`
- **Dataset:** [Civil Comments](https://huggingface.co/datasets/google/civil_comments)
- **Dataset license:** CC0 1.0 (as listed on the Hugging Face dataset card)
- **Label rule:** score >= 0.5 is positive; this is an experiment choice
- **Decision threshold:** 0.80, selected on validation data only
- **Train / validation / test sample sizes after duplicate checks:** 59,721 / 14,905 / 14,899
- **Seed:** 42
- **Trained at (UTC):** 2026-10-26T04:07:03.260666+00:00

## Held-out test metrics

- Accuracy: 0.9405
- Balanced accuracy: 0.7105
- Precision: 0.7075
- Recall: 0.4366
- F1: 0.5400
- Macro-F1: 0.7541
- False-positive rate: 0.0157
- ROC AUC: 0.9144
- Average precision: 0.6249

## Intended use

Research, demonstration, and prioritization of potentially toxic comments for further review in TrustGuard. This model must not be the sole basis for bans, removals, or other consequential user actions. Combine model signals with policy logic, contextual analysis, uncertainty handling, and human review.

## Limitations

The source dataset consists of historical English-language news-site comments. It is not representative of all online communities, dialects, contexts, or current moderation policies. Label scores represent crowd annotation fractions, not unquestionable ground truth. The score is not independently calibrated. The current release does not establish subgroup fairness or cross-domain generalization. Review false positives and false negatives before reuse.

## Reproducibility

The notebook downloads the source's published split files, uses bounded stratified samples with seed 42, removes normalized duplicate text across sampled splits, compares candidates using validation macro-F1, and evaluates the selected model on a separate test sample. See `toxicity_evaluation.json` for the detailed audit trail.
