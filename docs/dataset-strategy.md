# TrustGuard Dataset Strategy

## Decision

Use three dataset families for distinct experiments. Do not merge them into one undifferentiated label set.

### 1. SMS Spam Collection: spam baseline

- Source: <https://archive.ics.uci.edu/dataset/228/sms+spam+collection>
- Task: SMS spam versus legitimate message.
- Use: establish a small, quick, interpretable baseline and validate the model-loading/deployment path.
- Known limitation: SMS content and spam behavior differ from comment-platform moderation.
- License: UCI lists CC BY 4.0; preserve attribution.

### 2. Civil Comments: toxicity and subtype baseline

- Source/dataset card: <https://huggingface.co/datasets/google/civil_comments>
- Parquet source branch used by Notebook 02: `refs/convert/parquet`.
- Dataset card describes public comments from the Civil Comments archive (2015–2017), with target scores for toxicity, severe toxicity, obscenity, threats, insults, identity attacks, and sexual explicitness.
- The dataset card lists CC0 1.0.
- The labels are scores in [0, 1] representing annotation fractions. Notebook 02 uses score >= 0.5 as an explicit experiment rule. It is not a universal moderation policy.
- Use official train/validation/test source splits. Take bounded, reproducible, stratified samples for Colab. Check exact normalized text duplicates across sampled partitions before evaluation.
- Known limitations: historical, English-language, news-site comments; labels are imperfect crowd judgments; the basic release does not provide all identity-subgroup annotations needed for a complete fairness analysis.

### 3. Jigsaw Unintended Bias / identity subgroup evaluation: planned

- Source: <https://huggingface.co/datasets/google/jigsaw_unintended_bias>
- Dataset card lists CC0 1.0 and includes toxicity-related labels plus identity-attribute annotation fields.
- Use in a separate notebook for subgroup false-positive/false-negative analysis and bias-aware evaluation. Do not claim fairness based only on aggregate Civil Comments metrics.
- Before use, inspect the data card and annotation coverage. Identity attributes can be missing for many examples; subgroup results must report support counts and uncertainty.

## Experimental rules

1. Keep raw datasets out of Git. Record dataset identifier, source URL/revision, license, label rule, split/sample sizes, seed, and preprocessing in every report.
2. Do not mix datasets with different task definitions without an explicit harmonization study.
3. Do not make a new random split from validation/test rows that would break the source's published train/validation/test partitions.
4. Tune model/decision thresholds using validation data only. Do not tune on test metrics.
5. Remove or audit exact normalized text duplicates across sampled splits; log how many rows were removed.
6. Report per-label metrics, positive support, false-positive rate, false-negative rate, and macro-F1 where meaningful. For rare labels, emphasize support and uncertainty.
7. Keep row-level misclassified text local. Publish aggregate metrics and reviewed examples only when public disclosure is appropriate.
8. Treat annotation scores as imperfect labels. Clearly document any thresholding choices, especially conversion from crowd scores to binary targets.
9. Do not use benchmark accuracy to justify automatic bans or other consequential enforcement. Model outputs are signals for a policy engine and human-review workflow.

## Roadmap

- Notebook 01: SMS spam baseline (implemented).
- Notebook 02: Civil Comments toxicity and subtype baselines (new).
- Notebook 03: identity-subgroup metrics and robustness (planned).
- Notebook 04: contextual challenge set (quoted abuse, counterspeech, benign identity discussion, threats, prompt injection) (planned).
- Notebook 05: classifier-only versus LLM-only versus hybrid agent, latency/cost/error comparison (planned).
