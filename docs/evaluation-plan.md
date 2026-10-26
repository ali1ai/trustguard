# TrustGuard Evaluation Plan (v0.2.1)

## Questions

1. Does a trainable local baseline detect spam and toxicity better than simple heuristic patterns on a held-out split?
2. What precision/recall trade-off is produced by the chosen decision threshold?
3. What are the false-positive rate and the most common error patterns?
4. Does performance differ materially by class prevalence or dataset domain?

## Data

- Spam: UCI SMS Spam Collection, CC BY 4.0, official dataset record https://archive.ics.uci.edu/dataset/228/sms+spam+collection.
- Toxicity: Civil Comments dataset card, CC0 1.0, https://huggingface.co/datasets/google/civil_comments. The optional pipeline streams and samples a bounded number of records.

Do not commit raw datasets. Record source, version/selection process, threshold, seed, and sample size in every report. The two datasets test two different tasks and should not be merged into one metric.

## Protocol

- Normalize message casing/whitespace for duplicate detection, collapse same-label duplicate messages, and exclude normalized text groups with contradictory labels before splitting.
- Use stratified 60% train / 20% validation / 20% test on the remaining unique messages.
- Keep the test split untouched until selecting the candidate and operating threshold.
- Compare word TF-IDF + Multinomial Naive Bayes against word+character TF-IDF + Logistic Regression.
- Select the decision threshold on validation data by maximizing positive-class F1, subject to a positive precision floor of 0.80 when possible.
- Select the estimator with highest validation macro-F1.
- Evaluate once on the held-out test split.

## Required metrics

Accuracy, positive precision, positive recall, positive F1, macro-F1, false-positive rate, confusion matrix, per-class precision/recall/F1/support, and inspectable false-positive/false-negative examples.

## Guardrails

- Report metrics at the selected threshold, not only default threshold 0.5.
- Do not call model outputs calibrated probabilities.
- Do not describe the toxicity label as synonymous with harassment or hate.
- The SMS corpus is not representative of every spam domain; Civil Comments is historical English comment data.
- The misclassified examples CSV can contain raw text and is excluded from Git and public workflow artifacts by default; keep it local unless the content has been reviewed for disclosure.
- Synthetic unit fixtures validate code paths only; do not use them as a quality benchmark.
- No automatic account bans or permanent deletion are in scope.
