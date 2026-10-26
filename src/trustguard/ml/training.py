"""Shared train/validation/test pipeline and reports for TrustGuard classifiers."""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline

ARTIFACT_FORMAT_VERSION = 1


def build_candidates(seed: int) -> dict[str, Pipeline]:
    """Two transparent baselines: word-only Naive Bayes and word+character logistic regression."""
    word = TfidfVectorizer(
        lowercase=True, strip_accents="unicode", ngram_range=(1, 2), min_df=2,
        max_features=50_000, sublinear_tf=True, dtype=np.float32,
    )
    word_char = FeatureUnion([
        ("word", TfidfVectorizer(
            lowercase=True, strip_accents="unicode", ngram_range=(1, 2), min_df=2,
            max_features=50_000, sublinear_tf=True, dtype=np.float32,
        )),
        ("char", TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), min_df=2,
            max_features=60_000, sublinear_tf=True, dtype=np.float32,
        )),
    ])
    return {
        "word_tfidf_multinomial_nb": Pipeline([
            ("tfidf", word),
            ("classifier", MultinomialNB(alpha=0.25)),
        ]),
        "word_char_tfidf_logistic_regression": Pipeline([
            ("features", word_char),
            ("classifier", LogisticRegression(
                max_iter=1200, class_weight="balanced", random_state=seed, solver="liblinear", C=2.0,
            )),
        ]),
    }


def choose_threshold(y_true: Sequence[int], probabilities: Sequence[float]) -> tuple[float, dict[str, float]]:
    """Choose an operating point on validation data only; prioritize F1 with a precision floor."""
    best: tuple[float, float, float] | None = None
    fallback: tuple[float, float, float] | None = None
    for threshold in np.arange(0.10, 0.951, 0.01):
        predicted = (np.asarray(probabilities) >= threshold).astype(int)
        precision = float(precision_score(y_true, predicted, zero_division=0))
        recall = float(recall_score(y_true, predicted, zero_division=0))
        f1 = float(f1_score(y_true, predicted, zero_division=0))
        candidate = (f1, float(threshold), precision)
        if fallback is None or candidate[0] > fallback[0]:
            fallback = candidate
        if precision >= 0.80 and (best is None or candidate[0] > best[0]):
            best = candidate
    selected = best or fallback or (0.0, 0.5, 0.0)
    return selected[1], {
        "validation_f1": selected[0],
        "validation_precision": selected[2],
        "precision_floor_met": float(best is not None),
    }


def _metrics(y_true: Sequence[int], probabilities: Sequence[float], threshold: float) -> dict:
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred = (np.asarray(probabilities) >= threshold).astype(int)
    matrix = confusion_matrix(y_true_arr, y_pred, labels=[0, 1])
    tn, fp, fn, tp = [int(v) for v in matrix.ravel()]
    return {
        "threshold": round(float(threshold), 4),
        "n_test": int(len(y_true_arr)),
        "accuracy": round(float(accuracy_score(y_true_arr, y_pred)), 5),
        "precision_positive": round(float(precision_score(y_true_arr, y_pred, zero_division=0)), 5),
        "recall_positive": round(float(recall_score(y_true_arr, y_pred, zero_division=0)), 5),
        "f1_positive": round(float(f1_score(y_true_arr, y_pred, zero_division=0)), 5),
        "macro_f1": round(float(f1_score(y_true_arr, y_pred, average="macro", zero_division=0)), 5),
        "false_positive_rate": round(float(fp / max(1, fp + tn)), 5),
        "confusion_matrix_labels_0_1": [[tn, fp], [fn, tp]],
        "counts": {"true_negative": tn, "false_positive": fp, "false_negative": fn, "true_positive": tp},
        "classification_report": classification_report(
            y_true_arr, y_pred, labels=[0, 1], target_names=["negative", "positive"],
            output_dict=True, zero_division=0,
        ),
    }


def train_and_save(
    texts: Sequence[str],
    labels: Sequence[int],
    *,
    task: str,
    dataset_name: str,
    dataset_source: str,
    label_rule: str,
    dataset_license: str = "Unspecified; verify before redistribution",
    output_dir: str | Path = "models",
    report_dir: str | Path = "reports",
    seed: int = 42,
) -> dict:
    if task not in {"spam", "toxicity"}:
        raise ValueError("task must be 'spam' or 'toxicity'")
    if len(texts) != len(labels) or len(texts) < 50:
        raise ValueError("At least 50 aligned text/label rows are required")
    if len(set(int(value) for value in labels)) != 2:
        raise ValueError("The dataset must contain both negative class 0 and positive class 1")
    input_rows = len(texts)
    nonempty = [(str(text).strip(), int(label)) for text, label in zip(texts, labels) if str(text).strip()]
    empty_rows_removed = input_rows - len(nonempty)

    # Prevent leakage from repeated messages crossing train/validation/test boundaries.
    # Case-folding and whitespace normalization catches trivial formatting variants without
    # stripping punctuation or other potentially informative message content.
    groups: dict[str, list[tuple[str, int]]] = {}
    for text, label in nonempty:
        duplicate_key = " ".join(text.casefold().split())
        groups.setdefault(duplicate_key, []).append((text, label))

    deduplicated: list[tuple[str, int]] = []
    duplicate_rows_removed = 0
    conflicting_text_groups_removed = 0
    conflicting_rows_removed = 0
    for rows in groups.values():
        unique_labels = {label for _, label in rows}
        if len(unique_labels) > 1:
            # A normalized message with contradictory labels is ambiguous. Excluding the
            # entire group is safer than letting a label conflict contaminate evaluation.
            conflicting_text_groups_removed += 1
            conflicting_rows_removed += len(rows)
            continue
        deduplicated.append(rows[0])
        duplicate_rows_removed += len(rows) - 1

    if len(deduplicated) < 50 or len({label for _, label in deduplicated}) != 2:
        raise ValueError(
            "After removing empty rows, duplicates, and conflicting labels, "
            "at least 50 unique messages and both classes must remain"
        )
    texts_clean = [text for text, _ in deduplicated]
    labels_clean = np.asarray([label for _, label in deduplicated], dtype=int)
    indices = np.arange(len(texts_clean))
    train_val_idx, test_idx = train_test_split(
        indices, test_size=0.20, random_state=seed, stratify=labels_clean,
    )
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=0.25, random_state=seed, stratify=labels_clean[train_val_idx],
    )
    x_train = [texts_clean[i] for i in train_idx]
    y_train = labels_clean[train_idx]
    x_val = [texts_clean[i] for i in val_idx]
    y_val = labels_clean[val_idx]
    x_test = [texts_clean[i] for i in test_idx]
    y_test = labels_clean[test_idx]

    candidates = build_candidates(seed)
    candidate_results: dict[str, dict] = {}
    fitted: dict[str, Pipeline] = {}
    for name, estimator in candidates.items():
        estimator.fit(x_train, y_train)
        val_probabilities = estimator.predict_proba(x_val)[:, list(estimator.classes_).index(1)]
        threshold, validation_selection = choose_threshold(y_val, val_probabilities)
        val_metrics = _metrics(y_val, val_probabilities, threshold)
        candidate_results[name] = {
            "threshold": threshold,
            "validation_selection": validation_selection,
            "validation_macro_f1": val_metrics["macro_f1"],
            "validation_precision_positive": val_metrics["precision_positive"],
            "validation_recall_positive": val_metrics["recall_positive"],
        }
        fitted[name] = estimator

    selected_name = max(candidate_results, key=lambda name: candidate_results[name]["validation_macro_f1"])
    estimator = fitted[selected_name]
    threshold = float(candidate_results[selected_name]["threshold"])
    test_probabilities = estimator.predict_proba(x_test)[:, list(estimator.classes_).index(1)]
    final_metrics = _metrics(y_test, test_probabilities, threshold)
    y_pred = (test_probabilities >= threshold).astype(int)

    output_dir = Path(output_dir)
    report_dir = Path(report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / f"{task}_classifier.joblib"
    report_path = report_dir / f"{task}_evaluation.json"
    errors_path = report_dir / f"{task}_test_errors.csv"
    card_path = report_dir / f"{task}_model_card.md"

    metadata = {
        "model_name": selected_name,
        "dataset_name": dataset_name,
        "dataset_source": dataset_source,
        "dataset_license": dataset_license,
        "label_rule": label_rule,
        "rows_total": int(len(texts_clean)),
        "input_rows": int(input_rows),
        "empty_rows_removed": int(empty_rows_removed),
        "duplicate_rows_removed": int(duplicate_rows_removed),
        "conflicting_text_groups_removed": int(conflicting_text_groups_removed),
        "conflicting_rows_removed": int(conflicting_rows_removed),
        "rows_train": int(len(train_idx)),
        "rows_validation": int(len(val_idx)),
        "rows_test": int(len(test_idx)),
        "seed": seed,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_model_family": "scikit-learn text classification pipeline",
    }
    joblib.dump({
        "format_version": ARTIFACT_FORMAT_VERSION,
        "task": task,
        "threshold": threshold,
        "estimator": estimator,
        "metadata": metadata,
    }, artifact_path, compress=3)

    with errors_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["text", "expected_label", "predicted_label", "positive_score"])
        writer.writeheader()
        for text, expected, predicted, probability in zip(x_test, y_test, y_pred, test_probabilities):
            if int(expected) != int(predicted):
                writer.writerow({
                    "text": text,
                    "expected_label": int(expected),
                    "predicted_label": int(predicted),
                    "positive_score": round(float(probability), 6),
                })

    report = {
        "task": task,
        "dataset": {
            "name": dataset_name, "source": dataset_source, "license": dataset_license,
            "label_rule": label_rule,
        },
        "split": {
            "strategy": "deduplicate normalized message text, then stratified 60/20/20 train/validation/test",
            **{k: metadata[k] for k in (
                "input_rows", "rows_total", "empty_rows_removed", "duplicate_rows_removed",
                "conflicting_text_groups_removed", "conflicting_rows_removed",
                "rows_train", "rows_validation", "rows_test", "seed",
            )},
        },
        "selection": {"selected_model": selected_name, "candidate_validation_results": candidate_results},
        "test_metrics": final_metrics,
        "artifacts": {"model": str(artifact_path), "misclassified_examples": str(errors_path)},
        "important_notes": [
            "Normalized duplicate texts were removed before splitting to reduce train/test leakage.",
            "Rows with contradictory labels for the same normalized text were excluded.",
            "The test set was not used to select the model or decision threshold.",
            "Scores are model outputs and are not guaranteed to be calibrated probabilities.",
            "The benchmark measures the supplied dataset and does not establish production readiness or cross-platform generalization.",
        ],
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    card_path.write_text(
        f"# TrustGuard {task.title()} Model Card\n\n"
        f"- **Model:** `{selected_name}`\n"
        f"- **Dataset:** {dataset_name}\n"
        f"- **Source:** {dataset_source}\n"
        f"- **Dataset license:** {dataset_license}\n"
        f"- **Positive-label rule:** {label_rule}\n"
        f"- **Rows:** {input_rows} input; {empty_rows_removed} empty, {duplicate_rows_removed} duplicate, "
        f"and {conflicting_rows_removed} contradictory-label rows removed; {len(texts_clean)} unique messages remain\n"
        f"- **Split:** deduplicate normalized text before stratified 60/20/20 train/validation/test\n"
        f"- **Seed:** {seed}\n"
        f"- **Decision threshold:** {threshold:.3f}, selected on validation data with a positive precision floor of 0.80 when achievable\n"
        f"- **Test accuracy:** {final_metrics['accuracy']:.4f}\n"
        f"- **Test positive precision / recall / F1:** {final_metrics['precision_positive']:.4f} / {final_metrics['recall_positive']:.4f} / {final_metrics['f1_positive']:.4f}\n"
        f"- **Test macro-F1:** {final_metrics['macro_f1']:.4f}\n"
        f"- **Test false-positive rate:** {final_metrics['false_positive_rate']:.4f}\n\n"
        "## Intended use\n\n"
        "Research and demonstration of a text-classification signal inside TrustGuard. The classifier is not a standalone enforcement system. Its output must be interpreted with policy rules and context, and uncertain or high-impact cases should receive human review.\n\n"
        "## Limitations\n\n"
        "Dataset domain, language, label definitions, class balance, and historical collection methods constrain generalization. The held-out score is not a guarantee of real-world performance. Scores have not been independently calibrated. Review the machine-readable evaluation report and misclassified examples before deploying.\n",
        encoding="utf-8",
    )
    return report
