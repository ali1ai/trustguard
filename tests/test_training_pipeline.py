from __future__ import annotations

import json

from trustguard.ml.training import train_and_save


def test_training_pipeline_writes_artifact_reports_and_errors(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    texts = []
    labels = []
    for index in range(60):
        texts.append(f"hello friend meeting family plans normal note number {index}")
        labels.append(0)
        texts.append(f"win cash prize offer click now spam notice number {index}")
        labels.append(1)

    report = train_and_save(
        texts, labels, task="spam", dataset_name="synthetic test fixture",
        dataset_source="test only; not a benchmark dataset",
        label_rule="positive class 1 in this test fixture", seed=42,
    )
    assert "stratified 60/20/20 train/validation/test" in report["split"]["strategy"]
    assert report["test_metrics"]["n_test"] == 24
    assert report["selection"]["selected_model"] in report["selection"]["candidate_validation_results"]
    assert (tmp_path / "models" / "spam_classifier.joblib").is_file()
    assert (tmp_path / "reports" / "spam_test_errors.csv").is_file()
    card = (tmp_path / "reports" / "spam_model_card.md").read_text(encoding="utf-8")
    assert "Limitations" in card
    metrics = json.loads((tmp_path / "reports" / "spam_evaluation.json").read_text(encoding="utf-8"))
    assert "false_positive_rate" in metrics["test_metrics"]
    assert metrics["dataset"]["license"] == "Unspecified; verify before redistribution"
    assert "test set was not used" in " ".join(metrics["important_notes"]).lower()


def test_training_deduplicates_before_split_and_drops_conflicting_labels(tmp_path):
    texts = []
    labels = []
    for index in range(60):
        texts.append(f"ordinary friend meeting unique ham message {index}")
        labels.append(0)
        texts.append(f"limited prize cash offer unique spam message {index}")
        labels.append(1)
    # Add repeated rows; these must not be able to leak across evaluation splits.
    texts.extend([texts[0], texts[1], texts[2].upper()])
    labels.extend([0, 1, 0])
    # Same normalized text with contradictory labels: all such rows are excluded.
    texts.extend(["Conflict Sample Unique Marker", "  conflict   sample unique marker "])
    labels.extend([0, 1])

    report = train_and_save(
        texts, labels, task="spam", dataset_name="synthetic dedup fixture",
        dataset_source="test only; not a benchmark dataset",
        label_rule="positive class 1 in this test fixture", output_dir=tmp_path / "models",
        report_dir=tmp_path / "reports", seed=42,
    )
    split = report["split"]
    assert split["input_rows"] == len(texts)
    assert split["duplicate_rows_removed"] >= 3
    assert split["conflicting_text_groups_removed"] == 1
    assert split["conflicting_rows_removed"] == 2
    assert split["rows_total"] == split["rows_train"] + split["rows_validation"] + split["rows_test"]
    assert "deduplicate normalized message text" in split["strategy"]
