"""Train and evaluate TrustGuard's spam classifier on the UCI SMS Spam Collection."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from trustguard.ml.training import train_and_save

SOURCE = "https://archive.ics.uci.edu/dataset/228/sms+spam+collection (DOI: 10.24432/C5CC84)"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/raw/sms_spam.csv")
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--report-dir", default="reports")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    path = Path(args.input)
    if not path.exists():
        raise SystemExit("Dataset not found. Run: python scripts/download_sms_dataset.py")
    texts: list[str] = []
    labels: list[int] = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {"text", "label"}.issubset(reader.fieldnames):
            raise SystemExit("Input CSV must contain columns named 'text' and 'label'")
        for row in reader:
            label = (row.get("label") or "").strip().lower()
            text = (row.get("text") or "").strip()
            if not text or label not in {"ham", "spam", "legitimate", "not_spam"}:
                continue
            texts.append(text)
            labels.append(1 if label == "spam" else 0)
    report = train_and_save(
        texts, labels, task="spam", dataset_name="UCI SMS Spam Collection",
        dataset_source=SOURCE, label_rule="spam=1; ham/legitimate/not_spam=0",
        dataset_license="CC BY 4.0", output_dir=args.output_dir,
        report_dir=args.report_dir, seed=args.seed,
    )
    print(f"Selected model: {report['selection']['selected_model']}")
    print(f"Held-out test metrics: {report['test_metrics']}")
    print(f"Artifact: {report['artifacts']['model']}")
    print(f"Report: {Path(args.report_dir) / 'spam_evaluation.json'}")


if __name__ == "__main__":
    main()
