"""Train a binary toxicity baseline from the CC0 Civil Comments dataset.

This optional script uses Hugging Face Datasets in streaming mode, shuffles a bounded
buffer, then takes a reproducible sample to keep RAM and training time manageable.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from trustguard.ml.training import train_and_save

DATASET_ID = "google/civil_comments"
DATASET_SOURCE = "https://huggingface.co/datasets/google/civil_comments (CC0 1.0)"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-rows", type=int, default=50_000)
    parser.add_argument("--toxicity-threshold", type=float, default=0.5)
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--report-dir", default="reports")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.max_rows < 500:
        raise SystemExit("Use --max-rows >= 500 for a meaningful initial benchmark")
    if not 0.0 <= args.toxicity_threshold <= 1.0:
        raise SystemExit("--toxicity-threshold must be between 0 and 1")
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Install training dependencies first: python -m pip install -e '.[training]'") from exc

    print(f"Streaming {DATASET_ID}; sampling up to {args.max_rows:,} rows...")
    dataset = load_dataset(DATASET_ID, split="train", streaming=True)
    dataset = dataset.shuffle(seed=args.seed, buffer_size=min(10_000, args.max_rows))
    texts: list[str] = []
    labels: list[int] = []
    for row in dataset:
        text = str(row.get("text", "")).strip()
        toxicity = row.get("toxicity")
        if not text or toxicity is None:
            continue
        try:
            score = float(toxicity)
        except (ValueError, TypeError):
            continue
        texts.append(text)
        labels.append(int(score >= args.toxicity_threshold))
        if len(texts) >= args.max_rows:
            break
    report = train_and_save(
        texts, labels, task="toxicity", dataset_name="Civil Comments (streamed sample)",
        dataset_source=DATASET_SOURCE,
        label_rule=f"toxic=1 when source toxicity score >= {args.toxicity_threshold}; otherwise 0",
        dataset_license="CC0 1.0", output_dir=args.output_dir,
        report_dir=args.report_dir, seed=args.seed,
    )
    report["dataset"]["streamed_rows_requested"] = args.max_rows
    report["dataset"]["source_label_threshold"] = args.toxicity_threshold
    report_path = Path(args.report_dir) / "toxicity_evaluation.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Selected model: {report['selection']['selected_model']}")
    print(f"Held-out test metrics: {report['test_metrics']}")
    print(f"Artifact: {report['artifacts']['model']}")


if __name__ == "__main__":
    main()
