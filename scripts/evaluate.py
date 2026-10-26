"""Small offline evaluation harness for a CSV with columns: text, expected_label.

Example: python scripts/evaluate.py data/benchmark.csv
Labels should use the project's canonical labels: spam, harassment, hate, threat, sexual,
violence, self_harm, other, or none. The report is intentionally simple; use stratified
splits and confidence intervals for serious comparisons.
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

from trustguard.schemas import ModerationRequest
from trustguard.services.pipeline import get_pipeline


def main(path: str) -> None:
    rows = list(csv.DictReader(Path(path).open(encoding="utf-8", newline="")))
    if not rows or not {"text", "expected_label"}.issubset(rows[0]):
        raise SystemExit("CSV needs non-empty rows and columns: text, expected_label")
    counts = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "support": 0})
    pipeline = get_pipeline()
    for row in rows:
        expected = row["expected_label"].strip().lower()
        result = pipeline.run(ModerationRequest(content=row["text"], source="offline_eval"))
        predicted = set(result.labels)
        actual = set() if expected in {"", "none", "allowed"} else {expected}
        for label in actual | predicted:
            counts[label]["support"] += int(label in actual)
            counts[label]["tp"] += int(label in actual and label in predicted)
            counts[label]["fp"] += int(label not in actual and label in predicted)
            counts[label]["fn"] += int(label in actual and label not in predicted)
    print("label,precision,recall,f1,support")
    for label, c in sorted(counts.items()):
        precision = c["tp"] / max(1, c["tp"] + c["fp"])
        recall = c["tp"] / max(1, c["tp"] + c["fn"])
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        print(f"{label},{precision:.4f},{recall:.4f},{f1:.4f},{c['support']}")
    print("Note: this script is a smoke-test harness, not a statistically rigorous benchmark.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python scripts/evaluate.py path/to/benchmark.csv")
    main(sys.argv[1])
