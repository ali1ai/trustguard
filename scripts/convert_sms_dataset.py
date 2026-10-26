"""Convert the UCI tab-separated SMSSpamCollection file into text,label CSV."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Path to the extracted SMSSpamCollection file")
    parser.add_argument("--output", default="data/raw/sms_spam.csv")
    args = parser.parse_args()
    source = Path(args.input)
    if not source.is_file():
        raise SystemExit(f"Input file not found: {source}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with source.open(encoding="utf-8", errors="replace") as src, output.open("w", encoding="utf-8", newline="") as dst:
        writer = csv.DictWriter(dst, fieldnames=["text", "label"])
        writer.writeheader()
        for line in src:
            label, separator, text = line.rstrip("\r\n").partition("\t")
            if separator and label in {"ham", "spam"} and text.strip():
                writer.writerow({"text": text.strip(), "label": label})
                count += 1
    if count < 100:
        output.unlink(missing_ok=True)
        raise SystemExit(f"Only {count} valid rows found; check that input is the UCI data file")
    print(f"Converted {count:,} messages into {output}")


if __name__ == "__main__":
    main()
