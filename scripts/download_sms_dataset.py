"""Download the official UCI SMS Spam Collection and convert it to UTF-8 CSV."""
from __future__ import annotations

import argparse
import csv
import io
import urllib.request
import zipfile
from pathlib import Path

DATASET_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
DATASET_SOURCE = "UCI Machine Learning Repository, SMS Spam Collection, DOI: 10.24432/C5CC84"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/raw/sms_spam.csv")
    args = parser.parse_args()
    request = urllib.request.Request(DATASET_URL, headers={"User-Agent": "TrustGuard-portfolio-research/0.2"})
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            archive_bytes = response.read()
    except Exception as exc:
        raise SystemExit(
            f"Dataset download failed: {exc}\nDownload the SMS Spam Collection manually from "
            "https://archive.ics.uci.edu/dataset/228/sms+spam+collection and pass the extracted data through "
            "scripts/convert_sms_dataset.py."
        ) from exc
    try:
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            candidates = [name for name in archive.namelist() if Path(name).name == "SMSSpamCollection"]
            if not candidates:
                raise ValueError("SMSSpamCollection file not found inside the UCI archive")
            raw = archive.read(candidates[0]).decode("utf-8", errors="replace")
    except (zipfile.BadZipFile, ValueError) as exc:
        raise SystemExit(f"Could not parse the official dataset archive: {exc}") from exc

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["text", "label"])
        writer.writeheader()
        for line in raw.splitlines():
            label, separator, text = line.partition("\t")
            if not separator or label not in {"ham", "spam"} or not text.strip():
                continue
            writer.writerow({"text": text.strip(), "label": label})
            n += 1
    if n < 100:
        output.unlink(missing_ok=True)
        raise SystemExit(f"Only {n} rows parsed; refusing to create a suspiciously small dataset")
    print(f"Saved {n:,} rows to {output}")
    print(f"Source: {DATASET_SOURCE}")
    print("License: CC BY 4.0; preserve attribution when redistributing derived materials.")


if __name__ == "__main__":
    main()
