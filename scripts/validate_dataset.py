"""Milestone 1, tasks 1.6-1.7: validate the CSV using only the standard library."""
from collections import Counter
from datetime import date
from pathlib import Path
import csv

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "raw" / "documents.csv"
REQUIRED = ["id", "text", "category", "date", "source"]


def main() -> None:
    if not DATASET.is_file():
        raise FileNotFoundError(f"Dataset missing: {DATASET}")
    with DATASET.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != REQUIRED:
            raise ValueError(f"Expected columns in this order: {REQUIRED}; got {reader.fieldnames}")
        rows = list(reader)
    if len(rows) < 200:
        raise ValueError(f"Need at least 200 records; got {len(rows)}")
    ids = set()
    for line, row in enumerate(rows, start=2):
        for key in REQUIRED:
            if not row[key] or not row[key].strip():
                raise ValueError(f"Empty {key} at CSV line {line}")
        if row["id"] in ids:
            raise ValueError(f"Duplicate id {row['id']} at CSV line {line}")
        ids.add(row["id"])
        try:
            date.fromisoformat(row["date"])
        except ValueError as e:
            raise ValueError(f"Invalid ISO date at line {line}: {row['date']}") from e
    counts = Counter(row["category"] for row in rows)
    print(f"PASS: {len(rows)} valid rows, {len(ids)} unique IDs, {len(counts)} categories")
    print("Category distribution:")
    for name, count in sorted(counts.items()):
        print(f"  {name}: {count}")
    print(f"Dataset: {DATASET}")


if __name__ == "__main__":
    main()
