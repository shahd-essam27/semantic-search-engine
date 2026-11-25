# Milestone 1 — Dataset documentation

## Source and provenance

The included `data/raw/documents.csv` contains **288 original, English-language demonstration notes** about software, networking, cybersecurity, data, and related topics. These short notes were authored for this exercise by the assistant, not downloaded from a real external corpus. Each of 96 topics has three variations, so the dataset contains near-duplicates by design. The text is newly created for this project; it is not extracted from private documents or copyrighted articles. It may be stored in this project repository as sample material.

**Important distinction:** The roadmap and Task 1.5 request a **real corpus**. This demo CSV is useful for validating the project structure and running Milestone 2, but it does **not** fully satisfy the real-corpus requirement. Before marking Task 1.5 or all of Milestone 1 complete, replace or supplement this with 200+ permitted records from an actual source you select (your notes, a documentation site where reuse is permitted, or a legitimately licensed CSV). Record the original URL, title, license, retrieval date and any attribution requirements here, and confirm redistribution permission before committing it publicly.

## File format

- Path: `data/raw/documents.csv`
- Encoding: UTF-8
- Columns: `id`, `text`, `category`, `date`, `source`
- `id`: unique `DOC-####` identifier
- `text`: nonempty sentence or short note
- `category`: one of 12 technical topics
- `date`: ISO `YYYY-MM-DD` sample date for filtering demonstrations; **synthetic date**, not source publication date
- `source`: `original_demo_notes_v1`

## Preprocessing

Texts are short, plain UTF-8 records with no HTML or markup. No lowercasing, stemming, language translation, or chunking has been applied. Dates are synthetic for demonstrating filtering. Retain identical preprocessing for query and corpus experiments when appropriate.

## Limitations

This is a generated teaching dataset rather than a natural collection, is English-only, includes repeated concepts and formulaic sentence structures, and does not measure production or multilingual performance. Evaluate Arabic retrieval separately in Milestone 5 using an appropriately labeled Arabic set. Results on this dataset may overestimate performance on real-world records.

## Verify

From the project root in VS Code PowerShell:

```powershell
.\.venv\Scripts\python.exe scripts\validate_dataset.py
```

Expected summary: `PASS: 288 valid rows, 288 unique IDs, 12 categories` followed by the per-category counts.

## To complete Task 1.5 later

1. Choose a real, reusable source and verify permission/license.
2. Convert at least 200 *distinct text records* into the same CSV columns.
3. Re-run the validation script and inspect a sample of rows.
4. Update the provenance, dates, license, preprocessing, and limitations in this document.
