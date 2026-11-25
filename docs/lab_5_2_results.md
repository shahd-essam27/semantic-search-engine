# Lab 5.2 — Chroma Persistence & Filters

> Prepared template — not yet executed locally. Fill after testing.

## Settings
- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Collection: `week5_documents_minilm_cosine`
- Local index: `chroma_db/` (ignored by Git)
- Similarity: normalized embeddings, cosine index; `similarity ≈ 1 - distance`
- Metadata: `category`, `date` (ISO), `date_number` (integer YYYYMMDD), `source`
- Corpus: same `data/raw/documents.csv` from Milestones 1 and 2

## Test results (to fill)
- [ ] Repeated `--index` does not increase record count.
- [ ] `--verify-persistence` succeeds in a new terminal/process.
- [ ] Category filter returns only the requested category.
- [ ] Date start/end bounds are inclusive.
- [ ] Combined category + date filters work.
- [ ] Empty matches display a useful message.
- [ ] `--compare` generated a table of NumPy and Chroma results below.

## Manual observations
- Relevant results: TODO
- Date-filter example: TODO
- Category-filter example: TODO
- Persistence check result: TODO
- Approximate vs exact retrieval differences: TODO
- Limitations: synthetic demonstration notes, not a real external corpus

## Comparison

Run `python labs/lab_5_2_chroma_filters.py --compare` to populate this report with observed measurements. **The comparison command overwrites this template**, so make notes elsewhere first if you want to keep them.
