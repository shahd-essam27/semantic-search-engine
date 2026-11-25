# Milestone 6 — Semantic Search Engine: Final Project Note

## What the app does
- Load a dataset with `id,text,category,date,source`.
- Optionally chunk long records (default: 120 words / 20-word overlap). Short records stay as one chunk.
- Encode with the same local SentenceTransformer for both documents and queries.
- Save persistent cosine vectors in Chroma (generated `chroma_db/` folder).
- Display ranked matches with category and date metadata filters in Gradio.

## Configuration to confirm after running
- Selected model: `sentence-transformers/all-MiniLM-L6-v2` (starter choice; reconsider using Milestone 5 results).
- Embedding dimension: **384** for this starter model; verify on your device.
- Database: local Chroma; a hashed collection name keeps different model/chunk settings separate.
- Chunking: 120 words with 20-word overlap for long texts; no splitting on short demonstration notes.
- Model weights: cached locally by Hugging Face.
- Device, indexing runtime, query latency, total storage: **TODO — record measured values**.

## Required real-corpus decision
The Milestone 1 `documents.csv` is **synthetic demonstration data**, not a real corpus.
Task 6.10 / Week 5 portfolio acceptance is therefore **not satisfied** until you replace or supplement it with a properly licensed real dataset containing >=200 records, update `docs/dataset.md`, rerun indexing and evaluation, and inspect the results.

## Manual smoke tests
1. Search an account-security paraphrase; verify five ranked results.
2. Change top-k to 3; verify three or fewer results.
3. Apply a category filter; verify every returned category matches.
4. Apply date range; verify all dates fall inside the range.
5. Search empty string; verify informative message, no crash.
6. Choose filters matching no records; verify informative message.
7. Close the app, reopen it without rebuilding; verify persisted results.
8. Rename `chroma_db/` temporarily (with app stopped), restart the app; verify no-index guidance; restore folder afterward.
9. Use the held-out evaluation command; add a screenshot and explain failures.

## Screenshots to add later
- `assets/search_ui.png` — interface after a meaningful search.
- `assets/filtered_results.png` — category/date filter applied.
- `assets/evaluation_results.png` — terminal or generated evaluation report.

## Limitations
- Retrieval only, **not** RAG or answer generation.
- Retrieval quality depends on corpus and model; relevance labels are illustrative.
- Chroma top-k is approximate nearest-neighbor search, so ranking can differ from exhaustive cosine search.
- Chunking is word-based, not token-based or language-aware.
- If the corpus or model changes, rerun `python -m scripts.build_index`; for large corpora, a more incremental strategy would reduce re-embedding time.
- This is a local portfolio application, not a secure multi-user hosted service.
