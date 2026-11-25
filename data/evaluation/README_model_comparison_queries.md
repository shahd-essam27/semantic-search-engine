# Lab 5.4 — Evaluation Query Labels

`model_comparison_queries.csv` is a small **hand-authored teaching evaluation set**. It is not a validated benchmark and should not be used to claim production accuracy.

The file has 16 bilingual queries (8 English, 8 Arabic). All queries target English text in the Milestone 1 demo notes, `data/raw/documents.csv`. Each query has three labeled relevant document IDs: a core topic explanation, practical guidance, and an expanded example note.

- `query_id`: unique ID
- `language`: `en` or `ar`, language of the query
- `query`: the user query
- `relevant_ids`: semicolon-delimited document IDs from Milestone 1
- `note`: manual annotation intent

**Important caveat:** The labels are not exhaustive, and documents with related concepts may be genuinely relevant even if not labeled. So Recall@5 and Hit@5 are **teaching signals** limited by these labels. Arabic queries are cross-language requests for English documents; comparing Arabic-document retrieval would require a separate, licensed Arabic corpus with its own relevance labels.

Do not change `data/raw/documents.csv` between model runs. If you switch corpora, update `relevant_ids` accordingly, otherwise the validation step deliberately fails.
