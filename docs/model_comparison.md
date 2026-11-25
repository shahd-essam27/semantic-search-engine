# Lab 5.4 — English vs Multilingual Embedding Comparison

**Status: awaiting local execution.** This is a template. Run the lab with `--evaluate` to replace this file with measured results from your computer.

## Evaluation plan

- Compare `sentence-transformers/all-MiniLM-L6-v2` with `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
- Use the identical English document collection for both models.
- Use 8 English and 8 Arabic queries with labeled relevant document IDs.
- Re-encode the entire corpus for **each** model; do not reuse vectors across models.
- Measure dimension, cached weight size if available, embedding time, mean query time, Recall@5 and Hit@5; record process RSS if optional `psutil` is already available.
- Arabic queries over English documents test **cross-language** retrieval, *not* retrieval over Arabic documents.

## After running

The program writes a measured comparison table here, plus `docs/model_comparison_details.csv`.
Review mistakes and fill its **TODO** decision and limitations sections; do not mark Milestone 5 complete until this is done.
