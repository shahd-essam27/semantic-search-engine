# Milestone 0 — One-Page Quick Reference

| Term | Remember |
|---|---|
| Semantic search | Meaning-oriented retrieval, not just shared words |
| Embedding | Learned vector representation of a text |
| Dimension | Count of numeric coordinates; not an accuracy score |
| Cosine similarity | Directional similarity of two nonzero vectors |
| Normalization | Scale vector to unit length; dot product equals cosine after L2 normalization |
| Top-k | Return k highest-ranked candidates |
| Vector database | Store and search vectors with metadata |
| Metadata filter | Restrict eligible records, e.g., category/date |
| Chunk | Shorter searchable piece of a document |
| Overlap | Shared context between adjacent chunks |
| Near duplicate | Different texts with very similar content |
| KMeans | Unsupervised grouping into k clusters |
| Zero-shot label match | Compare text to label descriptions without new classifier training |
| Retrieval evaluation | Compare ranked outputs to known relevant documents |
| RAG | Retrieve evidence, then generate an answer using it |

## Our planned local stack

Python 3.11 + VS Code + sentence-transformers + Chroma + scikit-learn + Gradio; no paid API required. This pack is theory only.

## Remember these common mistakes

- Cosine score is **not** a probability of truth.
- Query and document vectors must use compatible model spaces.
- Similarity ranking and metadata filtering solve **different** problems.
- Top result can still be wrong or irrelevant.
- Comparing two embedding models means **re-embedding** with each model separately.
- Retrieval output is **not** a generated answer.
