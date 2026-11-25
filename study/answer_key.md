# Milestone 0 — Suggested Answer Guide

> Read **after** attempting the questions. These are concise reference answers, not scripts to memorize verbatim.

1. Semantic search ranks by approximate meaning, so "secure my login" may find "enable two-factor authentication" without shared query words.
2. A learned numeric representation of text; 384 means 384 coordinate values in the model's embedding space.
3. Generally no; dimensions jointly encode learned features and are not reliable standalone semantic labels.
4. Arabic support, measured retrieval quality on Arabic or Arabic-to-English queries, latency/memory and licensing.
5. `cos(a,b)=(a·b)/(||a||||b||)`. Scaling a nonzero vector by positive 5 leaves cosine unchanged.
6. Different model spaces are not aligned even if vectors have equal length; similarity scores then lack intended meaning.
7. Five highest-ranked candidates. No: ranking does not prove correctness or relevance.
8. Collection groups stored records; index structures fast similarity lookup; persistence survives process restart.
9. Chroma is the selected simple local persistent solution, with metadata filters; other products are alternatives.
10. Restrict eligible data by explicit fields such as category/date; ranking orders by semantic closeness within eligible results.
11. Long sources may exceed model limits or contain many topics. Small chunks lose context; large chunks dilute topics or truncate; overlap preserves context at extra cost.
12. Similarity can flag paraphrases with different strings but may wrongly flag different factual claims about the same subject.
13. KMeans partitions vectors into a requested number of groups; labels require human inspection and evaluation.
14. Compare input embedding to label-description embeddings; this similarity heuristic differs from an NLI classifier trained for entailment judgments.
15. Documents → clean/chunk → embed → store vectors+metadata → embed query with compatible model → filter/rank top-k → show passages. RAG adds a generative model using retrieved passages for an answer. Quality (Hit@5/Recall@5), query latency, and indexing/memory/storage are examples.
