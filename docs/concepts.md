# Week 5 — Milestone 0: Concepts First

> **Status:** Study guide, not proof of completion. Go through Tasks 0.1–0.12, write answers in `study/milestone_0_knowledge_check.md`, and mark the root README only after understanding each task.
>
> **Source alignment:** Week 5 roadmap (LO 5.1–5.5); original project README, Milestone 0 Tasks 0.1–0.12. This guide expands the roadmap with teaching examples; illustrative numbers are not project measurements.

## Task 0.1 — Semantic search vs keyword search

**Keyword search** matches exact words or patterns; e.g., a literal search for `reset password` can miss a note saying *recover access to my account*. **Semantic search** uses numerical representations of text meaning to rank conceptually related passages even if the wording differs.

**Example:** Query: "How can I stop someone entering my account?" Relevant note: "Enable two-factor authentication to protect logins." Little literal overlap, but related intent.

Semantic search can still fail: vague queries, ambiguous words, incomplete documents, and weak models may produce irrelevant results. It is *retrieval*, not guaranteed understanding or fact-checking.

**Explain it yourself:** Give a different keyword-mismatch example.

## Task 0.2 — Embeddings and embedding space

An **embedding** is a fixed-length numerical vector representing properties of text learned by a model. Example *toy* embedding: `[0.2, -0.7, 0.4]`. Real models often output hundreds or thousands of dimensions. A **dimension** is one numeric coordinate, **vector shape** is its length, and **embedding space** is the coordinate system learned by that particular model.

Similar sentences often map closer than unrelated sentences, but dimensions normally do not have individual human-readable meanings. Embeddings can encode biases and are not proof that statements are true.

**Key distinction:** Token count, text length, embedding dimensions, and embedding quality are different things.

## Task 0.3 — Embedding model families and Arabic

- `sentence-transformers` is a Python library to load/use sentence embedding models.
- **BGE** and **E5** are model families with different training, prompts, and usage conventions. Some E5 models expect query/passage prefixes; always read a chosen model's documentation.
- **English-focused models** usually prioritize English; **multilingual models** are designed for multiple languages. Confirm whether Arabic is supported and test actual Arabic and mixed-language queries.
- Our proposed labs use `sentence-transformers/all-MiniLM-L6-v2` and `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`. The comparison in Milestone 5 will measure results rather than assume a winner.

**Model selection:** quality on your task, language coverage, dimensions, speed, memory, license, and use restrictions.

## Task 0.4 — Similarity, dot product, cosine, normalization

For vectors **a** and **b**:

- Dot product: `a · b = Σ(a_i * b_i)`; magnitude affects the value.
- Cosine similarity: `cos(a,b) = (a · b) / (||a|| * ||b||)`; measures directional alignment, typically from -1 to +1 when defined.
- **L2 normalization** divides a nonzero vector by its length, giving length 1. For normalized vectors, dot product equals cosine similarity.

**Toy calculation:** `a=[1,0]`, `b=[0.8,0.6]`, both unit length → cosine = `0.8`. `c=[0,1]` → cosine(a,c)=`0`.

Scores are *relative ranking signals*, not calibrated probabilities. The same model (and compatible preprocessing/normalization) must encode both indexed documents and queries. Never compare vectors from two unrelated model spaces just because the dimensions happen to match.

## Task 0.5 — Top-k retrieval and relevance

A query is embedded, compared with document/chunk vectors, and ranked from most to least similar. **Top-k** returns the first `k` candidates; in Lab 5.1, `k=5`. **Nearest-neighbor search** finds close vectors, exactly or approximately depending on the index.

High similarity does not guarantee useful answers. Handle poor results by displaying scores, testing thresholds with labeled examples, allowing "no useful match," and improving corpus/model/chunking.

**Relevance** is judged against what the user asked, not simply whether text overlaps.

## Task 0.6 — Vector databases

A vector store manages vectors with **IDs**, original text or pointers, **metadata** (e.g., category/date), and search **indexes**. A **collection** is a named group of stored records. **Persistence** means the stored index survives process restarts.

- **FAISS:** nearest-neighbor indexing/search library; persistence and full application metadata behavior need extra implementation.
- **Chroma:** convenient local persistent store with collections and metadata filters; chosen for this project.
- **Qdrant:** vector database suitable for more production-style deployments.
- **Pinecone:** managed vector database service.

The roadmap names these as options, not requirements to install all four.

## Task 0.7 — Metadata filtering

Semantic ranking chooses documents by vector similarity. **Filtering** restricts the eligible records first or as part of retrieval, for example `category = cybersecurity` or `date >= 2026-01-01`. Filtering is an explicit condition, **not** a measure of meaning. If nothing meets the filter, return an empty result clearly. Store dates consistently; string-order comparisons work only with normalized sortable formats such as `YYYY-MM-DD`.

## Task 0.8 — Chunking

A **document** may be long; a **chunk** is a shorter segment indexed separately. **Chunk size** controls the amount of text in each segment; **overlap** repeats some material between neighbors to preserve context. **Truncation** silently cuts off excess text when an embedding model has an input limit.

**Trade-off:** very small chunks can lose context; very large chunks may mix topics or be truncated. Chunk boundaries affect relevance. Store document ID and chunk position to trace search results back to sources. Lab 5.1's 200 short sentences may need no splitting; the final app should document a deliberate chunking choice.

## Task 0.9 — Four practical NLP uses

1. **Semantic search:** rank documents close to the query embedding.
2. **Near-duplicate detection:** flag pairs whose embedding similarity exceeds a tuned threshold. **False positive:** related topic but meaningfully different fact.
3. **KMeans clustering:** partition embeddings into a chosen number `k` of groups by proximity; examine representative texts before naming clusters. Unsupervised clusters do not automatically correspond to true customer-review categories.
4. **Embedding-based zero-shot classification:** encode class descriptions and compare an input to them, then pick the best-matching description. This is a similarity heuristic, **not the same** as a dedicated natural-language-inference zero-shot model.

## Task 0.10 — Model evaluation

Compare models with **the same corpus, relevant query set, labeled answers, hardware context, and top-k setting**, while generating separate document embeddings for each model.

- **Retrieval quality:** e.g. **Hit@5** (fraction of queries retrieving ≥1 relevant document in top 5) and **Recall@5** (relevant documents found in top 5 divided by all known relevant documents, averaged appropriately).
- **Latency:** time per query, including or excluding embedding time—state which.
- **Embedding time:** time to encode the whole corpus.
- **Memory & storage:** RAM, cached model size, vector size, index size.
- **Dimension:** impacts storage and computation but is not an accuracy score.
- **Language:** evaluate English, Arabic, and cross-language queries separately.

Avoid claiming a best model until actual metrics and relevant examples support it. Synthetic test queries are useful for smoke tests but not a substitute for real user queries.

## Task 0.11 — How this becomes RAG

```text
INDEXING / PREPARATION (offline or once per update)
Documents → clean text → chunks → embedding model → vector DB (+ IDs + metadata)

SEARCH / RETRIEVAL (for each user query)
User query → SAME embedding model → vector DB top-k + filters → ranked source chunks

RAG (optional later stage)
Ranked source chunks + user question → generative LLM → generated answer with sources
```

Week 5's final Gradio app stops at **retrieval**: it displays matching passages. Week 6 may introduce answer generation. A retrieval hit is not itself a generated answer.

## Task 0.12 — Knowledge check and personal notes

Complete `study/milestone_0_knowledge_check.md` without looking at `study/answer_key.md` first. Add your own diagrams, analogies, and misunderstood concepts here, then explain the full pipeline out loud without code.

### My personal notes

- The concept I found easiest:
- The concept I found hardest:
- My own example of semantic search:
- Why document and query embeddings must be compatible:
- When I would use metadata filtering:
- What I want to revisit before Milestone 2:

### Completion rule

Mark a Task 0.x checkbox in the **root** README only when you can explain it in your own words. Milestone 0 is complete when all Tasks 0.1–0.12 are verified and you can explain the complete pipeline without code.
