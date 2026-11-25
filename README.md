# Week 5 — Semantic Search Engine

> **Track:** Embeddings, Vector Databases & Semantic Search  
> **Level:** Intermediate  
> **Source:** Helwan Career Center — 12-Week Industry Roadmap, Week 5 (pages 14–15)  
> **Status:** Planned — check boxes only after completing and testing the work.

## 1. Project Overview

Build a **local, semantic-search application** that retrieves documents by meaning rather than exact keyword matching. Progress from understanding embeddings to producing a searchable, persistent vector store, with metadata filters and a **Gradio** interface.

This is the **retrieval half of Retrieval-Augmented Generation (RAG)**. A full LLM-powered question-answering system is intentionally **out of scope** for this week.

### Learning Outcomes

- [ ] **LO 5.1:** Generate text embeddings with `sentence-transformers` and explain what embeddings represent.
- [ ] **LO 5.2:** Compute cosine similarity and build semantic search over a document set.
- [ ] **LO 5.3:** Store and query vectors in a vector database (Chroma / Qdrant / FAISS) with metadata filtering.
- [ ] **LO 5.4:** Apply embeddings to semantic search, deduplication, clustering, and zero-shot classification.
- [ ] **LO 5.5:** Choose an embedding model based on quality, dimension, speed, and language support, including Arabic.

### Implementation Choices

The roadmap allows multiple options. For this repository, the **proposed** stack is Python + VS Code, `sentence-transformers` for embeddings, **Chroma** as the local persistent vector database, `scikit-learn` for KMeans and evaluation, and **Gradio** for the interface. All models should be locally runnable after their initial download; no paid API key is required. The chosen models and dataset will be documented after benchmarking.

## 2. Milestones & Tasks

### Milestone 0 — Concepts First (No Code Required)

**Goal:** Be able to explain the concepts and design choices before implementing them.

- [ ] **Task 0.1 — What is semantic search?** Contrast keyword search and meaning-based search; explain one query that contains no exact matching words.
- [ ] **Task 0.2 — What are embeddings?** Define text embeddings, vectors, vector dimensions, embedding space, and why related sentences may be close together.
- [ ] **Task 0.3 — Embedding models.** Understand `sentence-transformers`, BGE, E5, English-focused vs multilingual models, and Arabic language support.
- [ ] **Task 0.4 — Similarity.** Understand dot product, cosine similarity, normalization, cosine-similarity scores, and why the same embedding model must be used for indexed documents and their queries.
- [ ] **Task 0.5 — Retrieval.** Understand nearest neighbors, top-k results, ranking, relevance, and what happens when none of the returned results is useful.
- [ ] **Task 0.6 — Vector databases.** Understand embedding storage, indexes, collections, IDs, metadata, persistence, and the roles of FAISS, Chroma, Qdrant, and Pinecone.
- [ ] **Task 0.7 — Metadata filters.** Explain category/date filters and the difference between filtering and semantic ranking.
- [ ] **Task 0.8 — Chunking.** Explain document vs chunk, chunk size, overlap, truncation, and the trade-off between short and long chunks.
- [ ] **Task 0.9 — NLP applications.** Explain semantic search, near-duplicate detection, KMeans clustering, and embedding-based zero-shot classification.
- [ ] **Task 0.10 — Model evaluation.** Understand quality, dimension, latency, memory, language coverage, and comparing models with the same test queries.
- [ ] **Task 0.11 — RAG connection.** Draw the retrieval pipeline: documents → chunks → embeddings → vector store → query embedding → ranked results. Distinguish retrieval from answer generation.
- [ ] **Task 0.12 — Knowledge check.** Answer 10–15 conceptual questions and write personal notes in `docs/concepts.md`.

**Deliverable:** `docs/concepts.md` with definitions, examples, diagrams (optional), and answers.  
**Exit check:** Explain the entire search pipeline without referring to code.

### Milestone 1 — Setup & Dataset

**Goal:** Create a reproducible repository and prepare reliable input data.

- [ ] **Task 1.1:** Create the `week5-semantic-search-engine` folder and open it in VS Code.
- [ ] **Task 1.2:** Create **one virtual environment at repository root** (`.venv`) and select that interpreter in VS Code.
- [ ] **Task 1.3:** Add `.gitignore`, `requirements.txt`, and the folders in the proposed structure below.
- [ ] **Task 1.4:** Install dependencies and verify imports with the root interpreter.
- [ ] **Task 1.5:** Choose a real corpus (notes, articles, documentation, or product descriptions) and prepare **at least 200 individual text records** for Lab 5.1. Avoid private/sensitive or non-redistributable data in a public repository.
- [ ] **Task 1.6:** Standardize dataset columns: `id`, `text`, `category`, `date`, `source`. Use non-empty, unique IDs and consistent dates (e.g. `YYYY-MM-DD`).
- [ ] **Task 1.7:** Validate dataset quality: no missing text/IDs; record counts and category distribution.
- [ ] **Task 1.8:** Write `docs/dataset.md` describing the source, license/permission, preprocessing, and limitations.

**Deliverables:** environment, dataset CSV, dataset documentation.  
**Exit check:** `data/raw/documents.csv` has at least 200 valid records and can be loaded by Python.

### Milestone 2 — Lab 5.1: Embeddings & Cosine Search

**Goal:** Implement semantic search **without** a vector database first.

- [ ] **Task 2.1:** Load the same 200+ records.
- [ ] **Task 2.2:** Load a small suitable `sentence-transformers` model; note model name, size, dimension, supported language(s), and device.
- [ ] **Task 2.3:** Generate one embedding per text and inspect shape and dtype.
- [ ] **Task 2.4:** Generate a query embedding using **the same model**.
- [ ] **Task 2.5:** Calculate cosine similarity to all 200+ records.
- [ ] **Task 2.6:** Rank the results and show **top 5**, including text, score, ID, and category.
- [ ] **Task 2.7:** Test at least 10 queries, including paraphrases and queries with few shared keywords.
- [ ] **Task 2.8:** Record examples, limitations, and latency in `docs/lab_5_1_results.md`.

**Deliverables:** `labs/lab_5_1_cosine_search.py`, results note.  
**Exit check:** Running the lab prints five ranked results for an entered query.

### Milestone 3 — Lab 5.2: Chroma & Metadata Filtering

**Goal:** Persist and search the **same** corpus and vectors in Chroma.

- [ ] **Task 3.1:** Create a local persistent Chroma database and collection.
- [ ] **Task 3.2:** Insert document IDs, texts, vectors, and metadata for the records from Milestone 2.
- [ ] **Task 3.3:** Prevent duplicate inserts by using stable document IDs and an appropriate update/upsert strategy.
- [ ] **Task 3.4:** Embed a user query with the same embedding model; retrieve top-k matching documents.
- [ ] **Task 3.5:** Implement a category metadata filter.
- [ ] **Task 3.6:** Implement a date filter (use consistent comparable date representation and validate its behavior).
- [ ] **Task 3.7:** Close and reopen the process; verify that the indexed records remain available.
- [ ] **Task 3.8:** Compare Chroma output against Lab 5.1 using identical queries and compatible similarity/distance settings; document differences.
- [ ] **Task 3.9:** Document collection settings, index location, and tests.

**Deliverables:** `labs/lab_5_2_chroma_filters.py`, `docs/lab_5_2_results.md`, locally persisted vector store.  
**Exit check:** A query returns relevant results **and** respects an applied filter after a restart.

### Milestone 4 — Lab 5.3: Other Embedding-Based NLP Tasks

**Goal:** Demonstrate reuse of embeddings beyond search.

- [ ] **Task 4.1:** Prepare a customer-review corpus with enough varied reviews to form meaningful groups.
- [ ] **Task 4.2:** Embed all reviews.
- [ ] **Task 4.3:** Apply KMeans with an explicitly chosen number of clusters.
- [ ] **Task 4.4:** Inspect representative reviews in each cluster and assign human-readable cluster labels.
- [ ] **Task 4.5:** Assess cluster coherence with examples (optionally a silhouette score when appropriate).
- [ ] **Task 4.6:** Implement simple near-duplicate detection using a cosine-similarity threshold and inspect false positives.
- [ ] **Task 4.7:** Explore embedding-based zero-shot classification: encode class descriptions, compare them with text embeddings, and assign the highest-scoring label. Explain how this differs from a dedicated zero-shot NLI model.
- [ ] **Task 4.8:** Save observations and limitations.

**Deliverables:** `labs/lab_5_3_clustering.py`, optional `labs/embedding_applications.py`, `docs/lab_5_3_results.md`.  
**Exit check:** Clusters have interpretable labels and the additional two applications can be demonstrated.

### Milestone 5 — Lab 5.4: English vs Multilingual Embeddings

**Goal:** Select the best embedding model for the project based on measured trade-offs.

- [ ] **Task 5.1:** Select one English-focused model and one multilingual model that supports Arabic.
- [ ] **Task 5.2:** Prepare a small evaluation set of English and Arabic queries with known relevant documents. Include Arabic paraphrases and, if possible, cross-language retrieval queries.
- [ ] **Task 5.3:** Run the **same** evaluation dataset through both models; generate fresh document embeddings for each model (do not mix embedding spaces).
- [ ] **Task 5.4:** Record embedding dimensions, model size, time to embed corpus, average query time, and CPU/memory usage where practical.
- [ ] **Task 5.5:** Evaluate retrieval quality with top-k relevance (e.g. Recall@5) and display representative success/failure cases.
- [ ] **Task 5.6:** Compare English and Arabic performance honestly. Do not assume the English-only model will always fail on Arabic or that multilingual always wins.
- [ ] **Task 5.7:** Explain the final model selection and expected trade-offs.

**Deliverables:** `labs/lab_5_4_model_comparison.py`, `docs/model_comparison.md`.  
**Exit check:** A results table supports a reasoned model selection for the final app.

### Milestone 6 — Mini Project: Complete Semantic Search Engine

**Goal:** Ship a portfolio-ready search application over a real corpus.

- [ ] **Task 6.1:** Organize code into reusable modules for loading, preprocessing, embedding, indexing, and searching.
- [ ] **Task 6.2:** Choose and document chunk size/overlap if records are long documents. Preserve document ID, chunk ID, source, and category; if records are already short, document why no chunking was needed.
- [ ] **Task 6.3:** Build a repeatable indexing command that creates/updates the persistent vector store from the real corpus.
- [ ] **Task 6.4:** Implement a search function: query → embedding → optional metadata filters → top-k results.
- [ ] **Task 6.5:** Build a **Gradio UI** with search input, top-k option, category/date filtering, and ranked result display.
- [ ] **Task 6.6:** Handle empty queries, empty results, missing database/index, and invalid filters without crashing.
- [ ] **Task 6.7:** Add a small held-out relevance test set and run final retrieval checks.
- [ ] **Task 6.8:** Document chosen model, vector dimension, chunk size, runtime, limitations, and how to rebuild the index.
- [ ] **Task 6.9:** Add screenshots of the working UI and final results to `assets/`.
- [ ] **Task 6.10:** Verify clean setup steps, tidy the repository, and publish to GitHub with the completed README checklist.

**Deliverables:** persistent indexed vector store (locally or reproducibly rebuilt), working UI, model/chunk-size decision note, tests, screenshots.  
**Exit check:** A new user can follow README instructions, index a dataset, open the app, search by meaning, and filter results.

## 3. Proposed Repository Structure

> This is the **target structure**, not a claim that these files already exist. Create files as you reach their milestones.

```text
week5-semantic-search-engine/
├── README.md
├── requirements.txt
├── .gitignore
├── .venv/                         # local only; do not commit
├── data/
│   ├── raw/
│   │   ├── documents.csv
│   │   └── customer_reviews.csv
│   └── evaluation/
│       └── relevance_queries.csv
├── labs/
│   ├── __init__.py
│   ├── lab_5_1_cosine_search.py
│   ├── lab_5_2_chroma_filters.py
│   ├── lab_5_3_clustering.py
│   ├── embedding_applications.py
│   └── lab_5_4_model_comparison.py
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── embeddings.py
│   ├── indexing.py
│   └── search.py
├── app.py                         # Gradio UI
├── scripts/
│   └── build_index.py
├── docs/
│   ├── concepts.md
│   ├── dataset.md
│   ├── lab_5_1_results.md
│   ├── lab_5_2_results.md
│   ├── lab_5_3_results.md
│   └── model_comparison.md
├── tests/
│   └── test_search.py
├── assets/
│   └── .gitkeep
└── chroma_db/                      # generated; do not commit by default
```

## 4. Getting Started (Windows, VS Code)

Use the **VS Code integrated terminal**, not a separate environment per lab.

```powershell
# From the repository root
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

In VS Code: **Ctrl+Shift+P → Python: Select Interpreter → `.venv\Scripts\python.exe`**.

**Planned dependencies** (add versions/pin them after the environment is verified):

```text
sentence-transformers
numpy
pandas
scikit-learn
chromadb
gradio
pytest
```

To run a lab **independently**, use the single root interpreter. Each lab will also require its corresponding input dataset and, where applicable, the model weights downloaded once.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m labs.lab_5_1_cosine_search
.\.venv\Scripts\python.exe -m labs.lab_5_2_chroma_filters
.\.venv\Scripts\python.exe -m labs.lab_5_3_clustering
.\.venv\Scripts\python.exe -m labs.lab_5_4_model_comparison
```

**Note:** These commands describe the intended entry points; they will work only after the corresponding files are implemented. Lab 5.2 may need to build or import the index on first run; its implementation should handle this or document the required step.

Planned final project commands, **after implementation**:

```powershell
.\.venv\Scripts\python.exe -m scripts.build_index
.\.venv\Scripts\python.exe app.py
.\.venv\Scripts\python.exe -m pytest
```

## 5. Git & Repository Hygiene

Example `.gitignore` entries:

```gitignore
.venv/
__pycache__/
*.py[cod]
.pytest_cache/
.vscode/
.env
chroma_db/
*.log
```

Commit documentation, code, and **only** datasets whose redistribution is allowed. Do not commit secrets, the Python virtual environment, or large generated vector stores by default. If raw data is not redistributed, include instructions or a small permitted sample to reproduce the index.

## 6. Testing & Evaluation Checklist

- [ ] Input data has valid IDs, text, categories, and dates.
- [ ] Query and document vectors come from the same embedding model.
- [ ] Cosine search returns a sorted top 5.
- [ ] Chroma indexes persist across runs.
- [ ] Category/date filters exclude incorrect records.
- [ ] English/Arabic model comparisons use the same labeled evaluation set.
- [ ] Final UI responds correctly to empty inputs and no matches.
- [ ] README setup/run instructions are tested from the repository root.

## 7. Final Deliverables

- [ ] Lab 5.1 — 200+ sentence embeddings and cosine top-5 search.
- [ ] Lab 5.2 — Chroma vector database and metadata filtering.
- [ ] Lab 5.3 — KMeans customer-review clustering with readable labels.
- [ ] Lab 5.4 — English-focused vs multilingual model comparison on Arabic queries.
- [ ] Mini Project 5 — persistent vector store, Gradio semantic-search UI, filtering, and model/chunk-size note.
- [ ] GitHub README, lab findings, final screenshots, and reproducible setup steps.

## 8. Suggested Commit Sequence

```text
docs: add week 5 project roadmap and milestone checklist
docs: explain embedding and vector search concepts
chore: initialize environment and project structure
data: add and validate semantic search corpus
feat: implement cosine similarity retrieval lab
feat: add persistent chroma store and metadata filters
feat: implement review clustering and embedding applications
experiment: compare english and multilingual embedding models
feat: build gradio semantic search interface
test: add retrieval and filter tests
docs: document evaluation results and final setup
```

## 9. References


- [Sentence-Transformers documentation](https://www.sbert.net/)
- [Chroma documentation](https://docs.trychroma.com/)
- [Qdrant documentation](https://qdrant.tech/documentation/)
- [Hugging Face NLP course](https://huggingface.co/learn/nlp-course/)
- [Gradio documentation](https://www.gradio.app/docs)

---

**Progress policy:** Do not check a task until its implementation, test, and documentation are complete. No milestones are marked complete in this initial README.
