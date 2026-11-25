# Week 5 — Semantic Search Engine

> **Track:** Embeddings, Vector Databases & Semantic Search  
> **Level:** Intermediate  
> **Source:** Helwan Career Center — 12-Week Industry Roadmap, Week 5 (pages 14–15)  
> **Status (2026-11-25):** Project scaffold uploaded to GitHub; environment, demonstration dataset and 31 automated tests verified. Model-based end-to-end labs and UI remain pending.
> **Repository:** https://github.com/shahd-essam27/semantic-search-engine
> **Local path:** `E:\semantic-search-engine`
> **Progress convention:** `[x]` means verified, not merely that a starter file exists.

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

The roadmap allows multiple options. For this repository, the **proposed** stack is Python + VS Code, `sentence-transformers` for embeddings, **Chroma** as the local persistent vector database, `scikit-learn` for KMeans and evaluation, and **Gradio** for the interface. All models should be locally runnable after their initial download; no paid API key is required. The current demonstration corpus has 288 synthetic records in 12 categories. The English `all-MiniLM-L6-v2` model download was confirmed complete; the multilingual model was still downloading in the last screenshot. The final model choice remains pending measured evaluation.

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

- [x] **Task 1.1:** Create the `semantic-search-engine` folder and open it in VS Code.
- [x] **Task 1.2:** Create **one virtual environment at repository root** (`.venv`) and select that interpreter in VS Code.
- [x] **Task 1.3:** Add `.gitignore`, `requirements.txt`, and the folders in the proposed structure below.
- [x] **Task 1.4:** Install dependencies and verify imports with the root interpreter.
- [ ] **Task 1.5:** Choose a real corpus (notes, articles, documentation, or product descriptions) and prepare **at least 200 individual text records** for Lab 5.1. Avoid private/sensitive or non-redistributable data in a public repository.
- [x] **Task 1.6:** Standardize dataset columns: `id`, `text`, `category`, `date`, `source`. Use non-empty, unique IDs and consistent dates (e.g. `YYYY-MM-DD`).
- [x] **Task 1.7:** Validate dataset quality: no missing text/IDs; record counts and category distribution.
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

## 3. Actual Repository Structure (Milestones 0–6)

> Based on the files shown in VS Code and the supplied milestone packages. **A file's existence does not mean its lab has been executed.** Generated artifacts appear after the corresponding commands run; they are not normally committed.

```text
E:\semantic-search-engine\
├── .git/                              # local Git history
├── .venv/                             # local Python 3.11 environment; ignored
├── .gitignore
├── README.md                          # this file
├── requirements.txt
├── app.py                             # M6: Gradio UI (not yet run end-to-end)
│
├── Downloaded_model/                  # early download utility, NOT model weights
│   ├── __init__.py
│   └── download_models.py
│
├── data/
│   ├── raw/
│   │   ├── documents.csv              # M1: 288 demo records / 12 categories
│   │   ├── customer_reviews.csv       # M4: synthetic demonstration reviews
│   │   └── customer_reviews_README.md
│   ├── evaluation/
│   │   ├── model_comparison_queries.csv         # M5: Arabic/English queries
│   │   ├── README_model_comparison_queries.md
│   │   └── final_relevance_queries.csv          # M6
│   └── processed/                     # generated by Lab 5.1, if run; ignored
│       └── lab_5_1/
│           ├── embeddings.npy         # generated
│           └── metadata.json          # generated
│
├── labs/                              # the four independent hands-on labs
│   ├── __init__.py
│   ├── .gitkeep                        # optional placeholder
│   ├── lab_5_1_cosine_search.py       # M2 / LAB 5.1
│   ├── lab_5_2_chroma_filters.py      # M3 / LAB 5.2
│   ├── lab_5_3_clustering.py          # M4 / LAB 5.3
│   └── lab_5_4_model_comparison.py    # M5 / LAB 5.4
│
├── scripts/
│   ├── __init__.py
│   ├── validate_dataset.py           # M1: verified passing
│   ├── build_index.py                # M6: index builder
│   └── evaluate_search.py            # M6: relevance evaluation
│
├── src/                               # M6: reusable app modules
│   ├── __init__.py
│   ├── .gitkeep                        # optional placeholder
│   ├── data_loader.py
│   ├── embeddings.py
│   ├── indexing.py
│   └── search.py
│
├── tests/
│   ├── .gitkeep                        # optional placeholder
│   ├── test_lab_5_1.py
│   ├── test_lab_5_2.py
│   ├── test_lab_5_3.py
│   ├── test_lab_5_4.py
│   └── test_search.py
│
├── docs/
│   ├── concepts.md                    # M0 study material
│   ├── dataset.md                     # M1
│   ├── lab_5_1_results.md            # M2 (template until run)
│   ├── lab_5_2_results.md            # M3 (template until run)
│   ├── lab_5_3_results.md            # M4 (template until run)
│   ├── model_comparison.md            # M5 (template until run)
│   ├── model_comparison_details.csv   # generated by M5, if evaluated
│   ├── final_project.md               # M6
│   └── final_evaluation.md            # generated by M6, if evaluated
│
├── study/                              # M0
│   ├── QUICK_REFERENCE.md
│   ├── milestone_0_knowledge_check.md
│   ├── answer_key.md
│   └── MILESTONE_0_CHECKLIST.md
│
├── Milestone_Instructure/            # exact folder name on your computer
│   ├── MILESTONE_0_INSTRUCTIONS.md
│   ├── MILESTONE_1_INSTRUCTIONS.md
│   ├── MILESTONE_2_INSTRUCTIONS.md
│   ├── MILESTONE_3_INSTRUCTIONS.md
│   ├── MILESTONE_4_INSTRUCTIONS.md
│   ├── MILESTONE_5_INSTRUCTIONS.md
│   └── MILESTONE_6_INSTRUCTIONS.md
│
├── assets/
│   ├── .gitkeep
│   ├── README.md
│   └── [actual UI screenshots to add after running M6]
└── chroma_db/                         # generated by M3/M6; ignored
```

### Lab map

| Milestone | Lab | Python file | Test file | Results file |
|---|---|---|---|---|
| M2 | Lab 5.1: Cosine search | `labs/lab_5_1_cosine_search.py` | `tests/test_lab_5_1.py` | `docs/lab_5_1_results.md` |
| M3 | Lab 5.2: Chroma | `labs/lab_5_2_chroma_filters.py` | `tests/test_lab_5_2.py` | `docs/lab_5_2_results.md` |
| M4 | Lab 5.3: Clustering | `labs/lab_5_3_clustering.py` | `tests/test_lab_5_3.py` | `docs/lab_5_3_results.md` |
| M5 | Lab 5.4: Model comparison | `labs/lab_5_4_model_comparison.py` | `tests/test_lab_5_4.py` | `docs/model_comparison.md` |
| M6 | Final Gradio search engine | `app.py` + `src/` | `tests/test_search.py` | `docs/final_project.md` |

## 4. Getting Started (Windows, VS Code)

Open `E:\semantic-search-engine` in VS Code. **Use the existing root `.venv`**, not a new virtual environment for each lab. The verified environment runs Python 3.11.9.

```powershell
# Execute from E:\semantic-search-engine
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

In VS Code: **Ctrl+Shift+P → Python: Select Interpreter → `.venv\Scripts\python.exe`**.

The root `requirements.txt` has `sentence-transformers`, `numpy`, `pandas`, `scikit-learn`, `chromadb`, `gradio`, and `pytest`.

### Verification commands (run in this order)

```powershell
# Milestone 1 — already passed on this computer
.\.venv\Scripts\python.exe scripts\validate_dataset.py

# Automated unit tests — 31 passed in 13.37 s (screenshot)
.\.venv\Scripts\python.exe -m pytest tests -q

# Milestone 2 — requires English MiniLM model, downloaded already
.\.venv\Scripts\python.exe -m labs.lab_5_1_cosine_search --evaluate

# Milestone 3 — Chroma indexing, search and persistence
.\.venv\Scripts\python.exe -m labs.lab_5_2_chroma_filters --index
.\.venv\Scripts\python.exe -m labs.lab_5_2_chroma_filters --verify-persistence

# Milestone 4 — review clustering, deduplication and classification
.\.venv\Scripts\python.exe -m labs.lab_5_3_clustering --all

# Milestone 5 — requires both model downloads to be complete
.\.venv\Scripts\python.exe -m labs.lab_5_4_model_comparison --evaluate

# Milestone 6 — build and evaluate the final index, then launch UI
.\.venv\Scripts\python.exe -m scripts.build_index
.\.venv\Scripts\python.exe -m scripts.evaluate_search
.\.venv\Scripts\python.exe app.py
```

Only mark the corresponding task complete after seeing and reviewing the actual output. The demonstration dataset has **288 valid rows, 288 unique IDs, and 12 categories**. It is not yet the real-world corpus requested by Task 1.5.

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

## 6. Verified Preparation and Outstanding Work

### Preparation and repository work — completed

- [x] Created `E:\semantic-search-engine` and configured a single `.venv` using Python 3.11.9.
- [x] Installed and imported the seven declared Python packages; `pip check` reported no broken requirements.
- [x] Integrated the seven milestone starter packages into the project folders.
- [x] Checked Python source syntax with `compileall` without reported errors.
- [x] Validated the 288-record, 12-category demonstration corpus with `scripts/validate_dataset.py`.
- [x] Ran automated unit tests: **31 passed in 13.37 s**.
- [x] Downloaded the English `all-MiniLM-L6-v2` model to the Hugging Face cache.
- [x] Initialized local Git, created and published `main` plus seven milestone branches.
- [x] Confirmed `main` tracks `origin/main` and the Git working tree was clean.

### Not yet confirmed

- [ ] Complete the Milestone 0 knowledge check and explain concepts independently.
- [ ] Replace/supplement the demonstration corpus with a redistributable **real** corpus of 200+ records.
- [ ] Confirm the multilingual model finished downloading.
- [ ] Run and review Lab 5.1 through Lab 5.4 model-based outputs.
- [ ] Index Chroma, test persistence and metadata filters end to end.
- [ ] Run and capture the final Gradio app and held-out retrieval evaluation.
- [ ] Add real screenshots, final measurements and verified conclusions to docs.

**Why the milestone task boxes are not all checked:** source files and tests exist, but the user previously confirmed that the full labs were not run. Unit tests do not replace the labs' exit checks.

## 7. Testing & Evaluation Checklist

- [x] Input demonstration data has valid IDs, text, categories, and dates (288 rows validated).
- [ ] Query and document vectors come from the same embedding model.
- [ ] Cosine search returns a sorted top 5.
- [ ] Chroma indexes persist across runs.
- [ ] Category/date filters exclude incorrect records.
- [ ] English/Arabic model comparisons use the same labeled evaluation set.
- [ ] Final UI responds correctly to empty inputs and no matches.
- [ ] README setup/run instructions are tested from the repository root.

## 8. Final Deliverables

- [ ] Lab 5.1 — 200+ sentence embeddings and cosine top-5 search.
- [ ] Lab 5.2 — Chroma vector database and metadata filtering.
- [ ] Lab 5.3 — KMeans customer-review clustering with readable labels.
- [ ] Lab 5.4 — English-focused vs multilingual model comparison on Arabic queries.
- [ ] Mini Project 5 — persistent vector store, Gradio semantic-search UI, filtering, and model/chunk-size note.
- [ ] GitHub README, lab findings, final screenshots, and reproducible setup steps (repository published; final lab findings/screenshots pending).

## 9. Git Branches & Commit History

The repository is published at **https://github.com/shahd-essam27/semantic-search-engine** with `main` and seven milestone branches:

```text
main
milestone-0-concepts
milestone-1-setup
milestone-2-embeddings
milestone-3-vector-db
milestone-4-nlp
milestone-5-model-comparison
milestone-6-final-project
```

Each source/documentation file was committed with a descriptive message on its milestone branch, and the prepared milestone files were integrated into `main`. **This history means starter code was delivered, not that every milestone was successfully run.** For future changes, update the appropriate branch with `main` before beginning work, run tests, push the branch, and merge reviewed changes by pull request.

Check repository status:

```powershell
git status
git branch -a
git remote -v
```

## 10. References



- [Sentence-Transformers documentation](https://www.sbert.net/)
- [Chroma documentation](https://docs.trychroma.com/)
- [Qdrant documentation](https://qdrant.tech/documentation/)
- [Hugging Face NLP course](https://huggingface.co/learn/nlp-course/)
- [Gradio documentation](https://www.gradio.app/docs)

---

**Progress policy:** Do not check a task until its implementation, test, and documentation are complete. No milestones are marked complete in this initial README.
