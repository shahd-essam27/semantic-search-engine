# Week 5 — Milestone 2 (Lab 5.1): Embeddings & Cosine Search

> **Reference:** The existing project `README.md`, Tasks **2.1–2.8**. We are not renumbering or redesigning the milestones. Your Windows project root is `E:\semantic-search-engine`.

## 1. First, understand exactly where things go

The ZIP contains only files for Milestone 2. **Copy its `labs/`, `docs/`, and `tests/` folders into the root of your EXISTING project**, not into a new nested `week5_m2` directory. Keep all existing files.

```text
E:\semantic-search-engine\
├── .venv\                               ← already created; keep it
├── .gitignore                           ← already created; keep it
├── README.md                            ← YOUR existing roadmap; keep it
├── requirements.txt                     ← already installed; keep it
├── Downloaded_model\                    ← YOUR existing model download script; keep it
│   ├── __init__.py
│   └── download_models.py
├── data\
│   ├── raw\
│   │   └── documents.csv                ← from MILESTONE 1, NOT included here
│   └── processed\                       ← automatic embeddings cache after run
│       └── lab_5_1\
│           ├── embeddings.npy           ← generated; DON'T commit
│           └── metadata.json            ← generated; DON'T commit
├── labs\
│   ├── __init__.py                       ← optional; can leave/create an empty file
│   └── lab_5_1_cosine_search.py          ← NEW milestone 2 lab
├── tests\
│   └── test_lab_5_1.py                  ← NEW fast tests
├── docs\
│   ├── dataset.md                       ← from Milestone 1
│   └── lab_5_1_results.md               ← NEW; auto updated after --evaluate
└── MILESTONE_2_INSTRUCTIONS.md          ← optional project documentation
```

**If you don't yet have `data/raw/documents.csv`, extract the Milestone 1 package first.** Its demonstration dataset has 288 rows. The real-corpus requirement is still pending.

## 2. Files in the downloaded ZIP

| ZIP path | Copy to your existing project | What it does |
| --- | --- | --- |
| `labs/lab_5_1_cosine_search.py` | `E:\semantic-search-engine\labs\` | Loads CSV, embeds documents, computes cosine similarities, prints ranked top 5 |
| `tests/test_lab_5_1.py` | `E:\semantic-search-engine\tests\` | Quick unit tests for cosine ranking, empty queries, and missing datasets |
| `docs/lab_5_1_results.md` | `E:\semantic-search-engine\docs\` | Lab report template; later overwritten by measured results |
| `MILESTONE_2_INSTRUCTIONS.md` | Root of project (optional) | These instructions |

Don't copy any `.venv` folders or model weight files from the ZIP (none are included).

## 3. Before running (VS Code)

1. Open `E:\semantic-search-engine` as your VS Code folder.
2. Select **Ctrl+Shift+P → Python: Select Interpreter → `E:\semantic-search-engine\.venv\Scripts\python.exe`**.
3. Open **Terminal → New Terminal**. Commands below use the explicit root interpreter and do not need shell activation.
4. Confirm Milestone 1 corpus exists and validate it:

```powershell
.\.venv\Scripts\python.exe scripts\validate_dataset.py
```

If this script does not exist, you have not merged Milestone 1 yet. Copy its `data/` and `scripts/` folders first.

5. The `all-MiniLM-L6-v2` model must be cached, or the first run needs internet access to download it. You already installed the relevant Python packages.

## 4. Run the lab, step by step

From `(.venv) PS E:\semantic-search-engine>`:

### A. Generate document embeddings and perform a top-5 search

```powershell
.\.venv\Scripts\python.exe labs\lab_5_1_cosine_search.py --query "How do I protect my online accounts?"
```

The initial run may take longer while loading the model and embedding 288 documents. Later runs reuse the saved embedding cache. You should see the model name, vector `shape` and `dtype`, and **five ranked results** with ID, score, category, and text.

### B. Enter your own queries interactively

```powershell
.\.venv\Scripts\python.exe labs\lab_5_1_cosine_search.py
```

Enter a question when prompted. Type `exit` to quit.

### C. Test 10 example queries and generate the results report

```powershell
.\.venv\Scripts\python.exe labs\lab_5_1_cosine_search.py --evaluate
```

This overwrites `docs/lab_5_1_results.md` with **real observed** results from your machine. Read the report and replace `TODO` items about good/bad results and paraphrase performance with your own observations.

### D. Optional: force new document embeddings (rather than reuse cache)

```powershell
.\.venv\Scripts\python.exe labs\lab_5_1_cosine_search.py --rebuild --query "How do I secure my Wi-Fi?"
```

### E. Run quick tests (model is mocked, so no model download required)

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_lab_5_1.py -q
```

## 5. How the code maps to Tasks 2.1–2.8

- **2.1** `load_records()` loads at least 200 records.
- **2.2** `SentenceTransformer(MODEL_NAME, device='cpu')` loads the English embedding model. Confirm model details in its official model card.
- **2.3** `load_or_create_embeddings()` generates one embedding per record; prints shape/dtype.
- **2.4** `search()` embeds each query with the *same* model.
- **2.5** `doc_vectors @ query_vec` calculates cosine similarity since both vectors were normalized to unit length.
- **2.6** Ranked output displays top five texts and score/ID/category.
- **2.7** `--evaluate` runs ten queries including paraphrases.
- **2.8** Generates `docs/lab_5_1_results.md` with actual scores/latency, plus human-review TODOs.

## 6. Git/GitHub multi-branch workflow

**Before creating a new milestone branch:** Check whether Milestone 1 work is committed on your current branch. Do not switch branches with uncommitted Milestone 1 edits. Do not use `git add .` blindly.

```powershell
git status
git branch
```

If you have **already merged Milestone 1 into `main`**:

```powershell
git switch main
git pull origin main
git switch -c milestone-2-embeddings
```

If you have **NOT created the Git repository yet**, initialize it first on your existing project root, commit the stable groundwork, and only then make the milestone branch:

```powershell
git init -b main
git add .gitignore README.md requirements.txt
git commit -m "chore: initialize semantic search project"
git switch -c milestone-2-embeddings
```

**Important:** This initial commit intentionally stages only the three named files. It does not mark Milestone 1 complete or include the Milestone 1 dataset. If Milestone 1 is uncommitted, organize its commit on `milestone-1-setup` first. Don't merge an incomplete milestone and label it completed.

After you run the lab, verify its output and edit the report:

```powershell
git status
git add labs/lab_5_1_cosine_search.py tests/test_lab_5_1.py docs/lab_5_1_results.md MILESTONE_2_INSTRUCTIONS.md
git commit -m "feat: implement cosine similarity semantic search lab"
git push -u origin milestone-2-embeddings
```

`git push` assumes you already have a GitHub remote named `origin`; if not, add one for your own repository first.

Go to GitHub → your repository → **Compare & pull request** → base `main`, compare `milestone-2-embeddings`. Merge **only after** the lab is tested, the manual report notes are complete, and any unfinished Milestone 1 prerequisites have been resolved. To verify a merge, run `git switch main` and `git pull origin main`.

**Update your existing README:** check Tasks 2.1–2.8 only when each is actually verified; do not pre-check them. Mention the added test file if you update the structure tree.

## 7. Gitignore (add if absent)

```gitignore
.venv/
__pycache__/
.pytest_cache/
data/processed/
*.safetensors
*.bin
```

Don't add `data/raw/documents.csv` to `.gitignore` if it is the original shareable demonstration dataset. Do not commit Hugging Face model weights or embeddings cache files.

## 8. Completion criteria

- `documents.csv` loads with **200+ unique, nonempty** document texts.
- CLI outputs **five ranked results** including scores, IDs and categories.
- Embedding shape/dtype displayed; query and corpus are encoded by the same model.
- Ten test queries evaluated with actual results saved in `docs/lab_5_1_results.md`.
- Report manually reviewed for relevance, paraphrases and limitations.
- Branch `milestone-2-embeddings` pushed and, after verification, merged into `main`.

**These files are prepared, not executed on your Windows laptop.** Complete the commands above before marking the milestone done.
