# Week 5 — Milestone 6: Final Semantic Search Engine

## 0. Prerequisites

Your project is `E:\semantic-search-engine` and uses the **existing** root `.venv` (Python 3.11). Do not create a new environment.
Ensure you have completed/copy-pasted files from Milestones 1–5, especially `data/raw/documents.csv`.

**Important:** Milestone 1's provided `documents.csv` contains synthetic demonstration notes, not the real corpus required by Tasks 1.5 and 6.10. The starter app can be tested on it, but for a portfolio-ready Week 5 completion you must provide a real corpus with sharing rights (200+ records), update dataset documentation, and rebuild/re-evaluate.

## 1. Exact file placement

Extract this ZIP into `E:\semantic-search-engine`, **merging folders** (do not nest another `week5_m6` folder). Do not overwrite earlier labs or README.

```text
E:\semantic-search-engine\
├── .venv\                         EXISTING (keep)
├── README.md                      EXISTING (update tasks after verification)
├── requirements.txt               EXISTING (keep)
├── .gitignore                     EXISTING (ensure chroma_db/ is ignored)
├── Downloaded_model\             EXISTING (keep)
├── data\
│   ├── raw\documents.csv          EXISTING (from M1)
│   └── evaluation\
│       ├── model_comparison_queries.csv  EXISTING (from M5)
│       └── final_relevance_queries.csv   NEW
├── labs\                          EXISTING (M2–M5: keep all)
├── src\
│   ├── __init__.py                NEW
│   ├── data_loader.py             NEW
│   ├── embeddings.py              NEW
│   ├── indexing.py                NEW
│   └── search.py                  NEW
├── scripts\
│   ├── __init__.py                NEW
│   ├── validate_dataset.py        EXISTING (M1)
│   ├── build_index.py             NEW
│   └── evaluate_search.py         NEW
├── app.py                         NEW
├── tests\
│   └── test_search.py             NEW (keep previous test files)
├── docs\
│   ├── final_project.md           NEW
│   └── final_evaluation.md        GENERATED on your computer
├── assets\
│   ├── README.md                  NEW
│   ├── search_ui.png              CAPTURE LATER
│   ├── filtered_results.png       CAPTURE LATER
│   └── evaluation_results.png     CAPTURE LATER
├── chroma_db\                     GENERATED LOCALLY, NEVER GIT COMMIT
└── MILESTONE_6_INSTRUCTIONS.md    NEW
```

## 2. Start from the project root in the VS Code integrated terminal

If a previous model download is still running in one terminal, use a second VS Code terminal.

```powershell
cd E:\semantic-search-engine
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest tests\test_search.py -q
```

## 3. Index your dataset (Task 6.3)

```powershell
.\.venv\Scripts\python.exe -m scripts.build_index
```

The first run encodes every record and writes a persistent `chroma_db/` folder. Re-running updates by stable chunk IDs. **Do not commit `chroma_db/` to Git**. Ensure your model cache is available and `sentence-transformers` and `chromadb` are installed.

Optional: test custom chunking using **matching values for both indexing and searching** (app currently uses the default 120/20 values):

```powershell
.\.venv\Scripts\python.exe -m scripts.build_index --chunk-words 120 --overlap-words 20
```

## 4. Launch the Gradio UI (Tasks 6.4–6.6)

```powershell
.\.venv\Scripts\python.exe app.py
```

Open the local URL printed in the terminal, typically `http://127.0.0.1:7860`.
Try a meaning-based query, category dropdown, date filters (`YYYY-MM-DD`), and top-k. Stop with Ctrl+C.

## 5. Run the final evaluation (Task 6.7)

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate_search
.\.venv\Scripts\python.exe -m pytest tests\test_search.py -q
```

Inspect `docs/final_evaluation.md` and write your own observations. Evaluation examples and labels are demonstration-only and must be revised for the real corpus. Capture genuine screenshots in `assets/` after running the UI.

## 6. Finish project docs and README (Tasks 6.8–6.10)

- Fill TODOs in `docs/final_project.md` with measured values and the model decision from Milestone 5.
- Take three real screenshots and put them in `assets/`.
- Do manual smoke tests (see `docs/final_project.md`).
- Replace synthetic data with your *licensed* real 200+ corpus to meet the original roadmap requirement.
- Edit your **existing root `README.md`**, marking Tasks 6.1–6.10 `[x]` **only after verifying each**; leave uncompleted Milestone 0 or earlier tasks unchecked.
- Add a small "Run the app" section to README with install/index/launch/evaluate commands.

## 7. GitHub multi-branch workflow

Only switch branches with a clean worktree, and after M5 is merged into `main`:

```powershell
git status
git switch main
git pull origin main
git switch -c milestone-6-final-project
```

If previous milestones are **not merged**, create the branch from the current verified M5 branch instead, **and do not merge this branch directly to main until dependencies are integrated**.

After verification:

```powershell
git add src/ app.py scripts/build_index.py scripts/evaluate_search.py scripts/__init__.py
git add tests/test_search.py data/evaluation/final_relevance_queries.csv
git add docs/final_project.md docs/final_evaluation.md assets/ MILESTONE_6_INSTRUCTIONS.md
git add README.md
git status
git commit -m "feat: build end-to-end Gradio semantic search engine"
git push -u origin milestone-6-final-project
```

Check `git status` to confirm you are **not** staging `.venv/`, the model cache, or `chroma_db/`. If you haven't generated `docs/final_evaluation.md` yet, omit it from `git add` until it exists. Publish a pull request `milestone-6-final-project` → `main`, review changes, then merge. Keep the feature branch or delete it after merging according to your preferred workflow.

## 8. Milestone 6 tasks (original README reference)

- [ ] 6.1 Reusable loading/preprocessing/embedding/indexing/search modules
- [ ] 6.2 Chunk-size/overlap decision with document/chunk IDs and metadata
- [ ] 6.3 Repeatable indexing command for **real corpus**
- [ ] 6.4 Query embedding, optional filters, and top-k retrieval
- [ ] 6.5 Gradio UI query/top-k/category/date/ranked results
- [ ] 6.6 Handle empty/missing/invalid conditions
- [ ] 6.7 Held-out relevance checks and results
- [ ] 6.8 Record model/dimension/chunking/runtime/limitations and rebuild instructions
- [ ] 6.9 Genuine UI and evaluation screenshots in assets/
- [ ] 6.10 Clean setup, tidy GitHub repo and completed README checklist

**Status:** Starter implementation and fast unit tests prepared here; **full model-based indexing, Chroma integration, Gradio visuals and runtime timings must be verified on your Windows laptop**.
