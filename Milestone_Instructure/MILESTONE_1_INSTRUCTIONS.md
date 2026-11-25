# Week 5 Milestone 1 — Review / merge instructions

This bundle **adds** to your existing `E:\semantic-search-engine` project. Do not delete or replace your `README.md`, `requirements.txt`, `.venv`, or `Downloaded_model/` folder. The bundled `.gitignore` can replace your existing one only after you compare them and preserve anything project-specific.

## What you already completed (verified from screenshots)

- Task 1.1: existing `E:\semantic-search-engine` folder opened in VS Code (different folder name from README target; functionally fine).
- Task 1.2: root `.venv` uses Python 3.11.9 and is active.
- Task 1.4: packages installed, `pip check` clean and imports succeeded.
- Task 1.3: `.gitignore` and `requirements.txt` created; remaining folders can be added from this bundle.

## Add the bundled files

Copy `data/`, `docs/`, `scripts/`, `labs/`, `src/`, `tests/`, and `assets/` into `E:\semantic-search-engine`. These create the dataset, validator, documentation, and empty placeholder folders. Do not overwrite any folder or file containing your own work. `Downloaded_model/download_models.py` stays where it is.

## Run Milestone 1 checks

Open the VS Code terminal with the project root as working directory, then run:

```powershell
.\.venv\Scripts\python.exe scripts\validate_dataset.py
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -c "import numpy,pandas,sklearn,chromadb,gradio,sentence_transformers,pytest; print('Imports OK')"
```

The validator should print `PASS: 288 valid rows, 288 unique IDs, 12 categories`. If the Hugging Face download is still running, leave that terminal alone and use a separate terminal for these checks.

**Important:** `documents.csv` is an original, generated demo corpus. Task 1.5's requirement for an *actual* licensed or permitted source remains outstanding until you select a real corpus. Task 1.8's source documentation is included, but must be updated when you choose the real corpus. Do not mark Milestone 1 fully complete before that.

## GitHub workflow (multiple branches)

**After** you verify the files, and after the model downloader has finished (or when it is safe to run a separate terminal), check your repository state:

```powershell
git status
```

If this says `not a git repository`, initialize Git:

```powershell
git init -b main
```

If your repository is already initialized, **do not** reinitialize it. Check the active branch:

```powershell
git branch --show-current
```

If there are uncommitted setup files on `main`, preserve them by committing the safe initial scaffold on `main` first (after reviewing `git status`):

```powershell
git add README.md requirements.txt .gitignore Downloaded_model
# Only if the staged files are correct and do not contain secrets/models:
git diff --cached --stat
git commit -m "chore: initialize week 5 project scaffold"
```

Then create the milestone branch. If you began on another branch, check first before switching:

```powershell
git switch -c milestone-1-setup
```

Stage and commit the new milestone data and documentation once validated:

```powershell
git add data/raw/documents.csv docs/dataset.md scripts/validate_dataset.py labs src tests assets
# Include .gitignore if updated
git add .gitignore
git status
git commit -m "data: add and validate sample semantic search corpus"
```

**Do not push or merge as "complete" yet** while Task 1.5 is pending. Once you replace/extend it with an eligible real dataset, update `docs/dataset.md`, validate again, and then mark all confirmed tasks in `README.md`.

When your GitHub remote is configured:

```powershell
git remote -v
# If origin is configured:
git push -u origin milestone-1-setup
```

Create a PR from `milestone-1-setup` into `main` on GitHub after completing the pending tasks and reviewing your changes. If no `origin` is set, create a GitHub repository first and use the `git remote add origin <your-actual-repository-URL>` command with its real URL; never use a guessed URL.

## Checklist status

- Task 1.1: verified (folder name differs from README; optional documentation note).
- Task 1.2: verified.
- Task 1.3: file set supplied; locally verify after copying.
- Task 1.4: verified.
- Task 1.5: **pending real corpus**; demo only.
- Task 1.6: demo schema prepared; validate after copying.
- Task 1.7: validator supplied; run it locally.
- Task 1.8: demo documentation drafted; update for actual source.

Milestone 0 remains incomplete by choice; it can be studied later.
