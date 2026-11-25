# Week 5 — Milestone 0: Concepts First — Instructions

This package follows **Tasks 0.1–0.12 exactly** as listed in the original Week 5 root README. It is a **study pack**, not a completed milestone; there is no model run or Python installation for Milestone 0.

## 1. Where the files go

Copy the **contents** of this ZIP into your existing `E:\semantic-search-engine\` project, merging folders, not making a second project. The ZIP adds:

```text
E:\semantic-search-engine\
├── README.md                             (existing; keep)
├── requirements.txt                      (existing; keep)
├── .venv\                                (existing; keep)
├── Downloaded_model\                    (existing; keep)
├── data\ ...                             (existing Milestone 1; keep)
├── labs\ ...                             (existing Milestones 2–5; keep)
├── src\ ...                              (existing Milestone 6; keep)
├── docs\
│   └── concepts.md                       (NEW — full 12-task theory guide)
├── study\
│   ├── QUICK_REFERENCE.md                (NEW — compact summary)
│   ├── milestone_0_knowledge_check.md    (NEW — 15 unanswered questions)
│   ├── answer_key.md                     (NEW — only read after attempting quiz)
│   └── MILESTONE_0_CHECKLIST.md          (NEW — study tracker)
└── MILESTONE_0_INSTRUCTIONS.md           (NEW — this walkthrough)
```

**Do not overwrite** an existing `docs/concepts.md` if you already wrote personal notes. Rename the incoming file or merge it manually instead.

## 2. Read in this order (whenever you have time)

1. Open `docs/concepts.md`. Read **one Task 0.1–0.12 at a time** and explain the example aloud.
2. Use `study/QUICK_REFERENCE.md` for recall.
3. Answer the 15 questions in `study/milestone_0_knowledge_check.md` **without copying**.
4. Compare to `study/answer_key.md` and write what needs revisiting.
5. Fill in `My personal notes` at the end of `docs/concepts.md`.
6. Check only the tasks you've genuinely understood in both the tracker and original root `README.md`.

No command is required to use these Markdown files. In VS Code use `Ctrl+Shift+V` to preview Markdown.

## 3. GitHub — separate milestone branch

**Important:** The earlier milestone packs have been prepared, not necessarily committed/merged. Never assume `main` contains them. Before commands, check:

```powershell
git status
git branch --show-current
git branch -a
git remote -v
```

If Git has never been initialized, follow the initial Git setup in `MILESTONE_1_INSTRUCTIONS.md` first. If this project is already initialized, **do not run `git init` again**.

Because Milestone 0 is conceptually first, the cleanest history is to branch from `main` **after the base scaffold exists** and before unrelated uncommitted changes. If your working tree contains uncommitted milestone work, commit it on its intended branch or stash safely; **do not discard changes**. On a clean `main`:

```powershell
git switch main
git switch -c milestone-0-concepts
```

Add this package on the feature branch. Only stage the files you mean to commit:

```powershell
git add docs/concepts.md study/QUICK_REFERENCE.md study/milestone_0_knowledge_check.md study/answer_key.md study/MILESTONE_0_CHECKLIST.md MILESTONE_0_INSTRUCTIONS.md
git diff --cached --stat
git commit -m "docs: add week 5 semantic search concepts study guide"
```

**Two choices:**

- **If studying later:** You may push the study files as a **draft/partial** branch, but leave Milestone 0 unchecked in the root README; don't merge/describe it as a completed concept milestone.
- **If studied and verified:** Update the original root `README.md` with only genuinely completed Tasks 0.1–0.12, then commit that update separately:

```powershell
git add README.md docs/concepts.md study/milestone_0_knowledge_check.md study/MILESTONE_0_CHECKLIST.md
git commit -m "docs: record milestone 0 knowledge check and completion"
```

If `origin` is already configured, push:

```powershell
git push -u origin milestone-0-concepts
```

On GitHub, create a pull request from `milestone-0-concepts` into `main` **once the milestone is genuinely complete**. If later milestone branches were already created, they may need merging or rebasing to include this documentation; handle that carefully, without rewriting shared history unnecessarily.

## 4. Milestone 0 completion conditions

- All 12 original README tasks understood.
- Personal notes and answers written.
- Able to sketch and explain the pipeline from documents to ranked results.
- README checkbox status reflects actual learning, not just file creation.
- Git branch created and pushed/merged as appropriate.

## 5. Scope and source

The uploaded Week 5 roadmap specifies embeddings, similarity, vector databases, chunking, embedding-based NLP and selection for Arabic; the **12 detailed Milestone 0 tasks** were established in the existing Week 5 root README. Examples and quiz explanations are educational expansions, not claims of measured model performance.
