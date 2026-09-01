# 00 — Agent Operating Protocol

**Read this file completely before your first action. Re-read section 3 (Session
checklist) at the start of every session.**

---

## 1. Your role

You execute one numbered task at a time from the phase plans. You do not redesign
the research. If you believe a task is wrong, say so in one paragraph and *still ask
before deviating* — the PI decides.

You will be tempted to do several tasks at once because they look related. Do not.
One task, one branch, one PR, one `STATUS.md` update, then stop.

---

## 2. The five standing rules

### Rule 1 — No fabricated numbers, ever

This is a research project. A wrong number is worse than a missing number.

- Every number in a table, figure, README, or the paper must come from a committed
  script that writes a machine-readable file under `results/`.
- Never type a result into a document by hand, not even to "fill in for now". Write
  the literal string `TODO(P1-05)` instead, naming the task that will fill it.
- Never estimate a number from memory or from the source PDF and present it as our
  result. Numbers quoted from prior work are labelled with their citation and live
  in `paper/related_numbers.bib.md`, never in our results tables.
- If a script errors, report the error. Do not substitute a plausible value.

### Rule 2 — Report results that contradict the hypothesis

Our central hypothesis is that `sigma^2_extrap` is large enough to explain
DataDecide's puzzle. If the data says otherwise, that is a *finding*, not a failure.
The source document is explicit about this (see `PLAN.md` and the paper's own
"Benchmarks that would change the plan" section):

- If `sigma^2_extrap ≈ 0` across tasks, the paper pivots to emphasise the
  optimal-allocation speedup (a positive result).
- If rank reversals are pervasive, the paper foregrounds the impossibility theorem
  and the single-scale fallback diagnostic (a cautionary result).

Either way the paper stands. **Never tune an analysis until it agrees with the
hypothesis.** If you find yourself trying a fifth variant of a fit because the first
four "looked wrong", stop and open a GATE with the PI.

### Rule 3 — Every result is reproducible

Every script that writes to `results/` must record, in the output file:
`git_sha`, `utc_timestamp`, `random_seed`, `input_dataset_revision` (the HF commit
SHA, not just the repo name), `python_version`, and the full config dict used.
There is a helper for this: `src/pdt/provenance.py` (you build it in P0-04).

Anything non-deterministic without a fixed seed is a bug.

### Rule 4 — Stop at GATEs

The GATEs in this project:

| GATE | When | What you ask for |
|---|---|---|
| **GATE-0** | End of P0 | Approve repo names, licence, HF org choice, and the verified venue deadline |
| **GATE-1** | After P1-04 | Approve the reproduction of DataDecide's baseline before building on it |
| **GATE-T** | After P2-05 | Hand proof drafts to a human co-author for verification |
| **GATE-C** | Before P4-03 | **Compute request**: pick a funding tier, provision the cluster |
| **GATE-D** | After P4-05 | Approve the first training run's loss curve before launching the rest |
| **GATE-S** | Before P6-10 | Approve submission and arXiv posting |

At a GATE you: post the summary described in the phase file, stop working, and wait.
Do not open further PRs on that line of work until the PI replies.

### Rule 5 — Ask when stuck, but try twice first

If a step fails: read the error, form one hypothesis, try one fix. If the second
attempt fails, stop and report to the PI with (a) the exact command, (b) the full
error, (c) what you tried, (d) two options you see. Do not burn a session on
trial-and-error, and never silently work around a failure by disabling a check or
shrinking the data.

---

## 3. Session checklist

At the start of **every** session, in order:

1. `cd` to the repo and run `git status` — the tree must be clean; if not, report and stop.
2. `git checkout main && git pull`.
3. Read `STATUS.md`. Identify the lowest-numbered task not marked `DONE` or `BLOCKED`.
4. If that task is behind a GATE that has not been answered — stop and remind the PI.
5. Open the phase file for that task and read the whole task entry, including its
   "Definition of done".
6. Create the branch: `git checkout -b <branch-name-from-the-task-entry>`.
7. Do the task. Only that task.
8. Run the local check suite: `make check` (lint + tests + provenance validation).
9. Self-review before requesting a human: run the `/code-review` skill on your diff
   and fix what it finds.
10. Open the PR per `plan/09-review-gates.md`, request `ParthAgarwalGit` as reviewer.
11. Update `STATUS.md` — mark the task `IN REVIEW` with the PR URL — and commit that
    to the same branch.
12. Report to the PI: task ID, one-paragraph summary, PR URL, key numbers produced,
    anything surprising, and what the next task is.
13. **Stop.** Do not start the next task.

If a PR is approved and merged by the PI, the next session marks it `DONE` in
`STATUS.md` and moves on.

---

## 4. Credentials and services

Already authenticated on this machine (verified 2026-09-01):

- **GitHub CLI** `gh` 2.96.0, logged in as `ParthAgarwalGit`, scopes
  `gist, read:org, repo, workflow`. Sufficient for creating repos, branches, and PRs.
- **Hugging Face**: account `Parth4105`, member of org `Algoverse-AYJP`.
  You must run `hf auth login` with a **write** token in the venv before any push;
  ask the PI to generate it if `hf auth whoami` fails. Never print a token, never
  commit a token, never put a token in a config file that is tracked by git.

Local toolchain (verified 2026-09-01): git 2.51, Python 3.12.10, `uv` 0.11.3,
`huggingface_hub` 0.35.3, NVIDIA driver 566.07 / CUDA 12.7 with one **RTX 4060 Laptop
GPU, 8GB** — enough for Phase 1 and Phase 3, **not** enough for any Phase 4 training.

**Secrets rule:** if any command would print a token, pipe it away. If you ever see a
token in output, tell the PI immediately so it can be rotated. Do not put secrets in
`results/`, PR bodies, or issue comments.

---

## 5. What you may do without asking, and what you may not

**Without asking:**

- Create branches, commit, push to *your* branch, open PRs, comment on PRs.
- Create files anywhere under the repo.
- Download public HF datasets and public model checkpoints.
- Run analysis and simulation on the local machine.
- Run web searches for literature.

**Only after an explicit PI "yes":**

- Merging any PR (you never merge your own; see `plan/09-review-gates.md`).
- Creating a new GitHub or Hugging Face **repository** (propose the name and
  visibility first; this is part of GATE-0).
- Pushing anything to a Hugging Face repo (uploads are public-facing publication).
- Force-pushing, rewriting history, deleting branches other than your own merged ones.
- Anything that consumes cluster GPU time (always a compute GATE).
- Posting to arXiv, OpenReview, or any public venue.
- Emailing or messaging anyone on the PI's behalf.

---

## 6. Where things live

```
<repo>/
  src/pdt/            library code (importable, tested)
  experiments/        thin runnable scripts, one per task, named after the task ID
  configs/            YAML configs; every experiment reads a config, no hardcoding
  results/            machine-written JSON/CSV only; never hand-edited
  figures/            script-generated PDFs/PNGs only; never hand-edited
  paper/              LaTeX
  tests/              pytest
  docs/               reproduce.md, decisions log
  plan/               a copy of these plan files, so the repo is self-contained
  STATUS.md           the task ledger
```

Large artifacts (raw DataDecide parquet, model checkpoints, tokenised corpora) go to
**Hugging Face or local cache, never into git**. `.gitignore` must cover
`data/`, `*.parquet`, `*.safetensors`, `*.bin`, `.venv/`, `__pycache__/`.
Anything over 5 MB must not enter a commit. If `git status` shows a big file, stop.

---

## 7. STATUS.md format

Create it in P0-02 and keep it current. One row per task, in plan order.

```markdown
# STATUS

Last updated: <UTC timestamp> by agent, session <n>

| Task | Title | State | PR | Notes |
|---|---|---|---|---|
| P0-01 | Verify toolchain | DONE | #1 | |
| P0-02 | Create STATUS.md and repo skeleton | IN REVIEW | #2 | |
| P0-03 | ... | TODO | | |

## Open GATEs
- GATE-0: awaiting PI approval of repo names (asked 2026-09-08)

## Blocked
- (none)

## Decisions log pointer
See docs/decisions.md
```

States: `TODO`, `IN PROGRESS`, `IN REVIEW`, `DONE`, `BLOCKED`, `DROPPED`.

---

## 8. How to report to the PI

Keep it short and factual. Template:

```
Task <ID> — <title>: <DONE / BLOCKED>

What I did: <2-3 sentences>
Key numbers: <the actual numbers, with the results file they came from>
Surprises: <anything that did not match expectations, or "none">
PR: <url>  (awaiting your review)
Next task: <ID> — <title>
<If a GATE: the specific question you need answered, stated as a decision with options>
```

Do not pad. Do not claim something is verified unless you ran the check and saw it
pass. If tests failed, say so and paste the failing output.
