# 09 — Git, GitHub, Hugging Face, and the Review Mechanism

**Core rule: you never merge your own pull request.** Every change reaches `main`
through a PR that the PI (or a delegate they name) approves. No exceptions, including
for "trivial" changes — a trivial change to an analysis script is exactly how a wrong
number reaches a paper.

---

## 1. Branch and commit conventions

**Branch names:** `phase<N>/<slug>` — the slug is given in each task entry. One branch
per task. Never reuse a branch after its PR is merged.

**Commits:** conventional-commits style, imperative mood.

```
feat(scaling): add two-step ladder extrapolator
fix(data): exclude step==0 checkpoints from the analysis frame
exp(p1-06): run bias/variance decomposition on all tasks
docs(findings): add phase 1 memo
theory(thm1): draft proof of the extrapolation-aware bound
```

Scopes: `data`, `scaling`, `theory`, `bai`, `train`, `viz`, `paper`, `ci`, `docs`, `plan`.

Every commit that changes analysis code and its results changes both in the *same*
commit, so a results file is never orphaned from the code that made it.

**Never** commit: tokens, datasets, checkpoints, files over 5 MB, or anything under
`data/`. If `git status` shows one, stop and fix `.gitignore` before committing.

---

## 2. Pull request workflow

Per task:

```bash
git checkout main && git pull
git checkout -b phase1/bias-variance
# ... do the task ...
make check                      # lint + tests + provenance validation
# run the /code-review skill on the diff and fix what it finds
git push -u origin phase1/bias-variance
gh pr create --fill --reviewer ParthAgarwalGit --draft=false
```

Then **stop**. Report the PR URL to the PI. Do not open the next task's PR until this
one is merged, unless the PI explicitly says to stack work.

If review comments arrive: address them on the same branch, push, reply to each comment
saying what you changed, and re-request review. Do not resolve a reviewer's thread
yourself unless you made the change they asked for.

---

## 3. PR template

Create `.github/pull_request_template.md` in P0-02:

```markdown
## Task
<Task ID and title from the plan>

## What changed
<2-4 sentences. What a reviewer needs to know to review, not a file list.>

## Results produced
<Each results/*.json written, with the 1-3 headline numbers from it. If none, say
"no results files".>

## How I verified this
- [ ] `make check` passes locally
- [ ] New/changed code has tests
- [ ] Every number in docs or paper traces to a results file
- [ ] Results were regenerated from a clean tree (provenance shows git_dirty=false)
- [ ] No file over 5 MB, no data, no secrets
- [ ] Ran `/code-review` on this diff and addressed the findings

## Compute used
<GPU-hours, or "none — laptop only". If cluster compute: spend against ceiling.>

## Surprises / things I am unsure about
<Anything that did not match expectations. If nothing, write "none". Do not leave
blank — a reviewer reads this field first.>

## Definition of done (from the plan)
<Paste the task's definition of done and confirm each part.>
```

---

## 4. Review checklists

### For every PR (the PI or delegate checks)

- [ ] The diff does only the one task
- [ ] Results are regenerable: does `python experiments/<script>.py` reproduce the JSON?
- [ ] No hardcoded numbers, paths, or magic constants; everything comes from a config
- [ ] Provenance block present and `git_dirty == false`
- [ ] Tests actually assert behaviour, not just that code runs without error
- [ ] The "Surprises" field is filled in honestly

### Extra checks for analysis PRs (Phase 1, 5)

- [ ] Is the estimator unbiased for what it claims to estimate, or is the bias corrected
      and documented?
- [ ] Are noise corrections applied in the *conservative* direction — the one that makes
      our hypothesis harder to confirm, not easier?
- [ ] Is the comparison at matched compute, matched seeds, matched tasks?
- [ ] Are error bars present, and computed by a method stated in the code?
- [ ] Is there any leakage of target-scale data into a fit?

### Extra checks for theory PRs (Phase 2)

- [ ] Is every proof step either cited, mechanical, or explicitly marked `\needshuman`?
- [ ] Does a numerical certificate exist for every claimed inequality?
- [ ] Do the constants in the statement match the constants in the code?

### Extra checks for training PRs (Phase 4)

- [ ] Config fully specified and committed; run reproducible from the config alone
- [ ] Spend counter updated; within ceiling
- [ ] Checkpoint and resume tested
- [ ] Target-scale results sealed from the decision code until the decision is committed

---

## 5. Hugging Face workflow

HF has no PR-blocking mechanism you should rely on, so treat every push as a release:

1. **Never push to HF without explicit PI approval for that push.** Uploads are
   public-facing publication.
2. Push from a clean tree only; `src/pdt/hub.py` enforces this.
3. Prefer HF's **pull-request/discussion** feature for substantive changes:
   `create_commit(..., create_pr=True)` opens a PR on the Hub that the PI can review and
   merge. Use it for anything beyond a typo fix in a card.
4. Every push message references the git commit SHA it came from.
5. Repos stay **private** until GATE-S, then flip to public together with the GitHub repo.
6. Dataset cards must credit DataDecide under ODC-BY and state our own licence.
   Model cards must state training data composition, compute used, and licence.

---

## 6. Issues and the decision log

- Open a GitHub issue for anything you notice but are not doing now: a bug, a smell, a
  follow-up experiment. Label `bug`, `analysis`, `theory`, `infra`, `paper`. Do not fix
  it inline — that is how one-task-per-session dies.
- `docs/decisions.md` is append-only. Log every choice that a future reader would
  otherwise have to reverse-engineer: which metric is canonical, which tie rule, which
  target scale, the approved compute tier, the chosen paper framing. Format:
  `## YYYY-MM-DD — <decision>` then `**Context** / **Decision** / **Rationale** / **Decided by**`.

---

## 7. If the PI is unavailable

You will sometimes be blocked at a GATE with no reply. Do:

- Mark the task `BLOCKED` in `STATUS.md` with the date and what you are waiting for.
- Move to the highest-priority task that is *not* downstream of the blocked GATE
  (Phase 2 and Phase 3 are largely independent of each other, and both are independent
  of GATE-C).
- If genuinely everything is blocked, do low-risk work: improve tests, improve
  documentation, tidy `docs/related_work.md`. Do not start speculative research.

Do **not**: proceed past a GATE, merge your own PR, spend compute, or push to HF because
the PI seems likely to say yes.
