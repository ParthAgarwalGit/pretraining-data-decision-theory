# Phase 0 — Setup

**Goal:** a working repo, a reproducible environment, the DataDecide data cached
locally, and a confirmed-current novelty check. No research results yet.

**Compute:** laptop only. No GATE for compute in this phase.
**Ends at:** GATE-0.

---

## P0-01 — Verify and record the toolchain

**Branch:** `phase0/toolchain`

1. Run and record output of: `git --version`, `gh --version`, `gh auth status`,
   `python --version`, `uv --version`, `nvidia-smi`.
2. Confirm `gh auth status` shows `ParthAgarwalGit` and scopes include `repo` and `workflow`.
3. Run `hf auth whoami`. If it fails, **stop and ask the PI for a Hugging Face write
   token** — do not proceed to any HF step. (Expected identity: `Parth4105`, org
   `Algoverse-AYJP`.)
4. Write findings into `docs/environment.md` in the repo you create in P0-02. For this
   first task, hold them in a scratch file and commit them as part of P0-02 if the
   repo does not exist yet.

**Definition of done:** `docs/environment.md` lists every tool with its version and
the date checked, plus one line stating the local GPU (RTX 4060 Laptop, 8 GB) is
sufficient for Phases 1/3 and insufficient for Phase 4.

**Known baseline as of 2026-09-01:** git 2.51.0, gh 2.96.0, Python 3.12.10, uv 0.11.3,
huggingface_hub 0.35.3, driver 566.07 / CUDA 12.7, RTX 4060 Laptop 8 GB.

---

## P0-02 — Propose repo names, then create the GitHub repo and skeleton

**Branch:** `phase0/repo-skeleton` (after the repo exists)

**This task contains part of GATE-0. Propose, then wait.**

1. Propose to the PI, and wait for approval:
   - GitHub repo name. Recommended: `pretraining-data-decision-theory`
     (alternatives: `extrapolated-bai`, `datadecide-theory`).
   - Visibility: recommend **private until GATE-S**, then flip to public at release.
     Rationale: novelty risk during a live submission cycle.
   - Licence: recommend **Apache-2.0** (permissive, patent grant, standard for ML
     research code). Note that DataDecide artifacts are ODC-BY / CC BY 4.0 and must be
     credited in `NOTICE`; our derived tables inherit ODC-BY attribution obligations.
   - HF namespace: personal `Parth4105` or org `Algoverse-AYJP`. Recommend the org if
     this is Algoverse work, personal otherwise. **Ask, do not guess.**
2. After approval: `gh repo create <name> --private --description "..." --clone`.
3. Create the skeleton exactly as laid out in `plan/00-agent-protocol.md` section 6.
4. Copy this entire `plan/` directory and `PLAN.md` into the repo so it is
   self-contained.
5. Create `STATUS.md` from the template in the protocol, pre-populated with **every**
   task ID from all phase files, all in state `TODO`.
6. Create `docs/decisions.md` — an append-only log: date, decision, rationale, who
   decided.
7. Add `.gitignore` covering `data/`, `.venv/`, `__pycache__/`, `*.parquet`,
   `*.safetensors`, `*.bin`, `*.pt`, `wandb/`, `.env`, `*.ipynb_checkpoints`.
8. Configure branch protection on `main`:
   `gh api -X PUT repos/{owner}/{repo}/branches/main/protection` requiring at least
   1 approving review and passing status checks, and disallowing force pushes. If the
   repo is private on a plan without protection support, instead record in
   `docs/decisions.md` that protection is enforced by convention, and **never push to
   `main` directly** regardless.

**Definition of done:** repo exists, skeleton committed via a PR (not a direct push to
`main` after the initial commit), `STATUS.md` lists all tasks, branch protection
configured or its absence documented.

---

## P0-03 — Reproducible Python environment

**Branch:** `phase0/env`

1. `uv init` / create `pyproject.toml` with `requires-python = ">=3.11,<3.13"`.
2. Pin these dependency groups:
   - core: `numpy`, `scipy`, `pandas`, `polars`, `pyarrow`
   - hub: `huggingface_hub`, `datasets`
   - fitting: `scipy.optimize` (in scipy), `statsmodels`
   - plotting: `matplotlib` (no seaborn; keep the figure style controlled by us)
   - dev: `pytest`, `pytest-cov`, `ruff`, `mypy`, `pre-commit`
   - Do **not** add torch yet; it arrives in P4-01 in a separate optional group
     `[project.optional-dependencies] train`.
3. `uv lock` and commit `uv.lock`.
4. Create `Makefile` with targets:
   - `make setup` — `uv sync --all-extras`
   - `make lint` — `ruff check . && ruff format --check .`
   - `make test` — `pytest -q`
   - `make check` — lint + test + `python -m pdt.provenance --validate results/`
   - `make figures` — regenerate every figure from `results/`
   - `make paper` — build the LaTeX
5. Add `.pre-commit-config.yaml` with ruff and a hook rejecting files over 5 MB.

**Definition of done:** `make setup && make check` passes on a clean clone.

---

## P0-04 — Provenance helper and results contract

**Branch:** `phase0/provenance`

Build `src/pdt/provenance.py`:

- `def stamp(config: dict, extra: dict | None = None) -> dict` returning
  `{git_sha, git_dirty, utc_timestamp, python_version, package_versions, seed, config, ...extra}`.
- `def write_result(path: str, payload: dict, config: dict) -> None` that writes
  `{"provenance": stamp(config), "data": payload}` as pretty JSON, creating parent
  directories, and refuses to overwrite unless `PDT_OVERWRITE=1` is set.
- `def validate(results_dir: str) -> int` (the CLI `--validate` path) checking every
  JSON under `results/` has a `provenance` block with a non-empty `git_sha` and a
  `git_dirty == false`. Exit non-zero if any file fails.
- Tests in `tests/test_provenance.py`.

**Why this exists:** it is the mechanical enforcement of Rule 1 and Rule 3. A result
produced from a dirty working tree is not reproducible and CI will reject it.

**Definition of done:** unit tests pass; `make check` fails loudly if you hand-write a
JSON into `results/` without provenance.

---

## P0-05 — CI

**Branch:** `phase0/ci`

`.github/workflows/ci.yml`, triggered on pull requests and pushes to `main`:

1. `ubuntu-latest`, Python 3.12, `uv sync --all-extras`.
2. `make lint`
3. `make test` with coverage; fail under 70% on `src/pdt/`.
4. `make check` (includes provenance validation).
5. A **fabrication guard** job: `python tools/check_no_orphan_numbers.py` — see below.
6. Cache the `uv` environment and the HF datasets cache keyed on `uv.lock`.

`tools/check_no_orphan_numbers.py`: scans `paper/**/*.tex`, `README.md`, and
`docs/**/*.md` for numeric literals with 2+ significant decimal digits that are not
(a) inside a `% NUMBER-OK` marked line, (b) inside a `\cite`-adjacent "prior work"
environment, or (c) present in some file under `results/`. Report violations as
warnings during P0-P2 and **hard failures from P6-01 onward** (flip the flag in the
workflow at P6-01). This is the automated backstop for "never invent a number".

**Definition of done:** CI green on a PR; a deliberately introduced fake number in
`README.md` makes the guard job flag it.

---

## P0-06 — Acquire and cache DataDecide

**Branch:** `phase0/data-datadecide`

Verified-available artifacts on the Hub (checked 2026-09-01):

| Repo | Type | Size | What it holds |
|---|---|---|---|
| `allenai/DataDecide-eval-results` | dataset | 1.4M rows / 693 MB parquet | **The primary table.** Columns: `params, data, task, step, seed, chinchilla, tokens, compute, metrics` |
| `allenai/DataDecide-ppl-results` | dataset | 22.7K rows / 2 MB | Per-domain validation perplexities; columns `step, eval/<domain>-validation/Perplexity (11 domains), data, params, seed` |
| `allenai/DataDecide-data-recipes` | dataset | — | The 25 corpora definitions |
| `allenai/DataDecide-eval-instances` | dataset | — | Per-instance eval records |
| `allenai/DataDecide-<recipe>-<size>` | models | many | The actual checkpoints (only needed if a task requires re-running evals; Phase 1 does not) |
| Collection `allenai/datadecide-67edb1d2bacba40b5d3ed633` | collection | — | Index of everything above |

Important schema facts you will need, confirmed from a live preview:

- `metrics` is a **stringified Python dict**, not a struct. Parse it with
  `ast.literal_eval`, not `json.loads` (it uses single quotes). Keys include
  `primary_metric`, `acc_raw`, `acc_per_char`, `acc_per_token`, `acc_uncond`,
  `correct_prob`, `correct_prob_per_char`, `norm_correct_prob`, `bits_per_byte_corr`,
  `logits_per_char_corr`, `margin`, and others.
  - **Discrete metrics** for the accuracy analysis: `acc_per_char`, `primary_metric`.
  - **Continuous likelihood proxies** for the metric ablation (P5-03):
    `correct_prob_per_char`, `norm_correct_prob`, `bits_per_byte_corr`.
- `seed` is a string with values like `default`, `small aux 2`, `small aux 3` — three
  seeds, but **not** named `0/1/2`. Do not assume integer seeds.
- `chinchilla` is a string token-budget multiplier, e.g. `5xC`.
- `step = 0` rows exist and are **untrained checkpoints** — they must be excluded from
  every fit. Filter `step > 0`.
- `params` is a string like `10M`, `150M`, `1B`. Write a parser to numeric params and
  keep the canonical ordering.

Steps:

1. Write `src/pdt/data/datadecide.py`:
   - `download(revision: str | None = None) -> Path` using `huggingface_hub.snapshot_download`
     into `data/raw/` (gitignored), **pinning and returning the dataset commit SHA**.
   - `load_eval_results() -> polars.DataFrame` with `metrics` exploded into typed
     columns, `params` parsed to int, `step > 0` filter applied, and a
     `is_final_checkpoint` boolean (max step within each `params, data, seed` group).
   - `load_ppl_results()`, `load_recipes()`.
   - Cache the parsed frame as a local parquet so later tasks load in seconds.
2. Write `experiments/p0_06_data_inventory.py` producing
   `results/p0_06_inventory.json` with: the list of distinct recipes (expect 25), the
   list of distinct `params` sizes with parsed values (expect 14, roughly 4M to 1B),
   distinct seeds, distinct tasks, distinct `chinchilla` values, row counts, and the
   pinned dataset revision SHA.
3. Sanity-assert in code (not in prose): 25 recipes, 14 sizes, 3 seeds. **If the
   actual counts differ, do not "fix" the assertion — report the real counts to the PI
   and record them in `docs/decisions.md`.** The source PDF itself warns that the
   "14 sizes / 25 recipes" figures come partly from a blog post and the HF collection
   rather than the arXiv abstract, so a mismatch is plausible and is information.

   **Outcome (2026-09-02):** 25 recipes and 14 sizes matched exactly. Seeds did not —
   5 distinct labels appear in the raw union, not 3, though every individual size still
   has exactly 3 (the extra labels are `large aux 2/3`, used only at 1B in place of
   `small aux 2/3`). Full detail in `docs/decisions.md`; see also the P1-01 update in
   `plan/02-phase1-datadecide.md` for what this resolves.

**Definition of done:** `results/p0_06_inventory.json` exists with a pinned revision
SHA; a second run reproduces it byte-identically except the timestamp.

---

## P0-07 — Novelty sweep #1

**Branch:** `phase0/novelty-sweep-1`

Search arXiv, OpenReview (ICLR/NeurIPS/ICML/COLM 2025-2027 cycles), Semantic Scholar,
and Google Scholar for concurrent work. Queries to run, at minimum:

- "best arm identification" + "scaling law"
- "sample complexity" + "data mixture" + "selection"
- "pure exploration" + "neural scaling"
- "extrapolation" + "best arm" + "bandit"
- "active experimental design" + "scaling law"
- "decision" + "pretraining data" + "confidence" / "guarantee"
- Citations *of* DataDecide (arXiv:2504.11393) — check every paper citing it
- Citations of Garivier & Kaufmann 2016 from 2025 onward that mention LLMs

Known near misses named in the source document, which you must read and characterise
precisely (each in one paragraph: what it does, what it does *not* do, why we are
distinct): arXiv:2605.17234 (active budget allocation for scaling-law *estimation*),
arXiv:2607.06879 (BAI with a same-scale generative proxy), arXiv:2601.21471 (BAI with
a biased LLM judge plus audits).

Also verify the exact citation metadata for every reference in the source PDF's
reference list — several are flagged as workshop papers or very recent preprints, and
the Muennighoff et al. Data-Constrained Scaling paper is cited as both 2023 and 2024
in different places. Produce `docs/related_work.md` with a table:
`citation | venue+year (verified) | arXiv id | one-line claim | how we differ`.

**Definition of done:** `docs/related_work.md` committed; a clear verdict sentence at
the top: "As of <date>, no work found that does X, Y, Z" or "Concurrent work found:
<citation> — assessment: <overlap>".

---

## P0-08 — Create the HF dataset repo shell

**Branch:** `phase0/hf-repo`

**Requires the namespace decision from P0-02 and PI approval to push.**

1. Create `<namespace>/pdt-datadecide-analysis` as a **dataset** repo, private
   initially.
2. Push only a `README.md` (dataset card) describing what will land there: derived
   analysis tables, fitted scaling-law parameters, per-task `sigma^2_extrap` estimates.
   Include the ODC-BY attribution to AI2's DataDecide, and state our licence.
3. Add `src/pdt/hub.py` with `push_results(local_dir, repo_id, revision_msg)` that
   refuses to run unless the working tree is clean and `results/` passes provenance
   validation.

**Definition of done:** the dataset repo exists and is empty except for the card;
`hub.py` has a unit test proving it refuses a dirty tree.

---

## GATE-0 — End of Phase 0

Post to the PI:

1. Repo URL, visibility, licence, and the HF namespace used.
2. `results/p0_06_inventory.json` summary: actual recipe/size/seed/task counts, and
   whether they match the expected 25/14/3.
3. The novelty verdict sentence from `docs/related_work.md`.
4. The **verified** deadline dates for ICML / COLM / NeurIPS next cycle, with the
   recommended target and why.
5. Confirmation that CI is green.

Then stop and wait for approval to enter Phase 1.
