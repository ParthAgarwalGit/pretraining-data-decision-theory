# Decisions Log

Append-only. Every entry: date, decision, rationale, who decided. Never
edit or delete a past entry — if a decision is reversed, add a new entry
that supersedes it and says so.

---

## 2026-09-01 — Repository name, visibility, licence

**Context:** Task P0-02 requires proposing repo name, visibility, licence,
and Hugging Face namespace to the PI before creating anything (part of
GATE-0).

**Decision:**
- GitHub repo name: `pretraining-data-decision-theory`
- Visibility: private, to be flipped to public at GATE-S (submission approval)
- Licence: Apache-2.0
- Hugging Face namespace: personal (`Parth4105`), not the `Algoverse-AYJP` org

**Rationale:** Private-until-submission avoids scoop risk given the novelty
caveats already flagged in `plan/01-phase0-setup.md` (P0-07) — the source
document's own novelty check was explicitly "not provably exhaustive."
Apache-2.0 is standard for ML research code and includes a patent grant.
Personal HF namespace was chosen because this project's HF affiliation with
Algoverse-AYJP was not confirmed at time of asking.

**Decided by:** Parth (PI), via `AskUserQuestion` at the start of P0-02 execution.

---

## 2026-09-01 — Branch protection unavailable; enforced by convention instead

**Context:** Task P0-02 step 8 requires branch protection on `main`
(required review, no force-push) or, if unavailable on the current GitHub
plan, a documented fallback.

**Decision:** `gh api -X PUT .../branches/main/protection` returned
`403 — "Upgrade to GitHub Pro or make this repository public to enable this
feature."` GitHub's free plan does not support branch protection rules on
private repositories. Protection is therefore **enforced by convention**:
no session pushes directly to `main` after the bootstrap commit
(`c394495`, LICENSE + README). Every subsequent change lands via a
`phase<N>/<slug>` branch and a pull request that the PI reviews and merges.
This is a discipline the agent protocol (`plan/00-agent-protocol.md`) already
mandates for other reasons (never self-merge), so the missing platform
enforcement is a reduced-safety-net situation, not a missing-process one.

**Rationale:** Upgrading to GitHub Pro or making the repo public solely to
unlock this feature is not worth doing before GATE-S, given the
private-until-submission decision above.

**Decided by:** Agent, following the explicit fallback instruction in
`plan/01-phase0-setup.md` task P0-02, step 8.

---

## 2026-09-02 — DataDecide: what P0-06 actually downloads, and real dataset facts

**Context:** Task P0-06 ("acquire and cache DataDecide") was planned around a
secondhand description of the Hub repos (from the source PDF / an earlier
tool-mediated preview). Building the real acquisition module surfaced
several things that description got wrong or omitted, checked directly
against the live Hub.

**Findings, all verified against the live Hub on 2026-09-02:**

1. **`allenai/DataDecide-data-recipes` is 19.2 TB** (6,194 files) of raw
   preprocessed tokenized-corpus `.npy` shards -- not "the 25 corpora
   definitions" as a small table. The actual human-readable recipe
   composition table lives in that repo's `README.md` as a markdown table,
   which is the only file `src/pdt/data/datadecide.py` ever fetches from it.
   `allenai/DataDecide-eval-instances` is similarly 123 GB and is never
   touched at all (the plan already correctly said Phase 1 doesn't need it).
2. **`allenai/DataDecide-eval-results` ships two extra tables** beyond the
   documented per-instance-family rows: `data/macro_avg-*.parquet`
   (task-macro-averaged rows, including a precomputed `olmes_10_macro_avg`
   "task") and `data/scaling_law_fit-*.parquet` -- **DataDecide's own
   baseline scaling-law fit results**, with a `decision_acc` column per
   (task, mix, metric, setup). The latter lets Phase 1 cross-check its own
   reproduction directly against the authors' own numbers, not just the
   ~80% figure quoted in the abstract. Both are now downloaded and cached by
   `load_macro_avg()` / `load_scaling_law_fit()`.
3. **The `metrics` column is inconsistently serialized across these files**:
   single-quoted Python dict repr (`ast.literal_eval` required) in the main
   `train-*.parquet` shards, but proper double-quoted JSON in `macro_avg`.
   `_parse_metrics()` tries JSON first and falls back to `ast.literal_eval`.
4. **25 recipes and 14 sizes are confirmed exactly** (independently, three
   ways: the recipes README table, its per-size model-link table, and the
   `data`/`params` columns of the real eval-results table). The 14 sizes are
   4M, 6M, 8M, 10M, 14M, 16M, 20M, 60M, 90M, 150M, 300M, 530M, 750M, 1B.
5. **Seeds: 5 distinct labels in the raw union (`default`, `small aux 2`,
   `small aux 3`, `large aux 2`, `large aux 3`), not the 3 the plan
   expected -- but every individual size still has exactly 3 seeds.** Sizes
   4M through 750M use `small aux 2/3`; **1B (the target scale) uses
   `large aux 2/3` instead of `small aux 2/3`**, with `default` present
   everywhere. This resolves, positively, a risk `plan/02-phase1-datadecide.md`
   task P1-01 flagged explicitly ("if `s* = 1B` has one seed, our ground
   truth ... has irreducible measurement noise") -- the target scale does
   have full 3-seed replication, just under different auxiliary-seed names.
6. **Only one `chinchilla` value (`5xC`) appears in the eval-results table**
   -- the multi-value design-axis concern P1-01 raised does not apply here.
7. **66 distinct `task` values**, not the ~10 OLMES task families the
   `macro_avg` table's task list suggested -- the full eval-results table is
   evaluated at much finer granularity. Recorded as data for P1-01 to select
   from; not resolved or filtered here.

Full counts are in `results/p0_06_inventory.json` (regenerated from a clean
tree, reproducible byte-for-byte aside from the timestamp).

**Decision:** `download_snapshot()` and `download_file()` in
`src/pdt/data/datadecide.py` require an explicit `allow_patterns` /
`filename` argument (no safe default) specifically because of finding 1 --
every call site is restricted to the small parquet/README files actually
needed, never a bare snapshot of an entire repo.

**Decided by:** Agent, while executing task P0-06, based on direct
inspection of the live Hub rather than the plan's secondhand description.

---

## 2026-09-02 — GATE-0 cleared: target venue is COLM 2027

**Context:** GATE-0 (`plan/01-phase0-setup.md`) required PI approval of the
repo/licence/namespace choices (already settled at P0-02), the P0-06
inventory results, the P0-07 novelty verdict, and a recommended submission
target given verified deadlines.

**Decision:** PI approved GATE-0 in full: merged PRs #6 (P0-07) and #7
(P0-08), and confirmed the target venue is **COLM 2027** (paper deadline
verified as ~March 31, 2027 in `docs/related_work.md` §5, moderate
confidence pending `colmweb.org` posting its own 2027 CFP page). ICML 2027
(~Jan 22, 2027) remains a stretch option only if Phase 1 and Phase 2 land
well ahead of schedule; COLM 2027 is the plan of record.

**Rationale:** COLM 2027 had the better-verified date of the two realistic
options and leaves roughly 7 months of schedule slack from 2026-09-02,
which matters given GATE-C (compute) and GATE-T (proof review) are both
hard to schedule tightly.

**How to apply:** `plan/07-phase6-paper.md`'s deadline-dependent tasks
(P6-09's "within 2 weeks of the deadline" novelty sweep, and the general
pacing in `PLAN.md` §3) should be read against a **~2027-03-31** target
from here forward, not the placeholder windows in `PLAN.md` §4. Re-verify
the exact COLM 2027 date directly from `colmweb.org` once its 2027 CFP page
exists — the current date is from aggregator sites, not the canonical
source.

**Decided by:** Parth (PI), in chat, after reviewing the GATE-0 summary.

---

## 2026-09-02 — P1-02: eval_results is the wrong granularity for the headline reproduction

**Context:** P1-01's frame (already merged, PR #9) was built entirely from
`eval_results`, which carries 66 "tasks" -- 9 core OLMES families plus each
of the 57 individual MMLU *subject* splits (`mmlu_abstract_algebra`,
`mmlu_marketing`, ...) as its own separate task. While building P1-02's
ground-truth computation, most individual MMLU-subject tasks showed exact
ties (`delta_min = 0.0`) between top recipes at 1B, which is what you'd
expect from small per-subject eval sets producing coarse accuracy
fractions.

**Finding:** checked directly against `scaling_law_fit` (DataDecide's own
baseline results, cached since P0-06) -- its `task` column contains
exactly 11 values: the 9 core OLMES families, one aggregated `mmlu`, and
`olmes_10_macro_avg`. **This confirms DataDecide's own headline numbers
are computed at this 11-task macro-averaged granularity, not the 66-task
fine-grained one.** `eval_results` is the wrong source to reproduce their
~80% figure against; `macro_avg` (also cached since P0-06, previously
unused past its own inventory count) is the right one.

**Decision:** `src/pdt/data/frame.py`'s `build_frame()` gained a `source`
parameter (`"eval_results"` or `"macro_avg"`, default unchanged at
`"eval_results"` for backward compatibility) and a `metrics` parameter
(previously a fixed whitelist -- `macro_avg` doesn't expose
`bits_per_byte_corr`, so a fixed whitelist would have broken it). P1-02's
ground truth is now computed from **both** sources: `macro_avg` as the
primary, DataDecide-comparable result, `eval_results` retained as a
fine-grained diagnostic explaining *why* ties happen. P1-03 must read
ground truth from the `macro_avg` branch of `results/p1_02_target.json`,
not `eval_results`.

**A second, more serious bug found while making this change:** the cache
key for `build_frame()` was keyed on `source` alone. P1-02's own script
calling `build_frame(source="eval_results", metrics=("primary_metric",
"acc_per_char"))` -- only 2 metrics -- silently wrote to the *same* cache
file P1-01's default 6-metric call also uses. Re-running P1-01's script
afterward silently returned the poisoned 2-metric cache: **8,078,400 rows
became 2,692,800, a 3x undercount, with no error** -- exactly the kind of
wrong-number-with-no-signal this project's provenance system exists to
prevent, except provenance only validates git-tree cleanliness, not
semantic correctness of a cached computation, so it did not catch this.
Fixed by keying the cache filename on a hash of the sorted `metrics` tuple
as well as `source`. Caught only by manually diffing a full regeneration
against the already-merged, already-PI-reviewed P1-01 output before
trusting the refactor -- confirmed the fix restores the original
8,078,400-row result exactly. A regression test now pins this specific
failure mode directly.

**How to apply:** any future caller of `build_frame()` with a non-default
`metrics` argument is safe now, but should still be aware that the cache
directory (`data/cache/pdt/`) can accumulate one file per distinct
(source, metrics) combination ever requested -- harmless (gitignored,
small), just worth knowing if the count looks surprising.

**A third bug, in `src/pdt/analysis/ground_truth.py` itself, found by the
same discipline** (diffing two clean-tree runs of the *same* code against
each other, not just checking each run's provenance individually): task
`compute_ground_truth()` sorted recipes by `mu` descending with **no
deterministic secondary key**. Given how pervasive exact ties in `mu`
turned out to be (P1-02's own headline finding -- see below), and that
polars gives no ordering guarantee across tied sort keys, two independent
clean runs of the identical computation on identical data picked a
*different* recipe as `k_star`/`runner_up` for every task with an exact
top-of-table tie -- roughly a dozen tasks differed between two runs before
the fix. Fixed by sorting on `["mu", "recipe"]` (mu descending, recipe name
ascending as a fixed, arbitrary-but-stable tiebreaker). A regression test
constructs an explicit 3-way exact tie and asserts the *same* winner is
chosen whether the input rows arrive in forward or reversed order.

**Why this is worth internalizing, not just fixing:** none of these three
bugs would have been caught by `pdt.provenance.validate()` alone -- it
checks that a result came from a clean git commit, not that the same
commit's code is *deterministic* or *semantically correct*. The "verify
byte-for-byte reproducibility by literally diffing two runs" step this
project's tasks have been doing isn't a formality; it is what actually
caught all three of these, and would not have caught any of them if
skipped in favor of trusting a single successful run.

**Decided by:** Agent, while executing task P1-02, verified by diffing
regenerated output against the already-merged P1-01 results before and
after the fix.

---

## 2026-09-02 — P1-04 scaling-law fitters: two scoping decisions

**Context:** `plan/02-phase1-datadecide.md` P1-04 specifies an
`Extrapolator` interface (`fit`/`predict`/`jacobian`) and six concrete
fitters, one of which (`TwoStepLadder`) explicitly follows DataDecide's own
baseline method (Bhagia et al., arXiv:2412.04403).

**Decision 1 — numerical jacobian, not six hand-derived analytic ones.**
`src/pdt/scaling/base.py`'s `Extrapolator.jacobian()` is implemented once,
generically, via central-difference numerical differentiation on the
fitted parameter vector -- every subclass gets it for free by implementing
`_predict_from_theta(theta, scale)` as a pure function. Hand-deriving six
analytic gradients (one per fitter, `TwoStepLadder`'s composed through two
chained functions) is six independent chances for a sign or chain-rule
error. `plan/02-phase1-datadecide.md` P1-07 already plans to cross-check
this exact machinery against bootstrap variance rather than assume perfect
analytic exactness, so numerical precision here is squarely within what
that later step is designed to catch if it's ever insufficient.

**Decision 2 — `TwoStepLadder` is a scoped adaptation, not a literal
reproduction of Bhagia et al.** Their method's step 1 target is actual
pretraining validation loss. This project's cached tables carry task
*accuracy* (`eval_results`/`macro_avg`, built in P0-06/P1-01), not
per-recipe loss -- that lives in a separate table, `allenai/DataDecide-ppl-results`
(also cached since P0-06, via `load_ppl_results()`, but never joined
against the accuracy tables by (recipe, scale, seed) anywhere in this
project yet). Implementing the literal method would require that join.
Instead, `TwoStepLadder` here fits step 1 (a power law in compute) directly
to the task metric as its own intermediate proxy, then step 2 reshapes that
proxy through a 4-parameter sigmoid -- the closest same-data-source
analogue to "compute -> loss -> metric" without pulling in a second table.

**How to apply:** if a future task (or a reviewer) needs the literal
Bhagia et al. method for a tighter DataDecide comparison, the join needed
is `ppl_results` x `eval_results`/`macro_avg` on (recipe `data`, `params`,
`seed`) -- `ppl_results` uses the same recipe/params/seed labels (confirmed
in P0-06), so the join keys already line up; the work is building the
loss-to-metric step 2 fit on real loss data instead of the metric-as-proxy
approximation used now. Not planned as a required task, but flagged here so
it isn't rediscovered from scratch if it becomes worth doing before the
paper is finalized.

**Decided by:** Agent, while executing task P1-04. All 6 fitters verified
against clean synthetic curves generated from each one's own functional
form before being trusted on real data (`tests/test_scaling.py`).

---

## 2026-09-02 — P1-04 experiment script: scope and matched-compute handling

**Context:** `experiments/p1_04_extrapolation_baselines.py` fits all 6
`pdt.scaling` extrapolators x 3 held-out designs (`S_fit` <=150M/<=300M/<=530M)
x 11 macro_avg tasks x 25 recipes (4,950 fits total) and compares against
the single-scale baseline from P1-03 at matched compute.

**Decision 1 — one variant only (primary_metric, seed-averaged), not all
four P1-03 sensitivity variants.** P1-03's four-variant sensitivity check
(metric x seed-handling) existed because *that* task's job was specifically
to stress-test the reproduction. P1-04's job is different: compare an
extrapolation frontier against *the* single-scale frontier at matched
compute, which requires both frontiers to use an identical metric/seed
definition or the comparison is meaningless. Scoped to P1-03's headline
definition (`primary_metric`, seed-averaged) -- the one DataDecide's own
~80% figure targets.

**Decision 2 — deterministic per-fit RNG seed via `sha256(fitter|design|task|recipe)`,
not a single shared `np.random.default_rng`.** A shared mutable generator
consumed sequentially across 4,950 fits would still be reproducible run to
run, but only by accident of a frozen iteration order -- adding, removing,
or reordering any fit anywhere would silently perturb every fit after it.
Hashing the four identifying strings (via `hashlib.sha256`, not Python's
built-in `hash()`, which is salted per-process by `PYTHONHASHSEED` and
would break reproducibility across separate runs) gives every fit an
independent, order-invariant seed. Verified: two independent full runs
produced byte-identical `results/p1_04_extrapolation.json` output
(excluding `provenance.utc_timestamp`).

**Decision 3 — matched-compute comparison is `null`/"out of range" rather
than extrapolated, when a design's total compute exceeds the largest
single-scale point available.** The plan asks to compare each design
against "the single-scale design of the same total compute" via
interpolation of P1-03's real per-size points. In practice the <=530M
design's total compute (sum of 6ND over 12 sizes, ~2.24e20) exceeds the
compute of the largest available single-scale *proxy* point (750M,
~1.39e20) -- extrapolating the comparison frontier itself past its own
observed range would be a second, unrequested extrapolation stacked on top
of the one actually being evaluated. `_log_interp_accuracy()` returns
`(None, out_of_range=True)` in this case rather than guessing. This is a
real result, not a bug: it means the <=530M design's accuracy (85.1%) has
no matched-compute single-scale comparison point at all within this
project's own data, only the unmatched observation that it exceeds every
directly observed single-scale point below it.

**Decision 4 — per-(fitter, design, task) results store aggregated fit
diagnostics (mean n_converged, mean objective_spread across the 25
recipes) plus every individual failure, not every individual fit's full
diagnostics.** Satisfies the plan's "log every failed fit, never drop
silently" requirement exactly (failures are rare and individually
informative -- Claim 3 treats a failure itself as evidence). Storing all
4,950 fits' full per-restart diagnostics would bloat the results file for
information that's only useful in aggregate once a design/fitter/task
group is healthy. In this run, 0 of 4,950 fits failed.

**Decided by:** Agent, while executing task P1-04. Verified via two
independent clean full runs (see Decision 2) and a hard consistency check
in the script itself: `ConstantExtrapolator`'s per-design accuracy must
exactly equal (not approximately) the matching P1-03 single-scale point,
since it is the same computation by construction; this passed on both runs.

---

## 2026-09-03 — P1-05 noise-floor: eval-instances source, composite-task noise, sigma2_target definition, and a real reproducibility bug

**Context:** `plan/02-phase1-datadecide.md` P1-05 asks for three variance
components (seed, checkpoint, eval-sampling) and a combined
`sigma^2_target(k,t)` "as the combination appropriate to how `mu_k(s*)` was
estimated in P1-02" -- deliberately left for the agent to define and
document, not spelled out.

**Decision 1 -- `allenai/DataDecide-eval-instances` is 123GB, but the one
number P1-05 needs (`n_instances` per task) lives in a single 269MB
root-level file, `summary-metrics.jsonl`.** Confirmed via
`HfApi().list_repo_files()` / `get_paths_info()` (no download) before
touching anything: the 123GB is per-instance model predictions
(`requests/*.jsonl.gz`, `models/*.tar.gz`, `sample-evals/**`), none of
which this task needs. `summary-metrics.jsonl` reports one row per (task,
model, size, seed, step) with a `num_instances` field confirmed constant
across every row sharing a task (`load_eval_instance_counts` in
`src/pdt/data/datadecide.py` raises if this ever stops holding). Fetched
via `download_file`, the same one-named-file pattern `load_recipes` already
established for the 19.2TB data-recipes repo -- never a snapshot of either
large repo. `datadecide.py`'s module docstring, which previously said this
repo is "similarly never touched," is updated to describe the distinction.

**Decision 2 -- `olmes_10_macro_avg` has no eval set of its own; its
eval-sampling noise is the variance of a mean, not `p(1-p)/n`.** Verified
empirically before assuming it (see the module docstring's usual "confirm,
don't trust a secondhand description" discipline): sampled 16
(recipe, size) combinations from the cached frame and confirmed
`olmes_10_macro_avg`'s `primary_metric` value equals the unweighted mean of
the 10 primitive OLMES tasks' own values exactly (0.0 max absolute
difference across all 16). `summary-metrics.jsonl` has no
`olmes_10_macro_avg` row at all (only the 10 primitives), confirming it's a
downstream aggregate, not a physical eval task. Its noise is therefore
`Var((1/10) sum_i X_i) = (1/100) sum_i p_i(1-p_i)/n_i`
(`noise.eval_sampling_noise_of_mean`), using each primitive task's own `p`
at the *same* (recipe, size) and its own `n_instances`. The combined
`mmlu` task (57 MMLU subjects) is treated as one primitive with its own
pooled `n_instances=14042` straight from `summary-metrics.jsonl`, not
re-derived from the 57 subtasks -- that file already reports it under the
same `"mmlu"` task label `macro_avg.parquet` uses, so no extra aggregation
step is needed or introduced.

**Decision 3 -- `sigma^2_target(k,t) = sigma^2_seed(k, s*, t) / n_seeds`,
*not* a combination with checkpoint jitter or eval noise.** P1-02's
`compute_ground_truth()` estimates `mu_k(s*)` as a plain average over the 3
seeds present at 1B -- no checkpoint or eval-noise correction is applied
anywhere in that code. The honest noise in *that specific estimator* is
therefore exactly its own standard error, the seed variance divided by the
seed count. `sigma2_ckpt` and `sigma2_eval` are still computed and reported
at every scale (satisfying the plan's three-component requirement, and
available as the "fallback when seeds are missing" the plan describes for
other uses), but are not folded into `sigma2_target` since P1-02's actual
estimator never used them. Flagged as a modeling choice worth a second look
if a reviewer disagrees, not treated as the only defensible answer.

**Decision 4 -- checkpoint jitter uses a fixed window of the last 4
checkpoints at every size, raw variance, no detrending.** 4, not 5 or more,
because the smallest run (6M params) has exactly 4 checkpoints total
(confirmed: `min == max == mean == 4` across all 25 recipes at 6M) -- any
larger window would silently become "every checkpoint" there while staying
a genuine tail window everywhere else, making the quantity mean something
different by size. No detrending because the plan's spec is literally
"variance across the last few checkpoints"; a visual check of one run's
last 8 checkpoints (150M, arc_challenge) showed noisy fluctuation without a
strong residual trend in that narrow window, supporting raw variance as a
reasonable-enough jitter proxy without adding an undocumented detrending
step the plan didn't ask for.

**Decision 5 (a real bug, not a modeling choice) -- `noise.py`'s
`group_by(...).agg(...)` calls now sort on a fully deterministic key and
pass `maintain_order=True` before every variance/mean reduction.** Found by
this project's standard 2-independent-runs diff: `checkpoint_jitter()`
originally produced different `sigma2_ckpt` values on 333 of 3850 cells
between two runs of identical code on identical cached data. Diagnosed by
comparing the two runs' rows directly -- same recipe/size/task keys, same
checkpoint counts, differences only in `sigma2_ckpt` itself, at
~1e-14 relative magnitude (e.g. `1.6259168229488267e-05` vs
`1.6259168229488264e-05`). This is IEEE-754 float addition's
non-associativity: `var()` summed the same 4 numbers in a different order
between runs (most likely because the cached parquet's parallel/chunked
read doesn't guarantee row order), not a logic error -- the row *sets*
were always identical. Setting `POLARS_MAX_THREADS=1` alone did not fix
it; adding an explicit `.sort([...])` immediately before the `group_by`,
plus `maintain_order=True` on the `group_by` itself, did -- verified
across 5 independent runs (2 initial + 3 more after the fix), all
byte-identical excluding `provenance.utc_timestamp`. Applied to both
`seed_variance()` and `checkpoint_jitter()` for consistency, though only
the latter was observed to actually fail (small 3-row groups apparently
don't trigger whichever parallel code path causes this). **This same risk
may be latent, unverified, in P1-01 through P1-04's own `group_by().agg()`
calls** (`ground_truth.py`, `decision_accuracy.py`, `frame.py`), which have
passed every reproducibility check run against them so far but were never
specifically probed for it -- flagged as a background task
(`task_2d6c6192`) rather than touched here, since fixing already-merged,
already-verified code is out of P1-05's scope unless the audit finds an
actual problem.

**Empirical finding (not a bug):** seed variance does **not**
monotonically shrink with scale in this data -- median `sigma2_seed`
across the 25 recipes ranges narrowly (~2.9e-5 to ~5.9e-5) from 4M through
1B with no clear downward trend (`monotonically_non_increasing: false` in
`results/p1_05_noise.json`). The plan asked this as an open empirical
question ("worth checking... as expected"); the answer here is "not as
expected" -- worth carrying into the P1-06/P1-08 write-up rather than
assuming noise simply shrinks with model size.

**Decided by:** Agent, while executing task P1-05. Verified via 5
independent full runs (see Decision 5) plus a completeness assertion in
`experiments/p1_05_noise_floor.py` itself (`eval_sampling_noise` row count
must equal `seed_variance` row count, or the script raises rather than
silently dropping a cell).
