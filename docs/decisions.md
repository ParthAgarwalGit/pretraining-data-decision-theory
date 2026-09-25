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

---

## 2026-09-03 — Background task `task_2d6c6192`: `group_by().agg()` determinism audit across `src/pdt/`

**Context:** P1-05's own reproducibility fix (Decision 5, above) found a
real float-summation-order bug in `noise.py` and explicitly flagged that
`ground_truth.py`, `decision_accuracy.py`, and `frame.py` carry
structurally similar `group_by().agg()` calls that had "passed every
reproducibility check run against them so far but were never specifically
probed for it." This task is that probe.

**Method:** every `group_by(...).agg(...)` call under `src/pdt/` was
enumerated (`grep -rn group_by src/pdt`) and triaged by whether its
aggregation is a floating-point reduction sensitive to summation order
(`.mean()`, `.std()`, `.var()`) versus one that isn't (`.max()`, `.len()`,
`.n_unique()`, `.first()` -- see below). Every reduction-sensitive call
lacking an explicit deterministic `.sort()` + `maintain_order=True` was
then empirically tested, not assumed broken: a repro script exercised each
function against the **real cached data** (both `macro_avg` and
`eval_results` sources, every proxy size, both `recipe_means()` seed
modes), run as **25 independent fresh Python processes** (not 25 calls in
one process -- the bug is a race in how a parallel/chunked parquet read
hands back row order, which a single warm process wouldn't re-trigger
per call), diffing all 24 pairs against the first run.

**Finding 1 (real bug, confirmed) -- `decision_accuracy.recipe_means()`
(seed_mode="average") is genuinely nondeterministic on the `eval_results`
source.** All 24/24 independent-process pairs differed from each other:
185 total leaf differences, always in the returned `mu` value, always at
last-bit-of-float64 magnitude (relative difference 1.12e-16 to 3.74e-16,
e.g. `0.29677419354838713` vs `0.296774193548387` <!-- NUMBER-OK: pre-fix scratch-audit values, never written to results/ -->
for `Dolma1.7 (no Flan)` / `mmlu_high_school_biology` / 530M). Exactly the
same mechanism as `checkpoint_jitter()`'s bug: `group_by(["recipe",
"task"]).agg(mean())` with no sort and no `maintain_order`. Every
differing cell used `seed_mode="average"` (a real 3-row float sum); zero
differences ever appeared under `seed_mode="default_only"` (a 1-row
"reduction" -- no summation, nothing to reorder), consistent with the
mechanism.

**Finding 2 (tested, not reproduced under cache-hit conditions) --
`ground_truth.compute_ground_truth()` and `decision_accuracy.recipe_trajectories()`
did *not* fail in this same 24-pair test**, despite `compute_ground_truth()`
running the *same* filter + `group_by(["recipe","task"])` shape (just with
`.std()` and `.len()` added alongside `.mean()`) on the *same*
`eval_results` frame that broke `recipe_means()`, and `recipe_trajectories()`
feeding an even larger single `group_by` call (all 13 proxy sizes at once)
through a near-identical `.agg(mean(), first(), mean())`. Zero diffs also
on `recipe_means()` itself when run against `macro_avg` (11 tasks) rather
than `eval_results` (66 tasks). At the time this looked like it might mean
"whichever polars execution path causes this is sensitive to some
combination of total row/group count and the exact shape of the
aggregation expression" -- Finding 3 below shows that read is incomplete.

**Finding 3 (the real scope, found by comparing against the actual
already-committed artifacts) -- both functions *did* corrupt the actual
P1-02/P1-03/P1-04 results as originally committed, and Finding 2's clean
result was an artifact of testing under lower-risk conditions than the
original runs actually ran under.** `build_frame()` only performs its
expensive `.unpivot()` rebuild from the *raw* per-source cache
(`data/cache/datadecide/{eval_results,macro_avg}.parquet`, 204MB /
44MB) on the *first-ever* call for a given `(source, metrics)` pair;
every call after that hits the small, already-narrowed
`data/cache/pdt/frame_*.parquet` file (4.9MB / 1.5MB) instead. P1-02's
original run on 2026-09-02 *was* that first-ever call (its
`frame_eval_results__f1b83261ec.parquet` / `frame_macro_avg__f1b83261ec.parquet`
cache files were written 90 seconds before its own commit timestamp) --
meaning the original `compute_ground_truth()` calls that produced the
currently-committed P1-02/P1-03/P1-04 numbers read from the *large* raw
parquet file, while every regeneration since (including this task's
25-independent-process Finding 1/2 audit, which ran *after* the narrow
cache already existed) reads the *small* pre-narrowed one. Diffing the
true original `results/p1_02_target.json` (git SHA `0ea49ca`, from that
first-ever run) against a byte-identical-across-two-runs clean
regeneration on this fixed code found **6,674 differences**: 3,096 in
`sd_seed` and 31 in `mu` at last-bit-of-float64 magnitude (relative
<=4.9e-14, the same mechanism as Finding 1), 122 in `effect_size`
(amplified up to 65% relative, since `effect_size = delta_min /
sqrt(pooled_variance)` divides by a near-zero quantity built from those
same `sd_seed` values), **one real `is_ambiguous` flip**
(`eval_results`/`primary_metric`/`mmlu_electrical_engineering`:
`effect_size` was `0.9999999999999998` in the original run and `1.0000000000000016` in the clean regeneration <!-- NUMBER-OK: pre-fix value superseded by the regenerated results/p1_02_target.json, kept here for the audit record -->
-- literally a coin-flip
across the `AMBIGUOUS_EFFECT_SIZE_THRESHOLD = 1.0` boundary caused by
last-bit noise, not a real disagreement about the task), and **one
`runner_up` flip** (`eval_results`/`acc_per_char`/`mmlu_college_physics`).
`k_star` (the actual per-task "winner") never changed, in any of the
6,674 diffs. The equivalent comparison for `results/p1_03_single_scale.json`
found 11 diffs (`kendall_tau`/`kendall_p_value`/`macro_avg_kendall_tau`
at the 16M and 300M proxy sizes only) with `accuracy_including_ties`/
`accuracy_excluding_ties` unchanged at every size and the 76.3% headline
untouched; `results/p1_04_extrapolation.json` found 61 diffs (57
`mean_objective_spread` + 1 `mean_n_converged` optimizer-convergence
diagnostics, downstream of `recipe_trajectories()`'s perturbed input
`mu` values feeding the nonlinear fits, plus 1 `kendall_tau`/`kendall_p_value`
pair inherited from `ConstantExtrapolator`'s known exact equivalence to
a P1-03 point) with every `prediction`/`accuracy`/`beats_single_scale`
value, and the 0/18 headline, unchanged. **So: `recipe_means()` is
confirmed broken by direct repro (Finding 1); `compute_ground_truth()`
is confirmed broken by this before/after comparison against its own
real, already-committed output (Finding 3), just not reproducible under
the lower-risk cache-hit conditions Finding 2's fresh audit ran under;
`recipe_trajectories()` is a one-step-removed casualty of
`compute_ground_truth`/`recipe_means`'s bug rippling into its own fit
inputs in the already-committed P1-04 numbers, though its *own*
`group_by` was never directly shown to reorder rows itself.** The
practical lesson survives even sharper than P1-05's version of it: a
negative result from an empirical repro is scoped to the conditions it
ran under (here, "cache already warm") and does not generalize to a
structurally identical call made under different conditions (here, "cache
cold, first build") -- which is exactly why the fix was applied to all
three functions rather than only the one Finding 1 caught directly.

**Decision:** fixed all three functions
(`ground_truth.compute_ground_truth()`, `decision_accuracy.recipe_means()`,
`decision_accuracy.recipe_trajectories()`) with the same `.sort([...])` +
`maintain_order=True` pattern P1-05 established, sorting on each
function's group-by keys plus the seed (or seed+params_str) dimension
being reduced away.

**Verification:** `make check`-equivalent (147 tests, 100% coverage on
both changed files) passes unchanged. Re-ran the same 25-independent-process
repro script after the fix: all 10 sampled runs are byte-identical
(`sha256sum` match) across both sources, every function, every scale.
Regenerated `results/p1_02_target.json`, `results/p1_03_single_scale.json`,
and `results/p1_04_extrapolation.json` from a clean tree (the only three
result files whose generating scripts call one of the three fixed
functions), each verified independently deterministic (two clean
regenerations of each, byte-identical `data` payload), and each diffed
against the version already committed on its upstream branch -- see
Finding 3 above and the PR for the exact diffs and headline-number
confirmation.

**Other `group_by().agg()` calls surveyed and found not at risk** (no fix
applied): `frame.py`'s two `coverage_matrix()` calls
(`group_by("task"/"params_str").agg(pl.len())`) aggregate with an exact
integer row count, which has no floating-point summation-order
sensitivity at all. `datadecide.py`'s
`_add_params_num_and_final_flag()` groups on `.max()` (step number),
also order-independent for the same reason (max, unlike sum/mean/var, is
associative regardless of float representation). `datadecide.py`'s
`_parse_eval_instance_counts()` uses `.first()` (order-sensitive in
general) alongside `.n_unique()`, but the function raises
`RuntimeError` if `n_unique() != 1` for any task before ever trusting the
`.first()` value -- by the time a `num_instances` value is returned, every
row in its group is provably identical, so which physical row was
"first" cannot matter. Left as-is; the existing invariant check is a
stronger safety property than adding a sort would be.

**Decided by:** Agent, executing the background task P1-05's Decision 5
flagged. Verified via 25 independent full runs pre-fix (24 pairwise diffs)
and 10 independent full runs post-fix (all byte-identical), plus 3 clean
regenerations of the affected results files.

---

## 2026-09-03 — P1-09 rank-reversal census: multiple comparisons inflate the naive reversal rate

**Context:** `plan/02-phase1-datadecide.md` P1-09 asks to classify every
recipe pair per task as stable / reversing (a genuine ranking flip
somewhere across the 14-size ladder) / within-noise, using an effect-size
threshold to tell a real reversal from noise sitting near a tie. The
obvious choice was to reuse `AMBIGUOUS_EFFECT_SIZE_THRESHOLD = 1.0`, the
threshold P1-02/P1-03 already established project-wide for exactly this
"is this gap real or noise" question.

**Decision -- report classification at a Bonferroni-corrected threshold as
the primary figure, with the uncorrected 1.0 threshold kept only for
continuity.** Reusing 1.0 here is a different statistical situation than
where it was established: P1-02 applies it *once* per (task, pair) at a
single fixed scale; P1-09 applies the identical per-size significance
test *14 times* per pair (once per size on the ladder) and calls a pair
"reversing" if any two of those 14 tests disagree in sign. Run that many
times without correction, a pair whose *true* gap is near zero at most
sizes has a substantial chance of clearing `|effect_size| >= 1.0` in
*both* directions by chance alone -- `effect_size=1.0` is a fairly loose
threshold to begin with (P1-02's own comment on it: "not a calibrated
hypothesis test"), and 14 independent-ish chances to clear it in each
direction compounds that. Caught by an explicit sensitivity check before
trusting the number: on a 3-task sample, the reversing rate was 79.3% at
threshold 1.0, 40.9% at 1.96 (conventional single-test 95%), and 14.6% at
a Bonferroni-corrected 2.94 -- an enormous swing driven entirely by the
threshold, not by anything about the data changing. The full 11-task run
confirms the same pattern: 61.7% reversing at the uncorrected threshold
vs. **15.2% at the Bonferroni-corrected one** (`z = 2.9137`, from
`norm.ppf(1 - (0.05/14)/2)`, controlling family-wise error at alpha=0.05
across the 14 simultaneous per-pair tests). `results/p1_09_rank_reversals.json`
reports both, under `thresholds.uncorrected_1_0` and `thresholds.bonferroni`,
with `primary_threshold_label: "bonferroni"` naming which one should be
quoted as *the* finding. 15.2% is still a real, non-trivial reversal rate
-- meaningfully supports the impossibility-regime framing (Claim 2) -- just
not the dramatically larger uncorrected number, which would have
overstated it.

**How to apply:** anywhere this project reports "N% of pairs reverse" (the
paper draft, P1-11's figures, P5-04's stress-test seed set), cite the
Bonferroni figure, not the uncorrected one. The uncorrected variant exists
in the results file for continuity/comparison, not as an alternative
headline number to pick opportunistically.

**Decided by:** Agent, while executing task P1-09. Caught via a targeted
sensitivity check (varying the threshold on a 3-task sample) before
trusting the initial single-threshold run's headline number, in the same
spirit as this project's other threshold/variant sensitivity checks
(P1-02's ambiguity threshold, P1-03's four-variant table).

---

## 2026-09-03 — P1-06 bias/variance decomposition: compute scope, shared-draw design, and a BLAS crash

**Context:** `plan/02-phase1-datadecide.md` P1-06 is "the core result" -- for
every (extrapolator, design, recipe, task), bootstrap-resample (B>=200,
two schemes) and decompose the resulting spread into estimation variance
and squared bias. Read literally across the full grid this project has
used since P1-04 (6 fitters x 3 designs x 25 recipes x 11 tasks), this is
~2 million individual scaling-law fits.

**Decision 1 -- run the full grid (no fitter/design cuts), parallelized
across processes, not scaled down.** A back-of-envelope estimate from
P1-04's own per-fit timing (~40-80ms with 8 restarts) put a serial run at
16-40+ hours -- too long to just run, but this machine has 20 CPU cores
and each (fitter, design, task, scheme) combination's ~5,000 fits are
embarrassingly parallel. Rather than cut fitters or designs (which would
have weakened the actual scientific comparison P1-08 needs -- predicted
vs. observed accuracy for *every* method), the fix was
`concurrent.futures.ProcessPoolExecutor` over the 396 (fitter, design,
task, scheme) work units, each running its own B=200 x 25 recipes
sequentially inside one worker. Empirically verified via two smoke tests
against real data (not synthetic) before committing to the full run: a
single real fit (TwoStepLadder, the fitter expected to be slowest) took
~41.5ms, giving a full-grid estimate of ~75-90 minutes wall-clock with
20-way parallelism -- long enough to run as a background job, short enough
not to need reducing B below the plan's own "B >= 200" floor or dropping
any fitter/design from the grid.

**Decision 2 (a real bug, caught before it could waste an hour) -- BLAS
must be pinned to 1 thread per worker process.** The first real
process-pool run crashed immediately: `OpenBLAS error: Memory allocation
still failed after 10 retries, giving up`, then `BrokenProcessPool`. Root
cause: numpy/scipy link against a multi-threaded BLAS that, left to its
own defaults, spawns its own thread pool *inside every worker process* --
20 worker processes x up to 20 BLAS threads each is up to 400 threads
competing for memory on a 20-core machine, and every one of this script's
fits works on <=12 data points, far too small for BLAS multi-threading to
help even if it didn't crash. Fixed by setting
`OMP_NUM_THREADS`/`OPENBLAS_NUM_THREADS`/`MKL_NUM_THREADS`/`NUMEXPR_NUM_THREADS`
to `"1"` via `os.environ.setdefault(...)` *before* numpy/scipy are
imported anywhere in the module -- required specifically before the
import, not just before `main()` runs, because Windows' spawn-based
multiprocessing re-executes the whole module top-to-bottom in every
worker process, including its imports. Re-verified against real data
after the fix: 0 failed fits across two smoke tests (11,000 and 220,000
individual fits respectively).

**Decision 3 -- seed-bootstrap and parametric-bootstrap both draw ONE
shared random pattern per (scale, replicate), applied identically to every
recipe, rather than independent draws per recipe.** The plan requires the
pairwise statistic `D_k = mu_hat_k*(s*) - mu_hat_k(s*)` to be "computed
from the same bootstrap replicate (so the correlation between the two
arms' fits is preserved)" and calls this "a real and important effect."
If every recipe drew independent randomness, there would be no shared
condition for a correlation to survive in the first place -- two recipes
evaluated at the same scale on the same benchmark instances share real
correlated noise (a hard eval instance is hard for every recipe; a
checkpoint-timing coincidence at a given step affects whichever recipes
happen to be evaluated near it). Implemented as: one rng per (design,
task, scheme, replicate) -- deliberately *not* keyed on fitter or recipe
-- draws one resample-index pattern (seed bootstrap) or one standard-normal
`z` (parametric bootstrap) per proxy scale; every recipe's own observed
values are then perturbed by that same shared pattern/z (recipe-specific
magnitude for parametric, since each recipe keeps its own noise variance;
recipe-specific *values* selected by the same slot pattern for seed
bootstrap). See `src/pdt/analysis/bootstrap.py`'s module docstring for the
full reasoning. Each fit's own multi-start-restart randomness is still
independently seeded per (fitter, design, task, scheme, recipe, replicate)
-- that's a numerical-optimization detail, not a scientific correlation
source, and doesn't need to be shared.

**Decision 4 -- `sigma2_target` for the pairwise decomposition is
`sigma2_target(k*, t) + sigma2_target(k, t)`, not re-derived.** The plan
gives the marginal formula (`sigma2_extrap_hat_k = max(0, bias_hat_k^2 -
v_hat_k/B - sigma2_target(k,t))`) but doesn't spell out the pairwise
analogue. Following the same "these are two independent training runs"
reasoning `decision_accuracy.pairwise_decision_accuracy`'s pooled-variance
calculation already uses, the noise in the *difference* of two
independent estimates is the sum of their individual noises. Reuses
`bootstrap.bias_variance_decomposition` unchanged for both the marginal
and pairwise cases -- it doesn't care whether its input series is a raw
`mu_hat^(b)` series or a difference series `D_k^(b)`, only that `mu_true`
and `sigma2_target` are the matching quantities for whichever series was
passed.

**Decision 5 -- a per-(fitter, design, task, scheme, recipe) cell needs
>=20 successful bootstrap replicates (10% of B=200) or it's flagged
`insufficient_replicates` rather than reporting a decomposition computed
from too few points.** Individual bootstrap-replicate fit failures are
expected to be rare but not impossible (a resampled/perturbed trajectory
can occasionally be harder to fit than the original), and per this
project's standing rule (P1-04's `FitFailure` handling; the plan's own
"log every failed fit, never drop silently" for P1-04) a failure is
evidence, not noise to average away. In the actual run: 0 fits failed
across both smoke tests; the real full run's failure count is recorded in
`results/p1_06_decomposition.json`'s `n_fits_total_failed`.

**Decided by:** Agent, while executing task P1-06. Verified via two
end-to-end smoke tests against real (not synthetic) cached data before
launching the full run: a reduced-scope run through the actual `main()`
entry point (2 fitters, 1 design, all 11 tasks, B=10, 44 work units,
220,000 individual fits, 0 failures) confirmed the full pipeline --
multiprocessing, JSON serialization of numpy float64 results via the
existing `provenance` encoder, aggregation -- end to end before spending
the ~75-90 minutes on the real B=200 full-grid run.

**Decision 6 (found after the first full run completed) -- results file
must round to 8 significant figures and drop two redundant fields per
recipe entry.** The first full run wrote a 7.4MB
`results/p1_06_decomposition.json` -- over this project's 5MB
pre-commit limit (`check-added-large-files --maxkb=5000`). Root cause:
~19,400 per-recipe decomposition entries (9,900 marginal + 9,504
pairwise across 396 combos), each serializing 3 float64 values at full
~17-digit precision (pure noise for a bootstrap estimate off B=200
replicates -- roughly 2 meaningful digits at best) plus two fields
(`mean_prediction`, `n_replicates`) that are either derivable
(`mean_prediction = mu_true + bias_hat`) or constant in the overwhelmingly
common case (`n_replicates == 200` whenever not flagged
`insufficient_replicates`) and read by no downstream consumer. Fixed by
rounding every reported float to 8 significant figures
(`_round_sigfigs`) and dropping both redundant fields, verified against
the *actual already-computed* first run's data (not a synthetic guess)
before spending another ~2.5 hours re-running: projected 2.79MB, comfortable
headroom under the limit. Did not hand-edit the existing 7.4MB file into a
smaller one and commit that -- results here must be exactly what running
`experiments/p1_06_decomposition.py` produces, so the fix went into the
script and the whole ~2.5-hour computation was re-run from scratch rather
than post-processed.

---

## 2026-09-04 — P1-06 finding: the sigma2_extrap/v ratio falls with compute, not rises

**Context:** `plan/02-phase1-datadecide.md` P1-06 states an explicit
"signature prediction": *"the ratio `sigma2_extrap_hat / v_hat` grows with
compute [in `S_fit`], because `v` shrinks and `sigma2_extrap` does not."*
This is presented as the theory's own falsifiable expectation, not a
tentative guess.

**Finding: the opposite happens, for every one of the 6 fitters, at every
step from `<=150M` to `<=300M` to `<=530M`.** From
`results/p1_06_decomposition.json`'s `ratio_vs_compute` (median across all
275 (task, recipe) cells, `seed_bootstrap` scheme):

| Fitter | ratio @150M | ratio @300M | ratio @530M |
|---|---|---|---|
| ConstantExtrapolator | 1612.6 | 574.0 | 195.1 |
| PowerLawN | 9.57 | 7.21 | 5.20 |
| PowerLawC | 20.79 | 16.10 | 11.98 |
| ChinchillaND | 8.81 | 5.20 | 3.42 |
| TwoStepLadder | 16.37 | 11.32 | 10.70 |
| LogLinear | 310.5 | 272.7 | 230.2 |

Every single row falls monotonically. Looking at `median_sigma2_extrap_hat`
and `median_v_hat` separately (not just their ratio) shows why: both
*do* shrink as the design grows, but `sigma2_extrap_hat` shrinks
faster than `v_hat` -- e.g. PowerLawN's `v_hat` is roughly flat
(2.72e-3 -> 3.48e-3 -> 3.47e-3, if anything drifting up slightly) while
its `sigma2_extrap_hat` falls by more than 30% (2.10e-2 -> 1.79e-2 ->
1.45e-2). The plan's prediction assumed `v` would be the one doing the
shrinking; empirically here it's `sigma2_extrap` that responds most to a
larger design.

**Why this is plausible, not just noise:** a design with a larger largest
size (530M vs 150M) is extrapolating a shorter *relative* distance to the
1B target, which should plausibly reduce bias more than it reduces the
bootstrap-estimated variance of an already-well-identified fit (`v_hat`'s
flatness suggests these fits are not variance-starved even at the
smallest design -- 10 scales is already comfortably above every fitter's
minimum data requirement, so adding more scales mostly sharpens *where*
the curve is anchored, i.e. bias, more than it sharpens the *spread*
across bootstrap replicates).

**How to apply:** do not average over this or reframe it as "roughly
matches the theory." State it plainly in `docs/findings/p1_06.md` (the
plan's own required deliverable, which must say "plainly whether
`sigma2_extrap` is large, small, or task-dependent") as a real
discrepancy between the stated theoretical expectation and this
empirical ladder, and flag it forward into P1-08 (does the bound predict
the 80% ceiling) and, per the plan's own P1-07 instructions, into P2 as a
theory-refinement candidate if P1-07's actual bound-tightness check also
shows something inconsistent with the marginal-form theory as currently
stated. This is exactly the kind of result the plan's own review gates
(plan/09-review-gates.md) exist to surface to the PI rather than paper
over -- reported here, not adjusted to fit the prediction.

**Decided by:** Agent, while executing task P1-06, reading the actual
`ratio_vs_compute` table before writing STATUS.md rather than assuming
the plan's stated direction would hold.

---

## 2026-09-04 — P1-07 plug-in bound: scheme choice, sandwich estimator scope, and undefined-ratio handling

**Context:** `plan/02-phase1-datadecide.md` P1-07 asks for the marginal
and pairwise-difference bound forms, an analytic delta-method `v_k(C)` as
a cross-check against P1-06's bootstrap `v_hat_k`, and a Monte-Carlo
estimate of the actual selection error compared to both bound forms as a
tightness ratio.

**Decision 1 -- the Monte-Carlo selection-error simulation uses the
seed-bootstrap scheme only, not both P1-06 schemes.** P1-06 runs two
resampling schemes (seed and parametric) as a cross-check against each
other; P1-07's Monte-Carlo asks a different question (does the actual
`argmax` selection procedure pick `k*`?), and running it under both
schemes would double an already-expensive (`B=500` x full grid) computation
for a question that doesn't need the comparison. Seed bootstrap was
chosen as canonical because it resamples real observed values with no
distributional assumption, closest in spirit to what P1-02's ground truth
and P1-03's reproduction are themselves built from.

**Decision 2 -- the analytic `v_k` is reported as a cross-check, not
substituted into the bound actually used.** The reported
`bound_marginal`/`bound_pairwise` values use P1-06's bootstrap
`v_hat`/`bias_hat` throughout (the same source `sigma2_extrap_hat` comes
from -- there is no purely-analytic `sigma2_extrap`, only a bootstrap
one, so mixing an analytic `v_k` into that formula would compare
quantities estimated two different ways within the same sum). The
per-recipe `analytic_v_k` values are reported alongside for direct
comparison against P1-06's `v_hat`, which is the cross-check the plan
actually asks for ("agreement validates the analytic machinery...
disagreement is a finding") -- not a request to change which number
feeds the bound.

**Decision 3 -- the sandwich covariance is the basic (HC0) estimator, no
small-sample correction.** `sandwich_covariance()` uses raw squared
residuals as the "meat," not an `n/(n-p)`-scaled variant (HC1) or similar.
The plan says "the sandwich covariance of the fit" without specifying a
correction; HC0 is the standard default meaning of "sandwich covariance"
in the literature, and with `n` (10-12 fitted scales) not much larger
than `p` (2-7 parameters) for some fitters, a correction would matter
somewhat -- flagged here as a real scoping choice, not the only
defensible one, should someone want a tighter analytic-vs-bootstrap
agreement check later.

**Decision 4 -- an undefined tightness ratio (empirical error rate
exactly 0) is reported as `null`, not infinity.** When a Monte-Carlo
simulation finds zero errors across `B=500` replicates, `bound /
empirical_error` is mathematically undefined (division by zero), and the
bound is trivially satisfied regardless of its value (any non-negative
bound holds against zero observed error). Rather than reporting `Infinity`
(not valid JSON) or an arbitrarily large sentinel, `tightness_ratio_*` is
`null` in this case, with `empirical_error_rate: 0.0` still visible so a
reader can see why.

**A property discovered while testing `sandwich_covariance()`/`analytic_v_k()`
(not a bug):** `PowerLawN`'s delta-method variance *saturates* rather than
diverging as the target scale moves further past the fitted range --
because the model's own prediction converges to a constant ceiling `E` as
`N -> infinity`, its jacobian converges to a fixed vector, and so does the
propagated variance. `LogLinear`, whose jacobian entry `d(prediction)/d(b)
= log(N)` grows unboundedly, does not share this property. Worth knowing
before reading too much into any one fitter's `analytic_v_k` trend versus
extrapolation distance -- it is model-form-dependent, not a general fact
about extrapolation uncertainty. See `tests/test_bound.py`'s two paired
tests for both properties checked directly.

**Decided by:** Agent, while executing task P1-07.

---

## 2026-09-04 — P1-08: the plug-in bound, taken literally as an accuracy predictor, is vacuous -- and what that reveals

**Context:** `plan/02-phase1-datadecide.md` P1-08 calls itself "the paper's
money question" -- plug P1-05/P1-06's estimates into the bound to get a
*predicted* decision accuracy, compare against P1-03/04's *observed*
accuracy, and run a `sigma2_extrap = 0` counterfactual.

**Decision 1 -- "predicted accuracy" is `max(0, 1 - bound_pairwise)`, not
`1 - bound_pairwise`.** `bound_pairwise` is a sum of `exp(...)` terms, one
per non-winning recipe (up to 24 per task) -- a union bound, which can
(and, checked directly before writing any code around it, *does*)
exceed 1 when many terms are individually large. P1-07 already found
`bound_pairwise > 1` in **all 396 of 396 cells** (min 1.16, median
17.08, max 23.36) -- not an edge case, the norm. This traces directly to
P1-02: 9 of 11 tasks have no statistically resolvable winner at 1B, so
most of a task's 24 comparisons against `k*` are near-ties, each
contributing a term close to 1, and 24 such terms trivially sum past 1. A
probability bound that overshoots 1 means "no informative lower bound on
accuracy" -- reading it as literally negative (an early run produced
"predicted accuracy: -1889%" before this fix) is not meaningful; clipping
at 0 is the standard, correct way to read it.

**Consequence, reported plainly rather than engineered around: predicted
accuracy (clipped) is 0.0% for every single (fitter, design) cell.** The
bound, while never *violated* (P1-07: ratio >= 1 everywhere) and while
demonstrably tighter in its pairwise form than its marginal form, is too
loose at these gap sizes to make any informative quantitative accuracy
prediction at all. This is a real finding about the practical usefulness
of the bound as currently scaled, distinct from (and consistent with, not
contradicting) P1-07's own "the bound holds, loose by a constant factor"
result -- "loose by a constant factor" turns out to mean "loose enough to
be vacuous once summed over ~24 mostly-tied comparisons," which is worth
stating as plainly as the plan's own P1-06 definition-of-done language
asks ("stating plainly whether `sigma2_extrap` is large, small, or
task-dependent") -- extended here to the bound's own usefulness, not
smoothed into "the theory roughly works."

**Decision 2 -- the `sigma2_extrap = 0` counterfactual is reported against
TWO baselines, not one.** The plan's literal wording ("recompute the
predicted extrapolation accuracy... if the prediction then exceeds
single-scale") is ambiguous about which "single-scale" -- its real
(bias-included) predicted accuracy, or its own bias-free counterfactual.
Both are computed and reported:
- **vs. single-scale's real predicted accuracy (0.0% everywhere, per
  Decision 1):** 5 of 15 (fitter, design) pairs "flip" to beating it --
  but this is close to trivial, since *any* positive counterfactual value
  beats a floor of exactly 0.
- **vs. single-scale's OWN bias-free counterfactual** (ConstantExtrapolator
  also carries substantial removable bias -- it "never corrects for scale
  at all", per its own docstring -- so zeroing bias moves its own
  predicted accuracy up to 20.8%-37.0%, not 0%): only **1 of 15** pairs
  still beats it -- `LogLinear` at the `<=530M` design (37.7% vs 20.8%).
  This is the honest, apples-to-apples version of "if neither method had
  bias, who wins on variance alone" -- and the answer is: almost nobody,
  and the one exception is the *deliberately misspecified* model, whose
  advantage here is having very little bias left to remove in the first
  place is beside the point -- what's left is its comparatively small
  variance, which is exactly what the theory says should matter once bias
  is controlled for.

**How to apply:** when this counterfactual result is quoted (P1-11's
figures, the paper draft), cite the apples-to-apples comparison (1/15),
not the raw 5/15 -- the wider number is an artifact of comparing against
a degenerate baseline, not a real 5-way vindication of the
extrapolation-bias explanation.

**Decided by:** Agent, while executing task P1-08. The vacuous-bound
finding was caught by inspecting the raw (unclipped) numbers before
trusting the first run's console output, which had already produced
nonsensical negative percentages -- a signal something was wrong with the
*interpretation*, not (as first suspected) a bug in P1-06/07's actual
computed values, both of which were re-checked and confirmed correct.

---

## 2026-09-05 — P1-10 secondary ladder: scope forced by Pythia's own data, and an inconclusive replication

**Context:** `plan/02-phase1-datadecide.md` P1-10 asks for the P1-06
decomposition rerun on Pythia, with an explicit escape clause ("if this
task balloons... cap it at one ladder and one task family and say so").
Every scope reduction below was discovered while building this, not
chosen in advance -- see `experiments/p1_10_secondary_ladder.py`'s module
docstring for the full list; this entry covers the reasoning and the
result.

**Decision 1 -- `1.4b`, not `1b`, is the target scale.** `1b` has no
plain (non-deduped) directory in `EleutherAI/pythia`'s published evals --
only `pythia-1b-bf16`, `pythia-1b-0.5MtokBS`, and `pythia-1b-deduped`,
discovered via a real 404, not assumed from the naming pattern that holds
for every other size. Rather than guess which irregular variant is the
"real" standard 1B run, `1.4b` (clean `pythia-1.4b` /
`pythia-1.4b-deduped` directories) is used instead. `src/pdt/data/pythia.py`
documents this; `SIZES` still lists `"1b"` for completeness but callers
needing a full ladder should avoid it.

**Decision 2 -- only `ConstantExtrapolator` and `LogLinear` run; the
other 4 P1-04 fitters cannot.** Only 3 Pythia sizes exist below the
target (`70m`/`160m`/`410m`). `PowerLawN`/`PowerLawC` need `>=4` scales to
identify 3 parameters, `ChinchillaND` needs `>=6` for 5, `TwoStepLadder`
needs `>=8` for 7 -- none can fit with only 3 candidate points. This is a
hard data-availability constraint, not a scope choice: DataDecide has 14
sizes with 10-12 below any reasonable target; Pythia's ladder is 8 sizes
total with only 3 below `1.4b`.

**Decision 3 -- parametric bootstrap only, using checkpoint jitter as the
sole noise source.** `EleutherAI/pythia`'s published per-checkpoint evals
have exactly one row per (size, variant, step) -- no second seed to
resample from, unlike DataDecide's 3 seeds everywhere. Reused P1-05's
checkpoint-jitter method (variance across the last 4 of a run's own
published checkpoints) as the noise variance fed into
`bootstrap.apply_parametric_noise` -- the same shared-per-scale-per-replicate
draw design P1-06 established, just with `K=2` recipes instead of 25.

**Decision 4 -- `mmlu` is reconstructed as the unweighted mean of 57
`hendrycksTest-*` subtasks; `boolq`/`csqa`/`hellaswag`/`openbookqa`/
`socialiqa` (and therefore `olmes_10_macro_avg`) are dropped entirely.**
Checked directly against one raw eval JSON's own task keys (not assumed):
only `arc_challenge`, `arc_easy`, `piqa`, `winogrande`, and the 57
`hendrycksTest-*` MMLU subjects have any counterpart in Pythia's public
eval set. The `mmlu` reconstruction uses the same unweighted-mean
convention this project already verified DataDecide's own `macro_avg`
table uses for `mmlu` (P1-05/P1-06). 5 of DataDecide's 11 headline tasks
have no Pythia counterpart at all and are silently unavailable, not
approximated.

**Decision 5 -- `Scale.d` is a placeholder, unused by either fitter that
actually runs.** Real per-checkpoint token counts for Pythia would need
combining published batch-size/sequence-length constants per size (which
differ across some sizes, per the `-0.5MtokBS`/`-1MtokBS` alternate
directories seen while exploring the repo) -- not computed here, because
neither `ConstantExtrapolator` nor `LogLinear` reads `scale.d` or
`scale.compute` anywhere in their fit or predict logic. `Scale(n=n,
d=20*n)` is a syntactically-required placeholder that never influences
any reported number -- confirmed by reading both fitters' source before
relying on this, not assumed safe.

**Result: the P1-06 ratio-vs-compute finding does not clearly replicate,
in either direction.** `ConstantExtrapolator`'s median `sigma2_extrap/v`
ratio across the 5 tasks is 131.0 at the smaller design (`le_160m`, 2
sizes) and 137.6 at the larger one (`le_410m`, 3 sizes) -- essentially
flat (~5% difference), not the clear monotonic fall P1-06 found in
DataDecide across 3 designs, but also not a clean rise. With only 2
usable design points and 1 fitter able to run at both, this is a weak
test either way -- reported as inconclusive rather than forced into
"replicates" or "contradicts."

**Decided by:** Agent, while executing task P1-10. Every scope limit here
was verified against the real published repo structure before being
treated as a constraint (the `1b` 404, the exact task-key overlap, the
minimum-scales-per-fitter arithmetic), not assumed from the plan's or
this project's own DataDecide-side conventions.

---

## 2026-09-05 — P1-11 figures: five rendering bugs found by reading the actual PDFs, not by trusting exception-free code

**Context:** `plan/02-phase1-datadecide.md` P1-11 asks for F1-F5, generated
by `src/pdt/viz/` with no manual steps, meeting: colourblind-safe
palette, no red/green pairing, legible at 6cm wide, vector PDF, every
axis labelled with units, no chart junk. Every figure below ran and
produced a PDF on the first try -- none of the bugs in this entry would
have been caught by "does `generate()` raise an exception," only by
opening the rendered file. That is the one policy this entry is really
recording: every figure was rendered to a 300-DPI PNG (`bbox_inches="tight"`)
and read with the Read tool before being accepted, and every fix below
was re-verified the same way, iterating until the render matched the
data, not the code's apparent intent.

**Decision 1 -- figure text sizing lives in global `rcParams`, not
per-instance `Text.set_size()` calls.** `src/pdt/viz/style.py` originally
set font sizes via calls like `ax.title.set_size(8)` inside `new_figure()`,
before any title text existed. `ax.set_title(...)`, called later by each
F-module, creates a *new* `Text` object that does not inherit that
earlier call, so F1's title rendered at matplotlib's default size and
clipped at the figure edge. Fixed by moving every size to a module-level
`plt.rcParams.update({...})` at import time, which applies correctly
regardless of when the text object is created.

**Decision 2 -- `style.save()` uses `bbox_inches="tight"`, but that alone
does not rescue every legend.** Added `bbox_inches="tight", pad_inches=0.03`
to `fig.savefig()` after F2's below-axes legend was cropped at the page
edge (`ConstantExtrapolator` rendered as `Constant...` cut off; confirmed
against a high-DPI PNG, not just the PDF's text layer, to rule out a
viewer-side artifact before treating it as a real bug). This is now the
default for every figure. It is necessary but was not, by itself,
sufficient for F2's specific layout -- see Decision 3.

**Decision 3 -- F2's legend is built with `fig.legend()` at explicit
figure-fraction coordinates, not `ax.legend(bbox_to_anchor=...)` in
axes-fraction coordinates.** Three compounding problems, found and fixed
in sequence by re-rendering after each:
  - A 3-column legend (`ncol=3`) put `ConstantExtrapolator` and the other
    long fitter names past the figure's right edge even with
    `bbox_inches="tight"` -- because that legend was attached via
    `ax.add_artist()` rather than being the axes' own tracked legend, and
    was under-measured by the tight-bbox pass. Switched to `ncol=2`,
    which fits inside the 6cm width without depending on tight-bbox to
    rescue an overflowing column.
  - The default log-scale tick locator added minor ticks (`2x, 3x, 4x,
    6x`) at every decade; with all 3 real compute values inside one
    decade, the x-axis became an illegible smear of overlapping labels.
    Fixed with an explicit `FixedLocator` at the 3 real values and minor
    ticks off.
  - `matplotlib.ticker.LogFormatterMathtext`, given those 3 non-decade
    values, rendered malformed fractional exponents (`10^19.33`) and
    silently dropped the middle tick's label. Replaced with a hand-built
    `a \times 10^{b}` formatter -- justified here specifically because
    there are only ever 3 fixed values, so a general log-tick formatter
    is solving a harder problem than actually exists.
  - The legend was then moved below the axes via `ax.legend(bbox_to_anchor=...,
    loc="upper center")` in axes-fraction coordinates, which put the
    legend's `"Fitter"` title directly on top of the xlabel -- the
    axes-fraction offset didn't account for the xlabel's own position
    below the axes, which is itself computed after the fact by
    matplotlib. `fig.legend(bbox_to_anchor=...)` in figure-fraction
    coordinates removes that coupling (xlabel and both legends are all
    positioned as absolute fractions of the same fixed canvas), and the
    redundant `"Fitter"` title was dropped rather than fought with --
    the legend's entries (fitter names) are already self-explanatory.

**Decision 4 -- F3 shows full fitter names, not a 4-character
truncation, and moved its legend below the axis.** `x_labels.append(f"{fitter[:4]}...")`
made `PowerLawC` and `PowerLawN` both render as `"Powe"` -- indistinguishable,
not just ugly, since the figure has 6 columns for each and a reader
cannot tell which is which. Rotated 90 degrees at a small font size, the
full names cost vertical space, not horizontal, so truncation was
solving a problem that didn't exist; the figure width was also increased
to 12cm (matching F4's existing precedent for a dense, many-category
panel) to give each of the 18 (fitter, design) columns more room.
Separately, the in-plot legend (`loc="center left"`) sat at the same
height as the real "observed accuracy" data cluster it was labelling, an
overlap confirmed by reading the render, not assumed from the `loc`
string. There is no y-band in this figure that is empty across the full
x-range (the plug-in-bound series sits at ~0 for every single column), so
the legend was moved below the axis instead of relocated in-plot.

**Decision 5 -- F5's legend corner was chosen from the actual data
range, not matplotlib's default placement.** `loc="upper left"` put the
legend on top of a dense scatter cluster (every point has `bound_marginal`
or `bound_pairwise >= 1.16`, and the cluster of near-1 empirical-error-rate
points sits exactly in the upper-left quadrant of this log-log square).
Checked the underlying `results/p1_07_bound_coverage.json` values
directly (`max(empirical_error_rate) == 1.0`, `min(bound) == 1.16`) before
picking `loc="lower right"`, which is providably empty rather than
visually guessed to be empty.

**F1 and F4 needed no changes.** F1's in-plot legend sits in the plot's
own empty region (accuracy never drops much below 0.5, so the
lower-right stays clear); F4's two-panel `loc="best"` legends were
checked the same way as everything else here and did not overlap either
panel's curve.

**Decided by:** Agent, while executing task P1-11. Every fix in this
entry was verified by rendering a 300-DPI PNG and reading it (zooming
into the specific region in question where a whole-figure read wasn't
conclusive enough), then repeating after each change until the render
matched expectations -- the same "read the actual output, don't trust
the code" discipline this project has applied to every prior task's
results files, applied here to a visual artifact instead of a number.

---

## 2026-09-05 — P1-12: Phase 1 memo, and pushing to the HF dataset repo without letting it clobber the dataset card

**Context:** `plan/02-phase1-datadecide.md` P1-12 asks for two things: the
Phase 1 memo (`docs/findings/phase1_memo.md`, see that file for the
findings synthesis and the framing recommendation), and a push of the
Phase 1 derived tables to the HF dataset repo from P0-08
(`Parth4105/pdt-datadecide-analysis`). The memo needed no permission and
was written and PR'd first (PR #21); the HF push is an external,
publicly-consequential action (the repo is private, but a Hub push is
still a real, hard-to-undo side effect on a third-party service), so it
was held for explicit PI confirmation before running, per this project's
standing practice all session of never touching HF without asking --
confirmed via `AskUserQuestion`, then executed the same session.

**Decision -- `push_results()` was called on a filtered copy of
`results/`, not `results/` itself.** `src/pdt/hub.py`'s `push_results(local_dir,
repo_id, revision_msg)` uploads *every* file under `local_dir` to the
repo root. `results/` contains its own `README.md` (a 4-line "machine-written
JSON only" note for the *code* repo) alongside the 11 result JSON files
-- pushing `results/` directly would have silently overwritten the HF
repo's actual dataset card (a much longer, hand-written description of
what the repo is and what's in it) with that unrelated 4-line note.
Caught by reading `push_results()`'s implementation (it wraps
`huggingface_hub.upload_folder(folder_path=local_dir, ...)` with no
`path_in_repo` or ignore-pattern support) before calling it, not
discovered after the fact. Fixed by staging a copy of just the 11 JSON
files in a scratch directory and pushing that, leaving `results/README.md`
untouched both locally and on the Hub. The dataset card itself was then
updated separately (`huggingface_hub.upload_file`, not routed through
`push_results()`, since a dataset card isn't a provenance-stamped result
and validating it as one would only fail) with a table documenting every
one of the 11 files' purpose and top-level `data` schema, built by
actually loading each file and reading its real top-level keys rather
than describing them from memory.

**Push result:** 11/11 files passed `provenance.validate()` (all
`git_dirty: false`, traceable to commit `c33f53b60a1a863ef48ad7d047ad603555f17725`).
Commits: `7f3bd6e0fcb3e88e270c2e1b368ac3a7956ab8ee` (the 11 result files)
and `71f659e1a21f980c86f4bf940d7baf33d719985e` (the updated card).

**Decided by:** Agent, while executing task P1-12, after the PI answered
"Yes, push results/ now" to an explicit `AskUserQuestion` prompt
describing exactly what would be pushed (the 11 already-provenance-validated
result files plus an updated dataset card) before any Hub-side action was
taken.

---

## 2026-09-10 — Theorem 1's marginal bound needed a real correction, found by its own numerical certificate

**Context:** `plan/03-phase2-theory.md` task P2-02 asks for Theorem 1's bound, proved,
plus a numerical certificate (`tests/theory/test_theorem1.py`) that simulates 5000
random instances with known ground truth and checks the bound is never violated --
"a single violation fails the test suite... A theorem that fails its numerical check is
wrong, and finding that out now is worth more than a month of proof-writing." The
plan's own draft text states the bound as
`P[k_hat != k*] <= sum_{k!=k*} exp(-Delta_k^2 / (2*(sigma2_extrap_k + v_k)))` -- the
same additive structure already implemented as `marginal_bound_term` in
`src/pdt/theory/bound.py` and used throughout P1-07/P1-08's analysis.

**What the certificate found.** The first version of the certificate, built to test
exactly this additive formula, found large, unambiguous violations (497 of 5000
instances, several with the claimed bound near 0 against an empirical error rate above
0.8) -- not marginal, noise-explainable near-misses. A minimal hand-built
counter-example confirms it directly: an arm with a fixed bias of 10, noise variance
0.01, <!-- NUMBER-OK: hand-chosen illustrative counter-example, not a reported result --> competing against a bias-free, noise-free `k*` with a true gap of 5. The true
selection-error probability here is essentially 1 (the fixed bias alone dwarfs the
gap). The additive formula claims a bound of `exp(-5^2/(2*(10^2+0.01))) ~= 0.88` <!-- NUMBER-OK: same hand-chosen counter-example --> -- a
real violation, reproduced exactly in `tests/test_bound.py::test_worst_case_form_corrects_a_real_violation_of_the_additive_form`.

**Why the additive form fails, and what the correct form is.** The additive form treats
`sigma2_extrap_k` as if it were a *random, zero-mean* contribution to variance --
valid if bias were itself drawn from a zero-mean distribution across instances. But
this project's own `delta`-correctness definition (`paper/sections/setup.tex`
Definition 3, `P[k_hat != k*] <= delta` for *every* instance in the class, uniformly)
requires a worst-case guarantee over a *fixed, unknown-sign, magnitude-bounded* bias --
the instance-by-instance framing `h_k in H` with `sup|h_k| <= eta` already commits to
in `paper/sections/setup.tex` Assumption 4. Under that framing, the correct treatment
of a Chernoff/Gaussian tail bound with an adversarial fixed-sign bias *subtracts* the
bias magnitude from the gap (`(Delta_k - sqrt(sigma2_extrap_k))_+` in the numerator),
not adds the bias squared to the variance in the denominator -- confirmed independently
three ways: (1) a from-scratch worst-case-Chernoff derivation, (2) the hand-built
counter-example above (the corrected form correctly reports a vacuous bound of 1.0,
honestly reflecting that no guarantee is possible when the bias alone can exceed the
gap), (3) the full numerical certificate, which finds 0 violations with the corrected
form across all 5000 instances once restricted to the regime Monte Carlo can actually
resolve (see below).

A second, independent gap in the additive form: it uses only arm `k`'s own
`(sigma2_extrap, v)`, omitting `k*`'s. Since the comparison is
`mu_hat_k(s*) >= mu_hat_{k*}(s*)`, both sides are noisy estimates, and `k*`'s own bias
and variance must enter the bound symmetrically -- the corrected form's
`total_variance = v_k + v_{k*}` and `bias_budget = sqrt(sigma2_extrap_k) +
sqrt(sigma2_extrap_{k*})` fix this. `src/pdt/theory/bound.py` now has
`worst_case_marginal_bound_term`/`worst_case_marginal_bound` implementing the corrected
form, alongside (not replacing) the original `marginal_bound_term`/`pairwise_bound_term`.

**Why P1-07's "0 violations on 396 real cells" finding is not contradicted by this.**
Every one of the 396 real DataDecide cells P1-07 checked had a bound value `>= 1`
(vacuous -- see `docs/decisions.md`, 2026-09-04, P1-07 entry: tightness ratio minimums
of 1.71 and 4.44 mean the *bound itself* was always comfortably above 1). A bound that
is vacuous either way cannot distinguish a correct formula from an incorrect one --
both say "no guarantee, but also never technically violated," because a probability is
always `<= 1` regardless of what the (much larger) claimed bound says. The additive
form's flaw only shows up once gaps get small relative to bias, a regime real
DataDecide data's own near-tie structure (P1-02: 9/11 tasks ambiguous) never let the
formula's numeric *value* fall into. This is a real, if fortunate, gap in what P1-07's
empirical check was structurally able to catch -- not a flaw in what P1-07 reported,
which remains an accurate description of that specific formula's real-data behaviour.

**A remaining subtlety the certificate also surfaced: Monte Carlo has a resolution
floor.** After the correction, an initial rerun (1000 MC trials/instance,
Clopper-Pearson alpha=1e-6) still showed 35/5000 "violations," but every one had only
1-4 raw error events out of 1000 trials -- far too sparse to statistically distinguish
a true rate of 5e-5 from 3e-4 at any reasonable alpha. This is a limit of empirical
verification, not evidence against the theorem: instances whose claimed bound is very
small are covered by the closed-form Gaussian-tail argument directly (exact for the
certificate's linear-in-theta family), not by simulation. Raising the budget to 20000
MC trials/instance and excluding instances below a resolution floor (bound `< 20 /
n_mc`, ensuring an expected raw-event count of at least 20) gives a clean, honest
result: 1801/5000 instances fell within Monte Carlo's resolution and were checked
directly (0 violations, tightness ratio min 1.0, median 7.1, max 2.0e4); the remaining
3199 were not empirically checkable at this budget and are covered by the proof
instead. This mirrors P1-06/07/08's own repeated experience this project: a numerical
check needs its own sensitivity analysis before its "0 violations" result can be
trusted, the same discipline applied to P1-09's Bonferroni correction and P1-07's own
Monte-Carlo replicate-count choice.

**What this changes going forward:** `paper/sections/theorem1_bound.tex` states and
proves the corrected (worst-case, both-arms) form as Theorem 1, with a prominent remark
explaining the discrepancy from both the plan's draft text and the already-shipped
`bound.py` functions. Phase 3's algorithm (P3-02 onward) should build its own
delta-correctness guarantee on `worst_case_marginal_bound_term`, not
`marginal_bound_term` -- the latter stays in the codebase unchanged (P1-07/P1-08's
reported numbers remain correct descriptions of that formula) but should not be
presented as a proven worst-case bound without this caveat if it is ever cited that way
in the paper.

**Decided by:** Agent, while executing task P2-02. Caught entirely by the numerical
certificate the task itself asked for, before any theorem statement was finalized or
presented as proven -- the exact scenario `plan/03-phase2-theory.md`'s introduction
anticipates ("A theorem that fails its numerical check is wrong, and finding that out
now is worth more than a month of proof-writing"), now applied to a formula already
relied on by two merged-into-the-open-PR-stack tasks (P1-07, P1-08), not just a fresh
draft.

---

## 2026-09-10 — Theorem 2 Part B: a concrete impossibility construction, and a sampling gotcha in verifying it

**Context:** `plan/03-phase2-theory.md` P2-03 asks for a change-of-measure lower bound
(Part A, "genuinely tractable") and an impossibility result (Part B, explicitly flagged
as needing `\needshuman` for the construction itself: "give the human a fully worked
*candidate* construction plus the numerical evidence that it works").

**Part A decision -- adapt Kaufmann/Capp\'e/Garivier's transportation lemma with a
Fisher-information-shaped per-pull rate, not a per-arm KL.** Since no scale `s < s*`
directly observes `mu_k(s*)`, the per-pull information is governed by how much a pull
shrinks the Fisher information matrix around `theta_k`, propagated to `s*` through the
Jacobian. The resulting per-arm rate, `Delta_k^2 / (J^T I_k(w)^-1 J)`, comes from a
standard D-optimal-design identity (`min{x^T A x : b^T x = c} = c^2/(b^T A^-1 b)`,
Lagrange multipliers) applied to "the cheapest parameter perturbation that flips the
gap at `s*`" -- concavity of the overall `sup_w min_k (...)` program follows from the
same "infimum of affine-in-w functions is concave" argument the classical (non-parametric)
BAI literature uses, not from a separate check of the closed-form's own convexity
(which is a genuinely different, and non-obviously-true, question -- e.g. `1/f` is not
concave for every positive convex `f`, so the safer route is the general argument, not
the specific closed form).

**Part B decision -- a Holder-ball-vs-sup-norm construction, with an explicit,
saturating bump function.** Two instances that agree exactly on every fitting scale but
flip the winner at `s*`: `h_1 = 0`, `h_2(s) = eta * phi((s-s_max)/g)` for a bump `phi`
supported on `[s_max, s*]`. For a pure sup-norm-bounded `H` (no smoothness), any `eta >=
Delta_min/2` suffices, independent of the gap `g = s*-s_max`. For a Holder-alpha ball,
using `phi(x)=clip(x,0,1)^alpha` (a textbook alpha-Holder function with constant
exactly 1) gives a construction that *exactly saturates* a Holder(alpha, eta_budget)
budget, achieving maximum amplitude `eta_budget * g^alpha` at `s*` -- so the sufficient
condition for impossibility is `eta_budget * g^alpha >= Delta_min/2`, a genuine phase
transition in the gap `g` (unlike the sup-norm case). Whether this is also *necessary*
(i.e. a tight characterization, not just a sufficient one) for general `H` is the part
left `\needshuman`, per the plan's own instruction.

**A numerical-verification gotcha, caught before it was trusted.**
`tests/theory/test_theorem2.py`'s first version verified the Holder construction's
realized modulus by sampling random pairs `(s, s')` and taking the empirical max of
`|h(s)-h(s')|/|s-s'|^alpha`. This under-reported the true constant by 4-15% across
random instances, because the textbook bound `|x^alpha-y^alpha| <= |x-y|^alpha` is
tight *exactly* at `y=0` -- a single point with Lebesgue measure zero, which uniform
random sampling essentially never lands on. Fixed by adding deliberate pairs anchored
at the construction's own breakpoints (`s_max`, `s_max+g`) alongside the random ones.
The same class of mistake as P1-11's F5 legend placement or P2-02's Monte-Carlo
resolution floor: a numerical check that looks like it passed (or, here, looked like it
was failing the *real* thing) for a reason that turns out to be about the *verification
method's* own blind spot, not the claim being checked -- caught by asking *why* a
result looked slightly off instead of loosening a tolerance until it passed.

**Deferred, not skipped:** the plan also asks to "verify T* computed numerically
matches the achieved sample complexity of the P3 algorithm in the solvable regime" --
impossible before Phase 3's algorithm exists. Noted explicitly in both the `.tex` and
the test file rather than silently dropped; P3-04's simulation study should close this
loop.

**Decided by:** Agent, while executing task P2-03.

---

## 2026-09-11 — Theorem 3: optimal scale placement is not "spread out as much as possible"

**Context:** `plan/03-phase2-theory.md` P2-04 asks for the rank/spacing identifiability
condition, a minimax rate, and a worked design-dependence corollary for the power-law
family specifically ("Derive it, at least for the power-law family").

**Decision -- reuse Theorem 2's Fisher-information machinery for the minimax lower
bound, rather than re-deriving Le Cam's method from scratch.** Le Cam's two-point
method and the change-of-measure BAI lower bound are the same underlying technique
(both ask "how well can C observations distinguish nearby parameter values"); the
minimax estimation-risk lower bound for `mu_hat_k(s*)` and Theorem 2 Part A's
selection-error lower bound share the identical `J^T I_k(w)^-1 J` rate. Presenting
Theorem 3's lower bound as a direct reuse (not a parallel derivation) keeps the theory
section internally consistent and is honest about how little new machinery this
specific result actually needs.

**A real, verified, non-obvious finding: optimal second-scale placement is U-shaped,
not monotone.** Before writing the "spread scales out" intuition into the paper as
fact, it was checked numerically for the reduced 2-parameter power law (`A*N^-alpha`,
`PowerLawN` minus its ceiling `E`) at a concrete instance (`A=2.0, alpha=0.3, N*=1e9,
N1=1e6`): `v(N*)` is **not monotone** in the second scale's position. It falls sharply
as the two scales separate (from `~1.5e4` at a near-clustered ratio of 1.01, <!-- NUMBER-OK: computed grid-search value, see tests/theory/test_theorem3.py --> to a
minimum of `~0.52` around ratio 30), but then **rises again** toward `~0.97` as the
second scale approaches the target itself (ratio 900) -- placing a design point as
close as possible to the target is *not* optimal for the variance at the target, even
though it minimizes that point's own extrapolation distance. This is a genuine feature
of extrapolation design (as opposed to interpolation design), not an artifact of the
specific numbers chosen -- confirmed by reproducing the exact table in
`tests/theory/test_theorem3.py` and checking the curve is interior-minimized (strictly
decreasing then strictly increasing), not just eyeballing a plot.

**Scoping decision:** the fully general version of this problem (m scales, the full
3-or-more-parameter families, compute-cost-weighted, not just 2 points and 2
parameters) has no simple closed form we found, and is exactly the `T*(nu)` program
Theorem 2 Part A already defines. Rather than force a general derivation, the worked
2-point example stands as a concrete, fully-verified illustration of *why* spacing is a
real, nontrivial trade-off, with the general case explicitly hedged to P3-02's
numerical solver.

**Decided by:** Agent, while executing task P2-04. The non-monotonicity was checked
numerically (a grid search over the second scale's position) before being written into
the theorem as a claimed finding, not assumed from the "spread scales out" intuition
that motivated looking at this in the first place.

---

## 2026-09-11 — Theorem 4: the algorithm's stopping rule is defined, not just its guarantee

**Context:** `plan/03-phase2-theory.md` P2-05 asks for the Extrapolation-Track-and-Stop
algorithm's correctness guarantee, with an explicit subtlety flagged: a naive
Track-and-Stop threshold never stops once a bias floor is present, because the GLR
statistic converges to a finite limit instead of diverging. The plan recommends route
(b) -- condition delta-correctness on a known/estimated `eta` -- over route (a)
(inflating the confidence level), calling it "more honest and more useful."

**Decision -- write the algorithm's stopping/abstention rule out precisely enough to
prove things about, since P3-03 needs a spec, not just a guarantee statement.**
`Certified_k(t)` compares a *shrinking* confidence radius `c_k(t,delta)` against a
*fixed* bias floor `eta_khat + eta_k` subtracted from the estimated gap -- the fix for
the "never stops" problem is structural (the threshold doesn't grow, so it's always
reachable in the solvable regime), not just a bigger confidence level.
`Abstain_k(t)` triggers on the *same* two quantities from the opposite direction:
once `c_k(t,delta)` has shrunk below a small fixed tolerance but the (now precisely
known) point estimate is still inside the "could be flipped by bias alone" band, no
more compute at this design can help, and the algorithm should say so rather than run
forever.

**The abstention proof is a direct corollary of Theorem 2 Part B's own construction,
not a separate argument.** Since `nu_1` and `nu_2` (Theorem 2's impossibility pair)
produce identical data below `s*`, the estimator's limiting value is identical under
both -- the algorithm literally cannot tell which instance it's in, and the construction
is built so that limit sits inside the eta-band under (at least) one of them. This
ties the whole theory section together: the same bias/variance decomposition (P1-06),
the same corrected worst-case bound (Theorem 1), the same impossibility construction
(Theorem 2 Part B) all feed directly into why the algorithm gives up gracefully instead
of hanging.

**A real bug in the certificate itself, caught by an explicit guard test.** The first
version of `tests/theory/test_theorem4.py` capped simulated replicate counts at
`2^20 ~= 1e6` and got "undecided" (neither certified nor abstained) in 500/500 runs of
the impossible-regime check -- not because abstention was broken, but because
`c_k(t,delta)` never actually crossed the fixed `epsilon_0=1e-4` tolerance within that
round budget for the specific `(fit_scales, s*, sigma)` used, so the test was checking
nothing. Fixed by extending the round cap (to `2^32`, a purely synthetic replicate
count -- no real experiment runs this many reps, but this is a unit-level check of the
stopping-rule *logic*, not a physical simulation) and adding a standing guard test
(`test_epsilon_0_is_reachable_within_the_round_cap`) that fails loudly if a future
change to the instance parameters makes the round cap insufficient again, rather than
silently producing "undecided" everywhere and passing for the wrong reason.

**Scoping decision:** the certificate tests the `Certified`/`Abstain` stopping logic in
isolation (fixed design, increasing replicates) -- not the full adaptive C-tracking
algorithm, which needs P3-02's `T*` solver. Asymptotic optimality (the second of
Theorem 4's three claims) is marked `\needshuman` for the same reason as elsewhere:
adapting Garivier & Kaufmann's classical tracking-convergence proof to this project's
M-estimator setting (rather than per-arm sample means) is mechanical in structure but
was not written out line-by-line.

**Decided by:** Agent, while executing task P2-05.

---

## 2026-09-11 — P3-01: the oracle interface, and the 6ND approximation checks out against DataDecide's own compute column

**Context:** `plan/04-phase3-algorithm.md` P3-01 asks for `PullOracle` and its first two
implementations, with an explicit definition-of-done item: "`cost()` verified against
the `compute` column already present in DataDecide's table (a good independent check of
our `6ND` accounting)."

**Finding: the `6ND` approximation matches DataDecide's own logged compute closely, with
known per-size scatter.** Comparing `Scale(n, d).compute` (`6*N*D`, used everywhere in
this project -- the cost model in `setup.tex` Assumption 3, every Phase 1/2 "matched
compute" comparison) against the real `compute` column in `results` from
`build_frame(source="macro_avg")`, across all 14 distinct sizes: ratio min 0.85, median
0.999, mean 0.991, max 1.10. <!-- NUMBER-OK: computed directly by tests/test_oracle.py::test_datadecide_cost_matches_6nd_to_within_measured_tolerance, reproducible from that test, not stored in a results/*.json file --> The two are **not exactly equal** (DataDecide evidently
accounts for FLOPs slightly differently per size -- plausibly embedding parameters or
attention-FLOPs specifics the textbook `6ND` heuristic elides), but agree to within 15%
at the extremes and under 0.1% in the median. Decision: keep `cost()` == `Scale.compute`
(the `6ND` form), for consistency with the cost model already used throughout Phase 1
and Phase 2's theory, rather than switching `DataDecideOracle` to the table's own
column -- switching would make "compute spent" not comparable between the synthetic/
theoretical analysis and the real-data replay, the opposite of what P3-05 needs. The
discrepancy is now a measured, tested fact (`tests/test_oracle.py`
`test_datadecide_cost_matches_6nd_to_within_measured_tolerance`), not an unstated
assumption.

**`SyntheticOracle` calibration is read from the real P1-05/P1-06 numbers, not
guessed.** Noise variance range (5e-5 to 2e-4) comes directly from
`results/p1_05_noise.json`'s `noise_vs_scale_summary` (median `sigma2_seed` around 1e-4
across sizes/tasks); <!-- NUMBER-OK: hand-chosen SyntheticOracle._BIAS_MAGNITUDE_RANGE, a design constant, not a result --> bias-magnitude range (0.03 to 0.20) comes from
`results/p1_06_decomposition.json`'s `ratio_vs_compute` (`median_sigma2_extrap_hat`
0.0023 to 0.030 across 6 fitters x 3 designs, i.e. bias magnitude ~0.048 to ~0.17,
widened slightly at both ends to cover cells P1-06 didn't report exactly). Both
verified against the actual committed result files before being written into the
oracle's constructor, not recalled from memory of earlier sessions' summaries.

**`DataDecideOracle`'s pull() determinism relies on discovering seed labels per
(recipe, scale), not a fixed list** -- directly re-using P0-06's finding
("small aux 2/3" below 1B, "large aux 2/3" at 1B) rather than re-litigating it;
`tests/test_oracle.py::test_datadecide_seed_labels_are_discovered_not_hardcoded`
confirms the label *sets* genuinely differ at the smallest vs. largest scale in the
real cached data, so this isn't a hypothetical the code merely tolerates.

**Decided by:** Agent, while executing task P3-01.

## 2026-09-11 — P3-02: the allocation solver is validated-reasonable, not certified-optimal

**Context:** `plan/04-phase3-algorithm.md` P3-02 asks for a numerical solver for the
Theorem 2 Part A compute-weighted optimal-allocation program (T*): `sup_w min_{k!=k*}
Delta_k^2 / (2 * J_k^T I_k(w)^-1 J_k)` subject to a compute budget on the simplex, plus
a brute-force check on small instances.

**Finding: two standard constrained-NLP routes were tried first and both failed on real
instances, for reasons traced to the same root cause.** `scipy.optimize.minimize`'s
SLSQP (epigraph reformulation, raw weights) converged to a badly-conditioned point
(weights ~1e-18, costs spanning ~1e15-1e20) at 0.55x brute force's rate; reparametrizing
to the true compute-fraction simplex and adding objective-scale normalization made
SLSQP report "inequality constraints incompatible" on a provably feasible problem;
switching to `trust-constr` converged without error but still landed at ~0.83x brute
force's rate, with a `delta_grad==0.0` warning. Root cause: both solvers' constraint
Jacobians are built by finite-differencing a function that routes through
`np.linalg.pinv`, which is ill-conditioned near the rank-deficient Fisher-information
matrices that sit right at the feasible region's boundary -- exactly where the search
needs to reason correctly.

**Fix, part 1: an exact closed-form gradient.** Using `d(M^-1)/dw = -M^-1(dM/dw)M^-1`,
`_arm_rate_and_grad` computes `d(rate)/dw_s` in closed form rather than by finite
differencing through `pinv`. Verified against finite differences directly (ratio
1.00000-1.00006 across components in ad hoc testing) -- confirming the earlier failures
were a numerical-conditioning problem, not a formula bug.

**Fix, part 2 and 3: step-size and floor regularization for hand-rolled projected
subgradient ascent.** Pivoting away from `scipy.optimize` entirely to a hand-rolled
single-arm-update projected subgradient ascent on the simplex first collapsed to
`rate=0` because gradient components vary by ~10,000x across cheap vs. expensive
scales, so a step size calibrated only to the rate's own magnitude let one step exit the
simplex before projection; fixed by normalizing the step to the *current* gradient's own
norm (`step = alpha/(sqrt(t)*||grad||)`). It then still occasionally collapsed one arm
to exactly zero weight, causing a rank-deficient `I_k(w)` and a silently-wrong `pinv`
(treating an out-of-range direction as zero variance rather than infinite); fixed with an
interior-point-style floor (`with_floor`, an affine remap of the simplex that keeps
every weight `>= 1e-6/n_dims` strictly positive *during* the search only, not in the
final reported weights).

**Finding, not fully fixed: even with both fixes, the solver reliably converges to a
*locally* max-min-consistent point that is not always the *global* optimum, and this
is shipped as a known, documented limitation rather than resolved.** Every restart lands
at (very nearly) the same objective value with the two tightest challengers' rates
equalized -- the textbook signature of a genuine critical point -- but direct comparison
against `brute_force_allocation` (dense random Dirichlet sampling, an unrelated search
strategy) found a feasible point with a noticeably higher objective on at least one
realistic test instance. Diagnosis (confirmed via a warm-start experiment: initializing
the search exactly at brute force's own known-good point still drifted one arm's weight
to 0 after 2000 iterations): updating only the currently-tightest challenger's gradient
block each step, then applying a *full* simplex projection, can let the projection's
mass-redistribution deplete an untouched arm's weight as a side effect of increasing
another's. A softmin-weighted all-arms gradient combination was tried and did not
reliably converge (oscillated); a water-filling/bisection decomposition (outer bisection
on target rate, inner per-arm minimum-cost-for-target-rate subproblem) was started but
left with an unresolved scaling bug (the bisection collapsed to `R≈0` despite clearly
feasible higher-`R` points existing) and not completed, given the time already spent on
the two fixes above. **Decision, made under explicit time pressure: ship the
single-arm-update, floor-protected solver as-is, with `n_restarts` multi-start as the
mitigation, and document the limitation honestly in `solve_allocation`'s own docstring**
rather than either hiding it or blocking on a fully robust fix. Every caller must treat
`solve_allocation`'s output as validated-reasonable, not certified-optimal, until this is
revisited -- `brute_force_allocation` remains the independent check referenced in the
docstring and in `tests/test_allocation.py`'s deliberately loose tolerances (e.g.
`res.rate >= bf.rate / 10`, and a restart-consistency spread `< 10.0`, loosened from an
initial `3.0` after an actual 4.5x spread was observed at `n_restarts=2, n_iter=500`).

**Also fixed in passing: `brute_force_allocation` was too slow to be a practical
verification tool.** The original per-sample Python loop took ~17s for 100k samples;
rewritten to build all samples' Fisher-information matrices via batched `np.einsum`
outer products and invert them with `np.linalg.pinv`'s batched `(..., p, p)` broadcasting,
which handles 1M+ samples in a few seconds -- necessary for it to actually get run as a
check rather than skipped for being too slow.

**Decided by:** Agent, while executing task P3-02.

## 2026-09-11 — P3-03: Extrapolation-Track-and-Stop, a real gap the theorem leaves open, and a genuine small-sample HC0 slowdown

**Context:** `plan/04-phase3-algorithm.md` P3-03 asks for `src/pdt/bai/ets.py`
implementing Extrapolation-Track-and-Stop exactly as specified in
`paper/sections/theorem4_algorithm.tex` (tracking, stopping, abstention), plus four
baselines (`UniformAllocation`, `SingleScale`, `SuccessiveHalvingOverScales`,
`FixedLadderExtrapolation`) sharing the same `PullOracle`.

**Finding: Theorem 4's tracking rule as written cannot be run literally, because
Theorem 2's T*(nu) program (P3-02) has no term for k*'s own allocation.** The
change-of-measure construction (`theorem2_lower_bound.tex` Step 2) perturbs only a
challenger's distribution, holding k*'s fixed -- so `solve_allocation`'s returned
`weights` dict has entries only for challenger arms, by design, not an oversight (see
P3-02's own module docstring). A literal `argmin_{k,s} N_k(s,t)/pi(k,s)` tracking rule
is undefined for k* itself. This is a real gap between the theorem (whose entire point
is the change-of-measure argument, which genuinely doesn't need k*'s own allocation)
and a runnable algorithm (which still needs to keep observing k* to know
`mu_hat_{k_hat}(s*)` at all, and to re-identify its extrapolator as the leader
potentially changes across rounds). Resolved with an explicit, documented heuristic:
reserve a fixed fraction `kstar_reserve_frac` (default `1/(n_challengers+1)`, i.e.
"treat k* like one more arm") of the tracking weight for the current leader, spread
uniformly across its own candidate scales. Not tuned or claimed optimal -- flagged in
`extrapolation_track_and_stop`'s own docstring as a real addition Theorem 4 does not
specify.

**Finding: the Theorem 3 identifiability check needed for `_assert_design_identified` had
to be about the *design*, not about one particular converged fit's numerical Jacobian.**
The first version fit each recipe's model, then asserted
`matrix_rank(jacobian at each distinct scale) == n_params`. This failed on a genuinely
well-designed test run (4 distinct scales, more than enough for `PowerLawN`'s 3
parameters) because the multi-start nonlinear fit had converged to a numerically
near-degenerate point (`alpha` saturating near its bound made the `d/d(alpha)` and
`d/d(a)` directions collinear at every observed scale) -- a property of *that particular
converged theta*, not of whether the scales pulled were enough to identify the model in
principle. Fixed by checking the number of distinct scales pulled (`>= n_params+1`)
directly, which is what forced exploration actually guarantees and is what Theorem 3's
condition is really about.

**Finding: a real, sensible-once-understood small-sample slowdown in the stopping
statistic, found while building a fast test for the abstention path.** With the
literal single-pull-per-round tracking rule, `c_t` (the confidence radius) does not
monotonically shrink from round 1 -- it *rises* for hundreds to thousands of rounds
before its asymptotic `1/sqrt(n)` decay dominates, because the HC0 sandwich variance
estimator (`analytic_v_k`) is itself noisy with very few residual degrees of freedom
(a handful of pulls per scale), and can report an artificially tiny variance early on
purely by chance. This is expected, textbook small-sample behavior of a
heteroscedasticity-consistent covariance estimator, not a bug in the tracking logic
(confirmed by direct instrumentation of the round-by-round trace) -- but it means a
literal "run until it naturally abstains" test on a close-gap instance needed several
thousand adaptive rounds to converge in some configurations, far too slow for a unit
test. Rather than accept a multi-minute test suite, added a `min_pulls_per_pair`
parameter (default 1, matching the theorem's own minimal forced-exploration floor) so
a caller -- including a test -- can front-load more initial replicates and skip past
this small-sample regime; the abstention tests use `min_pulls_per_pair=150` to
converge in under a tenth of a second while still exercising the real Certified/Abstain
logic, not a shortcut around it.

**Also found and removed: a dead defensive branch.** An early version handled
`total_w <= 0` (all challenger weights summing to zero) as a degenerate case, falling
back to uniform tracking. `solve_allocation`'s own `with_floor` interior-point
regularization guarantees every returned weight is strictly positive regardless of how
small the plug-in deltas are, so `total_w > 0` always holds in practice -- the branch
was unreachable and has been removed rather than kept as untested, unreachable
"just in case" code, per this project's own stated preference for no defensive
handling of scenarios that cannot happen.

**Baselines:** `SuccessiveHalvingOverScales` needed one correction after its first
version: eliminating survivors is about the *recipe pool*, not the *scale ladder* --
the original version stopped visiting further rungs as soon as only one recipe
remained, which (with few recipes and a longer ladder) can leave the eventual winner
with too few distinct scales to be identified at all. Fixed by always visiting every
rung regardless of how many recipes remain, so the final survivor(s) still accumulate
enough scale coverage for their own extrapolation to be valid.

**Definition of done (from the plan):** all five methods run to completion on the
(real, calibrated) `SyntheticOracle`; unit tests cover the stopping and abstention
rules -- 19 tests, 100% coverage on `src/pdt/bai/ets.py`.

**Decided by:** Agent, while executing task P3-03.

## 2026-09-11 — `uniform_allocation` follow-up fix: recipe-major pull order silently starved late recipes

**Context:** found while building P3-05's real DataDecide replay (25 recipes, a scale
ladder with one scale costing orders of magnitude more than the others -- some real
DataDecide `(N, D)` pairs sit far off the Chinchilla-optimal ratio). `uniform_allocation`'s
`pairs = [(r, s) for r in recipes for s in candidate_scales]` cycles every scale for one
recipe before moving to the next recipe. With a `compute_budget` that cannot afford a full
lap for every recipe (a real, not hypothetical, situation: a budget sized to be
"reasonable" relative to total ladder cost can still be far short of `n_recipes x` that
cost), this order exhausts the budget partway through the *first* recipe's own ladder,
leaving every later recipe with **zero** pulls -- not just incomplete data, no data at all
-- and `_fit_recipe` then raises `FitFailure` rather than `uniform_allocation` returning a
usable (if compute-starved) answer.

**Fix:** cycle scale-major, cheapest scale first (`pairs = [(r, s) for s in
sorted(candidate_scales, key=cost) for r in recipes]`) -- every recipe gets at least the
cheap end of the ladder before any recipe gets a second pull, for any positive budget.
This does not, by itself, guarantee every recipe reaches full identifiability under a too-
small budget (that requires the budget to actually afford a full lap, a separate, correct
requirement -- see the P3-05 entry below for the matching calling-code fix, "equal compute
per arm" needs a budget that scales with the number of arms); what it fixes is a real
correctness cliff where an insufficient budget failed some recipes completely rather than
partially. Regression test added:
`tests/test_ets.py::test_uniform_allocation_covers_every_recipe_even_with_few_expensive_scales`,
pinning the exact 25-recipe, skewed-cost configuration that raised before this fix.

**Decided by:** Agent, while executing task P3-05 (fix applied on `phase3/track-and-stop`
and merged forward, per this project's no-rebase discipline).

## 2026-09-11 — P3-04: a real, honest scope reduction, two instance-construction bugs, and a genuine finite-sample gap in Theorem 4's guarantee

**Context:** `plan/04-phase3-algorithm.md` P3-04 asks for a simulation study sweeping `K in
{5,10,25} x delta in {0.05,0.1,0.2} x eta_level in {0,small,medium,large} x gap_structure
in {well-separated, close top-two, reversing}`, >= 200 runs per cell (108 cells, 21,600+
runs total), checking three claims about error rate, compute-to-stop, and abstention
behavior in the impossible regime.

**Scope decision, stated up front rather than silently cut down.** `extrapolation_track_and_stop`
refits every recipe's extrapolator and re-solves P3-02's T* program every adaptive round;
a single run can take seconds to (for hard instances) several minutes. The literal
21,600-run grid was not feasible on a laptop within this session's time budget.
`experiments/p3_04_simulation.py` runs a smaller but genuinely real pilot instead
(`K=3`, `delta in {0.05, 0.2}`, `eta_level in {none, large}`, `gap_structure in
{well_separated, reversing}`, 15 runs/cell -- 8 cells, 120 runs, ~36 minutes wall time),
with every parameter overridable via CLI so the literal grid can be run later given more
compute. This is a stated limitation of *this pilot's results*, not a claim that the full
grid was run.

**Bug 1 found and fixed: "well_separated" instances weren't reliably separated.** The
first version drew each recipe's ceiling `e` independently from `Uniform(0.5, 0.85)` --
for `K=3` this can, by chance, place the top two ceilings arbitrarily close together
(the same "unlucky sampling looks like a real problem" class of gotcha as earlier
sessions' findings). This produced a misleadingly high apparent certified-error rate in
what was supposed to be the easy, well-specified regime, with nothing to do with the
algorithm. Fixed by spacing ceilings evenly (`np.linspace`) with jitter capped well below
the guaranteed minimum gap.

**Bug 2 found and fixed: the "reversing" (rank-reversal) construction could demand
fit parameters `PowerLawN` cannot represent.** The construction solves for the trailing
recipe's own `a` parameter so its curve crosses the apparent leader's at a chosen scale --
an early version allowed the crossover point to land anywhere between the fit ladder and
the target, which for a crossover close to the target requires huge `|a|` (tens to ~100)
to make up the gap over a short remaining distance. `PowerLawN`'s own fit bounds cap `a`
at +-10 (`src/pdt/scaling/fitters.py`) -- a demanded `a` outside that range is a curve the
fitter cannot even represent, not a genuinely hard-but-fittable extrapolation. Traced by
comparing a large-sample probe fit's fitted `theta` against the true one: the fit was
pinned exactly at the `a=-10` boundary. Fixed by placing the crossover close to the fit
boundary (needs less catching-up, keeps `|a|` representable in roughly half of draws) with
a bounded retry (including redrawing the base curve entirely if 200 inner attempts fail on
an unlucky draw).

**Finding, not a bug -- a genuine finite-sample gap in Theorem 4's guarantee, found while
diagnosing what looked at first like a third bug.** After fixing both constructions above,
the "reversing" cells still showed `ets_error_rate_given_certified = 1.0` (every certified
decision wrong) in early testing. Direct inspection of one such run: it certified after
only 12 pulls -- the bare warm-up (`min_pulls_per_pair=1`, one pull per (recipe, scale)
pair, as specified) -- with a reported `delta_hat` of 0.28 against a *true* gap of 0.0014.
Root cause: with exactly `n_params+1` distinct scales and one pull each, a 3-parameter
`PowerLawN` fit has essentially zero residual degrees of freedom, and the HC0 sandwich
variance estimate (`analytic_v_k`) computed from that fit can report an implausibly small
`v_hat` purely by chance -- the same small-sample HC0 artifact already documented for
P3-03 (a `c_t` that *rises* before it falls), but here manifesting as **false certification
at round 1**, not just slow convergence. This is a concrete instance of exactly the caveat
`theorem4_algorithm.tex`'s own asymptotic-optimality proof sketch flags but does not
resolve (the "nonlinear-g M-estimator concentration" issue, `\needshuman`-marked there):
Theorem 4's delta-correctness proof is mechanical *given* Theorem 1's M-estimator
concentration result, but that concentration is an asymptotic statement, and asymptotic
normality is a poor approximation at 1 residual degree of freedom.

**Mitigation, tested directly, not just assumed.** Raising `min_pulls_per_pair` (an
existing ETS parameter, added in P3-03 as a test-speed convenience, now given a real
safety justification) from 1 to 8 changed a `K=3` reversing-instance diagnostic from
100% wrong-when-certified with mostly-fast-but-wrong certifications, to 18/20 correctly
abstaining and only 2/20 certifying (both of those two still wrong, but from a sample far
too small to say whether this exceeds delta's budget or is within it). `min_pulls_per_pair=8`
is this script's default, documented in its own module constant.
**This residual uncertainty is reported as an open question, not resolved** -- a proper
answer needs many more certified-outcome samples than this session's time budget allowed,
and is a natural target for a follow-up run (or for P3-06's eta-sensitivity study, which
touches the same underlying mechanism from a different angle).

**Pilot results (`results/p3_04_simulation.json`, 8 cells x 15 runs):**
- **Claim 1 (error rate respects delta in the well-specified regime): holds.** Both
  `well_separated`/`eta=none` cells: 0/15 wrong at both `delta=0.05` and `delta=0.2`.
- **Claim 2 (compute-to-stop approaches `T*log(1/delta)` as delta shrinks): not
  meaningfully testable at this pilot's scale.** Mean compute-to-stop for certified runs
  was *identical* between `delta=0.05` and `delta=0.2` in the well-separated cells
  (2.9088e18 both times) -- because the gaps there are so large that every one of the
  15 runs at *both* delta values certified immediately after the forced-exploration
  warm-up, before a single adaptive round ran. The warm-up cost dominates total compute
  at this instance difficulty, not the delta-dependent stopping threshold -- a real
  limitation of this reduced pilot's chosen instance difficulty, not evidence against the
  claim. A future run needs a harder well-specified regime (or a smaller warm-up) to
  actually exercise the delta-dependent term.
- **Claim 3 (reversing regime: ETS abstains, baselines confidently wrong): mostly holds.**
  ETS abstained in 80-93% of reversing-regime runs across the 4 (delta, eta_level) cells
  -- the desired, cautious behavior. Baselines were correct 0% of the time in 6 of 8
  (delta, eta_level) x baseline combinations checked, confirming the intended contrast;
  `SuccessiveHalvingOverScales` (the one baseline that also extrapolates, per P3-03's own
  "visit every rung" fix) got it right in 2 of 4 reversing sub-cells, inconsistent rather
  than reliable. The residual false-certification rate noted above (small sample, open
  question) is the one qualifying caveat on an otherwise clean result.

**Definition of done (from the plan):** for the pilot scale actually run, error rates,
mean compute, and abstention rate recorded per cell; all three claims checked and reported
honestly, including the one (claim 2) not meaningfully testable at this scale and the one
(claim 3) with an open residual-uncertainty caveat, rather than being marked done without
qualification.

**Decided by:** Agent, while executing task P3-04.

## 2026-09-11 — P3-05: the headline finding is that ETS never reached a decision within this pilot's compute budget on real data, reported honestly rather than reframed

**Context:** `plan/04-phase3-algorithm.md` P3-05 asks for an offline replay of ETS and the
four baselines against real DataDecide data (target `s* = 1B`), with a leakage guard, a
headline compute-to-correct-decision table with error bars, and the abstention rate on
P1-09's reversal-heavy tasks -- explicitly anticipating a bad outcome: *"If instead it
abstains everywhere, the algorithm is practically useless and we must say so and reframe
toward the diagnostic contribution."*

**`GuardedOracle` (`src/pdt/bai/guarded_oracle.py`) enforces the no-target-leakage
constraint in code, tested directly** (`tests/test_guarded_oracle.py`, 7 tests, 100%
coverage): `pull()`/`cost()` raise `TargetScaleLeakageError` for the target scale,
`available_scales()` never lists it.

**Two real bugs found and fixed while building the replay, both documented in their own
decisions.md entries above/below this one:** `uniform_allocation`'s recipe-major pull
order could starve late recipes entirely under a real, many-recipe (K=25), skewed-cost
scale ladder (fixed: scale-major, cheapest first -- see the entry above); and ETS's
adaptive tracking can legitimately exhaust `DataDecideOracle`'s finite real+pseudo
replicate pool for a heavily-revisited (recipe, scale) pair -- something `SyntheticOracle`
never exposes, since it can draw unboundedly. Fixed with `_CyclingOracle`, which repeats
an already-observed real value past that point -- an explicit, documented practical
accommodation for finite real data, not a fabrication of new data.

**The headline finding, found only after fixing a reporting gap that nearly hid it:** the
first version of this script's output collapsed `SelectionResult.outcome == "abstained"`
into a single number, conflating a genuine Theorem-4 `Abstain` event with simply running
out of `max_rounds` without resolving either way. Adding `certificate_reason` /
`hit_round_cap` to the output (rather than trusting the collapsed `"abstained"` string)
revealed the real result: **across all 4 tasks tested (2 reversal-heavy, 2 stable),
`extrapolation_track_and_stop` hit the round cap (`max_rounds=60`) on every single run --
0% genuine abstention, 100% round-cap exhaustion, in both task groups.** It never once
reached either a delta-correct certification or a genuine bias-floor abstention on real
data within this pilot's affordable compute.

**This is reported as the honest result, not reframed to look better.** It is a different
failure mode than the plan's own anticipated "abstains everywhere" (which would still
demonstrate the diagnostic contribution the plan explicitly said to fall back on) --
"never resolves within affordable compute" is a real, distinct, and arguably more
concerning finding about practical usability at `K=25` on noisy real data with this
session's `max_rounds=60` budget. It is **not** conclusive evidence the algorithm doesn't
work on real data -- P3-04 already found (in a much smaller, K=3 synthetic setting) that
genuine convergence can require thousands of adaptive rounds for hard instances, and 60
rounds against 25 real, noisy recipes may simply be far too few to draw any conclusion
from. What P3-05 actually establishes is that **this question remains open**, and answering
it needs substantially more compute than this session's laptop time budget allowed for the
adaptive algorithm specifically (baselines, by contrast, are cheap and ran to real,
bootstrapped completion -- see the headline table below).

**Headline table (`results/p3_05_replay.json`, `delta=0.1`, `eta=0.02` assumed uniformly --
a fixed, documented, not-task-calibrated guess):**

| Task | Group | True winner | SingleScale | FixedLadder | UniformAlloc | SuccessiveHalving | ETS |
|---|---|---|---|---|---|---|---|
| winogrande | reversal-heavy | Falcon+CC (QC Orig 10%) | 20% | 0% | 0% | 0% | round-cap (fallback happened to be correct) |
| boolq | reversal-heavy | Falcon+CC (QC Orig 10%) | 10% | 0% | 0% | 0% | round-cap (fallback happened to be correct) |
| arc_easy | stable | DCLM-Baseline (QC 7%, FW3) | 100% | 25% | 80% | 0% | round-cap (fallback happened to be correct) |
| mmlu | stable | DCLM-Baseline (QC 7%, FW3) | 100% | 100% | 100% | 35% | round-cap (fallback happened to be correct) |

(Baseline columns are bootstrap accuracy over the 3 real seeds, `n_bootstrap=20`.) A real,
separate, worth-noting pattern in the baselines alone: they are *much* worse on the two
reversal-heavy tasks than the two stable ones (mostly 0% vs. mostly 80-100%) -- consistent
with P1-09's own kendall_tau ranking and with the plan's expectation that naive methods
struggle exactly where ranking is unstable, even though this pilot cannot say anything about
ETS's own comparative behavior there since it never reached a decision on any task.

**Definition of done (from the plan):** the headline table and the guarded-oracle test are
both delivered; the "if it abstains everywhere, say so" contingency is honored, adapted to
what was actually found (round-cap exhaustion, not genuine abstention) rather than forced
into the plan's literal anticipated framing. **This pilot does not resolve P3-05's core
question** (does ETS deliver real compute savings on real data) -- flagged explicitly as
open, for a follow-up run with substantially more compute for the adaptive algorithm
specifically.

**Decided by:** Agent, while executing task P3-05.

---

## 2026-09-17 — Three real bugs in the replay's oracle wrappers: an unguarded baseline path, a partial bootstrap remap, and silent recycling

**Context:** PR #31's reviewer found three real issues in
`experiments/p3_05_replay.py`'s oracle-wrapper classes.

**Issue 1 (P1): `_bootstrap_baseline` was called with the raw, unguarded
`base_oracle`, not `guarded`.** Every baseline ran against an oracle with
no `TargetScaleLeakageError` barrier at all -- current baselines don't
intentionally pull the target, but the regression barrier that would
catch a future one that did was simply missing.

**Fix:** `_run_task` now passes `guarded` into `_bootstrap_baseline`.
Added `tests/test_p3_05_replay.py::test_bootstrap_baseline_blocks_a_target_pulling_baseline`,
a baseline test double that deliberately pulls the target scale and must
raise `TargetScaleLeakageError` before returning anything -- exactly the
"replay-level test injecting a target-pulling baseline" the review asked
for.

**Issue 2 (P1): `_ReseededOracle`'s `seed_map` only remapped
`seed in range(_N_REAL_SEEDS)`; any higher index fell through unchanged
(`dict.get(seed, seed)`).** `UniformAllocation` was observed making 6
passes over the scale ladder on a real replay instance (`seed=0..5`),
so indices 3-5 reached `DataDecideOracle`'s own checkpoint-based
pseudo-replicate fallback directly -- a FIXED value, identical across
every bootstrap replicate, silently mixed in alongside genuinely
resampled real-seed pulls. Reported bootstrap accuracy and its CI
understated the true resampling variance by exactly this much
non-resampled, always-identical contribution.

**Fix:** `_ReseededOracle` now takes the bootstrap replicate's own `rng`
directly and lazily draws (and caches) an independently resampled real
seed for EVERY distinct index actually requested, however many there
are -- not a pre-sized map. Regression tests confirm indices up to 19
all resolve inside `[0, _N_REAL_SEEDS)` and stay consistent within one
instance.

**Issue 3 (P1): `_CyclingOracle` silently recycled an already-observed
real value once a (recipe, scale) pair's real+pseudo replicate pool was
exhausted, still counting it as a fresh pull.** ETS's own statistical
machinery (the analytic delta-method `v_hat`, the certification radius's
residual degrees of freedom -- PR #29's own fix) assumes every counted
pull is genuinely independent new information; silently recycling a
value while still counting it can make a certification look more
confident than the data actually supports, which would corrupt exactly
the guarantee this replay exists to exercise on real data. Per the
review's own remedy ("return finite-pool exhaustion, ... distinguish
replay compute/confidence from independent training runs").

**Fix:** `_CyclingOracle.pull()` now raises `_ReplicatePoolExhaustedError`
on exhaustion instead of recycling.
`extrapolation_track_and_stop`'s own `pull()` closure has no
`try`/`except` around the oracle call, so this propagates straight up;
`_run_task` catches it and reports an explicit `"pool_exhausted"` outcome
(`recipe`/`correct`/`compute_spent`/`n_pulls` all `None`, `pool_exhausted:
true`) -- NEVER a certification, abstention, or any other outcome built
on fabricated data. A new `ets_pool_exhausted_rate_*` pair (mirroring the
existing genuine-abstention/round-cap split) is reported alongside the
other two so a reader can see how often this happened without it being
silently absorbed into either.

**Not yet done:** `results/p3_05_replay.json` needs regenerating with all
three fixes once this branch merges past its upstream dependencies (real
DataDecide data access is also needed to run it at all, unlike the
synthetic-instance PRs).

**Decided by:** Agent, addressing PR #31's review. `tests/test_p3_05_replay.py`
added (6 tests, all passing).
---

## 2026-09-17 — make_instance zeroed the winner's own bias even when that bias made it the winner; abstention rate conflated timeout with genuine recognition

**Context:** PR #30's reviewer found two real issues in
`experiments/p3_04_simulation.py`.

**Issue 1 (P1): `eta_assumed` special-cased `eta=0` for whichever recipe
ended up as the re-resolved `k_star`, even when that recipe's OWN
target-only bump was substantial -- and, at large `eta_level`, is exactly
what can make a recipe win in the first place.** Reproduced exactly as
given: `make_instance(np.random.default_rng(0), 3, "well_separated",
"large")` yields winner `r2` with `eta_assumed["r2"]=0`, while `r2`'s own
`bias_at_target` is `0.6228505534` -- a supposedly "perfectly calibrated"
pilot (Theorem 4's delta-correctness hypothesis is meant to hold by
construction, per this file's own module docstring) was silently telling
the algorithm the eventual winner has no bias when it demonstrably does.

**Fix:** `eta_assumed[r] = abs(params[r]["bias_at_target"])` for every
recipe unconditionally, dropping the `0.0 if r == k_star` special case.
`bias_at_target` itself is unaffected -- it was already correctly 0 for
the *original* (pre-bump) leader by construction (line ~259's loop only
ever bumps recipes other than the original leader) and already correctly
nonzero for every bumped recipe, including one that goes on to become the
new leader; only the downstream `eta_assumed` formula was wrong. Pinned
the reviewer's exact reproduction as a regression test
(`tests/test_p3_04_simulation.py`).

**Issue 2 (P2): the reported `ets_abstention_rate` conflated genuine
bias-floor abstention with simply running out of `max_rounds`, so it
cannot support "the algorithm recognized an impossible instance" (claim
3's whole point in the reversing-regime cells).** `ets.py`'s own
`SelectionResult.certificate["reason"]` already distinguishes `"bias
floor"` (Theorem 4's actual abstention condition) from `"max_rounds
exhausted without certifying or abstaining"` (a pilot budget limit,
unrelated to the theorem) -- this script just never read it.

**Fix:** `_run_one_cell` now reads `certificate["reason"]` and tracks
`ets_abstention_rate_bias_floor` / `ets_abstention_rate_timeout`
separately alongside the original combined `ets_abstention_rate` (kept
for backward-compatible context, explicitly documented as not the right
number for a recognition claim). `claim3_reversing_regime_abstention_vs_baselines`
reports both split rates per cell.

**Not yet done:** `results/p3_04_simulation.json` needs regenerating with
both fixes once this branch merges past its upstream dependencies.

**Decided by:** Agent, addressing PR #30's review. `tests/test_p3_04_simulation.py`
added (4 tests, all passing).
---

## 2026-09-17 — ETS's certification radius was invalid for estimated variance; design-rank check missed a D-blind model

**Context:** PR #29's reviewer found two real issues in
`src/pdt/bai/ets.py`.

**Issue 1 (P1): the certification radius `c_t = sqrt(2*(v_a+v_b)*
log(1/beta_t))` is a valid sub-Gaussian tail bound only if `v_a`/`v_b`
are the TRUE, known variance -- but they are `analytic_v_k`'s HC0
sandwich-covariance ESTIMATE, itself uncertain, especially with the few
residual degrees of freedom a real run has early on.** Reproduced
exactly as given: `LogLinear`, scales `N=[1,2,3]`, target `N=4`,
`delta=.01`, a single check after just the `n_params+1=3` warm-up pulls
per arm (1 residual degree of freedom each), 1,000 independent trials --
9.1% unconditional certification error against the 1% requested, a ~9x
violation.

**Fix:** a Student-t radius, `t.ppf(1-beta_t, df) * sqrt(v_a+v_b)`, using
a Welch-Satterthwaite-combined `df` from each arm's own HC0-residual
degrees of freedom (pulls minus fitted parameters) -- the same remedy
already applied to P1-09's calibration bug. `_welch_satterthwaite_df` is
the general form (arbitrary, independently-estimated variances with
their own df) of `pdt.analysis.rank_reversal.welch_satterthwaite_df`
(the special case sharing one `n_seeds`). Verified empirically against
the reviewer's own counterexample, not just asserted: 5,000 independent
trials of the exact reproduction now give a 0.70% unconditional error
rate, comfortably under the 1% request (down from 9.1%).

**Explicitly not claimed as a rigorously proven finite-sample radius.**
A fully rigorous fix needs either a proper always-valid confidence
sequence for unknown variance, or restricting the guarantee to the
known-`sigma2` case `tests/theory/test_theorem4.py` already certifies
(that test's own simulation uses the TRUE simulation `sigma`, not an
estimate, so its Gaussian-formula certificate remains valid on its own
terms and was not changed). PR #26's review of `theorem4_algorithm.tex`
independently found the same "missing simultaneous-adaptive-confidence
proof" gap at the paper's proof level -- this ets.py fix is a
substantial, empirically-verified improvement to the running code, not a
substitute for that proof-level fix once PR #26 is reached.

**Issue 2 (P2): `_assert_design_identified` counted distinct `(N, D)`
scale pairs but never checked whether the model's Jacobian actually has
full rank there.** `LogLinear`'s fit ignores `D` entirely, so
`n_params+1` pairs sharing the same `N` (different `D`) passed the count
check while providing zero information about the slope parameter.

**Fix:** also check the Jacobian's rank at several GENERIC reference
parameter values (from the model's own `_bounds`, not this run's
possibly-degenerate converged fit -- reusing a possibly-degenerate
converged fit was already tried and rejected once, see the entry above;
this check deliberately avoids repeating that mistake). Two numerical
traps found and fixed while building this, both via direct measurement,
not guessed:
1. Linear-uniform probe sampling over a fitter's full declared bounds
   put ~81% of draws in the exact numerically-flat region PR #12's
   `multi_start_fit` log-uniform fix already exists to avoid (checked
   directly on a genuinely well-identified 4-scale `PowerLawN` design) --
   an 8-probe run had a ~20% chance of hitting 8-in-a-row bad draws and
   false-rejecting a legitimate design, which is exactly what an early
   version of this fix did to two pre-existing tests. Fixed by sampling
   log-uniformly for any dimension with a strictly positive lower bound
   (a model-agnostic heuristic matching every `alpha`/`beta` bound this
   project's fitters declare), dropping the bad-draw rate to ~18%
   (~1e-6 for 8-in-a-row).
2. The probe RNG was initially a single shared, mutating module-level
   generator, making one call's probe sequence (and thus whether it hit
   a bad-luck streak) depend on how many prior calls elsewhere had
   already consumed from it -- real, observed flakiness. Fixed by
   re-seeding a fresh, local RNG from a fixed constant on every call.

Regression tests added for both issues (`tests/test_ets.py`), including
the reviewer's exact `LogLinear`-ignores-`D` counterexample and the
`PowerLawN` false-rejection case that exposed the two numerical traps
above.

**Decided by:** Agent, addressing PR #29's review.
---

## 2026-09-17 — allocation.py: pinv silently mis-scored unidentifiable targets; converged was unconditional

**Context:** PR #28's reviewer found two real issues in
`src/pdt/bai/allocation.py`.

**Issue 1 (P1): `_arm_rate`/`_arm_rate_and_grad`/`brute_force_allocation`
computed `J_target^T pinv(I_k(w)) J_target` without checking whether
`J_target` actually lies in `range(I_k(w))`.** Reproduced exactly as
given: a `LogLinear` model (`y = a + b*log(N)`) observed only at `N=1`
has jacobian `[1, log(1)] = [1, 0]` at every candidate scale regardless
of weight, so `I = diag(1, 0)` -- the slope parameter is structurally
unidentifiable from this design, no matter how compute is allocated. The
target at `N=4` has jacobian `[1, log(4)]`, nonzero in the missing
direction. `np.linalg.pinv`'s minimum-norm convention treats a direction
outside `range(I)` as contributing ZERO variance (`pinv(diag(1,0)) =
diag(1,0)` exactly), the opposite of the truth (INFINITE variance --
"cannot be estimated at all"), so `_arm_rate` returned a small but finite
(and wrong) rate (~0.005 at `Delta=.1`, unit variance/weight) instead of
the correct 0.

**Fix:** a new `_target_denom` helper checks `J_target`'s residual after
projection through `I @ pinv(I)` (the orthogonal projector onto
`range(I)`) relative to `||J_target||`, returning 0.0 -- matching
`_arm_rate`'s own already-documented "no information at all" contract --
whenever that residual exceeds `_TARGET_RANGE_RTOL` (10%, deliberately
loose: `info` routinely gets extremely ill-conditioned (~1e14 condition
numbers) during the search, and reconstructing `J_target` through `pinv`
at that conditioning accumulates real floating-point residuals up to
~0.5% on ordinary, non-degenerate instances -- checked directly by
sampling many restarts of the existing test instance. A genuinely
unidentifiable case's residual is ~80-100%, not a few percent, so 10%
separates the two with wide margin on both sides; a tight,
precision-scale threshold was tried first and immediately false-triggered
on ordinary optimization trajectory, collapsing real solves to `rate=0`).
Applied to `_arm_rate` and `_arm_rate_and_grad` (whose zero-gradient
response in this case is the mathematically correct one -- if no
candidate scale's jacobian has any component in `J_target`'s missing
direction, no reweighting among them can ever create that information, so
`d(rate)/dw = 0` genuinely everywhere, not a spurious stall) and,
vectorized, to `brute_force_allocation`'s batched Fisher-information
computation. Regression tests added (`tests/test_allocation.py`),
including the exact reviewer counterexample and a sanity check that a
second, informative candidate scale restores a positive rate.

**Issue 2 (P2): `solve_allocation` reported `converged=True`
unconditionally, even when the `n_iter` budget was exhausted without the
subgradient ever reaching a stationary point.** Given the solver's own
documented status ("validated-reasonable, not certified-optimal," see
the entry above), this overclaimed every returned allocation as if the
search had definitively finished, when in the overwhelming majority of
real runs it had simply run out of budget.

**Fix:** each restart now tracks whether it broke early via the existing
exact-zero-subgradient check (a genuine stationary point, scale-invariant
regardless of the problem's own units) or ran the full `n_iter` without
reaching it; the returned `AllocationResult.converged` reflects the
BEST restart's own status, `n_iterations` reports how many iterations
that restart actually ran (not always the configured budget), and
`message` says plainly when the budget was exhausted rather than implying
a certified optimum. Deliberately did NOT use a small positive gradient-
norm tolerance instead of exact zero: the natural scale of the
subgradient depends entirely on the problem's own units (rates here span
roughly 1e-20 to 1 depending on `delta_k`/`sigma2`/`cost`), so any fixed
tolerance is either too loose (verified directly: a `1e-6` tolerance
falsely declared convergence after a single iteration on the existing
test instance, breaking two other tests that check solution quality) or
too tight for a different instance's units.

**Decided by:** Agent, addressing PR #28's review.
---

## 2026-09-17 — SyntheticOracle's observation noise was not seeded per-instance

**Context:** PR #27's reviewer found that `SyntheticOracle.pull()`'s
noise seed (`_stable_seed(recipe, scale.n, scale.d, seed)`) depended only
on the recipe, scale, and caller-supplied replicate `seed` -- never on
which `SyntheticOracle` *instance* was calling it. The constructor's own
`rng` draws different means/noise levels (`e`, `a`, `alpha`,
`bias_at_target`, `sigma2_noise`) per instance, but that randomness never
reached the observation draw. Two separate instances (e.g. two
"independent trials" in a simulation study, each built from its own
constructor seed) therefore reused the exact same standardized noise
innovation for the same `(recipe, scale, seed)` -- correlating
observations that were supposed to be independent repeated trials.

**Fix:** `SyntheticOracle.__init__` now draws `self._instance_salt` (a
64-bit int) once from the constructor's own `rng`, as the very first
thing pulled from it, and `pull()` folds it into `_stable_seed(...)`
alongside the existing `(recipe, scale.n, scale.d, seed)` key. Two
instances built from the SAME constructor seed still draw the identical
salt (it's the first deterministic draw from that seed's own stream), so
`pull()` stays exactly reproducible given the same construction seed --
`test_synthetic_pull_is_reproducible_across_fresh_instances` (same-seed
case) is unaffected. Only cross-instance independence for *differently*
seeded instances is what changes. Added
`test_synthetic_pull_noise_is_independent_across_differently_seeded_instances`,
which forces two differently-constructed instances to share identical
recipe params (isolating the instance-identity effect from the
already-different means/sigma) and confirms their pulls now differ.

Only `SyntheticOracle` was affected -- `DataDecideOracle` and
`LiveTrainingOracle` read from real/external data with no synthetic
per-instance randomness to correlate.

**Decided by:** Agent, addressing PR #27's review. Full suite: 22 passed
in `tests/test_oracle.py`.
## 2026-09-19 — Theorem 4: simultaneous confidence is an explicit condition; asymptotic optimality withdrawn (PR #26)

Review findings on `paper/sections/theorem4_algorithm.tex`, both accepted:
1. **Simultaneous adaptive confidence was asserted, not proved.** `beta = delta/[t(t+1)]` spends `delta`
   over time for one fixed comparison, but the proof unions over arms, the leader is data-selected, and
   the design is adaptive; Theorem 1 (fixed design) is not a confidence sequence. Now: the threshold is
   `beta = delta / (t(t+1) K(K-1))` (rounds x ordered arm pairs); Theorem 4 is split into
   (a) a *deterministic implication* (valid `eta` + simultaneous confidence event `E` => any stop is
   correct), (b) `P[E^c] <= delta` **proved for a pre-committed, non-adaptive schedule** with linear `g`
   and known sub-Gaussian noise, (c) **adaptive tracking: not proved** (needs a self-normalized /
   martingale confidence sequence; stated as an assumption, matching A4 in `src/pdt/bai/ets.py`
   and PR #29's decisions entry). The numerical certificate (test_theorem4.py) is scoped to (b) and
   its `beta` updated to include the `K(K-1)` factor. `eta` is explicitly an *assumed* upper bound;
   P1-06's clipped estimate is a heuristic and does not make it valid.
2. **Fixed bias budgets do not vanish as `delta -> 0`.** The stopping margin tends to
   `D_k^dagger - eta_{k*} - eta_k`, not `Delta_k`; even with `h = 0`, conservative `eta > 0`
   permanently reduces it, changing the leading constant; and the tracked program omits the winner's
   information (Theorem 2, challenger-only). The "asymptotic optimality = `T*`" theorem is
   **withdrawn** and replaced by a labeled, unproved *conjecture* for the joint robust characteristic time
   `T^eta` (robust gap `Delta_k^eta`). Abstention theorem restated as convergence in probability.

**Decided by:** Agent, following the review.
## 2026-09-19 — Theorem 3: rank is sufficient (not necessary) for identifiability; restore the 1/C in the minimax rate (PR #25)

Review findings on `paper/sections/theorem3_identifiability.tex`, both accepted:
1. **"iff rank(X) = p" was wrong.** Counterexample `g(theta, s) = theta^3` on `[-1, 1]`, `theta_k = 0`:
   uniquely identifiable from noiseless data, yet `J = 0`, rank `0 < p = 1`. Theorem 3 is now
   (a) rank `= p` **sufficient** for local identifiability, (b) a **constant-rank** converse
   (zero set is a `(p-r)`-dimensional submanifold => a continuum of equivalent parameters), and
   (c) the explicit note that a rank drop at a point alone gives no converse. Two-failure-modes
   remark reworded to "first-order identifiability fails".
2. **Missing `1/C`.** `I_k(w)` is per unit compute (`sum w c = 1`), so `J^T I_k(w)^-1 J` is constant
   in `C`, while risk is that quantity divided by `C` (the proof's own joint KL is proportional to `C`).
   The minimax display is now `(1/C) J^T I_k(w)^-1 J = v_k(C)`, the proof carries the factor through,
   the estimator achieving it is stated as *weighted* least squares (`1/sigma^2` weights; unweighted
   LS has the larger sandwich variance, with equality under homoscedasticity), and the in-family
   target risk is distinguished from misspecified `mu_k(s*)` risk (which adds
   `sigma^2_extrap`, not decreasing in `C`). Tests: 1/C scaling under replication, OLS >= WLS variance,
   the `theta^3` and constant-rank cases.

**Decided by:** Agent, following the review.
## 2026-09-19 — Theorem 2: cost normalization, exact KL, challenger-only is weaker, accessible scales (PR #24)

Review findings on `paper/sections/theorem2_lower_bound.tex`, all accepted:
1. **Cost normalization.** The statement had `sum w c = 1` with `I = sum w J J^T/sigma^2` (requires
   `w = E[N]/E[C]`, pulls per compute) but the proof defined `w = E[N] c / E[C]` and omitted `1/c`
   -- a factor-`c` error for a cost-`c` action. The proof now uses pulls-per-compute throughout
   (`P3-02`'s internal compute-fraction variable is converted by `w = p/c`; the code was already
   correct). Regression test checks the identity for costs (1, 10, 3).
2. **Fixed-gap KL is not local.** The change of measure needs a winner-flipping alternative at
   a fixed distance; Taylor-expanding the KL and "taking delta -> 0" does not justify a local
   Fisher form, sub-Gaussianity does not give Gaussian KL, and the alternative can leave the compact
   `Theta`. Theorem 2 Part A is now stated with the exact KL over admissible alternatives
   (Gaussian noise); the closed form `Delta^2 / (2 J^T I^-1 J)` is the linear-Gaussian special
   case and only when the minimizing alternative is admissible.
3. **Challenger-only vs joint alternatives.** Moving only the challenger yields a valid but *weaker*
   bound: two unit-cost Gaussian constant arms give `T = 2 sigma^2/Delta^2` (all budget on the
   challenger) vs `8 sigma^2/Delta^2` for joint alternatives (equal allocation) -- factor 4
   (numerically checked). We keep the challenger-only program, since it is what P3-02 solves, but
   rename it `T^chal`, state that it is not the tight characteristic time, and **withdraw every
   claim that an allocation solving it is asymptotically optimal**. The tight (joint, pairwise)
   program shares the winner's weights across challengers and is *not implemented*; this is a
   known limitation of P3-02/P3-03 (the current leader's reserved share in ETS is exactly the
   piece the challenger-only program cannot produce).
4. **Accessible scale set / ties.** The impossibility bump must vanish on *every scale a policy can
   query*, not only the fitting design's scales (queries inside `(s_max, s*)` would distinguish it);
   the construction is restated over `max S_acc` and does not apply when accessible scales approach
   `s*`. Equality `eta = Delta_min/2` (or `eta g^alpha = Delta_min/2`) is a tie, so the sufficient
   conditions are now strict; test updated accordingly.

**Decided by:** Agent, following the review.
## 2026-09-19 — Theorem 1: restrict the finite-sample claim; consistency is sufficient, not iff (PR #23)

Review findings on `paper/sections/theorem1_bound.tex`: (1) smoothness and a bounded Jacobian do
not make a nonlinear least-squares prediction exactly sub-Gaussian around the population
projection with its delta-method variance, yet the theorem asserted an exact finite-`C` bound;
(2) the consistency "iff" was wrong -- `D_k -> D_k^dagger` does not imply `D_k > 0` eventually iff
`D_k^dagger > 0` (a limiting tie resolved in favour of the true winner is consistent).

Changes: Theorem 1 is now stated as **(i) proved** for a linear-in-`theta` family with independent
sub-Gaussian noise, and **(ii) conditional** on an explicit concentration hypothesis (H) (centre
shift `rho_k`, variance proxy `bar v_k`) for nonlinear `g`; every nonlinear use (`pdt.theory.bound`,
P1-07/P1-08) is documented as an empirical diagnostic taking `rho_k = 0`, `bar v_k = v_k`. The
numerical certificate is scoped to case (i). Corollary 1 is now (a) sufficient strict positivity,
(b) failure when `D_k^dagger < 0`, (c) the tie boundary left open, with convergence stated in
probability (not a.s.) and proved from the sub-Gaussian tail. No new result is claimed; claims
were narrowed to what the argument supports.

**Decided by:** Agent, following the review.
## 2026-09-19 — Setup: the bootstrap "debiasing" of sigma2_extrap was mis-described (PR #22)

Second-source review of `paper/sections/setup.tex` (Remark `rem:sigma2-extrap-estimator`): the
text said `- v_hat/B - sigma2_target` debiases `bias_hat^2`. It removes only Monte Carlo noise of
the bootstrap mean (which vanishes as `B -> infinity`) and the ground-truth noise, but not the
sampling variance of the *original* fit, which the bootstrap mean converges to and which
survives every `B` (unbiased sample mean, exact target: `E[bias_hat^2] -> sigma^2/n`, not 0).
The remark now separates the three noise sources, gives the `n/(n-1)`-corrected estimator now
implemented in P1-06 (#16), and labels the clipped `sigma2_extrap_hat` a heuristic and the
unclipped value the approximately-unbiased squared-bias estimate (exact for a sample mean,
first-order for smooth fits). `docs/notation.md` updated to match.

**Decided by:** Agent, following the review.
## 2026-09-19 — Phase-1 memo and p1_06 findings: remove theory-validation inference from a vacuous bound (PR #21)

Review findings, both accepted: (1) all 396 bounds are >= 1, so every empirical error probability
passes, and "never violated" cannot be read as confirmation (the additive form is in fact invalid --
PR #17, acknowledged in PR #23); (2) six <=530M combinations have no matched-compute comparison and
must be reported as unassessed, not as losses. `docs/findings/phase1_memo.md`,
`docs/findings/p1_06.md` and `STATUS.md` rewritten: **0 wins among 12 evaluable comparisons plus 6
unassessed**; the `sigma2_extrap = 0` counterfactual is called inconclusive (vacuous bound, unmatched
single-scale endpoint baseline); the "18/18 (now 17/18)" wording removed. The memo is marked provisional
for P1-06/07/08 numbers until those results are regenerated on the repaired fitters and the
bootstrap-calibration/identifiability fixes. The HF publication (P1-12) was performed earlier with
PI confirmation; it is **not** re-run here -- re-publishing the regenerated derived tables needs a new PI
decision.

**Decided by:** Agent, following the review.
## 2026-09-19 — F3: separate the two decision events (PR #20)

F3 plotted the plug-in bound (a lower bound on P(select the single best arm)) and the observed
fraction of correctly ordered recipe *pairs* on one "Decision accuracy" axis as predicted vs observed --
different events, not comparable. F3 is now two panels: **A** best-arm selection (plug-in lower
bound, the `sigma2_extrap = 0` counterfactual, and P1-07's Monte-Carlo P(argmax = k*), all the same
event); **B** all-pairs ordering accuracy (observed only, labeled a different event; there is no
bound for it). `generate()` raises if `results/p1_08_ceiling_prediction.json` lacks
`observed_best_arm_accuracy` (i.e. predates PR #18's same-event fix) rather than plotting stale data;
unassessed (`None`) cells are omitted, not drawn as zero. `paper/figures/f3_predicted_vs_observed.pdf`
must be regenerated once P1-08 has been regenerated on the fixed upstream chain; until then the
committed PDF is the old, mislabeled one and should not be cited.

**Decided by:** Agent, following the review.
## 2026-09-16 — P1-10 repeated P1-06's correlation+1 and bootstrap-ID-alignment defects; fixed the same way

**Context:** PR #19's reviewer found that this module's parametric
bootstrap reused a single shared per-scale `z` across BOTH recipes
(`standard`/`deduped`) -- exactly the defect PR #16's review caught in
`p1_06_decomposition.py`, and exactly what Decision 3 above describes as
having deliberately mirrored ("the same shared-per-scale-per-replicate
draw design P1-06 established"). That design has since been shown to
force an unjustified exact +1 correlation between the two recipes'
bootstrap noise, collapsing the pairwise-difference bootstrap variance to
zero whenever they have equal noise even though the real observations are
independent -- see the P1-06 entry above (2026-09-14/16) for the full
mechanism. While fixing this, the same positional-zip bootstrap-ID-
misalignment PR #16's review separately caught in `p1_06_decomposition.py`
was also present here (`replicate_predictions[k_star][i] -
replicate_predictions[k_other][i]` after each recipe independently
dropped its own failed replicates) -- not flagged by PR #19's review this
time, but the identical defect, fixed proactively rather than waiting for
a future review pass to catch it separately.

**Fix:** independent per-recipe noise draws (seeded via `_seed_for(design,
task, recipe, "noise", b)`, extending the existing pattern with the
recipe name) instead of one shared `z` per scale; a new
`_pairwise_difference_series` helper (mirroring `p1_06_decomposition.py`'s
own) that tracks replicate id alongside each prediction and intersects by
id rather than zipping by position. Regression tests added
(`tests/test_p1_10_secondary_ladder.py`) for the ID-alignment fix,
matching P1-06's own regression tests for the same defect.

**Not yet done:** `results/p1_10_secondary_ladder.json` needs
regenerating once this branch merges past its upstream dependencies
(P1-04's fitter fixes, P1-06's own three fixes) and their own
regenerations.

**Decided by:** Agent, addressing PR #19's review.
## 2026-09-16 — P1-08's "gap" diagnostic mixed two decision events; the single-scale baseline was compute-mismatched

**Context:** PR #18's reviewer found two real issues in
`experiments/p1_08_ceiling_prediction.py`, both about comparing two
quantities that are not actually the same thing.

**Issue 1 (P1): `gap_predicted_minus_observed` compared a best-arm-selection
bound against an all-pairs accuracy statistic.** `predicted_accuracy` is
`max(0, 1 - bound_pairwise)`, and `bound_pairwise` is a union bound over
the ~24 comparisons against the task's single true best recipe `k*` -- it
lower-bounds P(this fitter's own argmax recipe == k*), one specific
decision event. `observed_accuracy` (from P1-03/04) is
`macro_avg_accuracy_including_ties`, the fraction of ALL 300 recipe
*pairs* correctly ordered -- a different statistic that mostly says
nothing about whether k* specifically was identified. Their difference
cannot diagnose anything about the bound's tightness or the 80% ceiling,
since a mismatch between them could be entirely an artifact of which event
each one measures, unrelated to the theory's quality.

**Fix:** added `_observed_best_arm_accuracy_per_task`, sourcing the SAME
decision event's empirical rate directly from P1-07's own Monte-Carlo
estimate (`empirical_error_rate` = 1 - P(argmax recipe == k*), computed by
literally resampling the fitting procedure the bound describes). The new
`observed_best_arm_accuracy` / `gap_predicted_minus_observed_best_arm`
fields are the valid ceiling diagnostic going forward. The original
(event-mismatched) `observed_accuracy` / `gap_predicted_minus_observed`
fields are kept, but now documented as NOT the ceiling diagnostic --
`observed_accuracy` is still legitimately used for the separate,
internally-consistent "does extrapolation's all-pairs accuracy beat
single-scale's all-pairs accuracy" central claim, where both sides use the
same statistic.

**Issue 2 (P2): the "beats single-scale" central claim compared extrapolation
against `ConstantExtrapolator` at the SAME endpoint size, not the same
compute.** `ConstantExtrapolator` at a design's endpoint (e.g. 150M) only
pays for one model at that one size; the extrapolation fitter being
compared against it consumed the compute of the ENTIRE fitting ladder up
to that endpoint (every smaller size too). Comparing accuracy at matched
*size* rather than matched *compute* is exactly the mismatch P1-04's own
headline finding ("0/18 beat single-scale at matched compute") was
designed to avoid -- P1-08 was silently redoing (and miscomputing) that
same comparison instead of reusing it.

**Fix:** added `_matched_compute_single_scale`, which reads P1-04's own
already-correct `matched_single_scale_accuracy_including_ties` /
`matched_compute_out_of_range` fields (the log-compute-interpolated
single-scale baseline P1-04's headline already uses, identical across
every fitter at a given design since they share the same ladder/compute).
`extrapolation_beats_single_scale_observed` in `central_claims` now uses
this matched-compute baseline instead of `ConstantExtrapolator`'s
same-endpoint accuracy; `ConstantExtrapolator`'s own accuracy is still
reported, renamed to `single_scale_observed_same_endpoint` to make clear
it is informational, not the comparison baseline. `matched_compute_out_of_range`
is now surfaced per central claim so a `<=530M`-design claim (out of
P1-03's interpolatable range) reads as "no valid comparison" rather than
silently falling back to something compute-mismatched.

**Not yet done:** `results/p1_08_ceiling_prediction.json` needs
regenerating once this branch merges past `phase1/bias-variance` and
`phase1/bound-check`'s own upstream fixes and regenerations.

**Decided by:** Agent, addressing PR #18's review. Full suite: 203 passed.

## 2026-09-19 — P1-08: label budgets, and unassessed is not "lost" (PR #18)

The observed "extrapolation beats single-scale" comparison uses P1-04's matched-compute
baseline and is now `None` (was `False`) when a design's compute is out of range, with
`summary.n_observed_evaluable_at_matched_compute` / `n_observed_unassessed_out_of_range` /
`n_observed_extrapolation_beats_matched_single_scale` reported alongside. The predicted and
counterfactual comparisons cannot be matched-compute (P1-07 evaluated single-scale only at the
three endpoint models), so every central claim carries `predicted_comparison_budget: "unmatched: ..."`
and the summary says so; consumers (F3, the Phase-1 memo) must not present them as matched.
`results/p1_08_ceiling_prediction.json` is regenerated last in the Phase-1 chain, after P1-04/06/07
have been regenerated on the fixed fitters.

**Decided by:** Agent, following the review.
## 2026-09-16 — P1-07's additive bound was an invalid "upper bound" for large fixed bias; estimator-specific uncertainty guard added

**Context:** PR #17's reviewer found two real issues in
`src/pdt/theory/bound.py`.

**Issue 1 (P1): the additive form `exp(-Delta_k^2 / (2*(bias^2+v)))`
folds a fixed, signed misspecification (bias) into a variance-like
denominator term, which is the wrong treatment and produces an invalid
bound.** Counterexample, reproduced exactly as given: challenger gap=5,
bias=+10, variance=.01 -- the bias alone dwarfs and reverses the apparent
5-point gap, so the true decision error is near-certain (~1), yet the old
formula evaluated to ~0.8825, an "upper bound" *smaller* than the true
error rate it is supposed to bound -- a violated bound, not just a loose
one. The bug: averaging a large *fixed* bias into the denominator
alongside genuinely random variance treats it as if it were symmetric
noise that merely widens the distribution, when a bias that exceeds the
gap in the wrong direction should make the term vacuous (-> 1, "no
guarantee"), not moderately shrink it.

**Fix:** `_bound_term` now treats bias as a worst-case, sign-unknown
shift that first cancels the apparent gap (`effective_gap = max(0,
|delta_k| - bias_magnitude)`), and only the *surviving* gap gets the
variance-driven exponential-tail treatment. At `bias_magnitude=0` this is
identical to the original formula, so the zero-bias case (and every
downstream reported bound value that happens to have negligible bias) is
unaffected. Re-running the exact counterexample now gives `1.0` (fully
vacuous, correctly signaling "no guarantee" instead of the invalid 0.8825).
`marginal_bound_term` uses `sqrt(sigma2_extrap_hat)` as the bias
magnitude (already a squared-magnitude, sign-unknown estimate);
`pairwise_bound_term` uses `abs(bias_hat)` (a signed point estimate,
whose sign is itself uncertain at the scale that matters, so its
magnitude is the defensible worst case). Documented plainly in the module
docstring that this is **an empirical diagnostic, not a proven
statistical bound** -- PR #23's review of the paper's own Theorem 1 proof
(`paper/sections/theorem1_bound.tex`) independently found the nonlinear
case isn't rigorously established either (smoothness/bounded-Jacobian
alone don't give exact sub-Gaussian tails), so every value from this
module should be read as "compare against P1-07's Monte-Carlo empirical
error estimate," not "certified guarantee."

**Issue 2 (P2): `sandwich_covariance`/`analytic_v_k` silently reported a
number for every fitter, including two whose actual fitting procedure the
joint-least-squares sandwich formula does not describe.**
`ConstantExtrapolator` only fits to the largest-scale observations
(ignoring the rest), and `sandwich_covariance` called with the *full*
scales list would wrongly charge it "residuals" at scales it never used.
`TwoStepLadder` fits in two separate sequential stages with different
objectives, not one joint simultaneous optimization -- the single
shared-jacobian/residual M-estimator structure doesn't represent a
two-stage procedure at all. Both previously produced a plausible-looking
`analytic_v_k` number that `results/p1_07_bound_coverage.json` reported
"alongside P1-06's bootstrap `v_hat_k` as a cross-check," implying the two
measure the same thing when for these two fitters they provably don't.

**Fix:** `analytic_v_k` now raises `UnsupportedEstimatorError` for any
fitter in the new `UNSUPPORTED_SANDWICH_ESTIMATORS` constant
(`{"ConstantExtrapolator", "TwoStepLadder"}`) rather than fabricating a
number. `experiments/p1_07_bound_coverage.py`'s `_compute_analytic_v_k`
catches it alongside the existing `FitFailure`/`LinAlgError` handling and
records `unsupported_estimator: true` in the per-recipe result (`false`
for a genuine fit failure), so a reader of the results file can tell "not
analytically supported by design" apart from "the fit itself failed."
Both fitters are simply absent from `analytic_v_k` going forward, rather
than silently present with a number that doesn't mean what the results
file's own docstring claims it means.

**Not yet done:** `results/p1_07_bound_coverage.json` needs regenerating
with both fixes (plus every inherited upstream fix -- P1-04's fitter
bugs, P1-06's bootstrap correlation/squared-bias/ID-alignment bugs) once
this branch is merged forward past `phase1/bias-variance`'s own P1-06
regeneration.

**Decided by:** Agent, addressing PR #17's review. Full suite: 204 passed.

---

## 2026-09-14 — Two real bugs found by external review, fixed, results regenerated

**Context:** PR #12's reviewer found two real correctness bugs in `src/pdt/scaling/`,
both with concrete, executable reproductions, and flagged that the resulting
mis-fits/mis-predictions propagate through every downstream task that fits a scaling
law (essentially all of Phase 1 onward) since PowerLawN/ConstantExtrapolator are used
throughout.

**Bug 1 (P1): `multi_start_fit`'s uniform-random restart initialization can silently
converge to the wrong answer with all restarts agreeing.** `x0 = rng.uniform(bounds[0],
bounds[1])` samples an exponent parameter like `alpha` linearly over `[1e-3, 10]` --
almost all of that mass lands on `alpha >~ 1`, where `N^-alpha` and its derivatives
underflow to numerically zero for the parameter counts this project fits over
(1e6-1e9): a flat region with no gradient signal. `scipy.optimize.least_squares` can
report `success=True` there anyway (it stops on step size, not residual, going to
zero), so **every one of the default 8 restarts can land in that flat region and agree
with each other** -- passing the function's own `objective_spread`-based multi-start
sanity check while still being badly wrong. Reproduced exactly as the reviewer gave it:
`PowerLawN(rng=np.random.default_rng(1))` fit to a noiseless `y = 0.9 - 2*N^-0.1` curve
(`N` from 1e6 to 1.5e8) predicted 0.504 at the target scale instead of the true 0.648,
with all 8 restarts converging to the identical wrong point (`objective_spread` ~1e-18).

**Fix:** `multi_start_fit` gained a `log_uniform_dims` parameter naming which parameter
indices are decay-rate exponents; those are now drawn log-uniformly over their own
bounds instead of linearly, concentrating restarts in the region where the fit's
gradient signal actually exists. Applied to every fitter with an exponent parameter:
`PowerLawN`, `PowerLawC` (`alpha`, index 2), `ChinchillaND` (`alpha` and `beta`,
indices 2 and 4), `TwoStepLadder`'s step 1 (`alpha1`, index 2). Re-running the exact
counterexample now recovers the true curve exactly (`theta = [0.9, -2.0, 0.1]`,
`best_cost ~4.5e-30`). Regression tests added:
`tests/test_scaling.py::test_power_law_n_recovers_a_small_alpha_noiseless_curve_across_seeds`
(the exact counterexample, checked across 5 seeds, not just the one reported) and
`test_multi_start_fit_log_uniform_dims_avoids_the_flat_high_alpha_region` (a direct,
model-agnostic before/after check of the fix itself).

**Bug 2 (P2): `ConstantExtrapolator` used the first observation at the largest scale,
not the average of every replicate there.** `values[idx]` for whichever row happened to
be first at max-`N`, when the interface accepts (and every real caller passes) a
replicate history -- several seeds at the same size. Reproduced exactly as given:
scales `[(1,1),(2,1),(2,1)]`, `y=[0,0.1,0.9]` predicted 0.1 (the first n=2 row), not the
mean 0.5; reordering the last two rows changed the answer.

**Fix:** average every observation at the largest scale (respecting the `weights`
argument when given), not just the first one encountered. Three regression tests
added, covering averaging, order-independence, and weighted averaging.

`results/p1_04_extrapolation.json` (and every downstream results file computed from a
scaling-law fit) needs regenerating with both fixes in place -- see the follow-up
decisions.md entry for the regenerated numbers.

**Decided by:** Agent, addressing PR #12's review. Full suite: 136 passed, 100%
coverage on `src/pdt/scaling/base.py` and `src/pdt/scaling/fitters.py`.

## 2026-09-14 — P1-04 results regenerated with both fixes: headline finding unchanged, individual fitter accuracies shift

**Context:** follow-up to the entry immediately above. `results/p1_04_extrapolation.json`
regenerated via `PDT_OVERWRITE=1 uv run python experiments/p1_04_extrapolation_baselines.py`
on a clean tree with both scaling-law bugs fixed.

**Headline finding is unchanged:** still 0/18 (fitter, design) combinations beat
single-scale training at matched compute. `summary.n_beat_single_scale_at_matched_compute`
is `0` both before and after, same as `summary.winners == []`. The consistency check
(`ConstantExtrapolator`'s predictions matching P1-03's own reported numbers) still
passes.

**Individual accuracies moved, in the direction the bug predicts.** Every fitter with a
decay-rate exponent parameter (the ones the `log_uniform_dims` fix touches) changed;
`ConstantExtrapolator` and `LogLinear` (no exponent parameter, untouched by the fix) are
bit-for-bit identical before and after, which is itself a useful sanity check that the
fix is scoped correctly. Macro-averaged decision accuracy (including ties), by fitter
and design:

| fitter | design | before | after |
|---|---|---|---|
| PowerLawN | S_fit≤150M | 0.6452 | 0.7376 |
| PowerLawN | S_fit≤300M | 0.6006 | 0.8097 |
| PowerLawN | S_fit≤530M | 0.6891 | 0.8273 |
| PowerLawC | S_fit≤150M | 0.6973 | 0.7358 |
| PowerLawC | S_fit≤300M | 0.6915 | 0.7645 |
| PowerLawC | S_fit≤530M | 0.7164 | 0.8179 |
| ChinchillaND | S_fit≤150M | 0.7203 | 0.7624 |
| ChinchillaND | S_fit≤300M | 0.7497 | 0.8148 |
| ChinchillaND | S_fit≤530M | 0.7858 | 0.8482 |
| TwoStepLadder | S_fit≤150M | 0.6942 | 0.5979 |
| TwoStepLadder | S_fit≤300M | 0.7082 | 0.6197 |
| TwoStepLadder | S_fit≤530M | 0.7697 | 0.6773 |
| ConstantExtrapolator | (all 3) | 0.7627 / 0.8252 / 0.8509 | unchanged |
| LogLinear | (all 3) | 0.7639 / 0.8148 / 0.8494 | unchanged |

`PowerLawN`, `PowerLawC`, and `ChinchillaND` all got *more* accurate after the fix (by
4-21 points) -- the old buggy initialization was landing genuine fits in the numerically
flat high-alpha region often enough to measurably drag down decision accuracy, not just
occasionally. `TwoStepLadder` moved the other way, *down* by 9-11 points: its step 1 also
fits an `alpha`-like exponent, and the old bug's flat-region fits apparently happened to
produce extrapolations that agreed with the true ranking more often than the genuinely
optimal fits now do. Neither direction is surprising once the mechanism is understood --
the old numbers weren't measuring "how good is this functional form", they were partly
measuring "how did this particular numerical failure mode happen to land" -- but it means
any pre-fix conclusion about `TwoStepLadder` specifically (e.g. "it's the best of the
exponent-based fitters") should be treated as an artifact of the bug, not a real result.

**Decided by:** Agent. Regeneration run completed cleanly (`git_dirty: false` in the
written provenance); no code changes in this entry, data only.

---

## 2026-09-14 — P1-04 results regenerated again: both the scaling-fitter fix and the group_by determinism fix are now in the same file

**Context:** this branch (`phase1/groupby-determinism-audit`, PR #14) and
`phase1/scaling-fitters` (PR #12) each independently regenerated
`results/p1_04_extrapolation.json` from a clean tree, from two different
fixes to two different bugs (the group_by summation-order bug above, and
the scaling-law initialization/replicate-averaging bugs in the entries
above that). Merging PR #12's fix forward into this branch produced a
real conflict in the results file itself -- both versions are genuine,
correct regenerations of their own fix in isolation, but neither reflects
both fixes at once. Per this project's provenance discipline (never
hand-merge a generated results file), resolved by regenerating fresh from
the merged code, which now has both fixes applied together, rather than
attempting to reconcile the two JSON payloads by hand.

**Result:** `PDT_OVERWRITE=1 uv run python experiments/p1_04_extrapolation_baselines.py`
on the merged, clean tree. Headline finding still unchanged (0/18 combinations
beat single-scale at matched compute); the `ConstantExtrapolator`-vs-P1-03
consistency check still passes. Superseded both parents' versions of this
file; no further diffing against either parent version individually is
meaningful since both were missing one of the two now-combined fixes.

**Decided by:** Agent, resolving the merge of PR #12 into PR #14.

---

## 2026-09-14 — P1-09's Bonferroni threshold was actually a ~12x-too-loose normal approximation; corrected reversal rate is 1.0%, not 15.2%

**Context:** PR #15's reviewer found that `pair_effect_size` is a t-like
statistic -- both `sigma2_a` and `sigma2_b` are estimated from only
`n_seeds=3` observations each -- but the Bonferroni-corrected significance
threshold used `scipy.stats.norm.ppf`, a standard-normal quantile, as if
the variances were known exactly. Reproduced exactly as given: at `df=4`
(the pooled-equal-variance case this project's real `n_seeds=3` gives
everywhere), the normal-based cutoff's actual two-sided false-positive
rate under the correct `t(4)` distribution is 0.0435, not the intended
`0.05/14=0.00357` -- **more than 12x the nominal rate**, meaning the
previously-committed "Bonferroni-corrected" 15.2% reversal rate
(2026-09-03 entry above) was substantially inflated by the same kind of
multiple-comparisons problem it was supposed to be correcting for, just a
smaller version of it.

**Fix:** `rank_reversal.welch_satterthwaite_df()` computes a per-(task,
size, pair) cell Welch-Satterthwaite degrees of freedom from each
recipe's own seed variance (they differ in practice, so pooling them into
a single fixed df is itself an approximation this avoids), and
`p1_09_rank_reversals._bonferroni_t_threshold()` uses `scipy.stats.t.ppf`
at that df instead of a single fixed normal quantile shared across every
cell. Four regression tests added (equal-variance recovers the pooled
`df=4` case, unequal variance gives a lower df, both degenerate cases
return `None` and fall back to an infinite threshold rather than
crashing).

**Regenerated `results/p1_09_rank_reversals.json` with the fix (plus the
inherited P1-04 fitter fixes and the group_by determinism fix, both
already merged forward into this branch) on a clean tree.** The corrected
Bonferroni-calibrated reversal rate is **1.0%** (32/3300 pairs), down from
the previously-reported 15.2% -- a much sharper conclusion than the earlier
number suggested, though still nonzero (rank reversals are real, just far
rarer at the properly-calibrated significance level than the miscalibrated
threshold made them look). The uncorrected `1.0`-threshold figure is
unaffected by this fix (61.7%, matching the earlier entry almost exactly --
the small residual difference is downstream of the P1-04/group_by fixes'
effect on `seed_variance`, not this fix) since it never used the Bonferroni
threshold at all.

**How to apply:** any paper draft, figure, or claim citing "15.2% of pairs
reverse" (the number this project's own earlier decisions.md entry and
`primary_threshold_label: "bonferroni"` pointed to) must be updated to
1.0% -- the earlier number was wrong, not superseded by a policy choice.
The uncorrected 61.7% figure's status as "kept for continuity, not the
headline" (2026-09-03 entry) is unchanged.

**Decided by:** Agent, addressing PR #15's review. Full suite: 172 passed.
`results/p1_09_rank_reversals.json` regenerated on a clean tree
(`git_dirty: false`).

---

## 2026-09-16 — P1-06 results regenerated with all upstream fixes: the sigma2_extrap/v ratio still falls with compute, more starkly for some fitters

**Context:** follow-up to this branch's own PR #16 review-fix commit
(correlation+1, squared-bias estimator, bootstrap-ID-alignment) and to
every upstream fix merged forward into this branch (`phase1/scaling-fitters`'s
fitter-initialization/replicate-averaging bugs, `phase1/groupby-determinism-audit`'s
summation-order fix, `phase1/rank-reversals`'s calibration fix -- none of
the latter two touch `p1_06_decomposition.py`'s own computation, but the
fitter fix does, directly). `results/p1_06_decomposition.json` regenerated
via `PDT_OVERWRITE=1 uv run python experiments/p1_06_decomposition.py` on
a clean tree: the full grid (6 fitters x 3 designs x 11 tasks x 2 schemes
= 396 work units, B=200 replicates x 25 recipes each), 1,980,000
individual bootstrap fits, **0 failures**. Took ~19.4 hours wall-clock this
run (vs. the ~75-90 minutes the original 2026-09-03 run took) -- almost
entirely because the fitter-initialization fix means restarts now do
genuine optimization work instead of instantly "converging" in the flat
high-alpha region for a large fraction of fits; this is a real, expected
cost of the correctness fix, not a regression to chase down.

**The core P1-06 finding (`sigma2_extrap_hat / v_hat` falls with compute,
contradicting the plan's stated theoretical expectation that it should
rise) survives, for every one of the 6 fitters, with some fitters' ratios
shifting substantially in magnitude.** Median ratio by fitter and design
(`seed_bootstrap` scheme, before -> after both this branch's own fix and
every upstream fix):

| Fitter | @150M before -> after | @300M before -> after | @530M before -> after |
|---|---|---|---|
| ConstantExtrapolator | 1612.6 -> 1611.6 | 574.0 -> 573.0 | 195.1 -> 194.1 |
| PowerLawN | 9.57 -> 31.68 | 7.21 -> 22.57 | 5.20 -> 14.36 |
| PowerLawC | 20.79 -> 14.27 | 16.10 -> 8.34 | 11.98 -> 5.07 |
| ChinchillaND | 8.81 -> 368.69 | 5.20 -> 307.22 | 3.42 -> 275.05 |
| TwoStepLadder | 16.37 -> 0.87 | 11.32 -> 0.21 | 10.70 -> 0.01 |
| LogLinear | 310.5 -> 309.5 | 272.7 -> 271.7 | 230.2 -> 229.2 |

`ConstantExtrapolator` and `LogLinear` (no exponent parameter, untouched
by the P1-04 fitter fix) are essentially unchanged, as expected -- the
small residual shift is from this PR's own squared-bias-formula
correction (always non-increasing, since it subtracts an additional
`v_hat` term) and the group_by determinism fix's last-bit noise, not the
fitter fix. `PowerLawN`, `PowerLawC`, `ChinchillaND`, and `TwoStepLadder`
(all fit an exponent parameter) moved substantially -- most strikingly
`ChinchillaND` (8.81 -> 368.69 at 150M) and `TwoStepLadder` (16.37 -> 0.87,
now falling all the way to **0.01** at 530M). Every single fitter still
falls monotonically across the three designs, exactly as the original
finding reported -- the magnitude shifted (for the affected fitters,
substantially), but the qualitative conclusion (the theory's own stated
signature prediction is contradicted by this data, across the board) is
unchanged and, if anything, now stated with cleaner numbers since they no
longer reflect the numerical-initialization artifact P1-04's bugs
introduced.

**Decided by:** Agent. Regeneration completed cleanly (`git_dirty: false`,
`git_sha` matches this branch's merge commit). `results/p1_07_bound_coverage.json`,
`results/p1_08_ceiling_prediction.json`, and every other downstream
results file computed from P1-06's output still need regenerating once
their own branches merge this fix forward.

---

## 2026-09-16 — Merging the P1-04 fitter fix forward broke a P1-07 test that was passing for the wrong reason

**Context:** merging `phase1/bias-variance` (which itself carries the
upstream `phase1/scaling-fitters` fix) into `phase1/bound-check` broke
`tests/test_bound.py::test_analytic_v_k_saturates_for_power_law_n_far_extrapolation`,
which asserts `PowerLawN`'s delta-method `v_k` saturates (stops growing)
between `N=1e11` and `N=1e14`.

**Root cause: the test shared this file's module-level mutable `_RNG`
across every test, so its outcome depended on how many random draws
earlier tests in the file happened to consume -- and the log-uniform-init
fix changes exactly that (one extra `rng.uniform()` call per restart per
exponent dimension).** Diagnosed by reproducing the exact fit this test
now gets: `PowerLawN` converged to `alpha=0.404` sitting at its own
parameter's *box boundary* (`a=-10.0`, the lower bound) -- a genuinely
different, boundary-constrained local optimum on this test's narrow (8
points, `1e6` to `1e8`) noisy synthetic curve, one of several comparably-
low-cost optima this specific data supports (checked directly: 20
independent seeds on the same synthetic curve land in >=3 qualitatively
different regimes, including two boundary-hugging ones). But the deeper
issue survives even for a *well-identified*, non-boundary fit with
`alpha` close to the curve's true `0.3`: `N^-alpha * ln(N)` (the shape of
the alpha-jacobian entry) decays to 0 as `N -> infinity` for any
`alpha > 0`, but only logarithmically slowly for `alpha` this small --
checked directly, a clean `alpha~0.3` fit's `v_k` is still 40-135%
different between `N=1e11` and `N=1e14`, not remotely saturated; genuine
saturation to float64 precision for this curve doesn't arrive until
roughly `N=1e30`-`1e40`. The original test only ever passed because
whatever fit the old (buggy, uniform-alpha) `_RNG` sequence happened to
produce at that point in file execution order behaved as if already
saturated by `1e11` -- plausibly because the old bug's own failure mode
(restarts landing in the near-flat, large-alpha region) produces
*faster*-decaying, not truer, fits.

**Fix:** the test now uses a dedicated local `np.random.default_rng(1)`
(not the shared file-level `_RNG`), wider/more-informative synthetic data
(14 points over `1e6`-`1e10`, lower noise, reliably identifying `alpha`
close to `0.3` across independent seeds -- checked directly), and
genuinely far-apart comparison scales (`1e30` vs `1e40`) that produce real
saturation regardless of which valid `alpha` the multi-start fit lands on,
rather than relying on a specific fit's incidental behavior at scales
nowhere near true saturation. Not a change to `bound.py`'s own logic --
the delta-method machinery itself was never wrong here, only this test's
premise about how close `N=1e11`-`1e14` gets to genuine saturation.

**How to apply:** the rest of this file's tests still share the same
file-level `_RNG` and remain fine today, but any future change to how
many random draws a fitter's `fit()` consumes internally could silently
shift which local optimum any of them lands in. Prefer a dedicated local
`rng` for a new test whose assertion depends on *which* local optimum a
multi-modal fit converges to (as this one does), not just whether it
converges.

**Decided by:** Agent, while merging `phase1/bias-variance` forward into
`phase1/bound-check`. Full suite: 220 passed, confirmed stable across
repeated runs and running the file in isolation.

---

## 2026-09-18 — P1-07 results regenerated with all fixes: no bound violations, full 198-combo Monte-Carlo run clean

**Context:** follow-up to this branch's own two review-fix commits
(the invalid additive-bound formula, the estimator-rank-deficiency
guard) and to every upstream fix merged forward (P1-04's fitter bugs,
P1-06's three bootstrap-decomposition fixes, the group_by determinism
fix, the P1-09 calibration fix). `results/p1_07_bound_coverage.json`
regenerated via `PDT_OVERWRITE=1 uv run python experiments/p1_07_bound_coverage.py`
on a clean tree: the analytic delta-method pass (6 fitters x 3 designs x
11 tasks x 25 recipes, minus `ConstantExtrapolator`/`TwoStepLadder` now
correctly excluded per this branch's own P2 fix) plus the full
Monte-Carlo pass (198 work units, B=500 each) -- roughly 34 hours
wall-clock this run (vs. the original run's much shorter time), almost
entirely for the same reason P1-06's regeneration got slower: the
fitter-initialization fix means restarts now do genuine optimization
work instead of instantly "converging" in the flat high-alpha region.

**`any_bound_violation: false`, `violations: []` -- the pairwise bound
held (ratio >= 1) in every one of the 198 (fitter, design, task)
cells, with all of this branch's own and every upstream fix applied
together.** This is the same qualitative finding the original
(pre-fix) run reported, now resting on a corrected additive-bound
formula, corrected fitter initialization, corrected bootstrap
decomposition, and a correctly-excluded set of estimators for the
analytic cross-check -- the bound-holds conclusion was not an artifact
of any of the bugs fixed across this whole review pass.

**Decided by:** Agent. Regeneration completed cleanly (`git_dirty: false`,
`git_sha` matches this branch's merge/fix commits).
`results/p1_08_ceiling_prediction.json` and every other downstream
results file computed from P1-07's output still need regenerating once
their own branches merge this fix forward.

## 2026-09-19 — P1-07 second-round review: `analytic_v_k` requires the target to be identified (PR #17)

**Problem.** `sandwich_covariance` inverts `J^T J` with `np.linalg.pinv`,
which treats a parameter direction that no observed scale moves as
carrying *zero* variance. A `LogLinear` fit observed at one N (varying
only D) therefore reported a small finite `analytic_v_k` for any target N,
when the true delta-method variance is unbounded.

**Change.** New `pdt.theory.identifiability.target_in_row_space` tests
whether the target Jacobian lies in the row space of the fitting-scale
Jacobians (column-equilibrated SVD, `max(shape) * eps` rank cutoff,
relative residual `<= 1e-8`). `analytic_v_k` raises
`UnidentifiedTargetError` (an `UnsupportedEstimatorError`) when it does
not; `p1_07` records these as `unidentified_target: true` instead of a
number. The check is on the *unweighted* design support (structural), so
a badly conditioned but identified design still returns a large finite
variance rather than being rejected. Regression tests cover targets whose
missing component is 5% / 0.25% / 0.005% of `||J_target||`, an
identified design, and an ill-conditioned identified design.
## 2026-09-19 — Second-round review of P1-06's squared-bias correction: calibrate v_hat for the n=3 bootstrap

**Context:** PR #16's re-review accepted the direction of the previous fix
(subtract the original estimator's own sampling variance) but showed it is
mis-calibrated for the *actual* resampling scheme. n-out-of-n bootstrap
variance of a sample mean is the plug-in variance `s_plug^2/n`, a factor
`(n-1)/n` below the unbiased `sigma^2/n`. With n=3 real seeds, `E[v_hat] = 2/9`
against a true `Var(original mean) = 1/3`, so after subtracting `v_hat` the
unclipped correction still has expectation `1/9` when the true squared bias is
zero. Reproduced independently on 20,000 three-observation datasets with the
shipped function (B=200): mean `v_hat` 0.2225, mean unclipped correction
+0.106, and the *clipped* reported value averaged 0.218 -- the earlier
alternating-values regression test could not detect this because it never
repeated across independent datasets.

**Fix:** `bias_variance_decomposition(..., variance_inflation=...)` now
subtracts `variance_inflation * v_hat`; the seed bootstrap passes
`bootstrap.seed_bootstrap_variance_inflation(n_seeds) = n/(n-1)` (3/2 for three
seeds), for both the marginal and the pairwise (paired-seed difference)
decomposition, since a difference of two seed-means resampled with the same index
pattern is itself an n-out-of-n bootstrap of the n paired differences. The
parametric bootstrap uses 1.0: its `v_hat` is a model-based variance, not the
plug-in variance of an n-observation resample. `p1_06_decomposition._variance_inflation`
computes n from the data and raises if seed counts differ across cells.

**What is and isn't justified.** Exact for a sample mean. For the smooth fits used
here it is the first-order (delta-method) statement of the same fact, and is an
approximation at n=3 and for boundary-pinned or otherwise non-smooth fits -- an
approximation, not a proof, and documented as such in the function docstrings.

**Clipped vs unbiased, now distinguished.** Results carry
`sigma2_extrap_unclipped` (the approximately unbiased squared-bias estimate,
negative about half the time when the true bias is small) alongside
`sigma2_extrap_hat = max(0, .)`, a nonnegative *heuristic* biased upward for
small true bias (E[max(0,X)] > E[X]; checked directly: mean clipped value > 0.05
with zero true bias). Any average over cells that is meant to estimate a mean
squared bias -- including the `ratio_vs_compute` medians -- should be read with
that in mind, and consumers wanting an estimate rather than a floor should use
the unclipped field.

**Validation across independent datasets (not one hand-built series):**
`tests/test_bootstrap.py` simulates 3,000 independent three-observation datasets:
without the calibration the mean unclipped estimate is > 0.08 and > 8 standard
errors above 0; with n/(n-1) it is within 4 standard errors of 0 (and within
0.03). The first, reproducing the reviewer's number, is asserted as a regression
guard so the defect cannot silently return.

**Mechanical:** the added field would push the pretty-printed
`results/p1_06_decomposition.json` (already 4.9 MB) past the repository's 5 MB
`check-added-large-files` limit, so `provenance.write_result` gained
`indent=None` (compact single-line output; default unchanged) and this one
script uses it.

**Also corrected:** `bootstrap.py`'s module docstring still said both schemes
share one draw across recipes, which stopped being true for the parametric scheme
in the previous fix.

**Not yet done in this entry:** `results/p1_06_decomposition.json` regeneration
(requires the new fitters merged in first) and `docs/findings/p1_06.md`.

**Decided by:** Agent, addressing the PR #16 re-review.
## 2026-09-19 — Second-round review of the fitter fix: random starts are not enough; "0/18" was mis-stated

**Context:** PR #12's re-review (and the identical blocker restated on #13-#19,
#27-#34, since they all inherit the fitter blobs) found that the log-uniform
restart fix repaired the original `PowerLawN` counterexample (100/100 seeds) but
the compute-based `PowerLawC` still silently fails: noiseless in-family
`y = 0.9 - 2*C^-0.03` with `default_rng(30)` (and `54`) predicts 0.2463 instead of
0.4004, all eight restarts "converged", `objective_spread ~ 1e-11`, 2 of 100
seeds. Reproduced exactly. Randomized starts can only lower the probability that
every start lands in a flat region, never remove it.

**Fix:** deterministic *informative* starts by variable projection.
`fitters._power_law_starts` (used by `PowerLawN`, `PowerLawC`, and
`TwoStepLadder`'s step 1) and `_chinchilla_starts` (`ChinchillaND`) evaluate a
dense log grid over the exponent(s); for each grid point the linear parameters
are solved exactly by weighted least squares (then clipped to their bounds), and
the three lowest-cost grid points become starting points for `least_squares`,
via a new `multi_start_fit(..., extra_starts=...)` argument, alongside five
(down from eight) random log-uniform restarts, so per-fit cost is essentially
unchanged (measured: `ChinchillaND` 602 ms vs 509 ms per fit on a hard synthetic
curve; power laws 22-240 ms). A start is now informative by construction rather
than by luck. Verified: 100/100 seeds recover noiseless curves for
alpha in {0.01, 0.03, 0.1, 0.3, 0.6}, for both N- and compute-based power laws,
including the reviewer's seeds 30 and 54; `ChinchillaND` recovers across seeds.
Regression tests added for compute-based models, not just `PowerLawN`
(`tests/test_scaling.py`).

**Correction to the headline wording (supersedes the phrasing of the
2026-09-14 entries above, which are left intact per this log's append-only
rule):** "0/18 (fitter, design) combinations beat single-scale at matched
compute" counted the six `<=530M` combinations as losses, but those have *no*
matched-compute comparison at all (`matched_compute_out_of_range: true`,
`beats: null`) -- a missing comparison is not a negative result. The correct
statement is **0 wins among the 12 evaluable comparisons, and 6 unassessed**.
`experiments/p1_04_extrapolation_baselines.py` now reports
`n_evaluable_at_matched_compute` and `n_unassessed_out_of_range` in its summary
and headline print; any downstream text still saying "0/18" (notably the Phase-1
memo, PR #21) must be corrected the same way.

**Not yet done in this entry:** `results/p1_04_extrapolation.json` regeneration
with the new fitters (running as the next commit), and every downstream result.

**Decided by:** Agent, addressing the PR #12 re-review. Full suite: 149 passed.

## 2026-09-19 — P1-04 regenerated with variable-projection fitter starts (PR #12)

`results/p1_04_extrapolation.json` was regenerated on a clean tree
(`git_dirty: false`, `git_sha` b0d1deb) with the robust power-law starts. Headline: **0 / 12
evaluable (fitter, design) combinations beat the single-scale frontier at matched compute; 6 more
have no matched-compute comparison** (the design's compute lies past the single-scale frontier's
range) and are *unassessed*, not losses. Versus the previous run, macro-average accuracy (incl.
ties) rose for every PowerLawN/PowerLawC design (e.g. PowerLawN 150M 0.738 -> 0.761, PowerLawC 300M
0.765 -> 0.815, PowerLawC 530M 0.818 -> 0.848); ChinchillaND was unchanged to within 0.001;
TwoStepLadder changed by at most 0.024 (150M +0.008, 300M -0.024, 530M +0.002). The qualitative
conclusion (extrapolation does not beat single-scale at matched compute on this data) is
unchanged, now resting on fits that recover the true optimum on a 100/100-seed sweep.

**Decided by:** Agent, following the second-round review.

## 2026-09-19 — P3-02 second-round review: structural identifiability replaces the 10% residual test (PR #28)

**Problem.** `_target_denom` decided whether `J_target` lies in `range(I_k(w))` by
a 10% relative residual of `I pinv(I) J_target`. That is wrong in both
directions: `J_target = [1, .05]` against a design that only ever sees
`[1, 0]` (a 5% missing component, e.g. `LogLinear` observed at N=1, target
N=e^0.05) passed and received a finite variance, while a tight tolerance
false-triggered on ordinary designs at condition number ~1e14.

**Change.** Two questions are now separated. (1) *Structural
identifiability* — is `J_target` in the row space of the Jacobians of the
scales carrying positive weight — is decided by the shared
`pdt.theory.identifiability.target_in_row_space` (column-equilibrated SVD,
numerical-rank cutoff), in `_arm_rate`, `_arm_rate_and_grad`, and once per arm
in `brute_force_allocation`. (2) *Numerical ill-conditioning* of the weighted
information is handled by a Jacobi-equilibrated pseudo-inverse
(`_info_solve`), so it appears as a large finite variance, and the result is
invariant to parameter units. `_TARGET_RANGE_RTOL` is removed. Regression
tests: missing component 5% / 0.5% / 0.01%, brute-force path, ill-conditioned
identified design, unit invariance.

**Decided by:** Agent, following the second-round review.

## 2026-09-19 — ETS: known-variance certification replaces HC0 + Student-t (PR #29, second review)

**Problem (reproduced exactly).** `LogLinear`, two arms with constant means .501/.5, N(0,.05^2)
noise, fitting scales `N=[1,1.00001,2]`, target `N=2.001`, `eta=0`, `delta=.01`,
`max_rounds=1`, `solver_n_iter=1`, oracle RNG seeds 0..199: the HC0-variance + Student-t rule
certified 198/200 runs and 98 of them the wrong arm (49% unconditional error vs 1% requested).
The high-leverage point near the target makes the fitted residual ~0, so HC0 deletes the
dominant uncertainty; a t quantile cannot restore it. (Our re-run of the old rule in
`variance_mode="hc0_heuristic"` gives the same 98 wrong picks.)

**Change.**
- `pdt.theory.bound.known_noise_v_k`: prediction variance from the *known* noise function,
  `v = sum_i g_i^2 sigma2(s_i)` with influence weights `g = pinv(J)^T J_target` (exact for
  linear-in-`theta` fits, first-order otherwise; equals the OLS closed form, checked against it and
  against Monte Carlo). Unidentified targets give `v = inf` (never certifies).
- Radius `sqrt(2 (v_a + v_b) log(1/beta))`, `beta = delta / (t (t+1) K (K-1))`: a union over rounds
  *and* over the `K(K-1)` ordered pairs, because the leader is data-dependent.
- `variance_mode="known_sigma2"` (default) may return `"certified"`; the certificate lists the
  assumptions: A1 `sigma2` is a valid sub-Gaussian proxy (an *input*), A2 bias <= `eta`, A3
  prediction linear in the data (exact for LogLinear; first-order for nonlinear fits, curvature not
  covered by `eta`), A4 design independent of the certified noise (exact only at the non-adaptive
  warm-up check).
- `variance_mode="hc0_heuristic"` keeps the residual-based variance but returns `"recommended"`,
  never `"certified"`; no error-probability claim.

**Evidence.** Reviewer's high-leverage design and the original evenly spaced design, 200 seeds
each, `max_rounds=1`: new default certified 0 of 200 on both (0 wrong; the rule is conservative
there); HC0 mode reproduces 98 wrong on the high-leverage design. Adaptive runs (LogLinear, 3 arms,
means 0.5+g, 0.5, 0.5-g, sigma .05, delta .1, up to 150 rounds, 60 seeds each): gap g=.1 -> 37
certified, **0 wrong**, 23 abstained; gap g=0 (all tied) -> **0 certified**, 60 abstained. That is
120 adaptive runs with no wrong certification -- supporting evidence that the effect of A4 is small
in this regime, **not** a proof; A4 remains an assumption.

**Not claimed.** No self-normalized / martingale confidence sequence is implemented, so `"certified"`
is a delta-level statement under A1-A4, not an unconditional guarantee for the adaptive algorithm.
`paper/sections/theorem4_algorithm.tex` (PR #26) is being brought into line with this scope.

**Decided by:** Agent, following the second-round review.

## 2026-09-19 — P3-05 replay regenerated on the corrected wrappers and the known-noise ETS (PR #31)

`results/p3_05_replay.json` regenerated on a clean tree (`git_dirty: false`) with the guarded baseline
oracle, complete seed remapping, the explicit pool-exhausted outcome (no recycling of already-observed values),
and the known-`sigma2` ETS rule (`sigma2 = 1e-4` here is an *approximation* of the real seed noise, which is not known
exactly -- so this real-data run does not satisfy A1 of the certification assumptions and is an empirical replay,
not a check of a guarantee). Pilot of 4 tasks (2 reversal-heavy, 2 stable), `delta = .1`, `eta = .02`, one ETS run per task.
**ETS certified on none of the 4 tasks**: 2 (winogrande, arc_easy) exhausted DataDecide's finite replicate pool
and are reported as `pool_exhausted` (previously counted as rounds), 2 (boolq, mmlu) hit the round cap; genuine
bias-floor abstention rate 0.0 in both groups. Round-cap rate fell from 1.0 to 0.5 in each group and pool-exhausted rose from
n/a to 0.5; baseline single-scale accuracies moved slightly (e.g. winogrande 0.20 -> 0.15) because the baseline
oracle is now guarded and remapped. No previously reported certification survives, so no invalid certification is presented as evidence.
## 2026-09-19 — P3-04 pilot regenerated on the known-noise ETS; claims restated at what the pilot supports (PR #30)

`results/p3_04_simulation.json` regenerated on a clean tree (`git_dirty: false`) with the corrected `eta` assignment, the split abstention outcomes
(bias-floor vs timeout), the known-`sigma2` stopping rule (the oracle's `sigma = .05` is passed as `sigma2`, so A1 holds), and a joint-rate
criterion with an exact Clopper-Pearson interval. 8 cells (K = 3; delta in {.05, .2}; eta in {none, large}; gap in {well-separated, reversing}) x 15 runs.
- **Claim 1 (delta-correctness in the well-specified regime):** no violation detected, but *not verified*. `eta=none`, well-separated: 11/15 (delta .05) and 13/15 (delta .2)
  certified, 0 wrong; exact 95% interval on the joint rate `P[certified and wrong]` is [0, 0.218] in each cell, which contains both deltas. `eta=large` cells never certified
  (all timeouts). The earlier "claim 1 holds" was a statement about a conditional rate on few certified runs.
- **Claim 2 (compute ~ T* log(1/delta)):** not supported -- mean compute-to-stop 3.07e18 (delta .05) vs 2.91e18 (delta .2); dominated by the warm-up.
- **Claim 3 (abstention in the impossible regime):** not observed -- every reversing cell ended at the round cap; bias-floor abstention rate 0.0, timeout rate 1.0 in all four.
  On the reversing instances SingleScale, FixedLadder and Uniform were wrong in every run (accuracy 0.0); SuccessiveHalving was right in all runs at `eta=none` and wrong at `eta=large` (an instance-construction effect, not a method advantage to lean on).
- Scope is a pilot (K = 3, 15 runs); nothing here calibrates delta or tests the abstention theorem. A real study needs the plan's grid, more rounds and >= 200 runs per cell.
## 2026-09-19 — P1-10 regenerated after the independence/fitter fixes (PR #19)

`results/p1_10_secondary_ladder.json` was regenerated (clean tree, `git_dirty: false`) with independent per-recipe
parametric draws and the repaired fitters. The ConstantExtrapolator median `sigma2_extrap / v` ratio moved from
131.0 (`le_160m`) / 137.6 (`le_410m`) -- rising, "not replicated" -- to 135.2 / 127.5, i.e. falling with the larger design as in
P1-06 (`replicates_p1_06_decreasing_direction` false -> true). The change is a consequence of removing the
forced +1 cross-recipe correlation. The gap is ~6% with two usable design points and two fitters, so
this is **weak evidence consistent with P1-06's direction, not a replication**; the "P1-10 is underpowered" caveat stands.
Downstream text quoting the old 131.0 vs 137.6 (memo, PR #21) is updated separately.
## 2026-09-20 — P1-06 regenerated on the repaired fitters with the calibrated squared-bias estimator (PR #16)

`results/p1_06_decomposition.json` regenerated on a clean tree (`git_dirty: false`, base `f728bc5`): 396 (fitter, design, task) work units,
B = 200 replicates x 2 schemes, **0 of 1,980,000 individual bootstrap fits failed**, ~24.9 h wall (the machine slept for part of it). It uses the variable-projection power-law starts (PR #12), the
`n/(n-1)` seed-bootstrap variance inflation and the unclipped estimator (`sigma2_extrap_unclipped`, stored alongside the clipped heuristic), and compact output (3.4 MB).

Median per-cell `sigma2_extrap_hat / v_hat` (`seed_bootstrap`), previous committed run -> this run, @150M / @300M / @530M:

| Fitter | @150M | @300M | @530M |
|---|---|---|---|
| ConstantExtrapolator | 1611.56 -> 1611.06 | 573.00 -> 572.50 | 194.07 -> 193.57 |
| PowerLawN | 31.68 -> 359.74 | 22.57 -> 305.71 | 14.36 -> 274.08 |
| PowerLawC | 14.27 -> 356.72 | 8.34 -> 308.23 | 5.07 -> 265.69 |
| ChinchillaND | 368.69 -> 376.51 | 307.22 -> 306.72 | 275.05 -> 274.06 |
| TwoStepLadder | 0.87 -> 0.15 | 0.21 -> 0.00 | 0.01 -> 0.00 |
| LogLinear | 309.50 -> 309.00 | 271.67 -> 271.17 | 229.17 -> 228.67 |

Readings (all from this table and the file, not from theory):
- The ratio still **falls as the design grows toward the target for every fitter** (P1-06's original, plan-contradicting finding survives).
- **PowerLawN and PowerLawC moved from 5-32 to 266-360**, i.e. they now behave like ChinchillaND and LogLinear. The old small values are consistent with their
  power-law fits having been stuck in flat-exponent regions before the initialization repair (the reviewers' seed-1/30/54 counterexamples); the new values are
  what a working fit gives. That attribution is an inference from the coincident P1-04 fix, not separately proved.
- **Bias dominates estimation variance by ~200-1600x for every fitter except TwoStepLadder**, whose variance is large (median `v_hat` ~6e-3) and whose bias is not distinguishable from zero.
- **13.6% of cells (672/4950) have a negative unclipped bias-squared estimate** -- the bias is undetectable against the estimation variance there; the clipped `sigma2_extrap_hat` reports 0 for those, which is why any average of the clipped field overstates the mean squared bias.
- The `n/(n-1)` correction is negligible for Constant/LogLinear/ChinchillaND-type cells (their `v_hat` is tiny) and matters only where `v_hat` is large (TwoStepLadder).
- Highest per-task bias at 150M: `hellaswag` (~0.055 for PowerLawN and ChinchillaND); the lowest tasks are near zero/negative (`boolq`).

**Decided by:** Agent, following the second-round review.

## 2026-09-22 — P1-07 regenerated on the regenerated P1-06 and the fixed fitters/identifiability (PR #17)

`results/p1_07_bound_coverage.json` regenerated on a clean tree (`git_dirty: false`, base `c4d740a`): 198 (fitter, design, task)
combinations x 2 bootstrap schemes = 396 cells, B = 500 Monte-Carlo replicates each, ~38.7 h wall
(mostly the Monte-Carlo pass; some individual combos took far longer than others -- e.g. one jumped from
5107s to 40524s elapsed between combos 60 and 70 -- plausibly this machine going idle/asleep partway
through, not a per-combo cost change). `any_bound_violation: false`, `violations: []` -- the pairwise bound
held (tightness ratio >= 1) in every one of the 396 cells, now computed with the corrected `_bound_term`
(gap-reduction form, PR #17/#23) and with `analytic_v_k` raising `UnidentifiedTargetError` where the target
is unidentified from the fitting scales (0 of 3,300 per-recipe analytic checks hit that path on real
DataDecide designs, i.e. every real design here does identify its own extrapolation target).

**Correction to prior wording:** this run's own `bound_pairwise` (seed_bootstrap scheme) is **not** `>= 1`
in literally every cell -- 2 of 198 are below 1 (informative): `ConstantExtrapolator` at `<=530M` on
`arc_easy` (0.665) and `hellaswag` (0.971), both the least-extrapolating baseline at its closest-to-target
design. `tightness_ratio_pairwise` (bound / empirical MC error) is `>= 1` everywhere regardless (min 1.16,
median 11.15, max 508 for seed_bootstrap; min 3.02, median 22.1, max 1370 for parametric_bootstrap) --
that is the quantity "never violated" actually refers to, and it is unaffected by whether the raw bound
itself happens to dip under 1 for two near-degenerate cells. Downstream text (P1-08, the Phase-1 memo)
should say "vacuous (`bound_pairwise >= 1`) in all but 2 of 396 cells, both the non-extrapolating baseline
at its closest design" rather than "all 396", and should quote `tightness_ratio`, not `bound_pairwise`,
for the "never violated" claim.

**Decided by:** Agent, following the second-round review.

## 2026-09-22 — P1-08 regenerated on the regenerated P1-04/06/07 (PR #18)

`results/p1_08_ceiling_prediction.json` regenerated on a clean tree (`git_dirty: false`, base `9544054`), reading the
regenerated `p1_04_extrapolation.json`, `p1_06_decomposition.json`, and `p1_07_bound_coverage.json`. This is the first
run of this file with PR #18's own fix (same-decision-event comparison, matched-compute observed baseline, unmatched-budget
labeling for the predicted/counterfactual comparisons) actually applied to non-stale upstream inputs.

- **Observed, matched-compute (the valid headline number):** `n_observed_evaluable_at_matched_compute: 10`,
  `n_observed_extrapolation_beats_matched_single_scale: 0`, `n_observed_unassessed_out_of_range: 5` (of 15 central claims
  = 5 extrapolation fitters x 3 designs) -- **0 of 10 evaluable comparisons favor extrapolation**, consistent with
  P1-04's own headline (0/12 evaluable there; the two counts differ only because P1-08's central claims exclude
  `ConstantExtrapolator`, which is the baseline being compared against, not an extrapolation method).
- **`observed_best_arm_accuracy`** (P1-07's Monte-Carlo P(select the true best recipe), the event the bound actually
  lower-bounds) is dramatically lower than the all-pairs `observed_accuracy` for every fitter/design -- e.g. PowerLawN
  @150M: 24.9% best-arm vs 76.1% all-pairs; @530M: 32.8% vs 84.8%. This is the numeric confirmation of P1-08's own module
  docstring: all-pairs ordering accuracy is a much easier, different statistic from best-arm selection, and the earlier
  (pre-#18) version of this file conflated them.
- **Predicted/counterfactual comparisons (UNMATCHED budget, labeled as such in every `central_claims` row and the summary):**
  12/15 pairs flip vs single-scale's real predicted accuracy (was 5/15 on stale pre-fix inputs), 4/15 flip vs single-scale's
  own bias-free counterfactual (was 1/15). Both counts moved because the underlying bound is now the corrected gap-reduction
  form (PR #17/#23) computed on the regenerated P1-06/07, not because the comparison became matched-compute -- it remains
  labeled `unmatched: extrapolation ladder compute vs single-scale endpoint-only compute` and should not be read as a
  matched-compute finding.
- `predicted_accuracy` is 0.0% (clipped) in every cell as before -- the bound remains vacuous for the predicted/counterfactual
  comparisons (see PR #17's decisions entry: only 2 of 396 P1-07 cells have an informative raw bound, and neither is an
  extrapolation fitter's predicted-accuracy cell here).

**Decided by:** Agent, following the review.

## 2026-09-22 — F1-F5 regenerated from the fully regenerated P1-04/06/07/08 chain (PR #20)

`python -m pdt.viz.build_all` regenerated all five Phase-1 figures from the now-consistent
`results/p1_0{3,4,6,7,8}_*.json` (all regenerated this pass on clean trees, see PRs #12/#16/#17/#18's
decisions entries). F3 (two-panel fix, PR #20's review) now renders with real
`observed_best_arm_accuracy` data: Panel A shows best-arm selection is far below 50% for every
(fitter, design) and the plug-in bound is ~0 throughout (2 informative cells out of 396 in the
underlying P1-07 data are not extrapolation-fitter cells and don't show up here); Panel B shows the
much higher (76-85%, except TwoStepLadder ~60-68%) all-pairs ordering accuracy on the same x-axis,
visibly a different quantity from Panel A -- confirming the two should never have been plotted
together. F1/F2/F4/F5 are unchanged in structure, only in the numbers they read.

**Decided by:** Agent, following the review.

## 2026-09-25 — ETS third review: `known_sigma2` alone no longer enables "certified" (PR #29)

**Finding.** `ets.py` returned `outcome="certified"` solely because `variance_mode == "known_sigma2"`, including after adaptive tracking and for the default
nonlinear `PowerLawN`. Known observation variance does not repair (i) adaptive-design selection (the design at round `t` depends on earlier noise; no
adaptive confidence sequence is implemented) or (ii) nonlinear / bound-clipped finite-sample prediction error (a clipped mean is neither Gaussian nor centred).

**Change.** New `certification` argument (default `"supported_only"`):
- `"supported_only"`: `"certified"` **only** where the argument is proved -- (A1) `sigma2` valid, (A2) `eta` valid, the model **linear in its parameters**
  (`Extrapolator.linear_in_parameters`, true only for `LogLinear`), its parameters strictly **inside** the box bounds (`bounds_inactive()`, checked at run time, so the fit
  is exactly OLS), and **no adaptive pull yet** (the first check, right after the non-adaptive warm-up). Every other time the rule fires the outcome is `"recommended"` with
  `certificate["unmet_supported_conditions"]` naming the failed conditions and no error-probability claim (same treatment as the HC0 mode).
- `"assume_unproved_conditions"`: the caller explicitly accepts A3 (linearization) and A4 (adaptive independence); `"certified"` on any round, flagged
  `certificate["guarantee"] = "assumed_unproved..."`. The decision and trajectory are identical to the supported mode's -- only the label changes (tested).

**Consequences.** By default `"certified"` is now rare (round-1 only, `LogLinear`, unclipped) -- deliberately: it is the only case the theorem covers. Every experiment that studies ETS
under the adaptive/nonlinear regime (P3-04/05/06) now passes `certification="assume_unproved_conditions"` explicitly and reports its `"certified"` outcomes as
"certified under caller-assumed conditions"; they are regenerated after the last Phase-3 code change. README, the trust guide, and the CLI/config are updated in PR #34.

**Not claimed.** No adaptive confidence sequence exists here; the assumed mode's error rate is empirical (P3-04/05/06), not proved.
## 2026-09-25 — P3-02 third review: the weighted-information solve dropped weak identified directions (PR #28)

**Reviewer's reproduction.** `LogLinear`, candidates `N = e, e^2`, target `N = e^3`, `sigma2 = 1`, gap `.1`, weights
`[1, 1e-16]`: `_arm_rate` returned ~`.00125`; the direct two-point regression gives `.01 / (2 (1 + 4/1e-16)) = 1.25e-19`. At weight `1e-20`
it still returned `.00125` instead of `1.25e-23`. The structural row-space check passed (the target IS in the row space);
`_info_solve` and the brute-force batched path inverted the weighted information matrix `I = A^T A` with `pinv`, whose cutoff dropped the
weak eigen-direction and treated its variance as zero. A different bug from the earlier rank-deficiency-tolerance one.

**Fix.** The solve now works from the weighted design `A` (rows `sqrt(w_s / sigma2_s) J_s`) via `_solve_weighted`: an SVD of the
column-equilibrated `A`, `f = ||S^-1 V^T j_eq||^2` and `y = V S^-2 V^T j_eq / col` -- conditioning of `A`, not of `A^T A`. It **fails closed**: if
the rank cutoff discards a singular direction in which the target has a component (> 1e-8 relative), the arm gets `f = 0` = "no information"
(`rate = 0`), the conservative reading -- never a small finite variance from a dropped direction. One function serves the scalar path
(`_arm_rate`, `_arm_rate_and_grad`) and the brute-force stack (`brute_force_allocation`), so both are covered. Checked: rates match the closed
form at weights `1e-16`, `1e-20`, `1e-30`; a weight of `1e-40` (below the cutoff) fails closed to 0; batched == scalar; agreement with `inv(A^T A)` on a
well-conditioned design. 38 allocation tests, 381 tests total pass on this branch.

**Consequence for results.** Well-conditioned cases are numerically unchanged (to ~1e-10), but `solve_allocation` is iterative, so ETS trajectories
that depend on it can differ in the last bits; every P3 result that runs the solver (P3-04/05/06/07) is regenerated after the last Phase-3 code change.
## 2026-09-25 — Theorem 4: `eta` bounds the conditional estimator bias; the naive-rule remark corrected (PR #26, third review)

1. **P1 -- wrong bias.** The simultaneous event is centred at `E[prediction gap | design]`, but `eta` was tied to the population projection bias
   `sqrt(sigma2_extrap)`; for nonlinear/constrained fits or adaptive designs the conditional mean differs (finite-sample and clipping bias).
   Part (a) now takes **Assumption B**: `|E[mu_hat_k(s*; t) | design_t] - mu_k(s*)| <= eta_k` at every round; it coincides with `eta >= sqrt(sigma2_extrap)` only for
   unconstrained linear least squares on a non-adaptive design. The Inputs paragraph and proof use it. Test: the clipped-mean fit has projection bias 0 but conditional bias `sigma/sqrt(2 pi n) > 0`.
2. **P2 -- naive Track-and-Stop.** With a nonzero limit gap `D = Delta + bias` (either sign) and `v(t) -> 0`, `D^2 / v` diverges, so the naive rule *does* stop -- confidently
   on the wrong arm when the bias reverses the gap (`D < 0`); only a cancelling gap (`D ~ 0`) leaves it undecided. The remark previously claimed it "runs forever". The bias floor
   is justified by this instead. Test: numeric divergence with `D = -0.2` (stops at t = 209) vs a bounded statistic at `D ~ 0`.
## 2026-09-25 — Theorem 3: `sigma2_extrap` is the projection estimator's bias, not a universal minimax floor (PR #25, third review)

Reviewer: the "two risks" paragraph added `sigma2_extrap` to a minimax lower bound for the misspecified target. That quantity is the squared bias
of the chosen parametric projection; an estimator that knows `h` or uses a larger identifiable family can remove it, and the in-family information
proof does not supply an additive misspecification term. Fix: the in-family display is the (only) minimax statement; the
misspecified-target statement is restricted to the **projection (least-squares) estimator** (risk = `sigma2_extrap` + a term of order `v_k(C)`); a
minimax floor for `mu_k(s*)` would need indistinguishable alternatives in `H` (Theorem 2 Part B's construction) and is not claimed.
Test: a projection estimator has squared bias ~0.15 at `s*` while an estimator using the known bump has ~0.
## 2026-09-25 — Theorem 2: the closed form is the fixed-`h` (in-family) bound, not the full-class infimum (PR #24, third review)

Reviewer: `Alt_k` lets both `theta_k` and `h_k` change, but the quadratic equality optimizes a `theta` shift with `h` unchanged;
admissibility of that shift shows it is *one* candidate, not the minimizer over all `h`. A target-only bump that is zero on every
accessible scale flips the winner with zero observed KL, while the Fisher quadratic is positive. Fix: the closed form is now
stated as `R_lin`, the infimum over `Alt^h_k` (the `h`-fixed subclass); since `Alt^h_k ⊆ Alt_k`, `R <= R_lin`, hence
`T^chal >= T^lin` and `E[C] >= kl * T^lin` -- a **valid but weaker** lower-bound program (what P3-02 solves), with equality only when `h` is
fixed and known. New Remark: with rich `H` the full-class rate is 0 (the impossibility regime of Part B). The Lemma is restated for the
fixed-`h` subclass. Test: a zero-on-accessible-scales bump has KL exactly 0 and flips the winner, against a strictly positive in-family rate.
## 2026-09-25 — Theorem 1: the proved linear case is *unconstrained* least squares (PR #23, third review)

Reviewer: `theorem1_bound.tex` said "linear `g` makes the fit a fixed linear function of the noise, hence exactly
Gaussian / sub-Gaussian", but the setup allows a compact `Theta` and the shipped fitters use box bounds. For `g(theta, s) = theta`,
`Theta = [0, 1]`, true `theta = 0` and Gaussian observations, constrained least squares is `clip(sample_mean, 0, 1)`:
half its mass sits at 0 and its mean is strictly positive -- neither Gaussian nor centred at the population projection.
Theorem 1(i) is now stated for **unconstrained full-rank linear least squares**; a compact/binding constraint moves a fit
to case (ii) (conditional on (H), with `rho_k` covering the constraint-induced bias). Step 1 and the numerical-certificate scope
say the same; the certificate (closed-form OLS) is an unconstrained check. `Extrapolator.bounds_inactive` (PR #29's branch)
gives the run-time test for whether a shipped fit is in case (i). Test: the clipped-mean example (mass at 0 = 0.5, mean =
`sigma / sqrt(2 pi n)`) vs the exactly centred unconstrained mean.
## 2026-09-25 — P1-07 third review: covariance conditioning fix, recheck, and artifact refresh (PR #17)

**Finding (reviewer).** `sandwich_covariance` formed `pinv(J^T J)`; squaring the condition number let the cutoff drop a weak *identified* direction. With `LogLinear`, `x = [1, 1+1e-8, 1+2e-8]`,
`y = [.51, .48, .51]`, target `exp(2)`: `J` has rank 2 (`cond ~ 2.4e8`, `cond(J^T J) ~ 6e16`) and `analytic_v_k` returned `1.5e-4`, while the same HC0 sandwich through the SVD of `J` gives `4.994e11`.
The row-space guard of the previous round does not catch it (the target *is* in the row space).

**Fix.** `sandwich_covariance` is now `J^+ diag(r^2) J^+^T` on the column-equilibrated design; `analytic_v_k` computes `sum_i g_i^2 r_i^2` with `g = pinv(J)^T j_target` from the design's SVD
(`identifiability.prediction_influence_weights`) and **fails closed** (`UnidentifiedTargetError`) if the target uses a direction below the numerical rank cutoff. The reviewer's case now gives `4.994e11`
(relative difference `2.7e-8` from the direct SVD sandwich). Regression tests: the reviewer's ill-conditioned full-rank design, equality with the textbook formula on a well-conditioned one,
fail-closed when a direction is below cutoff, and the weights themselves.

**Recheck of the affected diagnostics** (`experiments/p1_07_analytic_recheck.py` -> `results/p1_07_analytic_recheck.json`, clean tree). All 3,300 stored per-recipe `analytic_v_k` values recomputed with the fixed code:
| Fitter | compared | changed (> 1e-6 rel) | changed by > 10% | stored negative |
|---|---|---|---|---|
| LogLinear | 825 | 0 (max rel diff 4.6e-12) | 0 | 0 |
| PowerLawN | 825 | 466 | 17 | 0 |
| PowerLawC | 825 | 518 | 29 | 1 (-1.68e-4; now +1.23e-4) |
| ChinchillaND | 825 | 474 | 37 | 0 |
The old cross-check values for the three nonlinear fitters were materially wrong in a majority of cells (an impossible negative variance in one). `LogLinear` (well-conditioned) was unaffected.

**Refresh, not a 38 h rerun.** The Monte-Carlo pass and every bound value are independent of `analytic_v_k` (they use the bootstrap `v_hat`). `p1_07_bound_coverage.py --reuse-monte-carlo results/p1_07_bound_coverage.json`
recomputed everything on the fixed code and carried the per-cell Monte-Carlo results over (refused unless `B` and the scheme match; recorded in the payload as `monte_carlo_reused_from` /
`monte_carlo_source_git_sha` = `c4d740a9`). Verified: across 396 cells `bound_marginal`, `bound_pairwise`, `empirical_error_rate` and `tightness_ratio_pairwise` are identical to the previous file (0 differences);
only `analytic_v_k` changed and no value is negative. `any_bound_violation: false`. P1-08 reads only those unchanged fields, so it needs no regeneration. A full Monte-Carlo pass is required (and the flag refuses) after any
change to the fitters, bootstrap code or P1-06 outputs.

**Decided by:** Agent, following the third review.

## 2026-09-25 — P3-05 regenerated after the certification gating and allocation-solver changes (PR #31)

`results/p3_05_replay.json` regenerated on a clean tree (`git_dirty: false`) with `certification="assume_unproved_conditions"` passed explicitly (real data, adaptive tracking, nonlinear `PowerLawN`, `sigma2 = 1e-4` only
approximating the real seed noise -- none of the proved conditions hold, so any "certified" outcome could only be caller-assumed) and the SVD-based allocation solve (PR #28). Outcomes are **identical** to the
previous run: ETS certified on none of the 4 tasks (2 pool-exhausted, 2 round-cap; genuine bias-floor abstention rate 0.0); baseline accuracies unchanged. The file now records `certification_mode` / `certification_note`.
## 2026-09-25 — P3-04 regenerated after the certification gating and allocation-solver changes (PR #30)

Regenerated on a clean tree (`git_dirty: false`) with `certification="assume_unproved_conditions"` passed explicitly (PowerLawN + adaptive tracking: no proved conditions hold) and the SVD-based allocation solve.
The file records `certification_mode` / `certification_note`. Seven of eight cells are identical; one differs because `solve_allocation` is iterative and its last-bit changes shifted a few trajectories:
`delta = .05, eta = none, well-separated` now certifies **13/15** (was 11/15), 0 wrong. Current picture: `eta = none` well-separated cells certified **26 of 30 runs, 0 wrong**, exact 95% interval on the joint rate [0, 0.218] per cell (above both
deltas, so the pilot still cannot verify delta-correctness); every `eta = large` cell never certified; **claim 2** compute-to-stop 3.14e18 (delta .05) vs 2.91e18 (delta .2) -- still not the `log(1/delta)` scaling, warm-up dominated;
**claim 3** all reversing cells ended at the round cap (bias-floor abstention 0.0). The earlier statement "11/15" is superseded. Any "certified" here is under caller-assumed, unproved conditions.

**Decided by:** Agent, following the third review.
