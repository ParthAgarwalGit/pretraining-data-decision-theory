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
