# Phase 1 — DataDecide Re-analysis (the headline empirical result)

**Goal:** measure `sigma^2_extrap`, show it explains DataDecide's puzzle, and check
that our error bound predicts the observed ~80% ceiling.

**Compute:** laptop. The full parquet is 693 MB; use `polars` lazy scans, not pandas
`read_parquet` of the whole thing. No GPU needed.

**Why this phase is load-bearing:** it is a complete empirical validation on its own.
If Phase 4 never happens, this plus the theory is still a main-track paper. Treat it
with corresponding care.

---

## Notation used throughout this file

- `k` — a data recipe (an *arm*). `K` ≈ 25.
- `s` — a model scale (parameter count). `s*` — the target scale, **1B**.
- `t` — a downstream task (arc_challenge, mmlu, hellaswag, ...).
- `mu_k(s)` — true performance of recipe `k` at scale `s` on task `t`.
- `y_{k,i}` — the noisy observation from the DataDecide run of recipe `k` at scale `s_i`.
- `k*` — the true best recipe at `s*`; `Delta_k = mu_{k*}(s*) - mu_k(s*)` is the gap.
- `S_fit` — the set of proxy scales an estimator is allowed to see (always excludes `s*`).
- `v_k(C)` — estimation variance of the extrapolated value. `sigma^2_extrap,k` — the
  squared bias of the extrapolation. These are the two terms in our bound.

---

## P1-01 — Canonical analysis frame

**Branch:** `phase1/analysis-frame`

Build `src/pdt/data/frame.py` producing one tidy table, cached as parquet:

`recipe | params_str | params_num | seed | task | step | tokens | compute | metric_name | metric_value | is_final`

Rules baked into the builder (not left to callers):

1. Drop `step == 0` rows (untrained checkpoints).
2. `is_final` = max step within `(recipe, params, seed)`.
3. Parse `metrics` with `ast.literal_eval` (single-quoted Python dict, **not** JSON).
4. Keep a whitelist of metric names: `primary_metric`, `acc_per_char`, `acc_raw`,
   `correct_prob_per_char`, `norm_correct_prob`, `bits_per_byte_corr`.
5. Parse `params_str` to `params_num` (`10M` -> 1e7, `1B` -> 1e9). Fail loudly on an
   unrecognised format rather than guessing.
6. Emit a coverage matrix: for each `(recipe, params, seed, task)` whether a final
   checkpoint exists. **Report holes.** Do not silently drop; a missing cell changes
   which estimators are admissible.

Also record, in `results/p1_01_frame.json`:

- Which sizes have all 3 seeds and which have only `default`. **This matters:** the
  seed names (`default`, `small aux 2`, `small aux 3`) suggest the auxiliary seeds may
  exist only at small sizes. If `s* = 1B` has one seed, our ground truth `mu_k(s*)` has
  irreducible measurement noise that we must estimate a different way (P1-06) and must
  *subtract* when estimating bias. Getting this wrong inflates `sigma^2_extrap`
  and would make our headline result wrong in our favour — be careful here.
- Whether `chinchilla` (token multiplier) varies within a size, and if so whether we
  should condition on it or treat it as a second design axis.

**Definition of done:** cached frame + `results/p1_01_frame.json` with the coverage
matrix and the seed-availability table. Report the seed-availability finding to the PI
in your session summary even though it is not a GATE.

---

## P1-02 — Ground truth at the target scale, and the gaps

**Branch:** `phase1/target-truth`

`experiments/p1_02_target_truth.py` writing `results/p1_02_target.json`:

1. For each task `t`, compute `mu_k(s*)` for all `k` at `s* = 1B` from final
   checkpoints, averaged over available seeds.
2. Identify `k*(t) = argmax_k mu_k(s*)` per task.
3. Compute the gap vector `Delta_k(t)` per task, and the minimum gap `Delta_min(t)`.
4. Report the *effect size*: `Delta_min / sd_seed`. If the top two recipes are within
   noise at 1B, then **no method can identify a winner**, and that fact must be stated
   up front in the paper — it bounds how much of the 20% error is even attributable to
   extrapolation rather than to ties. This is a real risk to the headline framing;
   surface it explicitly.
5. Also compute a secondary target `s* = 530M` (or the largest size below 1B present)
   so that later tasks can use a second, independent target for validation.

**Definition of done:** per-task table of `k*`, gaps, and `Delta_min / sd`. Flag any
task where the winner is ambiguous.

---

## P1-03 — Reproduce DataDecide's single-scale baseline

**Branch:** `phase1/single-scale-baseline`

This is a **reproduction of a published number**. It is the check that our pipeline is
correct. Do not build anything else on top of Phase 1 until this passes.

`experiments/p1_03_single_scale.py` writing `results/p1_03_single_scale.json`:

1. For each proxy size `s_p` in all sizes below `s*`, each task `t`:
   - Pairwise decision accuracy:
     `acc(s_p, t) = mean over pairs (k,k') of 1{ sign(y_k(s_p) - y_k'(s_p)) == sign(mu_k(s*) - mu_k'(s*)) }`.
   - Handle ties explicitly (define and document the tie rule; recommend excluding
     pairs whose target-scale gap is below the seed noise, and reporting both with and
     without that exclusion).
   - Also compute Kendall's tau between the proxy ranking and the target ranking.
2. Aggregate across tasks (macro-average) and report per task.
3. **Target to reproduce:** roughly 80% of comparisons correct at about 150M params.

**If you do not get ~80%:** do not adjust the metric until you do. Investigate in this
order: (a) are you using final checkpoints, (b) is the metric `primary_metric` or
`acc_per_char` — try both and report both, (c) are you averaging seeds or using
`default` only, (d) is the target 1B with the same `chinchilla` multiplier. Report all
four variants and their accuracies. The published number is a target, not a
requirement; a documented discrepancy with an explanation is an acceptable outcome, a
silently tuned match is not.

**Definition of done:** table of decision accuracy by proxy size and task, with the
150M number highlighted, plus the four-variant sensitivity table.

---

## P1-04 — Scaling-law fitters, and reproduce "extrapolation does not win"

**Branch:** `phase1/scaling-fitters`

Build `src/pdt/scaling/` with a common interface:

```python
class Extrapolator(Protocol):
    def fit(self, scales, values, weights=None) -> "Extrapolator": ...
    def predict(self, scale) -> float: ...
    def jacobian(self, scale) -> np.ndarray: ...   # d predict / d theta, needed for v_k(C)
    n_params: int
```

Implement at least these, all with the same interface:

1. `ConstantExtrapolator` — predicts the value at the largest fitted scale. **This is
   the single-scale baseline expressed as an extrapolator**, which is conceptually
   important: single-scale ranking is extrapolation under a degenerate model with
   large level-bias but potentially small *ordering* bias. Say this in the paper.
2. `PowerLawN` — `y = E + A * N^(-alpha)`, fit by non-linear least squares on the task
   metric directly.
3. `PowerLawC` — same in compute `C = 6ND`.
4. `ChinchillaND` — `L = E + A*N^(-alpha) + B*D^(-beta)`, fit to *loss*.
5. `TwoStepLadder` — the Bhagia et al. (arXiv:2412.04403) method DataDecide uses:
   step 1 `compute -> loss`, step 2 `loss -> task metric` through a sigmoid link.
   Fit each step separately. This is the most important comparison because it is
   DataDecide's own baseline.
6. `LogLinear` — `y = a + b*log N`, a deliberately misspecified simple model, used in
   the misspecification ablation (P5-01).

Fitting hygiene, all enforced in code:

- Multi-start optimisation (at least 8 random restarts) with bounded parameters;
  record the number of restarts that converged and the spread of final objectives.
  Scaling-law fits are notoriously multi-modal — a single-start fit is a bug.
- Refuse to fit when `len(S_fit) < n_params + 1` and raise, do not silently regularise.
- Log every failed fit into the results file; never drop failures silently, because
  "the fit failed" is itself evidence about identifiability (Claim 3).

`experiments/p1_04_extrapolation_baselines.py` writing
`results/p1_04_extrapolation.json`:

- For each estimator, each held-out design `S_fit` (all sizes ≤ 150M; ≤ 300M; ≤ 530M),
  each task: fit per recipe, predict at `s*`, compute pairwise decision accuracy
  against the P1-02 ground truth.
- Compare against the single-scale baseline from P1-03 **at matched compute**: the
  compute used by a design `S_fit` is `sum over s in S_fit of 6*N*D`, so an
  extrapolating design must be compared to the single-scale design of the same total
  compute, not to the same largest size. Build a `compute_cost(S_fit)` helper and
  report accuracy-versus-compute curves, which is the "compute-decision frontier"
  DataDecide reports.

**Target to reproduce:** no extrapolation method exceeds the single-scale frontier.

**Definition of done:** an accuracy-versus-compute frontier plot data file where the
single-scale points and the extrapolation points are directly comparable.

---

## GATE-1 — Reproduction checkpoint

**Stop here. Post to the PI:**

1. The reproduced single-scale accuracy at 150M (all four variants) versus the ~80%
   published figure.
2. Whether extrapolation methods beat single-scale at matched compute in our
   reproduction, per method.
3. The ambiguity finding from P1-02: how many tasks have a statistically resolvable
   winner at 1B.
4. Any schema surprises (seed availability, missing cells, count mismatches).

**Ask:** "Do these reproductions look right to you, and should I proceed to the
bias/variance decomposition?" Wait for a reply. If the reproduction failed, propose
options rather than continuing.

---

## P1-05 — Noise-floor estimation

**Branch:** `phase1/noise-floor`

Before we can call anything "bias", we must know the noise. Estimate three variance
components and write `results/p1_05_noise.json`:

1. **Seed variance** `sigma^2_seed(s, t)` — variance across the 3 seeds at each size
   where they exist. If seeds exist only at small sizes, fit `sigma^2_seed` as a
   function of `s` (expect it to shrink with scale) and extrapolate cautiously to `s*`,
   clearly labelling it as an extrapolated noise estimate.
2. **Checkpoint jitter** `sigma^2_ckpt(s, t)` — variance of the metric across the last
   few checkpoints of a single run (using the `step` axis). This captures
   training-noise and is available at *every* size including 1B, which makes it the
   fallback when seeds are missing at `s*`.
3. **Eval sampling noise** `sigma^2_eval(t)` — binomial noise from a finite eval set:
   `p(1-p)/n_instances`. Get `n_instances` per task from
   `allenai/DataDecide-eval-instances`, or from the standard published sizes of each
   benchmark, and record which source you used.

Then define the **target-truth noise** `sigma^2_target(k,t)` used to de-bias the
`sigma^2_extrap` estimate in P1-06, as the combination appropriate to how `mu_k(s*)`
was estimated in P1-02.

**Definition of done:** all three components estimated with their method documented;
a plot-ready table of noise versus scale.

---

## P1-06 — The bias/variance decomposition (the core result)

**Branch:** `phase1/bias-variance`

`experiments/p1_06_decomposition.py` writing `results/p1_06_decomposition.json`.

For each `(extrapolator M, design S_fit, recipe k, task t)`:

1. **Resample.** Build `B >= 200` resamples of the fitting data. Two schemes, run both:
   - *Seed bootstrap* — resample seeds with replacement at each fitted scale (only
     valid where multiple seeds exist).
   - *Parametric/residual bootstrap* — perturb each observation by noise drawn from the
     P1-05 noise model, which works at every size.
2. For each resample `b`, refit and record `mu_hat_k^(b)(s*)`.
3. Compute:
   - `v_hat_k = Var_b[ mu_hat_k^(b)(s*) ]`  — the estimation variance.
   - `bias_hat_k = mean_b[ mu_hat_k^(b)(s*) ] - mu_k(s*)`  — the bias.
   - **`sigma2_extrap_hat_k = max(0, bias_hat_k^2 - v_hat_k / B - sigma2_target(k,t))`**
     — the bias-squared, corrected for the noise in the bootstrap mean *and* for the
     noise in the ground truth. **Do not skip either correction.** Omitting them
     inflates our headline quantity in the direction that flatters the hypothesis.
4. Also compute the **pairwise** versions, which is what the decision actually depends
   on: for each `k != k*`, the difference statistic `D_k = mu_hat_{k*}(s*) - mu_hat_k(s*)`,
   with `v_hat(D_k)` and `bias_hat(D_k)` computed from the *same* bootstrap replicate
   (so the correlation between the two arms' fits is preserved). Correlated errors can
   cancel in the difference — a real and important effect that the marginal version
   misses.
5. Report per task: the distribution of `sigma2_extrap_hat_k` across recipes, its
   median, and the ratio `sigma2_extrap_hat / v_hat` as a function of the compute in
   `S_fit`. **The signature prediction of our theory is that this ratio grows with
   compute**, because `v` shrinks and `sigma2_extrap` does not.
6. Run the same decomposition for `ConstantExtrapolator` (single-scale). Its
   `sigma2_extrap` in *level* will be large; its `sigma2_extrap` for the *pairwise
   difference* is the interesting quantity and is the formal statement of "single-scale
   accepts a fixed proxy gap instead of paying an extrapolation bias".

**Headline outputs:** `sigma2_extrap_hat` per (task, recipe, method, design), and the
`sigma2_extrap / v` versus compute curve. These are the numbers the paper is built on.

**Definition of done:** results JSON plus a written interpretation in
`docs/findings/p1_06.md` of at most one page, stating plainly whether
`sigma2_extrap` is large, small, or task-dependent.

---

## P1-07 — Plug-in bound and its empirical coverage

**Branch:** `phase1/bound-check`

1. Implement in `src/pdt/theory/bound.py`:
   - The **marginal form** from the source document:
     `P_err_bound = sum over k != k* of exp( - Delta_k^2 / (2 * (sigma2_extrap_k + v_k(C))) )`
   - The **pairwise-difference form** using the difference statistics from P1-06 step 4:
     `P_err_bound_pair = sum over k != k* of exp( - Delta_k^2 / (2 * (bias(D_k)^2 + v(D_k))) )`
     Report both. The pairwise form is the one that is actually tight; the marginal form
     is the one stated in the source document. If they differ materially, that is a
     theory refinement to feed back into P2 — flag it.
   - An analytic `v_k(C)` from the delta method:
     `v_k = J^T Sigma_theta J`, where `J = d g(theta,s*) / d theta` (the extrapolator's
     `jacobian`) and `Sigma_theta` is the sandwich covariance of the fit. Cross-check
     it against the bootstrap `v_hat_k` from P1-06 — agreement validates the analytic
     machinery the theory relies on; disagreement is a finding.
2. `experiments/p1_07_bound_coverage.py`:
   - Monte-Carlo the actual selection error: over `>= 500` resamples, run the full
     selection procedure and count `1{k_hat != k*}`.
   - Compare the empirical error rate to both bound forms, per task, per design.
   - Report the **tightness ratio** `bound / empirical`.

**Expected outcome and how to read it:** the bound should hold (ratio >= 1) and be
loose by a constant factor. If it is *violated* (ratio < 1), the theory as stated is
wrong for this setting — that is a critical finding, stop and report it to the PI as
an unscheduled GATE. Do not "fix" it by rescaling constants.

**Definition of done:** coverage table with the tightness ratio; explicit statement of
whether the bound ever fails.

---

## P1-08 — Does the bound predict the 80% ceiling?

**Branch:** `phase1/ceiling-prediction`

This is the paper's money question: *our theory explains DataDecide*.

1. Plug the P1-05/P1-06 estimates into the bound to get a **predicted** decision
   accuracy for (a) single-scale at 150M and (b) each extrapolation method at matched
   compute.
2. Compare predicted versus observed accuracy (from P1-03/P1-04). Report the gap.
3. Produce the central claim in a falsifiable form: "the theory predicts single-scale
   accuracy of X% and extrapolation accuracy of Y% at matched compute; observed X'%
   and Y'%".
4. Do a **counterfactual**: set `sigma2_extrap = 0` and recompute the predicted
   extrapolation accuracy. If the prediction then exceeds single-scale, we have
   demonstrated that the extrapolation-bias term is exactly what causes the failure —
   the cleanest possible version of the result.

**Definition of done:** one table with predicted versus observed, plus the
`sigma2_extrap = 0` counterfactual.

---

## P1-09 — Rank-reversal census

**Branch:** `phase1/rank-reversals`

Evidence for the impossibility regime (Claim 2).

1. For each task and each pair `(k, k')`, determine whether the sign of
   `mu_k(s) - mu_k'(s)` changes anywhere across the size ladder. Classify pairs as
   *stable*, *reversing*, or *within-noise*.
2. Report the fraction of reversing pairs per task, and whether reversals concentrate
   in small-gap pairs (expected) or occur for large gaps too (much more interesting,
   and directly supports the impossibility framing).
3. Kendall tau between the ranking at each size and at `s*`, as a curve in `s`.
4. Identify a small set of concrete **stress-test pairs**: recipes where the small-scale
   winner and the 1B winner genuinely differ with a gap exceeding noise. These become
   the case studies in the paper and the seed set for the P5-04 stress test.

**Definition of done:** reversal statistics table plus a named list of stress-test
recipe pairs with their gaps and the crossing scale.

---

## P1-10 — Secondary ladders: Pythia and OLMo 2

**Branch:** `phase1/secondary-ladders`

Independent replication on ladders we did not tune anything on.

- **Pythia** (Biderman et al. 2023): the deduped versus standard Pile split is a
  genuine `K = 2` data-recipe instance across 8 sizes with 154 checkpoints each.
  Small `K`, but a clean test of `sigma2_extrap` estimation on a different suite.
- **OLMo / OLMo 2** (arXiv:2501.00656): intermediate checkpoints across a size ladder.
- Use **published eval results where they exist**. Only re-run evals locally if the
  numbers are unavailable, and if so, restrict to tasks that fit in 8 GB and record
  that this is our own evaluation, not the authors'.

**Definition of done:** the P1-06 decomposition rerun on at least one secondary ladder,
with a statement of whether the `sigma2_extrap / v` behaviour replicates.

**Note:** if this task balloons (checkpoint downloads are large, evals are slow), cap
it at one ladder and one task family and say so. It is supporting evidence, not the
main result.

---

## P1-11 — Phase 1 figures

**Branch:** `phase1/figures`

Every figure is generated by `src/pdt/viz/` from a `results/*.json`, with no manual
steps, and regenerated by `make figures`. Draft figure list:

- **F1** — decision accuracy versus compute: single-scale frontier against every
  extrapolation method (the reproduction of DataDecide's central finding).
- **F2** — `sigma2_extrap` and `v` versus fitting compute, on a log axis, showing `v`
  shrinking and `sigma2_extrap` flat. **This is the paper's key figure.**
- **F3** — predicted versus observed decision accuracy, with the
  `sigma2_extrap = 0` counterfactual as a third series.
- **F4** — rank-reversal illustration: a handful of recipe curves that cross.
- **F5** — bound tightness: empirical error against bound, per task, on a log-log axis
  with the `y = x` line.

Style rules: colourblind-safe palette, no red/green pairing, legible at 6 cm wide,
vector PDF output, every axis labelled with units, no chart junk.

---

## P1-12 — Publish derived tables and write the Phase 1 memo

**Branch:** `phase1/publish-derived`

1. Push to the HF dataset repo from P0-08: the tidy frame schema, per-task
   `sigma2_extrap` estimates, fitted scaling-law parameters per (recipe, task, method,
   design), and the bound-coverage table. Include a dataset card documenting every
   column and crediting DataDecide under ODC-BY.
2. Write `docs/findings/phase1_memo.md`: at most three pages, no LaTeX, stating what
   we found, which of the two paper framings the evidence supports (large
   `sigma2_extrap` → mechanism explains the puzzle; small `sigma2_extrap` → pivot to
   the allocation speedup), and any threats to validity.
3. Report to the PI with the memo. This is not a formal GATE, but the PI should read it
   before Phase 2 theory drafting hardens around a framing.

**Definition of done:** HF dataset repo populated, memo committed, PI notified.
