# Phase 7 pre-registration (N7-05)

Frozen 2026-11-21, before any confirmatory data below was scored, opened or analysed.
The commit that adds this file is the freeze point. Any later change to this file
follows the deviation rule in section 6.

Numbers that appear here are **thresholds or exploratory values from committed results**:
`results/n7_02_reliability.json`, `results/n7_03_simulation.json` and
`results/n7_06_decomposition.json`. They are not new results.

## 1. What is exploratory and what is confirmatory

- **Exploratory (already examined):** DataDecide aggregate `primary_metric` scores at final
  checkpoints, i.e. everything in N7-02, N7-03 and N7-06. Results on this data are always
  labelled exploratory. The thresholds below were set from it.
- **Confirmatory (not yet examined).** As of the freeze, none of the following has been
  scored or analysed:
  1. **N7-11.** DataDecide final checkpoints (25 recipes × 3 seeds × {150M, 300M, 530M, 1B})
     re-scored on benchmarks that are not in OLMES-10:
     - **SciQ** (test split, 1,000 items; `acc_per_char`);
     - **LAMBADA-openai** (test, 5,153; greedy final-word accuracy);
     - **BLiMP** (all 67 subsets, mean of per-subset raw-log-prob accuracy).

     Formats, scorer and precision (fp16) are those in `src/pdt/eval/` at commit `0dfa334`.
     That code was validated against DataDecide's released log-likelihoods.
  2. **N7-09.** DataDecide per-instance predictions (`allenai/DataDecide-eval-instances`,
     revision `23f3b2e186ca6c39026e3efa00e4af397680c075`). These are downloaded but not
     analysed; only archive file listings were inspected.
  3. **N7-10.** The `allenai/signal-and-noise` `random_seeds` split (1B, 10 init seeds), and
     PolyPythia (14M–410M, 10 seeds) scored with the N7-04 harness.

## 2. Estimands and inference (frozen)

- **Estimator.** `pdt.analysis.decomposition.fit_decomposition`, Gaussian empirical Bayes
  (commit `62048c0` on `phase7/decomposition`, validated in N7-03).
- **CI-bearing estimands.** Each is a fraction of recipe pairs:
  - `da_obs` (DataDecide-style decision accuracy);
  - `da_latent`;
  - `ceiling` = 1 − `target_noise_err`;
  - `reorder` (true cross-scale reordering of recipe means);
  - `proxy_noise_err`.
- **Shares of observed error** are point summaries only. N7-03 found that their intervals
  under-cover (0.82–0.87).
- **Uncertainty.**
  - Recipe-cluster bootstrap, B = 200, 95% percentile intervals.
  - One resample matrix is shared across benchmarks and sizes, so every difference is paired.
  - `target_noise_err` is also reported with 99% intervals, because N7-03 measured its
    coverage at 0.91–0.93.
- **Reliability statistics.** Test–retest, k-seed proxy agreement and held-out-seed
  stratification use `pdt.analysis.reliability` at commit `a3a1f2f`. Bins are |z| < 1,
  1–2, 2–3 and ≥ 3.
- **Off-ladder size.** 750M is excluded from all scale comparisons, because its final
  checkpoints saw fewer tokens than 530M.
- **Benchmark mean.** The mean over the three confirmatory benchmarks is the unweighted
  mean of the per-benchmark estimates, computed on identical resamples.

## 3. Confirmatory hypotheses for N7-11 (new benchmarks)

**Decision rule.** A hypothesis is *supported* only if the criterion holds on the
three-benchmark mean **and** the point estimate has the predicted sign or direction on at
least 2 of the 3 benchmarks. Every hypothesis is reported whatever the outcome.

| ID | Hypothesis | Criterion | Exploratory value (OLMES-10 mean) |
|---|---|---|---|
| C1 | The 3-seed 1B target has a ceiling below 1. | 95% CI lower bound of `target_noise_err` > 0.02 | 0.085 [0.063, 0.109] |
| C2 | True reordering shrinks with proxy scale. | Paired 95% CI of `reorder(150M) − reorder(530M)` lies above 0 | 0.220 vs 0.078 |
| C3 | True reordering persists at 530M. | 95% CI lower bound of `reorder(530M)` > 0 | 0.078 [0.036, 0.118] |
| C4 | Proxies are accurate on target-resolvable pairs. | 3-seed 530M held-out top-bin agreement ≥ 0.90 (point) | 0.952 [0.885, 0.969] |
| C5 | Seeds and scale trade evenly at matched cost. | Paired 95% CI of `acc(300M×2) − acc(530M×1)` lies inside ±0.05 | −0.004 [−0.026, 0.017] |

If a benchmark has fewer than 2 distinct recipe orderings at the target, i.e. it is at
chance for all recipes at 1B, it is reported and dropped from the mean. The drop is
decided from the 1B scores alone, before any proxy is examined.

## 4. Confirmatory analyses for N7-09 (per-instance data)

- **I0 (sanity check, not a hypothesis).** Per (recipe, size, seed, task), the instance
  accuracy under the task's primary normalisation reproduces the frame's `metric_value`
  to within 1e-6. This also fixes the seed-name mapping between the two datasets.
  - Expected: `seed-2` = `default`.
  - `seed-14`/`seed-15` = `small aux 2/3`.
  - At 1B, `seed-4`/`seed-5` = `large aux 2/3`.

  If any cell fails, N7-09 stops and opens a GATE.
- **I1 (estimation).** For each task, the share of the 1B pairwise-gap variance that comes
  from item sampling versus seed noise:
  - item part: a paired bootstrap over items, with the same items for both recipes;
  - seed part: the N7-02 pooled seed variance.

  Reported per task as a median over pairs, with no threshold.
- **I2 (directional).** Define the population-level ceiling as the agreement of the 3-seed
  1B ranking with recipe means over the task's item population. It folds in seed noise
  *and* item sampling.
  - Hypothesis: on the 10-task mean it is lower than the fixed-item ceiling (N7-06).
  - Criterion: the paired 95% CI of the difference lies above 0.
- **I3 (directional).** Paired, item-level comparison (a sign test over items, pooled over
  seeds) resolves more 1B recipe pairs than the seed-only Welch comparison used in P1-09.
  - Criterion: on the 10-task mean, the resolved share at α = 0.05 (Bonferroni over pairs
    within a task) is at least 10 percentage points higher.

## 5. Confirmatory checks for N7-10 (noise model)

- **S1.** Compare the per-task 1B seed SD from the 10 init seeds in `signal-and-noise`
  `random_seeds` with the DataDecide 3-seed pooled 1B seed SD.
  - Criterion: the ratio lies in [0.5, 2] for at least 7 of the overlapping OLMES tasks.
  - This is descriptive only: the recipe and pipeline differ.
- **S2.** PolyPythia, 10 seeds per size (14M–410M), on the N7-11 benchmarks.
  - The 3-seed pooled-SD estimator, averaged over all 3-of-10 subsets, has median absolute
    relative error ≤ 0.30 against the 10-seed SD.
  - Shapiro–Wilk p > 0.01 for seed scores in at least 80% of (size, benchmark) cells.
  - If either fails, the paper reports the noise model as approximate and adds a
    non-Gaussian sensitivity analysis, using N7-03's `hetero_t` generator.

## 6. Deviation rule

- Any departure from sections 2–5 must be recorded in `docs/decisions.md` **before** the
  affected analysis is run. That includes a benchmark that will not load, a checkpoint that
  is missing, or a changed threshold.
- The entry gives the reason. The departure is reported in the paper.
- Analyses not listed here are exploratory and labelled as such.
- A failed hypothesis is a result, not a reason to change the analysis.


## Amendment 1 (2026-11-22): S2 criterion replaced before any PolyPythia data existed

- **Original criterion** (section 5): a median absolute relative error ≤ 0.30 for the
  3-seed SD estimator against the 10-seed SD. It is mis-calibrated: it fails by
  construction.
  - With 3 seeds, even i.i.d. Gaussian noise gives E|s/σ − 1| ≈ 0.39.
  - Against a 10-seed SD over all 3-of-10 subsets, the median over 15 cells is about 0.33
    under the Gaussian null, by Monte Carlo.
  - So a pass or fail would carry no information about the noise model.
- **Replacement (S2a).** The same statistic is now compared with its **exact Monte-Carlo
  null** under i.i.d. Gaussian seed noise, generated by the same subsetting procedure
  (`pdt.analysis.seed_noise.calibrated_sd_check`, 20,000 simulations).
  - It passes when the observed median over cells is at most the null's 95th percentile.
  - Rejection means the 3-seed estimates are worse than Gaussian theory allows, for
    example because of heavy tails or outlier seeds. Seed outliers are documented for
    PolyPythia.
- **Unchanged:** S2b (Shapiro–Wilk) and the consequence of a failure.
- **Approval and timing.** Approved by the PI on 2026-11-22. Committed before any
  PolyPythia output was produced: the N7-11 GPU jobs were running, but nothing had been
  copied back or analysed.
