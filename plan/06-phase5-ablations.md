# Phase 5 — Ablations and Stress Tests

Each ablation maps to a specific claim. If an ablation does not test a claim, drop it —
reviewers reward targeted ablations and ignore padding.

| Task | Tests | Needs Phase 4? |
|---|---|---|
| P5-01 | Claim 1 (bias term) | No |
| P5-02 | Claim 3 (identifiability) | No |
| P5-03 | Metric choice | No |
| P5-04 | Claim 2 (impossibility) | No |
| P5-05 | Robustness of the whole pipeline | No |
| P5-06 | Target-scale generalisation | No |
| P5-07 | Continuous mixtures (extension) | No |
| P5-08 | Live confirmation | Yes |
| P5-09 | Cross-suite agreement | Yes |
| P5-10 | Cost model accuracy | Yes |

---

## P5-01 — Correctly specified versus misspecified scaling forms

**Branch:** `phase5/misspecification`
**Tests: Claim 1 — the bias term is what breaks extrapolation.**

1. On the synthetic oracle where the truth *is* in the family, verify decision error
   goes to zero with compute and that extrapolation beats single-scale. This is the
   positive control, and it matters: it proves our pipeline is capable of showing
   extrapolation winning, so the DataDecide result is not an artefact of our code.
2. Inject controlled misspecification of increasing magnitude and trace the crossover
   point at which single-scale overtakes extrapolation.
3. On DataDecide, compare fitted families of increasing flexibility (log-linear,
   2-parameter power law, 3-parameter Chinchilla, two-step ladder). Report
   `sigma^2_extrap` and decision accuracy for each. **The expected and interesting
   result is non-monotone:** more flexible families reduce bias but inflate variance, so
   there is an optimum. If you find that, it is a paper figure.

---

## P5-02 — Number and spacing of scales

**Branch:** `phase5/scale-design`
**Tests: Claim 3 — identifiability.**

1. Vary `|S_fit|` from 2 to the maximum available, at matched total compute (fewer,
   larger scales versus more, smaller ones). Matched compute is essential — otherwise
   this measures budget, not design.
2. Vary the spacing: clustered scales versus log-spread scales.
3. Confirm the predicted failure at 2 scales for a 3-parameter family, and confirm the
   theoretical optimal spacing from P2-04 beats naive uniform-in-log spacing.

**Practitioner deliverable:** a concrete recommendation — "for a target of `X`, train at
these `n` sizes" — which belongs in the abstract if the result is clean.

---

## P5-03 — Continuous proxies versus discrete accuracy

**Branch:** `phase5/metric-choice`

DataDecide reports that continuous likelihood proxies make several benchmarks highly
predictable at the target scale with a tiny fraction of the compute. Test this against
our bound.

1. Rerun the P1-06 decomposition with `correct_prob_per_char`, `norm_correct_prob`, and
   `bits_per_byte_corr` in place of accuracy.
2. Decompose *why* they help: lower `sigma^2_extrap` (the metric is better behaved
   across scale), lower `v` (less eval-sampling noise), or both. The decomposition is
   the contribution here — "continuous metrics are better" is already known; *which
   term shrinks* is not.
3. Check whether a continuous proxy that ranks well at small scale still ranks the same
   winner as the discrete metric at `s*`. If it does not, continuous proxies buy
   predictability of the wrong quantity, which is a caveat worth stating plainly.

---

## P5-04 — Rank-reversal stress test

**Branch:** `phase5/rank-reversal`
**Tests: Claim 2 — the impossibility regime.**

1. Use the concrete reversing pairs identified in P1-09 to construct restricted
   `K`-armed instances that are provably or empirically in the impossible regime.
2. Run every method. The expected result: baselines commit confidently to the wrong
   recipe; our algorithm abstains and recommends the single-scale fallback.
3. Quantify: on reversal-heavy instances, what is each method's error rate, and what is
   ours *conditional on committing*?
4. Construct semi-synthetic instances that interpolate between stable and reversing, to
   exhibit the phase transition as a curve rather than an anecdote. A visible phase
   transition is the single most persuasive figure available for Claim 2.

---

## P5-05 — Robustness of the analysis pipeline

**Branch:** `phase5/robustness`

Vary every arbitrary choice made in Phase 1 and confirm the headline conclusions
survive:

- Metric: `primary_metric` versus `acc_per_char` versus `acc_raw`
- Seed handling: average versus `default`-only versus per-seed bootstrap
- Tie rule at the target scale
- Final checkpoint versus a mean over the last `n` checkpoints
- Bootstrap `B` from 100 to 2000
- Which tasks are included in the macro-average, and a leave-one-task-out sweep

Output a single robustness table: the headline number under each variation. If a
conclusion flips under a defensible variation, **say so in the paper**. Reviewers find
these; finding it yourself and reporting it is strictly better.

---

## P5-06 — Target-scale generalisation

**Branch:** `phase5/target-generalisation`

Repeat the core analysis with `s*` set to 530M, 300M, and 150M rather than 1B, fitting
only on smaller scales in each case. Questions:

- Does `sigma^2_extrap` grow with the extrapolation distance `s*/max(S_fit)`? Theory
  says it should. Measure the relationship and fit it — a usable formula for "how far
  can I safely extrapolate" would be a genuinely useful practitioner artefact.
- Is the ~80% decision-accuracy ceiling specific to the 1B target, or a general
  property at fixed extrapolation ratio?

---

## P5-07 — Continuous mixture extension (optional, only if time permits)

**Branch:** `phase5/continuous-mixtures`

Answers the anticipated reviewer objection that labs optimise continuous mixture weights
rather than choosing among `K` discrete recipes.

Scope it small: a linear or GP model over the mixture simplex, the analogous bias term,
and a simulation demonstrating that the same bias/variance phenomenon appears. Do **not**
attempt a full theory here. One paragraph in the paper plus a simulation figure is the
right size. If time is short, cut this and state the extension without experiments.

---

## P5-08 — Live confirmation of the ablations (requires Phase 4)

**Branch:** `phase5/live-confirmation`

Repeat P5-01 and P5-02 on the Phase 4 runs, at whatever scale the funded tier allows.

---

## P5-09 — Cross-suite agreement (requires Phase 4)

**Branch:** `phase5/cross-suite`

Compare `sigma^2_extrap` estimated on DataDecide, on the secondary ladders from P1-10,
and on our own Phase 4 runs. Report agreement or disagreement with error bars. Three
independent estimates agreeing is the strongest possible support for the mechanism.

---

## P5-10 — Cost-model accuracy (requires Phase 4)

**Branch:** `phase5/cost-model`

The whole allocation theory prices pulls with `C ≈ 6ND`. Check that against measured
GPU-hours from Phase 4. If real cost deviates systematically (small models are
memory-bound and get poor MFU, so `6ND` likely *understates* their relative cost), then
the optimal allocation computed with `6ND` is not optimal in wall-clock terms. Report
the discrepancy and recompute the optimal allocation under the measured cost model. This
is a small task with a disproportionately practical payoff.
