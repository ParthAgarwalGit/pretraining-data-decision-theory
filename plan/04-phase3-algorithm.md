# Phase 3 — The Algorithm and its Free Validation

**Goal:** implement Extrapolation-Track-and-Stop, and validate it **for free** by
replaying it against DataDecide's already-computed checkpoints.

**Compute:** laptop.

---

## The key idea in this phase

The algorithm's "pull arm `k` at scale `s`" action means *train recipe `k` at size `s`
and evaluate it*. In DataDecide, that result **already exists** for all 25 recipes ×
14 sizes × 3 seeds. So we can run the full adaptive algorithm against real data by
turning each pull into a table lookup, charging the pull its true FLOP cost
`c(s) = 6ND`, and comparing total compute spent against uniform allocation and against
single-scale.

**This gives us a real-data demonstration of the algorithm's compute savings at zero
GPU cost**, before spending a single hour of the PI's cluster. It is the highest
value-per-effort task in the whole project. Phase 4 then confirms the same behaviour on
runs we execute ourselves.

---

## P3-01 — The simulator and the oracle interface

**Branch:** `phase3/oracle-interface`

Define one interface with three implementations, so the algorithm code never knows
which world it is in:

```python
class PullOracle(Protocol):
    def pull(self, recipe: str, scale: Scale, seed: int) -> float: ...  # returns metric
    def cost(self, scale: Scale) -> float: ...                          # FLOPs
    def available_scales(self) -> list[Scale]: ...
```

1. `SyntheticOracle` — draws from a known `g(theta_k, s) + h_k(s) + noise`, with `theta`,
   the perturbation `h`, and the noise model **calibrated to the P1-05/P1-06 estimates**
   so the simulation resembles reality rather than a Gaussian toy.
2. `DataDecideOracle` — table lookup into the P1-01 frame; a repeated pull of the same
   `(recipe, scale, seed)` returns the same value (record how many distinct draws are
   available: 3 seeds, plus nearby checkpoints as pseudo-replicates if needed, and
   document that choice).
3. `LiveTrainingOracle` — Phase 4 only; launches a real run. Stub it now, raising
   `NotImplementedError`, so the interface is fixed before P4 starts.

**Definition of done:** interface fixed, first two oracles implemented and tested,
`cost()` verified against the `compute` column already present in DataDecide's table
(a good independent check of our `6ND` accounting).

---

## P3-02 — Solve the optimal-allocation program

**Branch:** `phase3/optimal-allocation`

Implement the `T*` program from Theorem 2 Part A numerically in
`src/pdt/bai/allocation.py`:

- Concave max-min over allocation weights `w` on `(arm, scale)` pairs subject to a
  compute normalisation.
- Solve with projected subgradient or a standard convex solver; verify the solution
  against a brute-force grid search on small instances (`K = 3`, 3 scales).
- Return both the optimal weights and `T*`.

Report the **shape** of the optimal allocation, which is a genuinely interesting result
for practitioners: does it concentrate on the largest affordable scale, spread across
scales for curvature, or focus pulls on the top-two contenders? Whatever it does,
explain it in one paragraph — this becomes a paper figure.

---

## P3-03 — The algorithm

**Branch:** `phase3/track-and-stop`

`src/pdt/bai/ets.py` — Extrapolation-Track-and-Stop:

- **Sampling rule:** at each round, refit each arm's extrapolator on data so far,
  recompute the plug-in optimal weights from P3-02, and pull the `(arm, scale)` pair
  that is furthest behind its target *compute-weighted* proportion. Include forced
  exploration so every arm gets a minimum number of pulls and the fits stay identified
  (Theorem 3's rank condition must hold at all times — assert it).
- **Stopping rule:** a GLR-style statistic for "the current leader is genuinely best at
  `s*`", with the bias floor `eta` entering the threshold as decided in P2-05. Support
  anytime-valid thresholds.
- **Abstention rule:** if the statistic's achievable ceiling given `eta` cannot reach
  the `1 - delta` level, stop and return
  `Abstain(reason="bias floor", recommended_fallback=single_scale_at(s_rec))`.
- **Outputs:** the recommended recipe or an abstention, the certificate, total compute
  spent, and the full pull log.

Baselines to implement alongside, sharing the oracle:

- `UniformAllocation` — equal compute per arm, spread over a fixed scale ladder.
- `SingleScale(s_p)` — one pull per arm at scale `s_p`, rank, done.
- `SuccessiveHalvingOverScales` — a natural strong baseline reviewers will ask for.
- `FixedLadderExtrapolation` — the DataDecide-style approach: fixed ladder, fit,
  extrapolate, rank.

**Definition of done:** all five run to completion on the synthetic oracle; unit tests
cover the stopping and abstention rules.

---

## P3-04 — Simulation study

**Branch:** `phase3/simulation-study`

`experiments/p3_04_simulation.py` writing `results/p3_04_simulation.json`:

- Sweep: `K` in {5, 10, 25}; `delta` in {0.05, 0.1, 0.2}; misspecification level `eta`
  in {0, small, medium, large}; gap structure {well-separated, close top-two, reversing}.
- For each cell, at least 200 independent runs. Record: empirical error rate (must be
  `<= delta` where the theory says it should be), mean compute to stop, abstention rate.
- **The three claims to check:**
  1. Error rate respects `delta` in the well-specified regime.
  2. Compute-to-stop approaches `T* log(1/delta)` as `delta` shrinks.
  3. In the reversing/impossible regime, the algorithm **abstains** rather than
     confidently returning the wrong recipe — and the baselines confidently return the
     wrong recipe. This contrast is the strongest practical argument in the paper.

---

## P3-05 — Offline replay on DataDecide (the free real-data result)

**Branch:** `phase3/datadecide-replay`

`experiments/p3_05_replay.py` writing `results/p3_05_replay.json`:

1. Instantiate `DataDecideOracle`. Target `s* = 1B`. Available pull scales: everything
   below 1B.
2. Run all five methods with matched `delta`. For each task, record: which recipe was
   selected, whether it matched the true 1B winner, total FLOPs charged, and whether
   the method abstained.
3. Headline table: **compute to a correct decision, by method, by task.**
4. Bootstrap over seeds and over task subsets to attach error bars — a single replay is
   one sample, and reporting it without uncertainty would be exactly the kind of
   overclaim this paper is about.
5. Report the abstention rate on tasks that P1-09 flagged as reversal-heavy. A high
   abstention rate there, paired with high accuracy where it does commit, is the
   result we hope for. **If instead it abstains everywhere, the algorithm is
   practically useless and we must say so** and reframe toward the diagnostic
   contribution.

**Constraint to respect:** the replay must never look at `s* = 1B` data except to score
the final decision. Enforce this in code with a guarded oracle that raises if `s*` is
pulled. Do not rely on discipline; rely on the assertion.

**Definition of done:** headline compute-savings table with error bars, plus the
guarded-oracle test proving no target-scale leakage.

---

## P3-06 — Sensitivity to the bias-floor estimate `eta`

**Branch:** `phase3/eta-sensitivity`

The guarantee is conditional on `eta`. Practitioners will not know it. So:

1. Sweep `eta` from 0 to 3× the P1-06 estimate; plot error rate and compute against
   `eta`.
2. Show what happens when `eta` is **under**-estimated (the dangerous direction:
   overconfident wrong answers) versus over-estimated (safe but wasteful).
3. Propose and test a practical plug-in: estimate `eta` from the arms' own residuals on
   the observed scales, with a small-sample correction. Report how well the plug-in
   version preserves the guarantee.

This section will be the first thing a good reviewer attacks. Get ahead of it.

---

## P3-07 — Figures for Phase 3

**Branch:** `phase3/figures`

- **F6** — compute to correct decision, our algorithm against baselines, on DataDecide
  replay (bar chart with error bars).
- **F7** — the shape of the optimal allocation across scales, as `delta` varies.
- **F8** — error rate and abstention rate against misspecification level `eta`, ours
  against baselines; the crossing point where baselines become confidently wrong.

---

## P3-08 — Reference implementation polish

**Branch:** `phase3/api-polish`

The algorithm is a deliverable in its own right; a practitioner should be able to use
it. Provide:

- A ten-line quickstart in the README: define recipes, define the scale ladder and
  costs, plug in a `pull` callback, get a decision plus certificate.
- A CLI: `pdt select --config configs/my_selection.yaml`.
- Docstrings on every public function stating the guarantee and its conditions.
- `docs/when_to_trust_extrapolation.md` — the practitioner-facing diagnostic: how to
  estimate your own `sigma^2_extrap` and decide between extrapolation and single-scale
  ranking. This is the "directly actionable advice" the source document promises, and
  it is cheap to write once the code exists.
