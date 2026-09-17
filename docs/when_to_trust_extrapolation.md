# When to trust extrapolation (and how to pick `eta`)

Practitioner-facing guide for using `pdt.bai.ets.extrapolation_track_and_stop`
(or the `pdt select` CLI) responsibly. See plan/04-phase3-algorithm.md P3-08.

## The one thing that matters most: `eta` is a promise you make, not a fact the algorithm discovers

Theorem 4's delta-correctness guarantee is a bound on a **joint**
probability, precisely: `P[the algorithm certifies AND the certified
recipe is wrong] <= delta`. It is **not** "whenever the algorithm
certifies, the certified recipe is right with probability `1 - delta`"
(that would be the *conditional* error rate, `P[wrong | certified]`, a
different and generally larger quantity) — and it is **not** "a valid
`eta` means certification is never wrong." Even a genuinely delta-correct
method allows rare errors by construction; `delta` is not zero. What the
guarantee actually promises is that wrong certifications, across the
full random path of the algorithm (including the cases where it
correctly abstains or runs out of rounds instead of certifying), happen
no more than a `delta` fraction of the time.

This guarantee holds **only if** `eta[k] >= sqrt(sigma2_extrap_k)` for
every recipe `k`: `eta` must be a genuine upper bound on how far each
recipe's extrapolated prediction can be from its true target-scale
value. The algorithm does not check this for you. It cannot — the whole
reason extrapolation is needed is that the true target-scale value is
never observed.

**If `eta` under-estimates the true bias, the guarantee gets much
worse, not just "silently fails" in some abstract sense.** This is not a
theoretical nicety: `experiments/p3_06_eta_sensitivity.py`
(`results/p3_06_eta_sensitivity.json`) found that giving the algorithm
`eta=0` (i.e. trusting the extrapolation completely) on a controlled
instance produced a confident, *wrong* certification in 20 out of 20
trials. Over-estimating `eta` is the safer direction — it costs more
compute (more abstention, more rounds before certifying) and reduces the
false-certification risk, but does **not** mean zero wrong
certifications are possible even with a perfectly conservative `eta`.

**A separate, real confidence-machinery bug, found and partially fixed:**
PR #29's review found that `ets.py`'s certification radius, as originally
implemented, plugged in an *estimated* variance as if it were known
exactly — with the shipped `LogLinear` fitter and a small sample, this
produced 91 wrong certifications in 1,000 independent, correctly-specified
trials at `delta=.01` (9.1% actual vs. 1% requested — the delta-correctness
bound itself was violated, not just "eta was wrong"). This has since been
fixed with a Student-t-based radius accounting for the variance estimate's
own degrees of freedom (see `docs/decisions.md`); the same reproduction
now gives a 0.70% actual error rate at `delta=.01` (5,000 trials), under
the requested bound. **This fix is a verified, substantial improvement,
not a proof.** It is documented in `ets.py` itself as a tested heuristic
correction, not a rigorously proven finite-sample guarantee — treat the
delta-correctness promise in this whole section as empirically
well-supported at the sample sizes tested, not as a mathematical
certainty, until a fully rigorous confidence-sequence treatment lands.

**When in doubt, over-estimate `eta`, and prefer more replicates per
scale over fewer** (the confidence-machinery issue above is worst with
very few residual degrees of freedom).

## How to estimate `sigma2_extrap` for your own recipes

Two approaches, in increasing order of rigor and cost:

1. **The residual plug-in** (cheap, one fit per recipe):
   `eta_hat_k = rmse(fit residuals on your observed scales) * sqrt(n /
   (n - p))`, where `n` is the number of (scale, replicate) points and
   `p` is your extrapolator's parameter count — the small-sample
   correction factor is the same one that turns a biased-low
   maximum-likelihood variance estimate into a less-biased one, applied
   here to a bias budget rather than a variance.
   `experiments/p3_06_eta_sensitivity.py::estimate_eta_plugin` implements
   this exactly.

   **Known limitation, found by testing it, not assumed:** in the same
   P3-06 experiment, this plug-in *itself* under-estimated the true bias
   (estimated ~0.04-0.05 against a true bias of 0.1) — because a bump
   that is, by construction, invisible on every observed scale cannot be
   seen by any residual-based estimator computed from those scales. A
   residual plug-in tells you how noisy your fit is *given the
   functional form you assumed*; it cannot tell you how wrong that
   functional form is far outside the range you fit it on. Treat it as
   a floor, not a ceiling, on your true `eta`.

2. **The bootstrap bias/variance decomposition** (more expensive, more
   trustworthy if you have multiple scaling-law fitters or designs to
   compare): `src/pdt/analysis/bootstrap.py` and
   `pdt.analysis.decision_accuracy`'s machinery, as used in this
   project's own P1-06 task, resample across scales/seeds and compare
   fits at different maximum training scales to estimate how much a
   recipe's ranking is likely to move as you extrapolate further. This
   is the same machinery that produced this project's own calibration
   numbers (`results/p1_06_decomposition.json`: median bias magnitude
   ranges from about 0.048 to about 0.17 across 6 fitters and 3 scale
   designs on real DataDecide data — a useful sanity-check range if your
   own recipes and metric are broadly similar in scale).

Neither approach is a substitute for the other; if you can afford it,
compute both and use the larger.

## When to skip extrapolation entirely and just rank at one scale

This project's own Phase 1 finding (`paper/figures/f1_accuracy_vs_compute.pdf`,
reproducing DataDecide's own result): **no scaling-law extrapolation
method beat the single-scale frontier at matched compute**, across every
fitter and design tested. If you don't have a specific reason to believe
your recipes' rankings will *reverse* between your affordable training
scale and your target scale (see `paper/sections/theorem3_identifiability.tex`'s
minimax-rate result, and `results/p1_09_rank_reversals.json` for how to
check whether your own task looks reversal-heavy), single-scale ranking
at the largest scale you can afford is a reasonable, simpler default —
`pdt.bai.ets.single_scale_recommendation`.

Extrapolation earns its complexity specifically when:
- you have reason to expect rank instability across scale (P1-09-style
  reversal-heavy evaluation tasks), and
- you can afford the (real, not free) compute cost of the adaptive
  algorithm reaching a decision — see the caveat below.

## A real, current limitation: genuine resolution can be expensive

`experiments/p3_05_replay.py` (real DataDecide data) and
`experiments/p3_06_eta_sensitivity.py` (controlled synthetic instances)
both found that `extrapolation_track_and_stop` can fail to reach *either*
a certified decision or a genuine Theorem-4 abstention within a modest
round budget — it simply runs out of rounds (`certificate["reason"]`
will say so explicitly; check it, don't just look at `outcome`). This
was found consistently across three independent experiments this
session (see `docs/decisions.md`, 2026-09-11 entries for P3-04/05/06).
If you see this in practice, it means the instance needs either more
compute (`max_rounds`) or a fundamentally easier problem (fewer, more
separated recipes; a target scale closer to your affordable training
range) — not that the algorithm's guarantee has failed. `outcome=
"abstained"` with `certificate["reason"]` starting with `"max_rounds"`
is an honest "did not resolve," not a claim of anything.
