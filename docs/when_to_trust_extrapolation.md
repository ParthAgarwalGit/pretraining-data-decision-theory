# When to trust extrapolation (and how to pick `eta`)

Practitioner-facing guide for using `pdt.bai.ets.extrapolation_track_and_stop`
(or the `pdt select` CLI) responsibly. See plan/04-phase3-algorithm.md P3-08.

## The one thing that matters most: `eta` and `sigma2` are promises you make, not facts the algorithm discovers

A `"certified"` outcome (default `variance_mode="known_sigma2"`) is a bound on a
**joint** probability, precisely: `P[the algorithm certifies AND the certified
recipe is wrong] <= delta` -- and it holds **only under stated assumptions**
(printed in `result.certificate["assumptions"]`):

- **A1** `sigma2` is a valid sub-Gaussian variance proxy for your oracle's noise.
  It is an *input*, used as the known noise level in the stopping rule. The
  algorithm does not estimate it, so under-stating it makes certification
  over-confident, exactly as under-stating `eta` does.
- **A2** `eta[k] >= sqrt(sigma2_extrap_k)` for every recipe `k`: `eta` must be a
  genuine upper bound on how far each recipe's extrapolated prediction can be
  from its true target-scale value. The algorithm cannot check this -- the whole
  reason extrapolation is needed is that the true target-scale value is never
  observed.
- **A3** the prediction is linear in the observations: exact for `LogLinear`, a
  first-order (delta-method) approximation for the nonlinear power-law fits,
  whose curvature error is *not* covered by `eta` unless you fold it in.
- **A4** the pulled design at each check is independent of the noise being
  certified: exactly true only for the non-adaptive warm-up check; once the
  tracking rule adapts scales to earlier noise it is a heuristic. This is **not
  proved** (a self-normalized confidence sequence would be needed). In the
  simulations recorded in `docs/decisions.md` it did not visibly matter (0
  wrong certifications in 120 adaptive runs), which is supporting evidence,
  not a guarantee.

It is **not** "whenever the algorithm certifies, the recipe is right with
probability `1 - delta`" (that is the *conditional* error rate,
`P[wrong | certified]`, a different and generally larger quantity), and it is
**not** "valid inputs mean certification is never wrong": `delta` is not zero.

**If `eta` under-estimates the true bias, the guarantee is no longer protected -- but
the evidence we have for how badly is weak.** `experiments/p3_06_eta_sensitivity.py`
(`results/p3_06_eta_sensitivity.json`) gave the algorithm `eta=0` (trusting the
extrapolation completely) on a controlled instance with true bias 0.1, 20 independent
runs: **1 of 20 runs certified, and that one was wrong** (joint rate
`P[certified AND wrong]` = 0.05, exact 95% interval about [0.001, 0.25], against
`delta = 0.1`) -- *not* a statistically detectable violation. An earlier version of this
guide quoted "20 of 20" wrong certifications; that number came from 20 copies of a
single noise realization and is withdrawn. The same experiment also shows that with
`eta` at or above half the true bias, no run certified within the round cap (so it says
nothing about calibration there), and that a residual-based `eta` estimate again
under-estimated the bias (~0.05 against 0.1). Over-estimating `eta` (and `sigma2`) is the
safer direction -- it costs more compute (more abstention, more rounds before certifying)
and lowers the false-certification risk, but does **not** mean zero wrong certifications
are possible even with perfectly conservative inputs. A proper calibration study needs
many more runs and rounds than this pilot used.

**A real confidence-machinery bug, found in review and replaced (not patched):**
the original stopping rule estimated each arm's prediction variance from the
fit's own residuals (an HC0 sandwich) and inflated the radius with a Student-t
quantile. Second-round review of PR #29 showed this is not a valid construction:
at a high-leverage observation near the target the fitted residual is almost
forced to zero, so HC0 deletes exactly the uncertainty that dominates the
prediction and no t quantile restores coverage. On a `LogLinear` design with
scales `N=[1, 1.00001, 2]` extrapolated to `N=2.001`, `delta=.01`, the old rule
certified the **wrong** arm in 98 of 200 runs (49% vs 1% requested). The default
rule now uses the known `sigma2` for the variance
(`pdt.theory.bound.known_noise_v_k`), a Gaussian/Chernoff radius, and a union
bound over rounds *and* over all ordered arm pairs (the leader is data-dependent);
the same reproduction certifies 0 of 200 runs. The old residual-based variance
survives only as `variance_mode="hc0_heuristic"`, which returns `"recommended"`
and never `"certified"`. The remaining unproved step is A4 above.

**When in doubt, over-estimate `eta` and `sigma2`, and prefer more replicates per
scale over fewer.**

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
